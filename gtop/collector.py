from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

from .accounting import process_jobs
from .constants import (
    ACTIVE_JOB_STATES,
    DEFAULT_TIMEOUT,
    SACCT_COMMAND,
    NODES_COMMAND,
)
from .models import ClusterState
from .partitions import normalize_partition_name, partition_names
from .runner import Command, CommandResult, CommandRunner, SubprocessRunner
from .slurm import parse_jobs, parse_nodelist, parse_nodes


@dataclass(frozen=True)
class CollectionOptions:
    sacct_command: Command = SACCT_COMMAND
    # Live jobs from slurmctld, merged over sacct's records (squeue wins).
    squeue_command: Optional[Command] = None
    timeout: int = DEFAULT_TIMEOUT
    gpu_only: bool = False
    partition_filter: Optional[tuple[str, ...]] = None
    store_allocations: bool = True
    allow_empty_servers: bool = False


class GTopError(RuntimeError):
    pass


class CommandExecutionError(GTopError):
    def __init__(self, command_name: str, result: CommandResult):
        super().__init__(f"{command_name} command failed")
        self.command_name = command_name
        self.result = result


class ClusterParseError(GTopError):
    pass


class NoMatchingServersError(GTopError):
    pass


def collect_cluster_state(
    *,
    runner: Optional[CommandRunner] = None,
    options: Optional[CollectionOptions] = None,
    stderr_console: Optional[Any] = None,
) -> ClusterState:
    active_options = options or CollectionOptions()
    active_runner = runner or SubprocessRunner()
    commands = {"scontrol": NODES_COMMAND, "sacct": active_options.sacct_command}
    if active_options.squeue_command is not None:
        commands["squeue"] = active_options.squeue_command
    results: dict[str, CommandResult] = {}
    parsed: dict[str, Any] = {}
    parse_errors: dict[str, ValueError] = {}
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {
            executor.submit(active_runner.run, command, active_options.timeout): name
            for name, command in commands.items()
        }
        # Parse each output as it arrives, while slower commands still run.
        for future in as_completed(futures):
            name = futures[future]
            result = results[name] = future.result()
            if result.returncode != 0:
                continue
            try:
                parsed[name] = (
                    parse_nodes(result.stdout)
                    if name == "scontrol"
                    else parse_jobs(result.stdout, states=ACTIVE_JOB_STATES)
                )
            except ValueError as error:
                parse_errors[name] = error
    # Report problems in a fixed order, whatever order the commands finished in.
    for name in commands:
        if results[name].returncode != 0:
            raise CommandExecutionError(name, results[name])
    for name in commands:
        if name in parse_errors:
            raise ClusterParseError(str(parse_errors[name])) from parse_errors[name]

    servers = parsed["scontrol"]
    if active_options.partition_filter:
        scope = set(active_options.partition_filter)
        servers = {
            name: server
            for name, server in servers.items()
            if server.partitions & scope
        }
    if not servers and not active_options.allow_empty_servers:
        if active_options.partition_filter:
            partitions = ", ".join(active_options.partition_filter)
            raise NoMatchingServersError(
                f"No servers found in partition scope: {partitions}."
            )
        raise ClusterParseError("Failed to parse any nodes from scontrol output.")
    if active_options.gpu_only:
        servers = {
            name: server
            for name, server in servers.items()
            if server.gpu.type != "null"
        }
        if not servers and not active_options.allow_empty_servers:
            raise NoMatchingServersError("No servers found matching the criteria.")

    active_jobs_by_id = {job.job_id: job for job in parsed["sacct"]}
    active_jobs_by_id.update((job.job_id, job) for job in parsed.get("squeue", ()))
    jobs = list(active_jobs_by_id.values())
    try:
        if active_options.partition_filter:
            partitions = {
                normalize_partition_name(partition)
                for partition in active_options.partition_filter
            }
            scoped_jobs = []
            for job in jobs:
                nodes = parse_nodelist(job.nodelist)
                # Shared partitions can allocate jobs on the same physical nodes.
                if nodes:
                    if not any(node in servers for node in nodes):
                        continue
                elif not partitions.intersection(partition_names(job.partition)):
                    continue
                scoped_jobs.append(job)
            jobs = scoped_jobs
        process_jobs(
            jobs,
            servers,
            stderr_console=stderr_console,
            store_allocations=active_options.store_allocations,
        )
    except ValueError as error:
        raise ClusterParseError(str(error)) from error

    return ClusterState(servers=servers, jobs=jobs)

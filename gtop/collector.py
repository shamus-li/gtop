from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .accounting import process_jobs
from .command_options import set_value_option
from .constants import (
    ACTIVE_JOB_STATES,
    DEFAULT_TIMEOUT,
    SACCT_COMMAND,
    SINFO_COMMAND,
)
from .models import ClusterState
from .partitions import normalize_partition_name, partition_names
from .runner import Command, CommandResult, CommandRunner, run_commands
from .slurm import parse_jobs, parse_nodelist, parse_sinfo


@dataclass(frozen=True)
class CollectionOptions:
    sinfo_command: Command = SINFO_COMMAND
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


def _sinfo_command_for_partitions(
    command: Command,
    partitions: tuple[str, ...],
) -> Command:
    # sinfo rejects -a with -p; named partitions show even if restricted.
    return set_value_option(
        tuple(token for token in command if token not in ("-a", "--all")),
        ("-p", "--partition", "--partitions"),
        ",".join(partitions),
    )


def collect_cluster_state(
    *,
    runner: Optional[CommandRunner] = None,
    options: Optional[CollectionOptions] = None,
    stderr_console: Optional[Any] = None,
) -> ClusterState:
    active_options = options or CollectionOptions()
    scoped_sinfo_command = (
        _sinfo_command_for_partitions(
            active_options.sinfo_command, active_options.partition_filter
        )
        if active_options.partition_filter
        else active_options.sinfo_command
    )
    commands = {"sinfo": scoped_sinfo_command, "sacct": active_options.sacct_command}
    if active_options.squeue_command is not None:
        commands["squeue"] = active_options.squeue_command
    results = run_commands(
        commands,
        timeout=active_options.timeout,
        runner=runner,
    )

    sinfo_result = results["sinfo"]
    sacct_result = results["sacct"]
    if sinfo_result.returncode != 0:
        raise CommandExecutionError("sinfo", sinfo_result)
    if sacct_result.returncode != 0:
        raise CommandExecutionError("sacct", sacct_result)
    squeue_result = results.get("squeue")
    if squeue_result is not None and squeue_result.returncode != 0:
        raise CommandExecutionError("squeue", squeue_result)

    try:
        servers = parse_sinfo(sinfo_result.stdout)
    except ValueError as error:
        raise ClusterParseError(str(error)) from error
    if not servers and not active_options.allow_empty_servers:
        if active_options.partition_filter:
            partitions = ", ".join(active_options.partition_filter)
            raise NoMatchingServersError(
                f"No servers found in partition scope: {partitions}."
            )
        raise ClusterParseError("Failed to parse any servers from sinfo output.")
    if active_options.gpu_only:
        servers = {
            name: server
            for name, server in servers.items()
            if server.gpu.type != "null"
        }
        if not servers and not active_options.allow_empty_servers:
            raise NoMatchingServersError("No servers found matching the criteria.")

    try:
        active_jobs_by_id = {
            job.job_id: job
            for job in parse_jobs(
                sacct_result.stdout,
                states=ACTIVE_JOB_STATES,
            )
        }
        if squeue_result is not None:
            active_jobs_by_id.update(
                (job.job_id, job)
                for job in parse_jobs(squeue_result.stdout, states=ACTIVE_JOB_STATES)
            )
    except ValueError as error:
        raise ClusterParseError(str(error)) from error
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

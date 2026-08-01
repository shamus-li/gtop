from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rich.text import Text

from .accounting import process_jobs
from .command_options import set_value_option
from .constraints import compile_constraint
from .constants import (
    ACTIVE_JOB_STATES,
    DEFAULT_TIMEOUT,
    SACCT_COMMAND,
    SINFO_COMMAND,
)
from .models import ClusterState
from .partitions import normalize_partition_name, partition_names
from .runner import Command, CommandResult, CommandRunner, run_commands
from .slurm import parse_jobs, parse_sinfo


@dataclass(frozen=True)
class CollectionOptions:
    sinfo_command: Command = SINFO_COMMAND
    sacct_command: Command = SACCT_COMMAND
    timeout: int = DEFAULT_TIMEOUT
    parallel: bool = True
    gpu_only: bool = False
    partition_filter: Optional[tuple[str, ...]] = None
    constraint: Optional[str] = None
    debug: bool = False
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
    return set_value_option(
        command,
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
    constraint = (
        active_options.constraint.strip() if active_options.constraint else None
    )
    matches = compile_constraint(constraint) if constraint is not None else None
    scoped_sinfo_command = (
        _sinfo_command_for_partitions(
            active_options.sinfo_command, active_options.partition_filter
        )
        if active_options.partition_filter
        else active_options.sinfo_command
    )
    results = run_commands(
        {"sinfo": scoped_sinfo_command, "sacct": active_options.sacct_command},
        timeout=active_options.timeout,
        runner=runner,
        parallel=active_options.parallel,
    )

    sinfo_result = results["sinfo"]
    sacct_result = results["sacct"]
    if sinfo_result.returncode != 0:
        raise CommandExecutionError("sinfo", sinfo_result)
    if sacct_result.returncode != 0:
        raise CommandExecutionError("sacct", sacct_result)

    if active_options.debug and stderr_console is not None:
        stderr_console.print(
            Text(
                f"sinfo returned {len(sinfo_result.stdout.splitlines())} lines",
                style="dim",
            )
        )
        stderr_console.print(
            Text(
                f"sacct returned {len(sacct_result.stdout.splitlines())} lines",
                style="dim",
            )
        )

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

    if matches is not None:
        servers = {
            name: info for name, info in servers.items() if matches(info.features)
        }
        if not servers and not active_options.allow_empty_servers:
            raise NoMatchingServersError(
                f"No servers found matching constraint '{constraint}'."
            )

    try:
        active_jobs_by_id = {
            job.job_id: job
            for job in parse_jobs(sacct_result.stdout)
            if job.state in ACTIVE_JOB_STATES
        }
    except ValueError as error:
        raise ClusterParseError(str(error)) from error
    jobs = list(active_jobs_by_id.values())
    if active_options.partition_filter:
        partitions = {
            normalize_partition_name(partition)
            for partition in active_options.partition_filter
        }
        jobs = [
            job
            for job in jobs
            if partitions.intersection(partition_names(job.partition))
        ]
    try:
        process_jobs(
            jobs,
            servers,
            debug_enabled=active_options.debug,
            stderr_console=stderr_console,
            store_allocations=active_options.store_allocations,
        )
    except ValueError as error:
        raise ClusterParseError(str(error)) from error

    return ClusterState(servers=servers, jobs=jobs)

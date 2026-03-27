from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rich.text import Text

from .accounting import process_jobs
from .constraints import matches_constraint
from .constants import DEFAULT_TIMEOUT, SACCT_COMMAND, SINFO_COMMAND
from .models import ClusterState
from .runner import CommandResult, CommandRunner, run_commands
from .slurm import parse_jobs, parse_sinfo


@dataclass(frozen=True)
class CollectionOptions:
    sinfo_command: str = SINFO_COMMAND
    sacct_command: str = SACCT_COMMAND
    timeout: int = DEFAULT_TIMEOUT
    parallel: bool = True
    gpu_only: bool = False
    constraint: Optional[str] = None
    debug: bool = False
    store_users: bool = True


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
    results = run_commands(
        {"sinfo": active_options.sinfo_command, "sacct": active_options.sacct_command},
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

    servers = parse_sinfo(sinfo_result.stdout, active_options.gpu_only)
    if not servers:
        raise (
            NoMatchingServersError("No servers found matching the criteria.")
            if active_options.gpu_only
            else ClusterParseError("Failed to parse any servers from sinfo output.")
        )

    if active_options.constraint:
        filtered = {
            name: info
            for name, info in servers.items()
            if matches_constraint(
                info.features,
                active_options.constraint.strip(),
                stderr_console=stderr_console,
            )
        }
        if not filtered:
            raise NoMatchingServersError(
                f"No servers found matching constraint '{active_options.constraint.strip()}'."
            )
        servers = filtered

    jobs = parse_jobs(sacct_result.stdout)
    process_jobs(
        jobs,
        servers,
        debug_enabled=active_options.debug,
        stderr_console=stderr_console,
        store_users=active_options.store_users,
    )

    return ClusterState(servers=servers, jobs=jobs)

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Any, Optional

from rich.text import Text

from .accounting import process_jobs
from .constraints import matches_constraint
from .constants import DEFAULT_TIMEOUT, SACCT_COMMAND, SINFO_COMMAND, SINFO_FEATURES_COMMAND
from .models import ClusterState
from .runner import CommandResult, CommandRunner, run_commands
from .slurm import parse_features_field, parse_jobs, parse_sinfo


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


def _override_command_option(command: str, option_names: tuple[str, ...], value: str) -> str:
    tokens = shlex.split(command)
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in option_names:
            skip_next = True
            continue
        if any(token.startswith(f"{name}=") for name in option_names):
            continue
        filtered.append(token)
    filtered.append(f"{option_names[0]}={value}")
    return shlex.join(filtered)


def _sinfo_command_for_nodes(command: str, nodes: list[str]) -> str:
    return _override_command_option(command, ("-n", "--nodes"), ",".join(nodes))


def _constraint_matching_nodes(
    output: str,
    *,
    constraint: str,
    stderr_console: Optional[Any] = None,
) -> list[str]:
    matching_nodes: list[str] = []
    for raw_line in output.strip().splitlines():
        if not raw_line.strip():
            continue
        node_name, _, features_raw = raw_line.partition("|")
        if not node_name:
            continue
        if matches_constraint(
            parse_features_field(features_raw.strip()),
            constraint.strip(),
            stderr_console=stderr_console,
        ):
            matching_nodes.append(node_name.strip())
    return matching_nodes


def collect_cluster_state(
    *,
    runner: Optional[CommandRunner] = None,
    options: Optional[CollectionOptions] = None,
    stderr_console: Optional[Any] = None,
) -> ClusterState:
    active_options = options or CollectionOptions()
    constraint = active_options.constraint.strip() if active_options.constraint else None
    constraint_fast_path = (
        constraint is not None
        and active_options.sinfo_command == SINFO_COMMAND
    )

    if constraint_fast_path:
        assert constraint is not None
        feature_results = run_commands(
            {"sinfo_features": SINFO_FEATURES_COMMAND},
            timeout=active_options.timeout,
            runner=runner,
            parallel=False,
        )
        feature_result = feature_results["sinfo_features"]
        if feature_result.returncode != 0:
            raise CommandExecutionError("sinfo", feature_result)
        matching_nodes = _constraint_matching_nodes(
            feature_result.stdout,
            constraint=constraint,
            stderr_console=stderr_console,
        )
        if not matching_nodes:
            raise NoMatchingServersError(
                f"No servers found matching constraint '{constraint}'."
            )
        results = run_commands(
            {
                "sinfo": _sinfo_command_for_nodes(active_options.sinfo_command, matching_nodes),
                "sacct": active_options.sacct_command,
            },
            timeout=active_options.timeout,
            runner=runner,
            parallel=active_options.parallel,
        )
    else:
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

    if constraint is not None and not constraint_fast_path:
        filtered = {
            name: info
            for name, info in servers.items()
            if matches_constraint(
                info.features,
                constraint,
                stderr_console=stderr_console,
            )
        }
        if not filtered:
            raise NoMatchingServersError(
                f"No servers found matching constraint '{constraint}'."
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

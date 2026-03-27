from __future__ import annotations

import argparse
import getpass
import json
import shlex
import sys
from typing import Any, Optional, Sequence, Set

from rich.console import Console
from rich.text import Text

from .accounting import (
    build_cluster_summary,
    build_top_users_summary,
    project_servers_for_users,
)
from .collector import (
    ClusterParseError,
    CollectionOptions,
    CommandExecutionError,
    NoMatchingServersError,
    collect_cluster_state,
)
from .constraints import matches_constraint
from .constants import (
    DEFAULT_TIMEOUT,
    EXIT_COMMAND_ERROR,
    EXIT_NO_MATCHES,
    EXIT_PARSE_ERROR,
    EXIT_SUCCESS,
    JOBS_DEFAULT_STATES,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
    SORT_CHOICES,
)
from .render import (
    _display_gpu_type,
    build_json_payload,
    print_filtered_users,
    render_jobs_view,
    print_summary,
    render_table,
    visible_servers,
)
from .runner import CommandRunner, SubprocessRunner
from .slurm import parse_jobs, parse_nodelist, parse_sinfo


def _write_json_output(json_text: str, console: Optional[Any]) -> None:
    if console is None:
        sys.stdout.write(json_text)
        sys.stdout.write("\n")
        return

    if isinstance(console, Console):
        console.print(json_text, markup=False, soft_wrap=True)
        return

    console.print(json_text)


def _job_states_arg(value: str) -> tuple[str, ...]:
    states = tuple(
        state.strip().upper()
        for state in value.split(",")
        if state.strip()
    )
    if not states:
        raise argparse.ArgumentTypeError("job states must not be empty")
    return states


def _override_command_option(command: str, option_names: Sequence[str], value: str) -> str:
    existing = _command_option_value(command, option_names)
    if existing == value:
        return command
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


def _remove_command_options(command: str, option_names: Sequence[str]) -> str:
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
    return shlex.join(filtered)


def _command_option_value(command: str, option_names: Sequence[str]) -> Optional[str]:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token in option_names and index + 1 < len(tokens):
            return tokens[index + 1]
        for name in option_names:
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def _jobs_sacct_command(
    command: str,
    *,
    states: Sequence[str],
    users: Optional[Set[str]] = None,
    partition: Optional[str] = None,
    apply_states: bool = True,
) -> str:
    updated = command
    if apply_states:
        updated = _override_command_option(updated, ("--state",), ",".join(states))
    else:
        updated = _remove_command_options(updated, ("--state",))
    if users:
        updated = _override_command_option(updated, ("--user", "--users"), ",".join(sorted(users)))
    if partition:
        updated = _override_command_option(updated, ("--partition", "--partitions"), partition)
    return updated


def _jobs_sinfo_command(command: str, *, nodes: Sequence[str]) -> str:
    return _override_command_option(command, ("-n", "--nodes"), ",".join(nodes))


def _filtered_jobs(
    jobs: Sequence[Any],
    *,
    target_users: Optional[Set[str]],
    partition_filter: Optional[Sequence[str]],
    states: Sequence[str],
) -> list[Any]:
    state_set = {state.upper() for state in states}
    filtered = []
    partition_needles = (
        {partition.lower() for partition in partition_filter}
        if partition_filter
        else None
    )
    for job in jobs:
        if target_users and job.user not in target_users:
            continue
        if partition_needles:
            haystacks = (job.partition.lower(), job.nodelist.lower())
            if not any(
                needle in haystack
                for needle in partition_needles
                for haystack in haystacks
            ):
                continue
        if job.state.upper() not in state_set:
            continue
        filtered.append(job)
    return sorted(
        filtered,
        key=lambda job: (
            job.user,
            job.partition,
            job.state,
            job.job_id,
        ),
    )


def _filter_jobs_by_constraint(
    jobs: Sequence[Any],
    resolved_servers: dict[str, Any],
    *,
    constraint: str,
    stderr_console: Optional[Any] = None,
) -> list[Any]:
    matching_nodes = {
        name
        for name, server in resolved_servers.items()
        if matches_constraint(
            server.features,
            constraint.strip(),
            stderr_console=stderr_console,
        )
    }
    filtered_jobs = []
    for job in jobs:
        assigned_nodes = [
            node
            for node in parse_nodelist(job.nodelist)
            if node and node.upper() not in {"N/A", "(NULL)", "NONE"}
        ]
        if not assigned_nodes:
            filtered_jobs.append(job)
            continue
        if any(node in matching_nodes for node in assigned_nodes):
            filtered_jobs.append(job)
    return filtered_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display SLURM cluster usage with detailed node and resource information",
        epilog=(
            "Legend:\n"
            "  magenta = priority   cyan = gpu   blue = default   dim = free\n"
            "  counts are free/total, then priority/gpu/default"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    core = parser.add_argument_group("Core options")
    debug = parser.add_argument_group("Debug options")
    core.add_argument(
        "-j",
        "--jobs",
        action="store_true",
        help="Show a job list view with account-style columns",
    )
    core.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show node-by-node details instead of the default GPU-type summary",
    )
    core.add_argument("-u", "--users", nargs="+", help="Filter by user netids")
    core.add_argument(
        "--me",
        action="store_true",
        help="Filter to your own jobs",
    )
    core.add_argument(
        "-U",
        "--top-users",
        nargs="?",
        const=25,
        type=int,
        help="Show only the largest users (default: 25)",
    )
    core.add_argument(
        "--constraint",
        help="Filter nodes by features matching the constraint expression (e.g., gpu-high)",
    )
    core.add_argument(
        "-p",
        "--partition",
        nargs="+",
        help="Filter jobs by partition in --jobs mode",
    )
    core.add_argument(
        "--states",
        type=_job_states_arg,
        default=JOBS_DEFAULT_STATES,
        help="Comma-separated job states for --jobs (default: RUNNING,PENDING,REQUEUED)",
    )
    core.add_argument(
        "--sort",
        choices=SORT_CHOICES,
        default="feature",
        help="Sort nodes by the specified attribute",
    )
    core.add_argument(
        "-s",
        "--shard",
        action="store_true",
        help="Display sharded GPU information instead of total GPU count",
    )
    core.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a Rich table",
    )
    debug.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel command execution"
    )
    debug.add_argument(
        "--debug", action="store_true", help="Enable debug output for troubleshooting"
    )
    debug.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for each SLURM command",
    )
    debug.add_argument(
        "--sinfo-command",
        default=SINFO_COMMAND,
        help="Override the sinfo command used to collect node data",
    )
    debug.add_argument(
        "--sacct-command",
        default=SACCT_COMMAND,
        help="Override the sacct command used to collect job data",
    )
    return parser


def cli_main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Optional[CommandRunner] = None,
    console: Optional[Any] = None,
    stderr_console: Optional[Any] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be a positive integer")
    if args.top_users is not None and args.top_users < 0:
        parser.error("--top-users must be zero or greater")

    active_console = console or Console()
    active_stderr = stderr_console or Console(stderr=True)
    target_users: Set[str] = set(args.users) if args.users else set()
    if args.me:
        target_users.add(getpass.getuser())
    target_user_filter: Optional[Set[str]] = target_users or None

    if args.jobs and not args.json:
        active_runner = runner or SubprocessRunner()
        jobs_base_command = (
            JOBS_SACCT_COMMAND if args.sacct_command == SACCT_COMMAND else args.sacct_command
        )
        sacct_command = _jobs_sacct_command(
            jobs_base_command,
            states=args.states,
            users=target_user_filter,
            apply_states=False,
        )
        sacct_result = active_runner.run(sacct_command, args.timeout)
        if sacct_result.returncode != 0:
            active_stderr.print(
                Text(
                    f"sacct command failed: {sacct_result.command}",
                    style="red",
                )
            )
            if sacct_result.stderr:
                active_stderr.print(Text(sacct_result.stderr.strip(), style="red"))
            return EXIT_COMMAND_ERROR
        jobs = _filtered_jobs(
            parse_jobs(sacct_result.stdout),
            target_users=target_user_filter,
            partition_filter=args.partition,
            states=args.states,
        )
        if not jobs:
            active_console.print(Text("No jobs found matching the criteria.", style="yellow"))
            return EXIT_NO_MATCHES
        job_nodes = sorted(
            {
                node
                for job in jobs
                for node in parse_nodelist(job.nodelist)
                if node and node.upper() not in {"N/A", "(NULL)", "NONE"}
            }
        )
        node_gpu_totals: dict[str, int] = {}
        node_gpu_types: dict[str, str] = {}
        if job_nodes:
            sinfo_command = _jobs_sinfo_command(args.sinfo_command, nodes=job_nodes)
            sinfo_result = active_runner.run(sinfo_command, args.timeout)
            if sinfo_result.returncode != 0:
                active_stderr.print(
                    Text(
                        f"sinfo command failed: {sinfo_result.command}",
                        style="red",
                    )
                )
                if sinfo_result.stderr:
                    active_stderr.print(Text(sinfo_result.stderr.strip(), style="red"))
                return EXIT_COMMAND_ERROR
            resolved_servers = parse_sinfo(sinfo_result.stdout, gpu_only=False)
            if args.constraint:
                jobs = _filter_jobs_by_constraint(
                    jobs,
                    resolved_servers,
                    constraint=args.constraint,
                    stderr_console=active_stderr,
                )
                if not jobs:
                    active_console.print(Text("No jobs found matching the criteria.", style="yellow"))
                    return EXIT_NO_MATCHES
                matching_nodes = {
                    node
                    for job in jobs
                    for node in parse_nodelist(job.nodelist)
                    if node and node.upper() not in {"N/A", "(NULL)", "NONE"}
                }
                resolved_servers = {
                    name: server for name, server in resolved_servers.items() if name in matching_nodes
                }
            node_gpu_totals = {
                name: server.gpu.shards if args.shard and server.gpu.shards > 0 else server.gpu.num
                for name, server in resolved_servers.items()
            }
            node_gpu_types = {
                name: _display_gpu_type(server, show_shards=args.shard)
                for name, server in resolved_servers.items()
            }
        title = "Jobs"
        if target_user_filter:
            title = "My Jobs" if len(target_user_filter) == 1 and args.me else "Filtered Jobs"
        active_console.print(
            render_jobs_view(
                jobs,
                title=title,
                node_gpu_totals=node_gpu_totals,
                node_gpu_types=node_gpu_types,
            )
        )
        return EXIT_SUCCESS

    if args.top_users is not None and not args.json and not args.shard:
        active_runner = runner or SubprocessRunner()
        sacct_result = active_runner.run(args.sacct_command, args.timeout)
        if sacct_result.returncode != 0:
            active_stderr.print(
                Text(
                    f"sacct command failed: {sacct_result.command}",
                    style="red",
                )
            )
            if sacct_result.stderr:
                active_stderr.print(Text(sacct_result.stderr.strip(), style="red"))
            return EXIT_COMMAND_ERROR
        summary = build_top_users_summary(
            parse_jobs(sacct_result.stdout),
            target_users=target_user_filter,
            show_shards=args.shard,
            top_users_limit=args.top_users,
        )
        print_summary(
            summary,
            console=active_console,
            show_overview=False,
            show_top_users=True,
            show_target_users=False,
        )
        return EXIT_SUCCESS

    options = CollectionOptions(
        sinfo_command=args.sinfo_command,
        sacct_command=(
            _jobs_sacct_command(
                args.sacct_command,
                states=("RUNNING",),
                users=target_user_filter,
            )
            if target_user_filter
            else args.sacct_command
        ),
        timeout=args.timeout,
        parallel=not args.no_parallel,
        gpu_only=not args.json,
        constraint=args.constraint,
        debug=args.debug,
        store_users=bool(args.jobs or target_user_filter or args.json),
    )

    try:
        state = collect_cluster_state(
            runner=runner,
            options=options,
            stderr_console=active_stderr,
        )
    except CommandExecutionError as error:
        active_stderr.print(
            Text(
                f"{error.command_name} command failed: {error.result.command}",
                style="red",
            )
        )
        if error.result.stderr:
            active_stderr.print(Text(error.result.stderr.strip(), style="red"))
        return EXIT_COMMAND_ERROR
    except NoMatchingServersError as error:
        active_console.print(Text(str(error), style="yellow"))
        return EXIT_NO_MATCHES
    except ClusterParseError as error:
        active_stderr.print(Text(str(error), style="red"))
        return EXIT_PARSE_ERROR

    servers = visible_servers(
        state.servers,
        target_users=target_user_filter,
        sort_by=args.sort,
        show_shards=args.shard,
    )
    if args.shard:
        servers = [server for server in servers if server.gpu.shards > 0]
        if not servers:
            active_console.print(Text("No sharded servers found matching the criteria.", style="yellow"))
            return EXIT_NO_MATCHES
    summary = build_cluster_summary(
        state,
        target_users=target_user_filter,
        show_shards=args.shard,
        top_users_limit=args.top_users or 0,
    )
    display_servers = (
        project_servers_for_users(servers, target_users=target_user_filter)
        if target_user_filter
        else servers
    )

    if args.json:
        payload = build_json_payload(
            summary,
            display_servers,
            sort_by=args.sort,
            show_jobs=args.jobs,
            show_shards=args.shard,
        )
        _write_json_output(json.dumps(payload, indent=2, sort_keys=True), console)
        return EXIT_SUCCESS

    if args.top_users is not None:
        print_summary(
            summary,
            console=active_console,
            show_overview=False,
            show_top_users=True,
            show_target_users=False,
        )
        return EXIT_SUCCESS

    if target_user_filter:
        print_filtered_users(
            summary,
            console=active_console,
        )
    active_console.print(
        render_table(
            display_servers,
            show_jobs=args.jobs or (args.verbose and bool(target_user_filter)),
            show_shards=args.shard,
            width=getattr(active_console, "width", None),
            show_used=bool(target_user_filter),
            overview_title="Filtered Usage" if target_user_filter else "Cluster Overview",
            verbose=args.verbose or args.jobs,
        )
    )
    return EXIT_SUCCESS


def main() -> None:
    try:
        raise SystemExit(cli_main())
    except KeyboardInterrupt:
        raise SystemExit(130)

from __future__ import annotations

import argparse
import getpass
import json
import shlex
import sys
from enum import Enum
from typing import Any, Optional, Sequence, Set

from rich.console import Console
from rich.text import Text

from .accounting import (
    project_servers_for_users,
    summarize_users,
)
from .collector import (
    ClusterParseError,
    CollectionOptions,
    CommandExecutionError,
    NoMatchingServersError,
    collect_cluster_state,
)
from .command_options import set_value_option
from .constraints import ConstraintSyntaxError, normalize_constraint_feature
from .constants import (
    DEFAULT_TIMEOUT,
    EXIT_COMMAND_ERROR,
    EXIT_NO_MATCHES,
    EXIT_PARSE_ERROR,
    EXIT_SUCCESS,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
)
from .models import JobRecord, UserUsage
from .partitions import partition_names
from .render import (
    help_legend,
    print_filtered_users,
    print_top_users,
)
from .render_cluster import render_table, visible_servers
from .render_json import jobs_json, nodes_json, summary_json, top_users_json
from .render_jobs import (
    render_jobs_view,
)
from .runner import Command, CommandRunner, SubprocessRunner
from .slurm import parse_nodelist


class View(Enum):
    SUMMARY = "summary"
    NODES = "nodes"
    JOBS = "jobs"
    TOP_USERS = "top-users"


class GtopArgumentParser(argparse.ArgumentParser):
    def print_help(self, file: Optional[Any] = None) -> None:
        super().print_help(file)
        Console(file=file).print(help_legend())


def _write_json_output(json_text: str, console: Optional[Any]) -> None:
    if console is None:
        sys.stdout.write(json_text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    if isinstance(console, Console):
        console.print(json_text, markup=False, soft_wrap=True)
        return

    console.print(json_text)


def _command_arg(value: str) -> Command:
    try:
        command = tuple(shlex.split(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not command:
        raise argparse.ArgumentTypeError("command must not be empty")
    return command


def _constraint_arg(value: str) -> str:
    try:
        return normalize_constraint_feature(value)
    except ConstraintSyntaxError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _sacct_command(
    command: Command,
    *,
    states: Optional[Sequence[str]],
    users: Optional[Set[str]] = None,
    partitions: Optional[Sequence[str]] = None,
) -> Command:
    updated = set_value_option(
        command,
        ("--state", "-s"),
        ",".join(states) if states is not None else None,
    )
    if users:
        updated = set_value_option(
            updated,
            ("--user", "--users"),
            ",".join(sorted(users)),
        )
    if partitions:
        updated = set_value_option(
            updated,
            ("--partition", "--partitions"),
            ",".join(partitions),
        )
    return updated


def _partition_scope_label(partitions: Sequence[str]) -> str:
    return ", ".join(partitions)


def _print_partition_scope(
    console: Any,
    *,
    partitions: Sequence[str],
) -> None:
    console.print(
        Text(f"Partition scope: {_partition_scope_label(partitions)}", style="dim")
    )


def _overview_title(
    *,
    target_users: Optional[Set[str]],
    partition_filter: Optional[Sequence[str]],
) -> str:
    if partition_filter:
        prefix = "Partition" if len(partition_filter) == 1 else "Partitions"
        label = _partition_scope_label(partition_filter)
        if target_users:
            return f"Usage ({prefix.lower()} {label})"
        return f"{prefix} {label}"
    return "Usage" if target_users else "Cluster Overview"


def _filtered_jobs(
    jobs: Sequence[JobRecord],
    *,
    target_users: Optional[Set[str]],
    server_names: Set[str],
    required_constraints: frozenset[str],
) -> list[JobRecord]:
    filtered = []
    for job in jobs:
        if target_users and job.user not in target_users:
            continue
        assigned_nodes = set(parse_nodelist(job.nodelist))
        if assigned_nodes and not (assigned_nodes & server_names):
            continue
        if (
            required_constraints
            and not assigned_nodes
            and not required_constraints.issubset(job.constraints)
        ):
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


def build_parser() -> argparse.ArgumentParser:
    parser = GtopArgumentParser(
        description="Show live SLURM GPU capacity and usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    core = parser.add_argument_group("Core options")
    debug = parser.add_argument_group("Debug options")
    views = core.add_mutually_exclusive_group()
    views.add_argument(
        "-j",
        "--jobs",
        dest="view",
        action="store_const",
        const=View.JOBS,
        help="Show active jobs and allocated resources",
    )
    views.add_argument(
        "-v",
        "--nodes",
        dest="view",
        action="store_const",
        const=View.NODES,
        help="Show node details",
    )
    users = core.add_mutually_exclusive_group()
    users.add_argument("-u", "--users", nargs="+", help="Filter by usernames")
    users.add_argument(
        "-m",
        "--me",
        action="store_true",
        help="Filter to your own usage",
    )
    views.add_argument(
        "-U",
        "--top-users",
        dest="view",
        action="store_const",
        const=View.TOP_USERS,
        help="Show the 25 highest GPU or shard users",
    )
    core.add_argument(
        "-C",
        "--constraint",
        nargs="+",
        type=_constraint_arg,
        metavar="FEATURE",
        help="Require one or more exact node features",
    )
    core.add_argument(
        "-p",
        "--partition",
        nargs="+",
        help="Scope results to one or more partitions",
    )
    core.add_argument(
        "-s",
        "--shard",
        action="store_true",
        help="Limit to sharded GPUs and count shards",
    )
    core.add_argument(
        "--json",
        action="store_true",
        help="Emit this view as JSON",
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
        type=_command_arg,
        help="Override the sinfo executable and arguments (no shell syntax)",
    )
    debug.add_argument(
        "--sacct-command",
        type=_command_arg,
        help="Override the sacct executable and arguments (no shell syntax)",
    )
    parser.set_defaults(view=View.SUMMARY)
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
    view = args.view
    if args.timeout <= 0:
        parser.error("--timeout must be a positive integer")
    if view is View.TOP_USERS and (args.users or args.me):
        parser.error("--top-users cannot be combined with a user filter")
    if view is View.JOBS and args.shard:
        parser.error("--shard is only valid for capacity and top-users views")

    active_console = console or Console()
    active_stderr = stderr_console or Console(stderr=True)
    diagnostic_console = active_stderr if args.json else active_console
    active_runner = runner or SubprocessRunner()
    target_users: Set[str] = set(args.users) if args.users else set()
    if args.me:
        target_users.add(getpass.getuser())
    target_user_filter: Optional[Set[str]] = target_users or None
    partition_filter = (
        tuple(
            partition
            for value in args.partition
            for partition in partition_names(value)
        )
        if args.partition
        else None
    )
    constraint_filter = frozenset(args.constraint or ())
    constraint = ",".join(args.constraint) if args.constraint else None

    sacct_base_command = args.sacct_command or (
        JOBS_SACCT_COMMAND if view is View.JOBS else SACCT_COMMAND
    )
    if view is View.JOBS:
        sacct_command = _sacct_command(
            sacct_base_command,
            states=None,
            users=target_user_filter,
            partitions=partition_filter,
        )
    else:
        sacct_command = (
            _sacct_command(
                sacct_base_command,
                states=("RUNNING",),
                users=target_user_filter,
                partitions=partition_filter,
            )
            if target_user_filter or partition_filter
            else sacct_base_command
        )

    options = CollectionOptions(
        sinfo_command=args.sinfo_command or SINFO_COMMAND,
        sacct_command=sacct_command,
        timeout=args.timeout,
        parallel=not args.no_parallel,
        gpu_only=True,
        partition_filter=partition_filter,
        constraint=constraint,
        debug=args.debug,
        store_allocations=bool(
            target_user_filter or args.json or view in {View.JOBS, View.TOP_USERS}
        ),
        allow_empty_servers=view is View.JOBS,
    )

    try:
        state = collect_cluster_state(
            runner=active_runner,
            options=options,
            stderr_console=active_stderr,
        )
    except CommandExecutionError as error:
        if (
            error.command_name == "sacct"
            and target_user_filter
            and "Invalid user id" in error.result.stderr
        ):
            users = ", ".join(sorted(target_user_filter))
            diagnostic_console.print(Text(f"Unknown user: {users}.", style="yellow"))
            return EXIT_NO_MATCHES
        active_stderr.print(
            Text(
                f"{error.command_name} command failed: "
                f"{shlex.join(error.result.command)}",
                style="red",
            )
        )
        if error.result.stderr:
            active_stderr.print(Text(error.result.stderr.strip(), style="red"))
        return EXIT_COMMAND_ERROR
    except NoMatchingServersError as error:
        diagnostic_console.print(Text(str(error), style="yellow"))
        return EXIT_NO_MATCHES
    except ClusterParseError as error:
        active_stderr.print(Text(str(error), style="red"))
        return EXIT_PARSE_ERROR

    if view is View.JOBS:
        jobs = _filtered_jobs(
            state.jobs,
            target_users=target_user_filter,
            server_names=set(state.servers),
            required_constraints=constraint_filter,
        )
        if not jobs:
            diagnostic_console.print(
                Text("No jobs found matching the criteria.", style="yellow")
            )
            return EXIT_NO_MATCHES
        if args.json:
            _write_json_output(
                json.dumps(jobs_json(jobs), indent=2, sort_keys=True),
                console,
            )
            return EXIT_SUCCESS

        title = "Jobs"
        if target_user_filter:
            title = (
                "My Jobs"
                if len(target_user_filter) == 1 and args.me
                else "Filtered Jobs"
            )
        if partition_filter:
            _print_partition_scope(
                active_console,
                partitions=partition_filter,
            )
        job_view = render_jobs_view(
            jobs,
            title=title,
            servers=state.servers,
            width=getattr(active_console, "width", None),
        )
        for renderable in job_view.renderables:
            active_console.print(
                renderable,
                soft_wrap=isinstance(renderable, Text),
            )
        return EXIT_SUCCESS

    servers = visible_servers(
        state.servers,
        target_users=target_user_filter,
        show_shards=args.shard,
    )
    if args.shard:
        servers = [server for server in servers if server.gpu.shards > 0]
        if not servers:
            diagnostic_console.print(
                Text("No sharded servers found matching the criteria.", style="yellow")
            )
            return EXIT_NO_MATCHES
    visible_server_map = {server.name: server for server in servers}
    unit = "shard" if args.shard else "GPU"
    all_users = (
        summarize_users(visible_server_map, show_shards=args.shard)
        if target_user_filter or view is View.TOP_USERS
        else {}
    )
    selected_users = {
        user: all_users.get(user, UserUsage(user=user)) for user in target_users
    }
    top_users = sorted(
        all_users.values(),
        key=lambda usage: (
            -usage.total_usage(),
            -max(usage.usage_by_partition.values(), default=0),
            usage.user,
        ),
    )[:25]
    display_servers = (
        project_servers_for_users(servers, target_users=target_user_filter)
        if target_user_filter
        else servers
    )

    if view is View.TOP_USERS:
        if not top_users:
            diagnostic_console.print(
                Text("No users found matching the criteria.", style="yellow")
            )
            return EXIT_NO_MATCHES
    if target_user_filter and not display_servers:
        diagnostic_console.print(
            Text("No usage found matching the criteria.", style="yellow")
        )
        return EXIT_NO_MATCHES

    if args.json:
        if view is View.NODES:
            payload = nodes_json(
                display_servers,
                show_shards=args.shard,
            )
        elif view is View.TOP_USERS:
            payload = top_users_json(top_users, unit=unit)
        else:
            payload = summary_json(
                display_servers,
                show_shards=args.shard,
                show_used=bool(target_user_filter),
            )
        _write_json_output(json.dumps(payload, indent=2, sort_keys=True), console)
        return EXIT_SUCCESS

    if partition_filter:
        _print_partition_scope(
            active_console,
            partitions=partition_filter,
        )

    if view is View.TOP_USERS:
        print_top_users(
            top_users,
            unit=unit,
            console=active_console,
        )
        return EXIT_SUCCESS

    if target_user_filter:
        print_filtered_users(
            selected_users,
            unit=unit,
            console=active_console,
        )

    if display_servers:
        active_console.print(
            render_table(
                display_servers,
                show_shards=args.shard,
                width=getattr(active_console, "width", None),
                show_used=bool(target_user_filter),
                overview_title=_overview_title(
                    target_users=target_user_filter,
                    partition_filter=partition_filter,
                ),
                verbose=view is View.NODES,
            )
        )
    return EXIT_SUCCESS


def main() -> None:
    try:
        raise SystemExit(cli_main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        raise SystemExit(EXIT_SUCCESS)

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
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
from .constants import (
    DEFAULT_TIMEOUT,
    EXIT_COMMAND_ERROR,
    EXIT_NO_MATCHES,
    EXIT_PARSE_ERROR,
    EXIT_SUCCESS,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
    SQUEUE_COMMAND,
)
from .history import (
    WINDOWS,
    HistoryError,
    UsageHistory,
    cached_history,
    fetch_history,
)
from .models import ClusterState, JobRecord, ServerState, UserUsage
from .partitions import partition_names
from .render import (
    help_legend,
    print_filtered_users,
    print_top_users,
    print_usage_history,
)
from .render_cluster import render_table, visible_servers
from .render_json import jobs_json, nodes_json, users_json
from .render_jobs import render_jobs_view
from .render_node import (
    node_json,
    queued_for,
    render_node,
)
from .render_ready import availability_json, build_availability, render_availability
from .runner import Command, CommandRunner, SubprocessRunner
from .scheduling import (
    MIN_JOB_CPUS,
    MIN_JOB_MEM_GB,
    PARTITION_COMMAND,
    parse_partitions,
    submittable_partitions,
    user_groups,
)
from .slurm import parse_nodelist

GPU_TIERS = ("high", "mid", "low", "all")
AVAILABLE_HELP = (
    "gtop columns, per partition you can submit to:\n"
    f"  Free      idle GPUs on nodes that also have {MIN_JOB_CPUS} CPUs and\n"
    f"            {MIN_JOB_MEM_GB}G RAM free\n"
    "  +Preempt  more GPUs by preempting lower-priority jobs\n"
    "  Short     idle GPUs on nodes without that much CPU or RAM"
)


class GtopArgumentParser(argparse.ArgumentParser):
    def print_help(self, file: Optional[Any] = None) -> None:
        super().print_help(file)
        Console(file=file).print(help_legend())


class _StdoutConsole(Console):
    def on_broken_pipe(self) -> None:
        # Rich exits 1 here; main() exits cleanly when head or a pager closes the pipe.
        raise BrokenPipeError


def _stdout_console() -> Console:
    # Rich lays out piped output at 80 columns, which wraps rows that grep then misses.
    if sys.stdout.isatty() or "COLUMNS" in os.environ:
        return _StdoutConsole()
    return _StdoutConsole(width=100_000)


def _no_matches(
    args: argparse.Namespace,
    console: Optional[Any],
    diagnostic_console: Any,
    message: str,
    empty_payload: dict[str, Any],
) -> int:
    """An empty result: a message, or with --json an empty payload for scripts."""
    diagnostic_console.print(Text(message, style="yellow"))
    if args.json:
        _write_json_output(empty_payload, console)
        return EXIT_SUCCESS
    return EXIT_NO_MATCHES


def _write_json_output(payload: Any, console: Optional[Any]) -> None:
    json_text = json.dumps(payload, indent=2, sort_keys=True)
    if console is None:
        sys.stdout.write(json_text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    if isinstance(console, Console):
        console.print(json_text, markup=False, soft_wrap=True)
        return

    console.print(json_text)


def _sacct_command(
    command: Command,
    *,
    states: Optional[Sequence[str]],
    users: Optional[Set[str]] = None,
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
    return updated


def _print_partition_scope(console: Any, partitions: Sequence[str]) -> None:
    console.print(Text(f"Partition scope: {', '.join(partitions)}", style="dim"))


def _overview_title(
    *,
    target_users: Optional[Set[str]],
    partition_filter: Optional[Sequence[str]],
) -> str:
    if partition_filter:
        prefix = "Partition" if len(partition_filter) == 1 else "Partitions"
        label = ", ".join(partition_filter)
        if target_users:
            return f"Usage ({prefix.lower()} {label})"
        return f"{prefix} {label}"
    return "Usage" if target_users else "Cluster Overview"


def _filtered_jobs(
    jobs: Sequence[JobRecord],
    *,
    target_users: Optional[Set[str]],
    server_names: Set[str],
) -> list[JobRecord]:
    filtered = []
    for job in jobs:
        if target_users and job.user not in target_users:
            continue
        assigned_nodes = set(parse_nodelist(job.nodelist))
        if assigned_nodes and not (assigned_nodes & server_names):
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


_FEATURE = re.compile(r"[a-z0-9_.:+-]+")


def _comma_list(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError(f"expected comma-separated names, got {value!r}")
    return names


def _feature_arg(value: str) -> list[frozenset[str]]:
    """A -C value: required features separated by ",", each "a|b" matching either."""
    groups = [
        frozenset(part.strip().lower() for part in item.split("|"))
        for item in _comma_list(value)
    ]
    if not all(_FEATURE.fullmatch(part) for group in groups for part in group):
        raise argparse.ArgumentTypeError(f"invalid feature {value!r}")
    return groups


def _positive_int(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {value!r}")
    return int(value)


def _node_matches(server: ServerState, args: argparse.Namespace) -> bool:
    if args.tier != "all" and f"gpu-{args.tier}" not in server.features:
        return False
    return all(
        alternatives & server.features for alternatives in args.constraint or ()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = GtopArgumentParser(
        prog="gtop",
        description="Show live SLURM GPU capacity and usage",
        epilog=f"{AVAILABLE_HELP}\n\n"
        "Options for the default view (-t, -C, -p, --json): gtop available -h",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    common = GtopArgumentParser(add_help=False)
    common.add_argument(
        "-p",
        "--partition",
        action="extend",
        type=partition_names,
        metavar="PARTITION[,...]",
        help="Limit to nodes in these partitions",
    )
    common.add_argument("--json", action="store_true", help="Emit this view as JSON")

    tier = GtopArgumentParser(add_help=False)
    tier.add_argument(
        "-t",
        "--tier",
        choices=GPU_TIERS,
        default="all",
        help="Only nodes with feature gpu-<tier> (default: all)",
    )
    tier.add_argument(
        "-C",
        "--constraint",
        action="extend",
        type=_feature_arg,
        metavar="FEATURE[,...]",
        help="Only nodes with every listed feature; a|b matches either "
        "(e.g. -C 'nvlink,ampere|ada')",
    )

    users_filter = GtopArgumentParser(add_help=False)
    selection = users_filter.add_mutually_exclusive_group()
    selection.add_argument(
        "-u",
        "--users",
        action="extend",
        type=_comma_list,
        metavar="USER[,...]",
        help="Filter by usernames",
    )
    selection.add_argument(
        "-m", "--me", action="store_true", help="Filter to your own usage"
    )

    commands.add_parser(
        "available",
        parents=[tier, common],
        help="GPUs free now and by preempting (default)",
        description=AVAILABLE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nodes = commands.add_parser(
        "nodes",
        parents=[tier, users_filter, common],
        help="Every node with free GPU, CPU and memory, or detail for named nodes",
    )
    nodes.add_argument(
        "names",
        nargs="*",
        metavar="NODE",
        help="Show state, partitions and running jobs for these nodes",
    )
    users = commands.add_parser(
        "users",
        parents=[users_filter, common],
        help="Who is using GPUs now, or GPU-hours over a past window",
    )
    window = users.add_mutually_exclusive_group()
    for name, days in (("week", 7), ("month", 30), ("year", 365)):
        window.add_argument(
            f"--{name}",
            dest="window",
            action="store_const",
            const=name,
            help=f"GPU-hours over the last {days} days (cached)",
        )
    users.add_argument(
        "-n",
        "--limit",
        type=_positive_int,
        default=25,
        metavar="N",
        help="Number of users to list (default: 25)",
    )
    users.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild the --week/--month/--year cache now",
    )
    commands.add_parser(
        "jobs",
        parents=[users_filter, common],
        help="Running, pending and requeued jobs",
    )
    return parser


def _commands_accepting(parser: argparse.ArgumentParser, option: str) -> list[str]:
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return [
        name
        for name, subparser in subparsers.choices.items()
        if option in subparser._option_string_actions
    ]


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Options alone run the default view; a mistyped command still reaches argparse.
    if not arguments or (
        arguments[0].startswith("-") and arguments[0] not in ("-h", "--help")
    ):
        arguments.insert(0, "available")
    parser = build_parser()
    args, unknown = parser.parse_known_args(arguments)
    if unknown:
        message = f"unrecognized arguments: {' '.join(unknown)}"
        commands = _commands_accepting(parser, unknown[0])
        if commands:
            message += f" ({unknown[0]} works with: " + ", ".join(
                f"gtop {command}" for command in commands
            ) + ")"
        parser.error(message)
    if getattr(args, "window", None) and (args.users or args.me or args.partition):
        parser.error("--week/--month/--year cannot be combined with -u, -m or -p")
    if args.command == "nodes" and args.names and (
        args.tier != "all" or args.constraint or args.users or args.me or args.partition
    ):
        parser.error("node names cannot be combined with -t, -C, -u, -m or -p")
    if getattr(args, "refresh", False) and not args.window:
        parser.error("--refresh needs --week, --month or --year")
    return args


def cli_main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Optional[CommandRunner] = None,
    console: Optional[Any] = None,
    stderr_console: Optional[Any] = None,
) -> int:
    args = _parse_args(argv)
    active_console = console or _stdout_console()
    active_stderr = stderr_console or Console(stderr=True)
    diagnostic_console = active_stderr if args.json else active_console
    active_runner = runner or SubprocessRunner()

    if args.command == "users" and args.window:
        return _history_view(
            args,
            runner=active_runner,
            console=console,
            active_console=active_console,
            diagnostic_console=diagnostic_console,
            stderr_console=active_stderr,
        )

    if args.command == "nodes" and args.names:
        return _node_detail_view(
            args,
            runner=active_runner,
            console=console,
            active_console=active_console,
            diagnostic_console=diagnostic_console,
            stderr_console=active_stderr,
        )

    target_users: Set[str] = set(getattr(args, "users", None) or ())
    if getattr(args, "me", False):
        target_users.add(getpass.getuser())
    target_user_filter: Optional[Set[str]] = target_users or None
    partition_filter = tuple(dict.fromkeys(args.partition)) if args.partition else None

    if args.command == "jobs":
        sacct_command = _sacct_command(
            JOBS_SACCT_COMMAND,
            states=None,
            users=target_user_filter,
        )
    else:
        sacct_command = (
            _sacct_command(
                SACCT_COMMAND,
                states=("RUNNING",),
                users=target_user_filter,
            )
            if target_user_filter
            else SACCT_COMMAND
        )

    options = CollectionOptions(
        sinfo_command=SINFO_COMMAND,
        sacct_command=sacct_command,
        gpu_only=True,
        partition_filter=partition_filter,
        allow_empty_servers=args.command == "jobs",
        # sacct misses requeued and array-range pending jobs that squeue lists.
        squeue_command=(
            set_value_option(
                SQUEUE_COMMAND,
                ("-u", "--user"),
                ",".join(sorted(target_user_filter)) if target_user_filter else None,
            )
            if args.command == "jobs"
            else None
        ),
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            partition_result = (
                executor.submit(active_runner.run, PARTITION_COMMAND, DEFAULT_TIMEOUT)
                if args.command == "available"
                else None
            )
            state = collect_cluster_state(
                runner=active_runner,
                options=options,
                stderr_console=active_stderr,
            )
            partitions_output = (
                partition_result.result() if partition_result is not None else None
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
        return _command_failed(active_stderr, error.command_name, error.result)
    except NoMatchingServersError as error:
        diagnostic_console.print(Text(str(error), style="yellow"))
        return EXIT_NO_MATCHES
    except ClusterParseError as error:
        active_stderr.print(Text(str(error), style="red"))
        return EXIT_PARSE_ERROR

    if args.command == "jobs":
        return _jobs_view(
            args,
            state,
            target_user_filter=target_user_filter,
            partition_filter=partition_filter,
            console=console,
            active_console=active_console,
            diagnostic_console=diagnostic_console,
        )

    if args.command == "available":
        assert partitions_output is not None
        if partitions_output.returncode != 0:
            return _command_failed(active_stderr, "scontrol", partitions_output)
        return _available_view(
            args,
            state,
            partitions_output=partitions_output.stdout,
            partition_filter=partition_filter,
            console=console,
            active_console=active_console,
            diagnostic_console=diagnostic_console,
        )

    servers = visible_servers(
        {
            name: server
            for name, server in state.servers.items()
            if args.command != "nodes" or _node_matches(server, args)
        },
        target_users=target_user_filter,
    )
    visible_server_map = {server.name: server for server in servers}
    unit = "GPU"
    all_users = summarize_users(visible_server_map)
    display_servers = (
        project_servers_for_users(servers, target_users=target_user_filter)
        if target_user_filter
        else servers
    )
    if not display_servers:
        return _no_matches(
            args,
            console,
            diagnostic_console,
            "No usage found matching the criteria."
            if target_user_filter
            else "No nodes match these filters.",
            users_json([], unit=unit) if args.command == "users" else nodes_json([]),
        )

    if args.command == "users" and not target_user_filter:
        top_users = sorted(
            all_users.values(),
            key=lambda usage: (
                -usage.total_usage(),
                -max(usage.usage_by_partition.values(), default=0),
                usage.user,
            ),
        )[: args.limit]
        if not top_users:
            return _no_matches(
                args,
                console,
                diagnostic_console,
                "No users found matching the criteria.",
                users_json([], unit=unit),
            )
        if args.json:
            _write_json_output(users_json(top_users, unit=unit), console)
            return EXIT_SUCCESS
        if partition_filter:
            _print_partition_scope(active_console, partition_filter)
        print_top_users(top_users, unit=unit, console=active_console)
        return EXIT_SUCCESS

    if args.json:
        payload = (
            nodes_json(display_servers)
            if args.command == "nodes"
            else users_json(
                [all_users.get(user, UserUsage(user=user)) for user in sorted(target_users)],
                unit=unit,
            )
        )
        _write_json_output(payload, console)
        return EXIT_SUCCESS

    if partition_filter:
        _print_partition_scope(active_console, partition_filter)
    if target_user_filter:
        print_filtered_users(
            {user: all_users.get(user, UserUsage(user=user)) for user in target_users},
            unit=unit,
            console=active_console,
        )
    active_console.print(
        render_table(
            display_servers,
            width=getattr(active_console, "width", None),
            show_used=bool(target_user_filter),
            overview_title=_overview_title(
                target_users=target_user_filter,
                partition_filter=partition_filter,
            ),
            verbose=args.command == "nodes",
        )
    )
    return EXIT_SUCCESS


def _command_failed(console: Any, name: str, result: Any) -> int:
    console.print(
        Text(f"{name} command failed: {shlex.join(result.command)}", style="red")
    )
    if result.stderr:
        console.print(Text(result.stderr.strip(), style="red"))
    return EXIT_COMMAND_ERROR


def _node_detail_view(
    args: argparse.Namespace,
    *,
    runner: CommandRunner,
    console: Optional[Any],
    active_console: Any,
    diagnostic_console: Any,
    stderr_console: Any,
) -> int:
    nodelist = ",".join(args.names)
    options = CollectionOptions(
        sinfo_command=(*SINFO_COMMAND, "-n", nodelist),
        # Pending jobs have no node yet, so fetch the whole active queue.
        sacct_command=JOBS_SACCT_COMMAND,
        squeue_command=SQUEUE_COMMAND,
        allow_empty_servers=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            partition_future = executor.submit(
                runner.run, PARTITION_COMMAND, DEFAULT_TIMEOUT
            )
            state = collect_cluster_state(
                runner=runner, options=options, stderr_console=stderr_console
            )
            partition_result = partition_future.result()
    except CommandExecutionError as error:
        return _command_failed(stderr_console, error.command_name, error.result)
    except ClusterParseError as error:
        stderr_console.print(Text(str(error), style="red"))
        return EXIT_PARSE_ERROR
    if partition_result.returncode != 0:
        return _command_failed(stderr_console, "scontrol", partition_result)

    missing = [name for name in args.names if name not in state.servers]
    if missing:
        diagnostic_console.print(
            Text(f"Unknown node: {', '.join(missing)}.", style="yellow")
        )
        return EXIT_NO_MATCHES
    partitions = parse_partitions(partition_result.stdout)
    servers = [state.servers[name] for name in dict.fromkeys(args.names)]
    if args.json:
        _write_json_output(
            {
                "view": "node",
                "nodes": [
                    node_json(
                        server,
                        partitions,
                        queued_for(server.name, state.jobs, partitions),
                    )
                    for server in servers
                ],
            },
            console,
        )
        return EXIT_SUCCESS
    for index, server in enumerate(servers):
        if index:
            active_console.print()
        active_console.print(
            render_node(
                server,
                partitions,
                queued_for(server.name, state.jobs, partitions),
            )
        )
    return EXIT_SUCCESS


def _available_view(
    args: argparse.Namespace,
    state: ClusterState,
    *,
    partitions_output: str,
    partition_filter: Optional[Sequence[str]],
    console: Optional[Any],
    active_console: Any,
    diagnostic_console: Any,
) -> int:
    partitions = parse_partitions(partitions_output)
    servers = {
        name: server
        for name, server in state.servers.items()
        if _node_matches(server, args)
    }
    if partition_filter:
        shown = [
            partitions[name]
            for name in partition_filter
            if name in partitions and partitions[name].nodes & servers.keys()
        ]
    else:
        shown = [
            partition
            for partition in submittable_partitions(
                partitions, user_groups(getpass.getuser())
            )
            if partition.nodes & servers.keys()
        ]
    if not shown:
        return _no_matches(
            args,
            console,
            diagnostic_console,
            "No partitions you can use have matching GPU nodes.",
            availability_json([]),
        )

    results = build_availability(servers, partitions, shown)
    if args.json:
        _write_json_output(availability_json(results), console)
        return EXIT_SUCCESS
    active_console.print(render_availability(results, tier=args.tier))
    return EXIT_SUCCESS


def _history_view(
    args: argparse.Namespace,
    *,
    runner: CommandRunner,
    console: Optional[Any],
    active_console: Any,
    diagnostic_console: Any,
    stderr_console: Any,
) -> int:
    history = None if args.refresh else cached_history(args.window)
    if history is None:
        if not args.json:
            diagnostic_console.print(
                Text(f"Querying {args.window} history from sreport...", style="dim")
            )
        try:
            history = fetch_history(args.window, runner=runner)
        except HistoryError as error:
            stderr_console.print(Text(str(error), style="red"))
            return EXIT_COMMAND_ERROR

    payload = {
        "view": f"users-{args.window}",
        "generated_at": history.generated_at,
        "unit": "GPU-hours",
        "users": [
            {
                "rank": rank,
                "user": user,
                "total": round(hours, 1),
                "accounts": history.accounts.get(user, []),
            }
            for rank, (user, hours) in enumerate(
                history.ranked()[: args.limit], start=1
            )
        ],
    }
    if not payload["users"]:
        return _no_matches(
            args, console, diagnostic_console, "No GPU usage in this window.", payload
        )
    if args.json:
        _write_json_output(payload, console)
        return EXIT_SUCCESS
    title = f"GPU use, last {args.window}" + _age_label(args.window, history)
    print_usage_history(
        history,
        title=title,
        limit=args.limit,
        console=active_console,
    )
    return EXIT_SUCCESS


def _age_label(window: str, history: UsageHistory) -> str:
    seconds = history.age_seconds()
    minutes = int(seconds // 60)
    if minutes < 1:
        return ""
    if minutes < 120:
        age = f"{minutes} min"
    elif minutes < 48 * 60:
        age = f"{minutes // 60} h"
    else:
        age = f"{minutes // 1440} days"
    max_age = WINDOWS[window][1]
    if max_age is None:
        return f" (as of {age} ago; --refresh to update)"
    if seconds > max_age:
        return f" (as of {age} ago; updating in the background)"
    return f" (as of {age} ago)"


def _jobs_view(
    args: argparse.Namespace,
    state: ClusterState,
    *,
    target_user_filter: Optional[Set[str]],
    partition_filter: Optional[Sequence[str]],
    console: Optional[Any],
    active_console: Any,
    diagnostic_console: Any,
) -> int:
    jobs = _filtered_jobs(
        state.jobs,
        target_users=target_user_filter,
        server_names=set(state.servers),
    )
    if not jobs:
        return _no_matches(
            args,
            console,
            diagnostic_console,
            "No jobs found matching the criteria.",
            jobs_json([]),
        )
    if args.json:
        _write_json_output(jobs_json(jobs), console)
        return EXIT_SUCCESS

    title = "Jobs"
    if target_user_filter:
        title = "My Jobs" if args.me and len(target_user_filter) == 1 else "Filtered Jobs"
    if partition_filter:
        _print_partition_scope(active_console, partition_filter)
    job_view = render_jobs_view(
        jobs,
        title=title,
        servers=state.servers,
        width=getattr(active_console, "width", None),
    )
    for renderable in job_view.renderables:
        active_console.print(renderable, soft_wrap=True)
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

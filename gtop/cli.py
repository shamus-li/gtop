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
    process_jobs,
    project_servers_for_users,
)
from .collector import (
    ClusterParseError,
    CollectionOptions,
    CommandExecutionError,
    NoMatchingServersError,
    collect_cluster_state,
)
from .command_options import (
    override_command_option,
    remove_command_options,
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
    build_jobs_overview,
    print_filtered_users,
    render_jobs_view,
    print_summary,
    render_table,
    visible_servers,
)
from .models import ClusterState
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
        updated = override_command_option(updated, ("--state",), ",".join(states))
    else:
        updated = remove_command_options(updated, ("--state",))
    if users:
        updated = override_command_option(updated, ("--user", "--users"), ",".join(sorted(users)))
    if partition:
        updated = override_command_option(updated, ("--partition", "--partitions"), partition)
    return updated


def _jobs_sinfo_command(command: str, *, nodes: Sequence[str]) -> str:
    return override_command_option(command, ("-n", "--nodes"), ",".join(nodes))


def _user_partition_assoc_command(user: str) -> str:
    return (
        "sacctmgr show assoc "
        f"where user={shlex.quote(user)} "
        "format=Account,Partition -Pn"
    )


def _known_partitions_command() -> str:
    return 'sinfo -h -o "%P"'


def _normalize_partition_name(value: str) -> str:
    return value.strip().rstrip("*")


def _known_partition_names(output: str) -> set[str]:
    return {
        normalized
        for normalized in (
            _normalize_partition_name(line)
            for line in output.splitlines()
        )
        if normalized
    }


def _detect_user_partitions(
    assoc_output: str,
    *,
    known_partitions: set[str],
) -> tuple[str, ...]:
    detected: set[str] = set()
    for raw_line in assoc_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        account, _, partition = line.partition("|")
        normalized_partition = _normalize_partition_name(partition)
        if normalized_partition:
            detected.add(normalized_partition)
            continue
        normalized_account = _normalize_partition_name(account)
        if normalized_account and normalized_account in known_partitions:
            detected.add(normalized_account)
    return tuple(sorted(detected))


def _partition_scope_label(partitions: Sequence[str]) -> str:
    return ", ".join(partitions)


def _print_partition_scope(
    console: Any,
    *,
    partitions: Sequence[str],
    source: str,
) -> None:
    mode = "auto-detected" if source == "auto" else "explicit"
    console.print(
        Text(f"Partition scope ({mode}): {_partition_scope_label(partitions)}", style="dim")
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
            return f"Filtered Usage ({prefix.lower()} {label})"
        return f"{prefix} {label}"
    return "Filtered Usage" if target_users else "Cluster Overview"


def _filter_jobs_to_servers(
    jobs: Sequence[Any],
    *,
    server_names: Set[str],
) -> list[Any]:
    return [
        job
        for job in jobs
        if any(node in server_names for node in parse_nodelist(job.nodelist))
    ]


def _matches_partition_scope(value: str, partition: str) -> bool:
    candidate = value.strip().lower()
    needle = partition.strip().lower()
    if not candidate or not needle:
        return False
    return candidate == needle or candidate.startswith(f"{needle}-") or candidate.startswith(
        f"{needle}_"
    )


def _resolve_servers_for_nodes(
    runner: CommandRunner,
    *,
    sinfo_command: str,
    nodes: Sequence[str],
    timeout: int,
) -> tuple[dict[str, Any], Optional[Any]]:
    targeted_command = _jobs_sinfo_command(sinfo_command, nodes=nodes)
    targeted_result = runner.run(targeted_command, timeout)
    if targeted_result.returncode != 0:
        return {}, targeted_result

    resolved_servers = parse_sinfo(targeted_result.stdout, gpu_only=False)
    missing_nodes = [node for node in nodes if node not in resolved_servers]
    if not missing_nodes:
        return resolved_servers, None

    full_result = runner.run(sinfo_command, timeout)
    if full_result.returncode != 0:
        return {}, full_result

    full_servers = parse_sinfo(full_result.stdout, gpu_only=False)
    return {name: server for name, server in full_servers.items() if name in nodes}, None


def _filtered_jobs(
    jobs: Sequence[Any],
    *,
    target_users: Optional[Set[str]],
    partition_filter: Optional[Sequence[str]],
    states: Sequence[str],
) -> list[Any]:
    state_set = {state.upper() for state in states}
    filtered = []
    partition_needles = tuple(partition_filter) if partition_filter else None
    for job in jobs:
        if target_users and job.user not in target_users:
            continue
        if partition_needles:
            if not any(
                _matches_partition_scope(job.partition, needle)
                or any(
                    _matches_partition_scope(node, needle)
                    for node in parse_nodelist(job.nodelist)
                )
                for needle in partition_needles
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


def _job_group_name(job: Any) -> str:
    if job.state.upper().startswith("PEND"):
        return "Pending / Unassigned"
    nodelist = job.nodelist.strip()
    if not nodelist or nodelist.upper() in {"N/A", "(NULL)", "NONE"}:
        return "Pending / Unassigned"
    return nodelist


def _job_view_node_metadata(
    jobs: Sequence[Any],
    resolved_servers: dict[str, Any],
    *,
    show_shards: bool,
) -> tuple[dict[str, int], dict[str, str], dict[str, str], dict[str, float]]:
    group_nodes: dict[str, set[str]] = {}
    for job in jobs:
        group_name = _job_group_name(job)
        if group_name == "Pending / Unassigned":
            continue
        nodes = {
            node
            for node in parse_nodelist(job.nodelist)
            if node and node.upper() not in {"N/A", "(NULL)", "NONE"} and node in resolved_servers
        }
        if nodes:
            group_nodes.setdefault(group_name, set()).update(nodes)

    node_gpu_totals: dict[str, int] = {}
    node_gpu_types: dict[str, str] = {}
    node_gpu_units: dict[str, str] = {}
    node_shards_per_gpu: dict[str, float] = {}
    for group_name, grouped_nodes in group_nodes.items():
        group_servers = [resolved_servers[name] for name in sorted(grouped_nodes)]
        all_sharded = all(server.gpu.shards > 0 for server in group_servers)
        unit = "shard" if all_sharded else "GPU"
        node_gpu_totals[group_name] = sum(
            server.gpu.shards if all_sharded else server.gpu.num
            for server in group_servers
        )
        if all_sharded:
            node_gpu_units[group_name] = unit

        gpu_types = {
            _display_gpu_type(server, show_shards=show_shards)
            for server in group_servers
        }
        node_gpu_types[group_name] = next(iter(gpu_types)) if len(gpu_types) == 1 else "Mixed"

        if all_sharded:
            shards_per_gpu = {
                server.gpu.shards / server.gpu.num
                for server in group_servers
                if server.gpu.num > 0
            }
            if len(shards_per_gpu) == 1:
                node_shards_per_gpu[group_name] = next(iter(shards_per_gpu))
    return node_gpu_totals, node_gpu_types, node_gpu_units, node_shards_per_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display SLURM cluster usage with detailed node and resource information",
        epilog=(
            "Legend:\n"
            "  colored segments reflect detected partition groups or partitions   dim = free\n"
            "  Cornell-style priority/gpu/default buckets are preserved when present"
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
        help="Scope results to one or more partitions",
    )
    core.add_argument(
        "--mine",
        action="store_true",
        help="Auto-detect your accessible partitions and scope results to them",
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
    if args.partition and args.mine:
        parser.error("--partition and --mine cannot be combined")

    active_console = console or Console()
    active_stderr = stderr_console or Console(stderr=True)
    active_runner = runner or SubprocessRunner()
    target_users: Set[str] = set(args.users) if args.users else set()
    current_user = getpass.getuser() if (args.me or args.mine) else None
    if args.me:
        assert current_user is not None
        target_users.add(current_user)
    target_user_filter: Optional[Set[str]] = target_users or None
    partition_filter = tuple(args.partition) if args.partition else None
    partition_scope_source = "explicit" if partition_filter else None
    if args.mine:
        assert current_user is not None
        assoc_command = _user_partition_assoc_command(current_user)
        assoc_result = active_runner.run(assoc_command, args.timeout)
        if assoc_result.returncode != 0:
            active_stderr.print(
                Text(
                    f"sacctmgr command failed: {assoc_result.command}",
                    style="red",
                )
            )
            if assoc_result.stderr:
                active_stderr.print(Text(assoc_result.stderr.strip(), style="red"))
            return EXIT_COMMAND_ERROR
        known_partitions_result = active_runner.run(_known_partitions_command(), args.timeout)
        if known_partitions_result.returncode != 0:
            active_stderr.print(
                Text(
                    f"sinfo command failed: {known_partitions_result.command}",
                    style="red",
                )
            )
            if known_partitions_result.stderr:
                active_stderr.print(Text(known_partitions_result.stderr.strip(), style="red"))
            return EXIT_COMMAND_ERROR
        partition_filter = _detect_user_partitions(
            assoc_result.stdout,
            known_partitions=_known_partition_names(known_partitions_result.stdout),
        )
        if not partition_filter:
            active_console.print(Text("No accessible partitions detected.", style="yellow"))
            return EXIT_NO_MATCHES
        partition_scope_source = "auto"

    if args.jobs and not args.json:
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
            partition_filter=partition_filter,
            states=args.states,
        )
        if not jobs:
            active_console.print(Text("No jobs found matching the criteria.", style="yellow"))
            return EXIT_NO_MATCHES
        title = "Jobs"
        if target_user_filter:
            title = "My Jobs" if len(target_user_filter) == 1 and args.me else "Filtered Jobs"
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
        node_gpu_units: dict[str, str] = {}
        node_shards_per_gpu: dict[str, float] = {}
        if job_nodes:
            resolved_servers, sinfo_result = _resolve_servers_for_nodes(
                active_runner,
                sinfo_command=args.sinfo_command,
                nodes=job_nodes,
                timeout=args.timeout,
            )
            if sinfo_result is not None:
                active_stderr.print(
                    Text(
                        f"sinfo command failed: {sinfo_result.command}",
                        style="red",
                    )
                )
                if sinfo_result.stderr:
                    active_stderr.print(Text(sinfo_result.stderr.strip(), style="red"))
                return EXIT_COMMAND_ERROR
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
            (
                node_gpu_totals,
                node_gpu_types,
                node_gpu_units,
                node_shards_per_gpu,
            ) = _job_view_node_metadata(
                jobs,
                resolved_servers,
                show_shards=args.shard,
            )
        if partition_filter:
            _print_partition_scope(
                active_console,
                partitions=partition_filter,
                source=partition_scope_source or "explicit",
            )
        active_console.print(build_jobs_overview(jobs, title=title))
        active_console.print(
            render_jobs_view(
                jobs,
                title=title,
                node_gpu_totals=node_gpu_totals,
                node_gpu_types=node_gpu_types,
                node_gpu_units=node_gpu_units,
                node_shards_per_gpu=node_shards_per_gpu,
                include_overview=False,
            )
        )
        return EXIT_SUCCESS

    if (
        args.top_users is not None
        and not args.json
        and not args.shard
        and not args.constraint
        and partition_filter is None
    ):
        top_users_sacct_command = _jobs_sacct_command(
            args.sacct_command,
            states=("RUNNING",),
            users=target_user_filter,
        )
        sacct_result = active_runner.run(top_users_sacct_command, args.timeout)
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
        jobs = parse_jobs(sacct_result.stdout)
        if any(job.shard > 0 for job in jobs):
            job_nodes = sorted(
                {
                    node
                    for job in jobs
                    for node in parse_nodelist(job.nodelist)
                    if node and node.upper() not in {"N/A", "(NULL)", "NONE"}
                }
            )
            if job_nodes:
                resolved_servers, sinfo_result = _resolve_servers_for_nodes(
                    active_runner,
                    sinfo_command=args.sinfo_command,
                    nodes=job_nodes,
                    timeout=args.timeout,
                )
                if sinfo_result is not None:
                    active_stderr.print(
                        Text(
                            f"sinfo command failed: {sinfo_result.command}",
                            style="red",
                        )
                    )
                    if sinfo_result.stderr:
                        active_stderr.print(Text(sinfo_result.stderr.strip(), style="red"))
                    return EXIT_COMMAND_ERROR
                process_jobs(jobs, resolved_servers, store_users=True)
                summary = build_cluster_summary(
                    ClusterState(servers=resolved_servers, jobs=jobs),
                    target_users=target_user_filter,
                    show_shards=False,
                    top_users_limit=args.top_users,
                )
            else:
                summary = build_top_users_summary(
                    jobs,
                    target_users=target_user_filter,
                    show_shards=args.shard,
                    top_users_limit=args.top_users,
                )
        else:
            summary = build_top_users_summary(
                jobs,
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
                states=JOBS_DEFAULT_STATES if (args.verbose and target_user_filter) else ("RUNNING",),
                users=target_user_filter,
            )
            if target_user_filter
            else args.sacct_command
        ),
        timeout=args.timeout,
        parallel=not args.no_parallel,
        gpu_only=not args.json,
        partition_filter=partition_filter,
        constraint=args.constraint,
        debug=args.debug,
        store_users=bool(args.jobs or target_user_filter or args.json or args.top_users is not None),
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
    filtered_verbose_jobs = []
    if args.verbose and target_user_filter and not args.jobs:
        filtered_verbose_jobs = _filtered_jobs(
            state.jobs,
            target_users=target_user_filter,
            partition_filter=partition_filter if partition_filter else None,
            states=JOBS_DEFAULT_STATES,
        )
        filtered_verbose_jobs = [
            job for job in filtered_verbose_jobs if job.gpu > 0 or job.shard > 0
        ]
        if partition_filter:
            filtered_verbose_jobs = _filter_jobs_to_servers(
                filtered_verbose_jobs,
                server_names=set(state.servers),
            )
        if args.constraint:
            filtered_verbose_jobs = _filter_jobs_by_constraint(
                filtered_verbose_jobs,
                state.servers,
                constraint=args.constraint,
                stderr_console=active_stderr,
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

    if partition_filter:
        _print_partition_scope(
            active_console,
            partitions=partition_filter,
            source=partition_scope_source or "explicit",
        )

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

    if display_servers:
        active_console.print(
            render_table(
                display_servers,
                show_jobs=args.jobs,
                show_shards=args.shard,
                width=getattr(active_console, "width", None),
                show_used=bool(target_user_filter),
                overview_title=_overview_title(
                    target_users=target_user_filter,
                    partition_filter=partition_filter,
                ),
                verbose=args.verbose or args.jobs,
            )
        )
    if filtered_verbose_jobs:
        (
            node_gpu_totals,
            node_gpu_types,
            node_gpu_units,
            node_shards_per_gpu,
        ) = _job_view_node_metadata(
            filtered_verbose_jobs,
            state.servers,
            show_shards=args.shard,
        )
        title = (
            "My Jobs"
            if target_user_filter is not None and len(target_user_filter) == 1 and args.me
            else "Filtered Jobs"
        )
        if display_servers:
            active_console.print()
        active_console.print(
            render_jobs_view(
                filtered_verbose_jobs,
                title=title,
                node_gpu_totals=node_gpu_totals,
                node_gpu_types=node_gpu_types,
                node_gpu_units=node_gpu_units,
                node_shards_per_gpu=node_shards_per_gpu,
            )
        )
    return EXIT_SUCCESS


def main() -> None:
    try:
        raise SystemExit(cli_main())
    except KeyboardInterrupt:
        raise SystemExit(130)

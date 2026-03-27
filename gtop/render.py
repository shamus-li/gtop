from __future__ import annotations

from functools import lru_cache
import math
import pwd
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from .accounting import partition_bucket
from .models import ClusterSummary, JobRecord, ResourceUsageSplit, ServerState, to_jsonable

GROUP_HEADER_WIDTHS = {
    "node": 20,
    "gpu": 27,
    "cpu": 29,
    "mem": 31,
}
BAR_WIDTHS = {"gpu": 8, "cpu": 12, "mem": 10}
CELL_GAP = 1
GROUP_GAP = 2

GPU_CAPABILITY_PATTERNS = (
    ("T4", 0),
    ("GTX 1080 TI", 1),
    ("GTX TITAN X", 2),
    ("TITAN X PASCAL", 3),
    ("TITAN XP", 4),
    ("TITAN X", 5),
    ("2080 TI", 6),
    ("RTX 2080 TI", 6),
    ("TITAN RTX", 7),
    ("RTX 3090", 8),
    ("QUADRO RTX 6000", 9),
    ("L4", 10),
    ("A40", 11),
    ("A5000", 12),
    ("A5500", 13),
    ("A6000", 14),
    ("6000 ADA", 15),
    ("PRO 6000 BLACKWELL MAX-Q", 16),
    ("PRO 6000 BLACKWELL SERVER EDITION", 17),
    ("V100S", 18),
    ("V100", 19),
    ("A100", 20),
    ("H100", 21),
    ("H200", 22),
    ("GB10", 23),
    ("B200", 24),
)
TOP_USER_BAR_WIDTH = 12
# OKLCH(0.72 0.19 320)
PRIORITY_PARTITION_COLOR = "#c764f4"
# OKLCH(0.80 0.13 165)
GPU_PARTITION_COLOR = "#4fd3a1"
# OKLCH(0.78 0.09 255)
DEFAULT_PARTITION_COLOR = "#88b4ff"
NODE_COUNT_COLOR = "white"


@lru_cache(maxsize=None)
def _lookup_full_name(user: str) -> str:
    try:
        gecos = pwd.getpwnam(user).pw_gecos.strip()
    except KeyError:
        return ""
    if not gecos:
        return ""
    return gecos.split(",", 1)[0].strip()


def _top_user_label(user: str) -> str:
    full_name = _lookup_full_name(user)
    if not full_name or full_name == user:
        return user
    return f"{user} ({full_name})"


def _partition_color(partition: str) -> str:
    bucket = partition_bucket(partition)
    if bucket == "priority":
        return PRIORITY_PARTITION_COLOR
    if bucket == "gpu":
        return GPU_PARTITION_COLOR
    return DEFAULT_PARTITION_COLOR


def _pluralize(count: int, singular: str) -> str:
    if count == 1:
        return singular
    return f"{singular}s"


def _pad(value: str, width: int) -> str:
    if len(value) >= width:
        return value
    return value.ljust(width)


def _resource_totals(
    server: ServerState, *, show_shards: bool = False
) -> tuple[float, float]:
    if show_shards and server.gpu.shards > 0:
        usage = server.usage["shard"]
        total = float(server.gpu.shards)
    else:
        usage = server.usage["gpu"]
        total = float(server.gpu.num)

    used = float(usage.priority) + float(usage.gpu) + float(usage.default)
    return used, total


def _resource_split(
    server: ServerState, resource: str, show_shards: bool
) -> ResourceUsageSplit:
    if resource == "gpu" and show_shards and server.gpu.shards > 0:
        return server.usage["shard"]
    return server.usage[resource]


def format_resource(server: ServerState, res: str, show_shards: bool = False) -> str:
    usage_info = _resource_split(server, res, show_shards)
    priority = usage_info.priority
    gpu = usage_info.gpu
    default = usage_info.default

    if res == "cpu":
        idle = server.cpu.idle
        return f"{int(priority):2d}/{int(gpu):2d}/{int(default):2d}/{int(idle):2d}"
    if res == "gpu":
        gpu_info = server.gpu
        total = gpu_info.shards if show_shards and gpu_info.shards > 0 else gpu_info.num
        idle = max(total - priority - gpu - default, 0)
        return f"{int(priority)}/{int(gpu)}/{int(default)}/{int(idle)}"

    idle = server.mem.idle / 1024.0
    return f"{int(priority):3d}/{int(gpu):3d}/{int(default):3d}/{int(idle):3d}"


def sort_server_names(
    servers: Mapping[str, ServerState], sort_by: str, *, show_shards: bool = False
) -> List[str]:
    items = list(servers.items())

    if sort_by == "feature":

        def feature_key(server: ServerState) -> tuple[str, float]:
            used, total = _resource_totals(server, show_shards=show_shards)
            return (",".join(sorted(server.features)), -(total - used))

        return [
            name
            for name, _ in sorted(
                items,
                key=lambda item: (
                    *feature_key(item[1]),
                    item[0],
                ),
            )
        ]
    if sort_by == "free-gpu":
        return [
            name
            for name, _ in sorted(
                items,
                key=lambda item: (
                    -(
                        _resource_totals(item[1], show_shards=show_shards)[1]
                        - _resource_totals(item[1], show_shards=show_shards)[0]
                    ),
                    item[0],
                ),
            )
        ]
    if sort_by == "used-gpu":
        return [
            name
            for name, _ in sorted(
                items,
                key=lambda item: (
                    -_resource_totals(item[1], show_shards=show_shards)[0],
                    item[0],
                ),
            )
        ]
    if sort_by == "free-shard":

        def free_shards(server: ServerState) -> float:
            if server.gpu.shards <= 0:
                return 0.0
            used, total = _resource_totals(server, show_shards=True)
            return total - used

        return [
            name
            for name, _ in sorted(
                items,
                key=lambda item: (-free_shards(item[1]), item[0]),
            )
        ]
    return sorted(servers.keys())


def visible_servers(
    servers: Mapping[str, ServerState],
    *,
    target_users: Optional[Set[str]],
    sort_by: str,
    show_shards: bool = False,
) -> List[ServerState]:
    sorted_names = sort_server_names(servers, sort_by, show_shards=show_shards)
    visible = [servers[name] for name in sorted_names]
    if not target_users:
        return visible
    return [server for server in visible if server.has_target_users(target_users)]


def print_summary(
    summary: ClusterSummary,
    *,
    console: Optional[Any] = None,
    show_overview: bool = True,
    show_top_users: bool = True,
    show_target_users: bool = True,
) -> None:
    active_console = console or Console()

    if show_overview:
        if summary.gpu_total > 0:
            active_console.print(
                "[bold yellow]Cluster GPU Overview: "
                f"{int(summary.gpu_used)}/{summary.gpu_total} GPUs Used "
                f"({summary.gpu_utilization_pct:.1f}%)[/]"
            )
        else:
            active_console.print("[bold yellow]Cluster GPU Overview: No GPUs detected[/]")

    target_user_summary = summary.target_users
    if show_top_users and summary.top_users:
        active_console.print(Text("Top Users", style="bold cyan"))
        active_console.print(_build_top_users_table(summary))
        active_console.print()

    if not show_target_users or not target_user_summary:
        return

    active_console.print("\n" + "=" * 80, style="bright_white")
    active_console.print("[bold]Summary of Resources Used by Specified Users[/]")
    active_console.print("=" * 80, style="bright_white")


def print_filtered_users(
    summary: ClusterSummary,
    *,
    console: Optional[Any] = None,
) -> None:
    active_console = console or Console()
    visible_users = [
        (user, stats)
        for user, stats in sorted(summary.target_users.items())
        if stats.total_usage() > 0
    ]
    if not visible_users:
        return

    active_console.print(Text("Filtered Users", style="bold cyan"))
    for user, stats in visible_users:
        node_count = len(stats.nodes)
        total_usage = stats.total_usage()
        line = Text()
        line.append(_top_user_label(user), style="cyan")
        line.append("  ")
        line.append(str(total_usage), style="bold yellow")
        line.append(f" {_pluralize(total_usage, summary.resource_label)} across ")
        line.append(str(node_count), style=f"bold {NODE_COUNT_COLOR}")
        line.append(f" {_pluralize(node_count, 'node')}")
        for partition, count in sorted(stats.usage_by_partition.items()):
            line.append("  ")
            line.append(f"{partition}:", style="white")
            line.append(str(count), style=f"bold {_partition_color(partition)}")
        active_console.print(line)
    active_console.print()


def _build_jobs_table(
    server: ServerState,
    *,
    show_shards: bool,
    group_name_width: Optional[int] = None,
    gpu_type_width: Optional[int] = None,
) -> Group:
    jobs = sorted(
        server.users.values(),
        key=lambda job: (
            job.state,
            job.partition,
            job.job_id,
        ),
    )
    records = [
        JobRecord(
            user=job.netid,
            job_id=job.job_id,
            job_name=job.job_name,
            state=job.state,
            partition=job.partition,
            nodelist=job.nodelist,
            usage_str=job.usage_str,
            time_limit=job.time_limit,
            cpu=job.cpu,
            gpu=job.gpu,
            mem=job.mem,
            shard=job.shard,
        )
        for job in jobs
    ]
    use_shards = server.gpu.shards > 0 and (
        show_shards or any(job.shard > 0 for job in records)
    )
    capacity = server.gpu.shards if use_shards else server.gpu.num
    widths = _job_column_widths(records)
    return Group(
        _build_job_group_header(
            server.name,
            records,
            node_gpu_totals={server.name: capacity},
            gpu_type=_display_gpu_type(server, show_shards=show_shards),
            group_name_width=group_name_width,
            gpu_type_width=gpu_type_width,
            node_gpu_units={server.name: "shard" if use_shards else "GPU"},
            node_shards_per_gpu=(
                {server.name: server.gpu.shards / server.gpu.num}
                if use_shards and server.gpu.num > 0
                else None
            ),
        ),
        _build_job_rows_renderable(records, widths),
    )


def _job_state_label(state: str) -> str:
    normalized = state.upper()
    if normalized.startswith("RUN"):
        return "RUN"
    if normalized.startswith("PEND"):
        return "PEND"
    if normalized.startswith("REQUEUE") or normalized.startswith("REQ"):
        return "REQUEUE"
    return normalized[:7]


def _job_state_style(state: str) -> str:
    normalized = state.upper()
    if normalized.startswith("RUN"):
        return "green"
    if normalized.startswith("PEND"):
        return "yellow"
    if normalized.startswith("REQUEUE") or normalized.startswith("REQ"):
        return "magenta"
    return "white"


def _format_job_count(value: float, *, suffix: str = "") -> str:
    number = int(value) if float(value).is_integer() else round(value, 1)
    return f"{number}{suffix}"


def _job_gpu_count(job: JobRecord) -> int:
    return int(round(job.shard if job.shard > 0 else job.gpu))


def _job_group_resource_count(job: JobRecord, *, shards_per_gpu: float | None) -> int:
    if shards_per_gpu and shards_per_gpu > 0:
        if job.shard > 0:
            return int(round(job.shard))
        if job.gpu > 0:
            return int(round(job.gpu * shards_per_gpu))
    return _job_gpu_count(job)


def _job_group_name(job: JobRecord) -> str:
    if job.state.upper().startswith("PEND"):
        return "Pending / Unassigned"
    nodelist = job.nodelist.strip()
    if not nodelist or nodelist.upper() in {"N/A", "(NULL)", "NONE"}:
        return "Pending / Unassigned"
    return nodelist


def build_jobs_overview(jobs: Sequence[JobRecord], *, title: str) -> Text:
    running = sum(1 for job in jobs if job.state.upper().startswith("RUN"))
    pending = sum(1 for job in jobs if job.state.upper().startswith("PEND"))
    requeued = sum(1 for job in jobs if job.state.upper().startswith("REQ"))

    line = Text()
    line.append(title, style="bold cyan")
    line.append("  ")
    line.append(str(running), style="green")
    line.append(" running", style="white")
    if pending:
        line.append("  ")
        line.append(str(pending), style="yellow")
        line.append(" pending", style="white")
    if requeued:
        line.append("  ")
        line.append(str(requeued), style="white")
        line.append(" requeued", style="white")
    return line


def _build_job_group_header(
    group_name: str,
    jobs: Sequence[JobRecord],
    *,
    node_gpu_totals: Optional[Mapping[str, int]] = None,
    gpu_type: Optional[str] = None,
    group_name_width: Optional[int] = None,
    gpu_type_width: Optional[int] = None,
    node_gpu_units: Optional[Mapping[str, str]] = None,
    node_shards_per_gpu: Optional[Mapping[str, float]] = None,
) -> Text:
    shards_per_gpu = (
        None if node_shards_per_gpu is None else node_shards_per_gpu.get(group_name)
    )
    priority = sum(
        _job_group_resource_count(job, shards_per_gpu=shards_per_gpu)
        for job in jobs
        if partition_bucket(job.partition) == "priority"
    )
    gpu = sum(
        _job_group_resource_count(job, shards_per_gpu=shards_per_gpu)
        for job in jobs
        if partition_bucket(job.partition) == "gpu"
    )
    default = sum(
        _job_group_resource_count(job, shards_per_gpu=shards_per_gpu)
        for job in jobs
        if partition_bucket(job.partition) == "default"
    )
    used = priority + gpu + default
    total_capacity = None if node_gpu_totals is None else node_gpu_totals.get(group_name)
    resource_name = (
        "shard"
        if node_gpu_units is not None and node_gpu_units.get(group_name) == "shard"
        else "GPU"
    )

    summary = Text()
    if total_capacity is not None and total_capacity > 0:
        summary.append(str(used), style="bold yellow")
        summary.append("/", style="dim")
        summary.append(str(total_capacity), style="white")
        summary.append(f" {_pluralize(total_capacity, resource_name)} used  ", style="white")
    else:
        summary.append(str(used), style="bold yellow")
        summary.append(f" {_pluralize(used, resource_name)}  ", style="white")
    summary.append_text(_build_top_user_bar(priority, gpu, default))
    summary.append("  ")
    summary.append_text(_build_split(priority, gpu, default))
    summary.append("  ")
    summary.append(str(len(jobs)), style="white")
    summary.append(f" {_pluralize(len(jobs), 'job')}", style="white")
    line = Text()
    line.append(group_name, style="bold cyan")
    if group_name_width and len(group_name) < group_name_width:
        line.append(" " * (group_name_width - len(group_name)))
    if gpu_type_width is not None:
        line.append("  ")
        gpu_value = gpu_type or ""
        line.append(gpu_value, style="white")
        if len(gpu_value) < gpu_type_width:
            line.append(" " * (gpu_type_width - len(gpu_value)))
    elif gpu_type:
        line.append("  ")
        line.append(gpu_type, style="white")
    line.append("  ")
    line.append_text(summary)
    return line


def _job_column_widths(jobs: Sequence[JobRecord]) -> dict[str, int]:
    widths = {
        "job_id": len("ID"),
        "state": len("State"),
        "user": len("User"),
        "partition": len("Partition"),
        "gpu": len("GPU"),
        "cpu": len("CPU"),
        "mem": len("MEM"),
        "time": len("Time"),
    }
    for job in jobs:
        widths["job_id"] = max(widths["job_id"], len(job.job_id))
        widths["state"] = max(widths["state"], len(_job_state_label(job.state)))
        widths["user"] = max(widths["user"], len(job.user))
        widths["partition"] = max(widths["partition"], len(job.partition))
        gpu_value = (
            _format_job_count(job.gpu)
            if job.gpu > 0
            else ("-" if job.shard <= 0 else _format_job_count(job.shard, suffix="s"))
        )
        widths["gpu"] = max(widths["gpu"], len(gpu_value))
        widths["cpu"] = max(widths["cpu"], len(_format_job_count(job.cpu) if job.cpu > 0 else "-"))
        widths["mem"] = max(
            widths["mem"],
            len(f"{_format_job_count(job.mem)}G" if job.mem > 0 else "-"),
        )
        widths["time"] = max(widths["time"], len(job.time_limit or "-"))
    return widths


def _append_job_field(
    line: Text,
    value: str,
    *,
    width: int,
    style: str,
    justify: str = "left",
) -> None:
    padding = max(width - len(value), 0)
    if justify == "right" and padding:
        line.append(" " * padding)
    line.append(value, style=style)
    if justify == "left" and padding:
        line.append(" " * padding)


def _build_job_rows_header(widths: Mapping[str, int]) -> Text:
    line = Text()
    _append_job_field(line, "ID", width=widths["job_id"], style="bold white")
    line.append(" ")
    _append_job_field(line, "State", width=widths["state"], style="bold white")
    line.append(" ")
    _append_job_field(line, "User", width=widths["user"], style="bold white")
    line.append(" ")
    _append_job_field(line, "Partition", width=widths["partition"], style="bold white")
    line.append(" ")
    _append_job_field(line, "GPU", width=widths["gpu"], style="bold white", justify="right")
    line.append(" ")
    _append_job_field(line, "CPU", width=widths["cpu"], style="bold white", justify="right")
    line.append(" ")
    _append_job_field(line, "MEM", width=widths["mem"], style="bold white", justify="right")
    line.append(" ")
    _append_job_field(line, "Time", width=widths["time"], style="bold white", justify="right")
    line.append(" ")
    line.append("Name", style="bold white")
    return line


def _build_job_row_line(job: JobRecord, widths: Mapping[str, int]) -> Text:
    gpu_value = (
        _format_job_count(job.gpu)
        if job.gpu > 0
        else ("-" if job.shard <= 0 else _format_job_count(job.shard, suffix="s"))
    )
    cpu_value = _format_job_count(job.cpu) if job.cpu > 0 else "-"
    mem_value = f"{_format_job_count(job.mem)}G" if job.mem > 0 else "-"
    time_value = job.time_limit or "-"

    line = Text()
    _append_job_field(line, job.job_id, width=widths["job_id"], style="dim")
    line.append(" ")
    _append_job_field(
        line,
        _job_state_label(job.state),
        width=widths["state"],
        style=_job_state_style(job.state),
    )
    line.append(" ")
    _append_job_field(line, job.user, width=widths["user"], style="cyan")
    line.append(" ")
    _append_job_field(
        line,
        job.partition,
        width=widths["partition"],
        style=_partition_color(job.partition),
    )
    line.append(" ")
    _append_job_field(line, gpu_value, width=widths["gpu"], style="white", justify="right")
    line.append(" ")
    _append_job_field(line, cpu_value, width=widths["cpu"], style="white", justify="right")
    line.append(" ")
    _append_job_field(line, mem_value, width=widths["mem"], style="white", justify="right")
    line.append(" ")
    _append_job_field(line, time_value, width=widths["time"], style="white", justify="right")
    line.append(" ")
    line.append(job.job_name or "-", style="white")
    return line


def _build_job_rows_renderable(
    jobs: Sequence[JobRecord],
    widths: Mapping[str, int],
    *,
    show_header: bool = True,
) -> Group:
    lines: list[Text] = []
    if show_header:
        lines.append(_build_job_rows_header(widths))
    lines.extend(_build_job_row_line(job, widths) for job in jobs)
    return Group(*lines)


def render_jobs_view(
    jobs: Sequence[JobRecord],
    *,
    node_gpu_totals: Optional[Mapping[str, int]] = None,
    node_gpu_types: Optional[Mapping[str, str]] = None,
    node_gpu_units: Optional[Mapping[str, str]] = None,
    node_shards_per_gpu: Optional[Mapping[str, float]] = None,
    include_overview: bool = True,
    title: str = "Jobs",
) -> Group:
    grouped: Dict[str, List[JobRecord]] = {}
    for job in jobs:
        grouped.setdefault(_job_group_name(job), []).append(job)

    def group_sort_key(item: tuple[str, List[JobRecord]]) -> tuple[int, float, int, str]:
        name, group_jobs = item
        is_pending = 1 if name == "Pending / Unassigned" else 0
        shards_per_gpu = (
            None if node_shards_per_gpu is None else node_shards_per_gpu.get(name)
        )
        total_gpu = sum(
            _job_group_resource_count(job, shards_per_gpu=shards_per_gpu)
            for job in group_jobs
        )
        total_capacity = None if node_gpu_totals is None else node_gpu_totals.get(name)
        utilization = (total_gpu / total_capacity) if total_capacity else float(total_gpu)
        return (is_pending, -utilization, -total_gpu, name)

    grouped_items = sorted(grouped.items(), key=group_sort_key)
    group_name_width = max(len(name) for name, _ in grouped_items)
    gpu_type_width = max(
        [0]
        + [
            len(node_gpu_types.get(name, ""))
            for name, _ in grouped_items
            if node_gpu_types is not None
        ]
    )
    widths = _job_column_widths(jobs)

    renderables: list[Any] = []
    if include_overview:
        renderables.append(build_jobs_overview(jobs, title=title))
    pending_jobs: list[JobRecord] = []
    last_rows_jobs: Optional[list[JobRecord]] = None
    for group_name, group_jobs in grouped_items:
        sorted_jobs = sorted(
            group_jobs,
            key=lambda job: (
                job.state,
                job.partition,
                job.user,
                job.job_id,
            ),
        )
        if group_name == "Pending / Unassigned":
            pending_jobs = sorted_jobs
            continue
        renderables.append(Text(""))
        renderables.append(
            _build_job_group_header(
                group_name,
                sorted_jobs,
                node_gpu_totals=node_gpu_totals,
                gpu_type=None if node_gpu_types is None else node_gpu_types.get(group_name),
                group_name_width=group_name_width,
                gpu_type_width=gpu_type_width if gpu_type_width > 0 else None,
                node_gpu_units=node_gpu_units,
                node_shards_per_gpu=node_shards_per_gpu,
            )
        )
        last_rows_jobs = list(sorted_jobs)
        renderables.append(_build_job_rows_renderable(last_rows_jobs, widths))
    if pending_jobs:
        if last_rows_jobs is None:
            renderables.append(Text(""))
            renderables.append(_build_job_rows_renderable(pending_jobs, widths))
        else:
            last_rows_jobs.extend(pending_jobs)
            renderables[-1] = _build_job_rows_renderable(last_rows_jobs, widths)
    return Group(*renderables)


def _display_gpu_type(server: ServerState, *, show_shards: bool) -> str:
    gpu_type = server.gpu.type
    if gpu_type.startswith("Shard(") and gpu_type.endswith(")"):
        gpu_type = gpu_type[6:-1]
    replacements = [
        ("nvidia_geforce_", ""),
        ("nvidia_rtx_", ""),
        ("nvidia_", ""),
        ("tesla_", ""),
        ("geforce_", ""),
    ]
    for source, target in replacements:
        gpu_type = gpu_type.replace(source, target)
    gpu_type = gpu_type.replace("_generation", "")
    gpu_type = gpu_type.replace("_workstation_edition", "")
    gpu_type = gpu_type.replace("_", " ")
    words = gpu_type.split()
    known_tokens = {
        "rtx": "RTX",
        "gtx": "GTX",
        "hbm": "HBM",
        "nvl": "NVL",
        "pcie": "PCIe",
        "sxm": "SXM",
        "mxm": "MXM",
        "gb": "GB",
        "tb": "TB",
    }
    compact_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in known_tokens:
            compact_words.append(known_tokens[lower_word])
            continue
        if any(char.isdigit() for char in word):
            compact_words.append(word.upper().replace("PCIE", "PCIe"))
            continue
        compact_words.append(word.title())
    compact = " ".join(compact_words)
    compact = compact.replace("Rtx ", "").replace("Geforce ", "").replace("Tesla ", "")
    compact = compact.replace(" TI", " Ti")
    return compact or "GPU"


def _resource_numbers(
    server: ServerState,
    resource: str,
    *,
    show_shards: bool,
    show_used: bool = False,
) -> Tuple[int, int, int, int, int]:
    usage = _resource_split(server, resource, show_shards)
    priority = int(round(usage.priority))
    gpu = int(round(usage.gpu))
    default = int(round(usage.default))
    used = priority + gpu + default
    if resource == "gpu":
        total = server.gpu.shards if show_shards and server.gpu.shards > 0 else server.gpu.num
        count = used if show_used else max(int(total) - used, 0)
        return count, int(total), priority, gpu, default
    if resource == "cpu":
        total = used + server.cpu.idle
        count = used if show_used else server.cpu.idle
        return count, total, priority, gpu, default

    total = int(round(server.mem.total / 1024.0))
    free = int(round(server.mem.idle / 1024.0))
    count = used if show_used else free
    return count, total, priority, gpu, default


def _split_segments(
    priority: int, gpu: int, default: int, free: int, width: int
) -> tuple[int, int, int, int]:
    total = priority + gpu + default + free
    if width <= 0:
        return 0, 0, 0, 0
    if total <= 0:
        return 0, 0, 0, width

    raw = [
        priority / total * width,
        gpu / total * width,
        default / total * width,
        free / total * width,
    ]
    base = [math.floor(value) for value in raw]
    remainder = width - sum(base)
    order = sorted(
        range(4),
        key=lambda index: (raw[index] - base[index], raw[index]),
        reverse=True,
    )
    for index in order:
        if remainder <= 0:
            break
        if [priority, gpu, default, free][index] > 0:
            base[index] += 1
            remainder -= 1
    if remainder > 0:
        base[3] += remainder
    return base[0], base[1], base[2], base[3]


def _build_bar(priority: int, gpu: int, default: int, free: int, width: int) -> Text:
    priority_len, gpu_len, default_len, free_len = _split_segments(
        priority, gpu, default, free, width
    )
    bar = Text("[", style="dim")
    if priority_len:
        bar.append("█" * priority_len, style=PRIORITY_PARTITION_COLOR)
    if gpu_len:
        bar.append("█" * gpu_len, style=GPU_PARTITION_COLOR)
    if default_len:
        bar.append("█" * default_len, style=DEFAULT_PARTITION_COLOR)
    if free_len:
        bar.append("·" * free_len, style="dim")
    bar.append("]", style="dim")
    return bar


def _build_split(priority: int, gpu: int, default: int) -> Text:
    split = Text()
    split.append(
        str(priority),
        style=f"dim {PRIORITY_PARTITION_COLOR}"
        if priority == 0
        else PRIORITY_PARTITION_COLOR,
    )
    split.append("/", style="dim")
    split.append(
        str(gpu),
        style=f"dim {GPU_PARTITION_COLOR}" if gpu == 0 else GPU_PARTITION_COLOR,
    )
    split.append("/", style="dim")
    split.append(
        str(default),
        style=f"dim {DEFAULT_PARTITION_COLOR}"
        if default == 0
        else DEFAULT_PARTITION_COLOR,
    )
    return split


def _availability_style(count: int, total: int, *, show_used: bool = False) -> str:
    if total <= 0:
        return "white"
    if show_used:
        if count <= 0:
            return "bright_black"
        if count >= total:
            return "bright_white"
        return "white"
    if count <= 0:
        return "red"
    if count >= total:
        return "green"
    return "yellow"


def _build_counts(
    count: int,
    total: int,
    width: int,
    *,
    show_used: bool,
    show_total: bool = True,
) -> Text:
    counts = Text()
    counts.append(str(count), style=_availability_style(count, total, show_used=show_used))
    if show_total:
        counts.append("/", style="dim")
        counts.append(str(total), style="white")
    return counts


def _build_top_user_bar(priority: int, gpu: int, default: int) -> Text:
    return _build_bar(priority, gpu, default, 0, TOP_USER_BAR_WIDTH)


def _build_top_users_table(summary: ClusterSummary) -> Table:
    resource_header = _pluralize(2, summary.resource_label)
    table = Table(
        box=None,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        padding=(0, 1),
    )
    table.add_column("User", header_style="bold white", no_wrap=True)
    table.add_column(resource_header, header_style="bold white", justify="right", no_wrap=True)
    table.add_column("Nodes", header_style="bold white", justify="right", no_wrap=True)
    table.add_column("", header_style="bold white", no_wrap=True)
    table.add_column(
        "",
        header_style="bold white",
        no_wrap=True,
    )
    for stats in summary.top_users:
        table.add_row(
            Text(_top_user_label(stats.user), style="cyan"),
            Text(str(stats.total_usage()), style="bold yellow"),
            Text(str(len(stats.nodes)), style=NODE_COUNT_COLOR),
            _build_top_user_bar(
                stats.priority_usage,
                stats.gpu_usage,
                stats.default_usage,
            ),
            _build_split(
                stats.priority_usage,
                stats.gpu_usage,
                stats.default_usage,
            ),
        )
    return table


def _resource_layout(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool = False,
    grouped_servers_list: Optional[Sequence[Tuple[str, Sequence[ServerState]]]] = None,
) -> Dict[str, Dict[str, int]]:
    layout: Dict[str, Dict[str, int]] = {}
    for resource in ("gpu", "cpu", "mem"):
        counts_width = 0
        split_width = 0
        for server in servers:
            count, total, priority, gpu, default = _resource_numbers(
                server, resource, show_shards=show_shards, show_used=show_used
            )
            counts_width = max(counts_width, len(f"{count}/{total}"))
            split_width = max(split_width, len(f"{priority}/{gpu}/{default}"))
        if grouped_servers_list:
            for _, grouped_servers in grouped_servers_list:
                count, total, priority, gpu, default = _group_resource_numbers(
                    grouped_servers,
                    resource,
                    show_shards=show_shards,
                    show_used=show_used,
                )
                counts_width = max(counts_width, len(f"{count}/{total}"))
                split_width = max(split_width, len(f"{priority}/{gpu}/{default}"))
        bar_width = BAR_WIDTHS[resource]
        column_width = (
            counts_width
            + CELL_GAP
            + bar_width
            + 2
            + CELL_GAP
            + split_width
        )
        layout[resource] = {
            "counts_width": counts_width,
            "split_width": split_width,
            "bar_width": bar_width,
            "column_width": max(column_width, GROUP_HEADER_WIDTHS[resource]),
        }
    return layout


def _resource_cell(
    count: int,
    total: int,
    priority: int,
    gpu: int,
    default: int,
    *,
    counts_width: int,
    split_width: int,
    bar_width: int,
    show_used: bool,
    show_total: bool = True,
) -> Text:
    split_text = _build_split(priority, gpu, default)
    split_value = f"{priority}/{gpu}/{default}"
    bar = _build_bar(priority, gpu, default, max(total - priority - gpu - default, 0), bar_width)
    cell = Text()
    cell.append_text(
        _build_counts(
            count,
            total,
            counts_width,
            show_used=show_used,
            show_total=show_total,
        )
    )
    count_value = f"{count}/{total}" if show_total else str(count)
    if len(count_value) < counts_width:
        cell.append(" " * (counts_width - len(count_value)))
    cell.append(" " * CELL_GAP)
    cell.append_text(bar)
    cell.append(" " * CELL_GAP)
    cell.append_text(split_text)
    if len(split_value) < split_width:
        cell.append(" " * (split_width - len(split_value)))
    return cell


def _resource_cell_for_server(
    server: ServerState,
    resource: str,
    *,
    counts_width: int,
    split_width: int,
    bar_width: int,
    show_shards: bool,
    show_used: bool,
) -> Text:
    count, total, priority, gpu, default = _resource_numbers(
        server, resource, show_shards=show_shards, show_used=show_used
    )
    return _resource_cell(
        count,
        total,
        priority,
        gpu,
        default,
        counts_width=counts_width,
        split_width=split_width,
        bar_width=bar_width,
        show_used=show_used,
    )


def _resolve_layout(
    width: Optional[int],
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool = False,
    resource_layout: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, int]:
    layout = dict(GROUP_HEADER_WIDTHS)
    if resource_layout is None:
        resource_layout = _resource_layout(
            servers,
            show_shards=show_shards,
            show_used=show_used,
        )
    layout["group"] = max(
        len("GPU Type"),
        *(
        len(_display_gpu_type(server, show_shards=show_shards))
        for server in servers
        ),
    )
    layout["node"] = max(
        layout["node"],
        len("Node"),
        *(len(server.name) for server in servers),
    )
    layout["gpu"] = resource_layout["gpu"]["column_width"]
    layout["cpu"] = resource_layout["cpu"]["column_width"]
    layout["mem"] = resource_layout["mem"]["column_width"]
    return layout


def _split_group_gpu_types(gpu_type: str) -> list[str]:
    if gpu_type.startswith("(") and gpu_type.endswith(")") and "|" in gpu_type:
        return [part.strip() for part in gpu_type[1:-1].split("|") if part.strip()]
    return [gpu_type]


def _single_gpu_capability_rank(gpu_type: str) -> int:
    normalized = gpu_type.upper()
    for pattern, rank in GPU_CAPABILITY_PATTERNS:
        if pattern in normalized:
            return rank
    return len(GPU_CAPABILITY_PATTERNS)


def _gpu_capability_rank(gpu_type: str) -> tuple[int, int, str]:
    members = _split_group_gpu_types(gpu_type)
    ranks = sorted(_single_gpu_capability_rank(member) for member in members)
    strongest = max(ranks)
    weakest = min(ranks)
    return strongest, weakest, gpu_type.upper()


def _group_servers(
    servers: Sequence[ServerState], *, show_shards: bool, show_used: bool = False
) -> List[Tuple[str, List[ServerState]]]:
    grouped: Dict[str, List[ServerState]] = {}
    for server in servers:
        grouped.setdefault(_display_gpu_type(server, show_shards=show_shards), []).append(server)

    def gpu_free(server: ServerState) -> int:
        count, _, _, _, _ = _resource_numbers(
            server,
            "gpu",
            show_shards=show_shards,
            show_used=show_used,
        )
        return count

    groups = []
    for gpu_type, members in grouped.items():
        groups.append(
            (
                gpu_type,
                sorted(
                    members,
                    key=lambda server: (
                        -gpu_free(server),
                        -_resource_numbers(
                            server,
                            "cpu",
                            show_shards=show_shards,
                            show_used=show_used,
                        )[0],
                        server.name,
                    ),
                ),
            )
        )
    return sorted(
        groups,
        key=lambda item: (
            *_gpu_capability_rank(item[0]),
            -sum(
                _resource_numbers(
                    server,
                    "gpu",
                    show_shards=show_shards,
                    show_used=show_used,
                )[0]
                for server in item[1]
            ),
        ),
    )


def _group_header_summary(
    count: int,
    total_count: int,
    *,
    show_shards: bool,
    show_used: bool,
) -> Text:
    summary = Text()
    summary.append(
        str(count),
        style=f"bold {_availability_style(count, total_count, show_used=show_used)}",
    )
    if not show_used:
        summary.append("/")
        summary.append(str(total_count), style="white")
    summary.append(
        f" {_pluralize(count, 'shard' if show_shards else 'GPU')} "
        f"{'used' if show_used else 'free'}"
    )
    return summary


def _build_group_header_table(
    gpu_type: str,
    count: int,
    total_count: int,
    *,
    group_width: int,
    show_shards: bool,
    show_used: bool,
) -> Table:
    table = Table.grid(padding=(0, GROUP_GAP), expand=False)
    table.add_column(width=group_width, no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_row(
        Text(_pad(gpu_type, group_width), style="bold cyan"),
        _group_header_summary(
            count,
            total_count,
            show_shards=show_shards,
            show_used=show_used,
        ),
    )
    return table


def _cluster_overview_summary(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool,
) -> Text:
    count = sum(
        _resource_numbers(
            server,
            "gpu",
            show_shards=show_shards,
            show_used=show_used,
        )[0]
        for server in servers
    )
    total_count = sum(
        _resource_numbers(
            server,
            "gpu",
            show_shards=show_shards,
            show_used=show_used,
        )[1]
        for server in servers
    )
    summary = Text()
    summary.append(
        str(count),
        style=f"bold {_availability_style(count, total_count, show_used=show_used)}",
    )
    if not show_used:
        summary.append("/", style="dim")
        summary.append(str(total_count), style="white")
    summary.append(
        f" {_pluralize(count, 'shard' if show_shards else 'GPU')} "
        f"{'used' if show_used else 'free'}"
    )
    return summary


def _build_cluster_overview_table(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool,
    overview_title: str,
) -> Table:
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_row(
        Text(overview_title, style="bold cyan"),
        _cluster_overview_summary(
            servers,
            show_shards=show_shards,
            show_used=show_used,
        ),
    )
    return table


def _build_nodes_table(
    grouped_servers: Sequence[ServerState],
    *,
    layout: Mapping[str, int],
    resource_layout: Mapping[str, Mapping[str, int]],
    show_shards: bool,
    show_header: bool,
    show_used: bool,
) -> Table:
    table = Table(
        box=None,
        show_header=show_header,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        padding=(0, 1),
    )
    table.add_column(
        "",
        header_style="bold white",
        no_wrap=True,
        width=layout["node"],
        overflow="ignore",
    )
    table.add_column(
        "Shard" if show_shards else "GPU",
        header_style="bold white",
        no_wrap=True,
        width=layout["gpu"],
        overflow="ignore",
    )
    table.add_column(
        "CPU",
        header_style="bold white",
        no_wrap=True,
        width=layout["cpu"],
        overflow="ignore",
    )
    table.add_column(
        "Memory",
        header_style="bold white",
        no_wrap=True,
        width=layout["mem"],
        overflow="ignore",
    )
    for server in grouped_servers:
        table.add_row(
            Text(_pad(server.name, layout["node"]), style="bright_black"),
            _resource_cell_for_server(
                server,
                "gpu",
                counts_width=resource_layout["gpu"]["counts_width"],
                split_width=resource_layout["gpu"]["split_width"],
                bar_width=resource_layout["gpu"]["bar_width"],
                show_shards=show_shards,
                show_used=show_used,
            ),
            _resource_cell_for_server(
                server,
                "cpu",
                counts_width=resource_layout["cpu"]["counts_width"],
                split_width=resource_layout["cpu"]["split_width"],
                bar_width=resource_layout["cpu"]["bar_width"],
                show_shards=show_shards,
                show_used=show_used,
            ),
            _resource_cell_for_server(
                server,
                "mem",
                counts_width=resource_layout["mem"]["counts_width"],
                split_width=resource_layout["mem"]["split_width"],
                bar_width=resource_layout["mem"]["bar_width"],
                show_shards=show_shards,
                show_used=show_used,
            ),
        )
    return table


def _group_resource_numbers(
    grouped_servers: Sequence[ServerState],
    resource: str,
    *,
    show_shards: bool,
    show_used: bool,
) -> tuple[int, int, int, int, int]:
    count = 0
    total = 0
    priority = 0
    gpu = 0
    default = 0
    for server in grouped_servers:
        server_count, server_total, server_priority, server_gpu, server_default = _resource_numbers(
            server,
            resource,
            show_shards=show_shards,
            show_used=show_used,
        )
        count += server_count
        total += server_total
        priority += server_priority
        gpu += server_gpu
        default += server_default
    return count, total, priority, gpu, default


def _build_summary_table(
    grouped_servers_list: Sequence[Tuple[str, Sequence[ServerState]]],
    *,
    layout: Mapping[str, int],
    resource_layout: Mapping[str, Mapping[str, int]],
    show_shards: bool,
    show_used: bool,
) -> Table:
    table = Table(
        box=None,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        padding=(0, 1),
    )
    table.add_column(
        "GPU Type",
        header_style="bold white",
        no_wrap=True,
        width=layout["group"],
        overflow="ignore",
    )
    table.add_column(
        "Shard" if show_shards else "GPU",
        header_style="bold white",
        no_wrap=True,
        width=layout["gpu"],
        overflow="ignore",
    )
    table.add_column(
        "Nodes",
        header_style="bold white",
        no_wrap=True,
        justify="right",
        width=5,
        overflow="ignore",
    )
    for gpu_type, grouped_servers in grouped_servers_list:
        gpu_count, gpu_total, gpu_priority, gpu_gpu, gpu_default = _group_resource_numbers(
            grouped_servers,
            "gpu",
            show_shards=show_shards,
            show_used=show_used,
        )
        table.add_row(
            Text(_pad(gpu_type, layout["group"]), style="bold cyan"),
            _resource_cell(
                gpu_count,
                gpu_total,
                gpu_priority,
                gpu_gpu,
                gpu_default,
                counts_width=resource_layout["gpu"]["counts_width"],
                split_width=resource_layout["gpu"]["split_width"],
                bar_width=resource_layout["gpu"]["bar_width"],
                show_used=show_used,
                show_total=not show_used,
            ),
            Text(str(len(grouped_servers)), style=NODE_COUNT_COLOR),
        )
    return table


def render_table(
    servers: Sequence[ServerState],
    *,
    show_jobs: bool,
    show_shards: bool = False,
    width: Optional[int] = None,
    show_used: bool = False,
    overview_title: str = "Cluster Overview",
    verbose: bool = False,
) -> Any:
    grouped_servers_list = _group_servers(
        servers,
        show_shards=show_shards,
        show_used=show_used,
    )
    resource_layout = _resource_layout(
        servers,
        show_shards=show_shards,
        show_used=show_used,
        grouped_servers_list=grouped_servers_list,
    )
    layout = _resolve_layout(
        width,
        servers,
        show_shards=show_shards,
        show_used=show_used,
        resource_layout=resource_layout,
    )
    renderables: list[Any] = []
    renderables.append(
        _build_cluster_overview_table(
            servers,
            show_shards=show_shards,
            show_used=show_used,
            overview_title=overview_title,
        )
    )
    if not verbose:
        renderables.append(
            _build_summary_table(
                grouped_servers_list,
                layout=layout,
                resource_layout=resource_layout,
                show_shards=show_shards,
                show_used=show_used,
            )
        )
        return Group(*renderables)

    renderables.append(
        _build_nodes_table(
            [],
            layout=layout,
            resource_layout=resource_layout,
            show_shards=show_shards,
            show_header=True,
            show_used=show_used,
        )
    )
    for index, (gpu_type, grouped_servers) in enumerate(grouped_servers_list):
        count = sum(
            _resource_numbers(
                server,
                "gpu",
                show_shards=show_shards,
                show_used=show_used,
            )[0]
            for server in grouped_servers
        )
        total_gpus = sum(
            _resource_numbers(
                server,
                "gpu",
                show_shards=show_shards,
                show_used=show_used,
            )[1]
            for server in grouped_servers
        )
        if index > 0:
            renderables.append(Text(""))
        renderables.append(
            _build_group_header_table(
                gpu_type,
                count,
                total_gpus,
                group_width=layout["group"],
                show_shards=show_shards,
                show_used=show_used,
            )
        )
        renderables.append(
            _build_nodes_table(
                grouped_servers,
                layout=layout,
                resource_layout=resource_layout,
                show_shards=show_shards,
                show_header=False,
                show_used=show_used,
            )
        )

    if show_jobs:
        for server in servers:
            if server.users:
                renderables.append(Text(""))
                renderables.append(
                    _build_jobs_table(
                        server,
                        show_shards=show_shards,
                        group_name_width=layout["node"],
                        gpu_type_width=layout["group"],
                    )
                )

    return Group(*renderables)


def build_json_payload(
    summary: ClusterSummary,
    servers: Sequence[ServerState],
    *,
    sort_by: str,
    show_jobs: bool,
    show_shards: bool,
) -> Dict[str, Any]:
    return {
        "summary": to_jsonable(summary),
        "servers": to_jsonable(servers),
        "options": {
            "show_jobs": show_jobs,
            "show_shards": show_shards,
            "sort_by": sort_by,
        },
    }

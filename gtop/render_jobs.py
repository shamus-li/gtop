from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .models import JobRecord, ServerState
from .render import (
    TOP_USER_BAR_WIDTH,
    _build_bar,
    _build_split,
    _data_table,
    _detail_policy,
    _display_gpu_type,
    _fits_width,
    _max_width,
    _partition_color,
    _pluralize,
)
from .slurm import parse_nodelist


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


def _job_gpu_value(
    job: JobRecord,
    *,
    unit: str = "GPU",
    servers: Mapping[str, ServerState] | None = None,
) -> str:
    if unit == "shard" and servers is not None:
        shards = sum(
            server.allocations[job.job_id].usage.shard
            for node in parse_nodelist(job.nodelist)
            if (
                (server := servers.get(node)) is not None
                and job.job_id in server.allocations
            )
        )
        if shards > 0:
            return _format_job_count(shards, suffix="s")
    if job.usage.gpu > 0:
        return _format_job_count(job.usage.gpu)
    if job.usage.shard > 0:
        return _format_job_count(job.usage.shard, suffix="s")
    return "-"


def _job_cpu_value(job: JobRecord) -> str:
    return _format_job_count(job.usage.cpu) if job.usage.cpu > 0 else "-"


def _job_mem_value(job: JobRecord) -> str:
    return f"{_format_job_count(job.usage.mem)}G" if job.usage.mem > 0 else "-"


def _job_assignment(job: JobRecord) -> str:
    return job.nodelist if parse_nodelist(job.nodelist) else "-"


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


@dataclass(frozen=True)
class _JobGroup:
    assignment: str
    jobs: tuple[JobRecord, ...]
    total: int | None
    gpu_type: str
    unit: str
    partitions: Mapping[str, float]
    used: int


def _build_job_groups(
    jobs: Sequence[JobRecord],
    *,
    servers: Mapping[str, ServerState],
) -> list[_JobGroup]:
    grouped: dict[str, list[JobRecord]] = {}
    for job in jobs:
        assignment = _job_assignment(job)
        if assignment != "-":
            grouped.setdefault(assignment, []).append(job)

    groups: list[_JobGroup] = []
    for assignment, group_jobs in grouped.items():
        nodes = {
            node
            for job in group_jobs
            for node in parse_nodelist(job.nodelist)
            if node in servers
        }
        group_servers = [servers[node] for node in sorted(nodes)]
        all_sharded = bool(group_servers) and all(
            server.gpu.shards > 0 for server in group_servers
        )
        unit = "shard" if all_sharded else "GPU"
        total = (
            sum(
                server.gpu.shards if all_sharded else server.gpu.num
                for server in group_servers
            )
            if group_servers
            else None
        )
        gpu_types = {_display_gpu_type(server) for server in group_servers}
        gpu_type = next(iter(gpu_types)) if len(gpu_types) == 1 else "Mixed"
        if not group_servers:
            gpu_type = "-"
            unit = (
                "shard"
                if any(job.usage.shard > 0 for job in group_jobs)
                and all(job.usage.gpu <= 0 for job in group_jobs)
                else "GPU"
            )

        job_ids = {job.job_id for job in group_jobs}
        partitions: dict[str, float] = {}
        for server in group_servers:
            for job_id, allocation in server.allocations.items():
                if job_id in job_ids:
                    amount = (
                        allocation.usage.shard if all_sharded else allocation.usage.gpu
                    )
                    partition = allocation.job.partition
                    partitions[partition] = partitions.get(partition, 0.0) + amount
        if not group_servers:
            for job in group_jobs:
                amount = job.usage.shard if unit == "shard" else job.usage.gpu
                partitions[job.partition] = partitions.get(job.partition, 0.0) + amount

        groups.append(
            _JobGroup(
                assignment=assignment,
                jobs=tuple(group_jobs),
                total=total,
                gpu_type=gpu_type,
                unit=unit,
                partitions=partitions,
                used=int(round(sum(partitions.values()))),
            )
        )

    return sorted(
        groups,
        key=lambda group: (
            -(group.used / group.total if group.total else float(group.used)),
            -group.used,
            group.assignment,
        ),
    )


def _capacity_value(group: _JobGroup) -> Text:
    value = Text()
    value.append(str(group.used), style="bold yellow")
    if group.total is not None and group.total > 0:
        value.append("/", style="dim")
        value.append(str(group.total), style="white")
        count = group.total
    else:
        count = group.used
    value.append(f" {_pluralize(count, group.unit)} used", style="white")
    return value


def _capacity_table(groups: Sequence[_JobGroup], *, width: Optional[int]) -> Table:
    capacities = [_capacity_value(group) for group in groups]
    job_labels = [
        f"{len(group.jobs)} {_pluralize(len(group.jobs), 'job')}" for group in groups
    ]
    required_widths = [
        _max_width("Node", [group.assignment for group in groups]),
        _max_width("GPU Type", [group.gpu_type for group in groups]),
        _max_width("Capacity", [value.plain for value in capacities]),
        _max_width("Jobs", job_labels),
    ]
    split_width = max(len(_build_split(group.partitions).plain) for group in groups)
    include_bar, include_split = _detail_policy(
        width,
        required_widths=required_widths,
        bar_widths=[TOP_USER_BAR_WIDTH + 2],
        split_widths=[split_width],
    )

    if not _fits_width(width, required_widths):
        table = _data_table(show_header=True)
        table.add_column("Node / GPU capacity", header_style="bold white")
        for group, capacity, jobs_label in zip(groups, capacities, job_labels):
            card = Text(group.assignment, style="bold cyan")
            if group.gpu_type != "-":
                card.append("\n")
                card.append(group.gpu_type, style="white")
            card.append("\n")
            card.append_text(capacity)
            card.append("  ")
            card.append(jobs_label, style="white")
            table.add_row(card)
        return table

    table = _data_table(show_header=True)
    for label in ("Node", "GPU Type", "Capacity"):
        table.add_column(label, header_style="bold white", no_wrap=True)
    if include_bar:
        table.add_column("Usage", header_style="bold white", no_wrap=True)
    if include_split:
        table.add_column("Partitions", header_style="bold white", no_wrap=True)
    table.add_column("Jobs", header_style="bold white", no_wrap=True)
    for group, capacity, jobs_label in zip(groups, capacities, job_labels):
        row: list[Text] = [
            Text(group.assignment, style="bold cyan"),
            Text(group.gpu_type, style="white"),
            capacity,
        ]
        if include_bar:
            row.append(
                _build_bar(
                    group.partitions,
                    free=max((group.total or group.used) - group.used, 0),
                    width=TOP_USER_BAR_WIDTH,
                )
            )
        if include_split:
            row.append(_build_split(group.partitions))
        row.append(Text(jobs_label, style="white"))
        table.add_row(*row)
    return table


_DETAIL_HEADERS = (
    "ID",
    "Node",
    "GPU",
    "CPU",
    "MEM",
    "Partition",
    "State",
    "User",
    "Time",
    "Name",
)


def _job_rows(
    jobs: Sequence[JobRecord],
    *,
    groups: Sequence[_JobGroup],
    servers: Mapping[str, ServerState],
) -> list[tuple[str, ...]]:
    units = {group.assignment: group.unit for group in groups}
    rows = []
    for job in jobs:
        assignment = _job_assignment(job)
        rows.append(
            (
                job.job_id,
                assignment,
                _job_gpu_value(
                    job,
                    unit=units.get(assignment, "GPU"),
                    servers=servers,
                ),
                _job_cpu_value(job),
                _job_mem_value(job),
                job.partition or "-",
                _job_state_label(job.state),
                job.user,
                job.time_limit or "-",
                job.job_name or "-",
            )
        )
    return rows


def _job_cards(rows: Sequence[tuple[str, ...]]) -> Text:
    cards = Text()
    for row_index, row in enumerate(rows):
        if row_index:
            cards.append("\n\n")
        values = dict(zip(_DETAIL_HEADERS, row))
        for label in ("ID", "Node", "Partition", "State", "User", "Time", "Name"):
            if label != "ID":
                cards.append("\n")
            cards.append(f"{label}: ", style="bold white")
            style = {
                "ID": "bright_black",
                "Partition": _partition_color(values[label]),
                "State": _job_state_style(values[label]),
                "User": "cyan",
            }.get(label, "white")
            cards.append(values[label], style=style)
        cards.append("\nGPU/CPU/MEM: ", style="bold white")
        cards.append("/".join(row[2:5]), style="white")
    return cards


def _job_details(rows: Sequence[tuple[str, ...]], *, width: Optional[int]) -> Any:
    widths = [
        _max_width(header, [row[index] for row in rows])
        for index, header in enumerate(_DETAIL_HEADERS)
    ]
    if not _fits_width(width, widths):
        return _job_cards(rows)

    table = _data_table(show_header=True)
    for index, label in enumerate(_DETAIL_HEADERS):
        table.add_column(
            label,
            header_style="bold white",
            justify="right" if label in {"GPU", "CPU", "MEM", "Time"} else "left",
            no_wrap=True,
        )
    for row in rows:
        table.add_row(
            Text(row[0], style="dim"),
            Text(row[1], style="cyan"),
            *(Text(value, style="white") for value in row[2:5]),
            Text(row[5], style=_partition_color(row[5])),
            Text(row[6], style=_job_state_style(row[6])),
            Text(row[7], style="cyan"),
            Text(row[8], style="white"),
            Text(row[9], style="white"),
        )
    return table


def render_jobs_view(
    jobs: Sequence[JobRecord],
    *,
    servers: Mapping[str, ServerState],
    title: str = "Jobs",
    width: Optional[int] = None,
) -> Group:
    groups = _build_job_groups(jobs, servers=servers)
    state_order = {"RUN": 0, "PEND": 1, "REQUEUE": 2}
    sorted_jobs = sorted(
        jobs,
        key=lambda job: (
            state_order.get(_job_state_label(job.state), 3),
            _job_assignment(job) == "-",
            _job_assignment(job),
            job.partition,
            job.user,
            job.job_id,
        ),
    )
    rows = _job_rows(sorted_jobs, groups=groups, servers=servers)
    renderables: list[Any] = [build_jobs_overview(jobs, title=title), Text("")]
    if groups:
        renderables.extend((_capacity_table(groups, width=width), Text("")))
    renderables.append(_job_details(rows, width=width))
    return Group(*renderables)

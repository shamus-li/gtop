from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any, Collection, Mapping, Optional, Sequence

from rich.cells import cell_len, chop_cells
from rich.console import (
    Console,
    ConsoleOptions,
    Group,
    RenderResult,
)
from rich.segment import Segment
from rich.style import Style
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
    _max_width,
    _partition_color,
    _pluralize,
)
from .slurm import array_task_count, parse_nodelist


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
            for node in _job_nodes(job.nodelist)
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


@lru_cache(maxsize=4096)
def _job_nodes(nodelist: str) -> tuple[str, ...]:
    return tuple(parse_nodelist(nodelist))


def _job_assignment(job: JobRecord) -> str:
    return job.nodelist if _job_nodes(job.nodelist) else "-"


def build_jobs_overview(jobs: Sequence[JobRecord], *, title: str) -> Text:
    running = sum(1 for job in jobs if job.state.upper().startswith("RUN"))
    pending = sum(
        array_task_count(job.job_id)
        for job in jobs
        if job.state.upper().startswith("PEND")
    )
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
            for node in _job_nodes(job.nodelist)
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


@dataclass(frozen=True)
class _PlainTable:
    """Lays out rows like a borderless rich Table without per-cell rendering.

    A cell is a ``(value, style)`` pair or a multi-style ``Text`` in a column
    that never shrinks. When rows are too wide, ``shrink`` columns narrow down
    to their minimum widths, wrapping or, for ``ellipsize`` columns, truncating
    with an ellipsis. If that is not enough, ``drop`` columns are
    removed in order.
    """

    headers: Sequence[str]
    rows: Sequence[Sequence[Text | tuple[str, str]]]
    right: Collection[str] = ()
    shrink: Collection[str] = ()
    ellipsize: Collection[str] = ()
    drop: Sequence[str] = ()
    min_widths: Mapping[str, int] = field(default_factory=dict)

    @cached_property
    def _natural_widths(self) -> list[int]:
        return [
            max([len(header), *(_cell_width(row[index]) for row in self.rows)])
            for index, header in enumerate(self.headers)
        ]

    def _layout(self, max_width: int) -> Optional[tuple[list[int], list[int]]]:
        """Returns the kept column indexes and their widths with padding."""
        kept = list(range(len(self.headers)))
        for dropped in (None, *self.drop):
            if dropped is not None:
                kept.remove(self.headers.index(dropped))
            pads = [int(position < len(kept) - 1) for position in range(len(kept))]
            widths = [self._natural_widths[index] + pad for index, pad in zip(kept, pads)]
            minimums = [
                min(
                    self.min_widths.get(self.headers[index], len(self.headers[index])),
                    self._natural_widths[index],
                )
                + pad
                if self.headers[index] in self.shrink
                else width
                for index, pad, width in zip(kept, pads, widths)
            ]
            if sum(minimums) > max_width:
                continue
            # Narrow the widest shrinkable column first, as rich's Table does.
            for _ in range(sum(widths) - max_width):
                position = max(
                    (
                        position
                        for position, minimum in enumerate(minimums)
                        if widths[position] > minimum
                    ),
                    key=lambda position: widths[position],
                )
                widths[position] -= 1
            return kept, widths
        return None

    def fits(self, width: Optional[int]) -> bool:
        return width is None or self._layout(width) is not None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        layout = self._layout(options.max_width)
        assert layout is not None
        kept, widths = layout
        headers = [self.headers[index] for index in kept]
        rows = [
            [(header, "bold white") for header in headers],
            *([row[index] for index in kept] for row in self.rows),
        ]
        pads = [int(position < len(kept) - 1) for position in range(len(kept))]
        content_widths = [width - pad for width, pad in zip(widths, pads)]
        rights = [header in self.right for header in headers]
        ellipsized = [header in self.ellipsize for header in headers]
        styles: dict[str, Style] = {}
        fitted: dict[tuple[str, int, bool, bool], list[str]] = {}
        # rich styles header padding with the header style.
        header_gaps = [
            (Segment(" ", console.get_style("bold white")),) if pad else ()
            for pad in pads
        ]
        gaps = [(Segment(" "),) if pad else () for pad in pads]
        new_line = Segment.line()
        for row_index, row in enumerate(rows):
            cells = [
                _cell_lines(console, styles, fitted, cell, width, right, ellipsis)
                for cell, width, right, ellipsis in zip(
                    row, content_widths, rights, ellipsized
                )
            ]
            row_gaps = header_gaps if row_index == 0 else gaps
            for line_index in range(max(len(lines) for lines in cells)):
                for lines, width, gap in zip(cells, content_widths, row_gaps):
                    if line_index < len(lines):
                        yield from lines[line_index]
                    else:
                        yield Segment(" " * width)
                    yield from gap
                yield new_line


def _cell_width(cell: Text | tuple[str, str]) -> int:
    return cell.cell_len if isinstance(cell, Text) else cell_len(cell[0])


def _cell_lines(
    console: Console,
    styles: dict[str, Style],
    fitted: dict[tuple[str, int, bool, bool], list[str]],
    cell: Text | tuple[str, str],
    width: int,
    right: bool,
    ellipsis: bool,
) -> Sequence[Sequence[Segment]]:
    cell_width = _cell_width(cell)
    if isinstance(cell, Text):
        style = console.get_style(cell.style)
        padding = Segment(" " * (width - cell_width), style)
        segments = list(Segment.apply_style(cell.render(console), style))
        return ([padding, *segments] if right else [*segments, padding],)
    value, style_name = cell
    style = styles.get(style_name)
    if style is None:
        style = styles[style_name] = console.get_style(style_name)
    if cell_width > width:
        key = (value, width, right, ellipsis)
        lines = fitted.get(key)
        if lines is None:
            if ellipsis:
                text = Text(value)
                text.truncate(width, overflow="ellipsis")
                lines = [text.plain]
            else:
                lines = [
                    line.plain
                    for line in Text(value).wrap(
                        console,
                        width,
                        justify="right" if right else "left",
                        overflow="fold",
                    )
                ]
            fitted[key] = lines
        return [(Segment(line, style),) for line in lines]
    padding = " " * (width - cell_width)
    return ((Segment(padding + value if right else value + padding, style),),)


def _capacity_table(
    groups: Sequence[_JobGroup], *, width: Optional[int]
) -> Table | _PlainTable:
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
    splits = [_build_split(group.partitions) for group in groups]
    split_width = max(len(split.plain) for split in splits)
    include_bar, include_split = _detail_policy(
        width,
        required_widths=required_widths,
        bar_widths=[TOP_USER_BAR_WIDTH + 2],
        split_widths=[split_width],
    )

    headers = ["Node", "GPU Type", "Capacity"]
    if include_bar:
        headers.append("Usage")
    if include_split:
        headers.append("Partitions")
    headers.append("Jobs")
    rows = []
    for group, capacity, jobs_label, split in zip(
        groups, capacities, job_labels, splits
    ):
        row: list[Text | tuple[str, str]] = [
            (group.assignment, "bold cyan"),
            (group.gpu_type, "white"),
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
            row.append(split)
        row.append((jobs_label, "white"))
        rows.append(row)
    plain_table = _PlainTable(
        headers,
        rows,
        shrink={"GPU Type"},
        ellipsize={"GPU Type"},
        drop=["GPU Type"],
    )
    if plain_table.fits(width):
        return plain_table

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


_DETAIL_HEADERS = (
    "ID",
    "Node",
    "State",
    "User",
    "Partition",
    "GPU",
    "CPU",
    "MEM",
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
                _job_state_label(job.state),
                job.user,
                job.partition or "-",
                _job_gpu_value(
                    job,
                    unit=units.get(assignment, "GPU"),
                    servers=servers,
                ),
                _job_cpu_value(job),
                _job_mem_value(job),
                job.time_limit or "-",
                job.job_name or "-",
            )
        )
    return rows


@dataclass(frozen=True)
class _JobCards:
    """Packs each job's labeled fields onto as few lines as the width allows.

    A field moves whole to an indented continuation line, and only a field
    wider than a line folds.
    """

    rows: Sequence[tuple[str, ...]]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        max_width = options.max_width
        label_style = console.get_style("bold white")
        styles: dict[str, Style] = {}
        new_line = Segment.line()
        indent = Segment("  ")
        for row in self.rows:
            job_id, node, state, user, partition, gpu, cpu, mem, time, name = row
            position = 0
            for label, value, style_name in (
                ("ID: ", job_id, "bright_black"),
                ("State: ", state, _job_state_style(state)),
                ("User: ", user, "cyan"),
                ("Time: ", time, "white"),
                ("Node: ", node, "white"),
                ("Partition: ", partition, _partition_color(partition)),
                ("GPU/CPU/MEM: ", f"{gpu}/{cpu}/{mem}", "white"),
                ("Name: ", name, "white"),
            ):
                style = styles.get(style_name)
                if style is None:
                    style = styles[style_name] = console.get_style(style_name)
                field_width = len(label) + cell_len(value)
                if position and position + 2 + field_width <= max_width:
                    yield indent
                    position += 2
                elif position:
                    yield new_line
                    yield indent
                    position = 2
                yield Segment(label, label_style)
                position += len(label)
                pieces = chop_cells(value, max(max_width - position, 1))
                for piece_index, piece in enumerate(pieces):
                    if piece_index:
                        yield new_line
                        yield indent
                        position = 2
                    yield Segment(piece, style)
                position += cell_len(pieces[-1])
            yield new_line


def _job_details(
    rows: Sequence[tuple[str, ...]], *, width: Optional[int]
) -> _JobCards | _PlainTable:
    show_node = len({row[1] for row in rows}) > 1
    headers = [
        header for header in _DETAIL_HEADERS if show_node or header != "Node"
    ]
    styled_rows = []
    for row in rows:
        cells = [
            (row[0], "dim"),
            (row[1], "cyan"),
            (row[2], _job_state_style(row[2])),
            (row[3], "cyan"),
            (row[4], _partition_color(row[4])),
            *((value, "white") for value in row[5:]),
        ]
        if not show_node:
            cells.pop(1)
        styled_rows.append(cells)
    table = _PlainTable(
        headers,
        styled_rows,
        right={"GPU", "CPU", "MEM", "Time"},
        shrink={"ID", "User", "Partition", "Name"},
        ellipsize={"Name", "Partition"},
        drop=["CPU", "MEM", "Partition"],
        # Only pending array ranges like 123_[1-99%5] wrap; job IDs stay whole.
        min_widths={
            "ID": max(
                (len(row[0]) for row in rows if "[" not in row[0]),
                default=len("ID"),
            ),
            "Name": 8,
        },
    )
    return table if table.fits(width) else _JobCards(rows)


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

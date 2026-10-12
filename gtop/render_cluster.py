from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from .models import ServerState
from .render import (
    NODE_COUNT_COLOR,
    _availability_style,
    _build_bar,
    _build_counts,
    _build_split,
    _data_table,
    _detail_policy,
    _display_gpu_type,
    _fits_width,
    _max_width,
    _pluralize,
    _usage_partitions,
)

BAR_WIDTHS = {"gpu": 8, "cpu": 12, "mem": 10}
# Rough strength order: generation first, then memory. The first matching pattern
# wins, so specific names come before names they contain (V100S before V100).
GPU_CAPABILITY_PATTERNS = (
    ("T4", 0),
    ("GTX 1080 TI", 1),
    ("GTX TITAN X", 2),
    ("TITAN X PASCAL", 3),
    ("TITAN XP", 4),
    ("TITAN X", 2),
    ("2080 TI", 5),
    ("TITAN RTX", 6),
    ("QUADRO RTX 6000", 7),
    ("V100S", 9),
    ("V100", 8),
    ("RTX 3090", 10),
    ("L40S", 16),
    ("L4", 11),
    ("A5000", 12),
    ("GB10", 13),
    ("A5500", 13),
    ("A40", 14),
    ("A6000", 15),
    ("6000 ADA", 17),
    ("A100", 18),
    ("PRO 6000 BLACKWELL MAX-Q", 19),
    ("PRO 6000 BLACKWELL SERVER EDITION", 20),
    ("H100", 21),
    ("H200", 22),
    ("B200", 23),
)


def visible_servers(
    servers: Mapping[str, ServerState],
    *,
    target_users: Optional[set[str]],
) -> list[ServerState]:
    visible = sorted(
        servers.values(),
        key=lambda server: (
            ",".join(sorted(server.features)),
            -(
                server.gpu.num - server.gpu.occupied()
                if server.accepts_jobs
                else 0
            ),
            server.name,
        )
    )
    if not target_users:
        return visible
    return [server for server in visible if server.has_target_users(target_users)]


def _resource_numbers(
    server: ServerState,
    resource: str,
    *,
    show_used: bool = False,
) -> tuple[int, int, dict[str, float], int, int]:
    """Return (shown count, total, partition usage, occupied, unavailable).

    Unavailable is capacity that is neither in use nor schedulable: everything
    spare on a down or draining node, plus CPUs counted as "other".
    """
    partition_amounts = _usage_partitions(server.usage[resource])
    used = int(round(sum(partition_amounts.values())))
    accepts = server.accepts_jobs
    if resource == "gpu":
        total = server.gpu.num
        occupied = min(server.gpu.occupied(), total)
        unavailable = 0 if accepts else total - occupied
        free = total - occupied - unavailable
    elif resource == "cpu":
        total = server.cpu.total
        free = server.cpu.idle if accepts else 0
        unavailable = server.cpu.other + server.cpu.idle - free
        occupied = total - server.cpu.idle - server.cpu.other
    else:
        total = int(round(server.mem.total / 1024.0))
        occupied = min(
            int(round((server.mem.total - server.mem.idle) / 1024.0)), total
        )
        idle = total - occupied
        free = idle if accepts else 0
        unavailable = idle - free
    if show_used:
        return used, total, partition_amounts, used, unavailable
    return free, total, partition_amounts, occupied, unavailable


def _split_group_gpu_types(gpu_type: str) -> list[str]:
    return [part.strip() for part in gpu_type.split(" + ") if part.strip()]


def _single_gpu_capability_rank(gpu_type: str) -> int:
    normalized = gpu_type.upper()
    for pattern, rank in GPU_CAPABILITY_PATTERNS:
        if pattern in normalized:
            return rank
    return len(GPU_CAPABILITY_PATTERNS)


def _gpu_capability_rank(gpu_type: str) -> tuple[int, int, str]:
    ranks = sorted(
        _single_gpu_capability_rank(member)
        for member in _split_group_gpu_types(gpu_type)
    )
    return max(ranks), min(ranks), gpu_type.upper()


def _group_servers(
    servers: Sequence[ServerState],
    *,
    show_used: bool = False,
) -> list[tuple[str, list[ServerState]]]:
    grouped: dict[str, list[ServerState]] = {}
    for server in servers:
        grouped.setdefault(
            _display_gpu_type(server),
            [],
        ).append(server)

    def count(server: ServerState, resource: str) -> int:
        return _resource_numbers(
            server,
            resource,
            show_used=show_used,
        )[0]

    groups = [
        (
            gpu_type,
            sorted(
                members,
                key=lambda server: (
                    -count(server, "gpu"),
                    -count(server, "cpu"),
                    server.name,
                ),
            ),
        )
        for gpu_type, members in grouped.items()
    ]
    # Strongest GPU types first, matching the availability view.
    return sorted(
        groups,
        key=lambda item: (
            *(
                -rank if isinstance(rank, int) else rank
                for rank in _gpu_capability_rank(item[0])
            ),
            -sum(count(server, "gpu") for server in item[1]),
        ),
    )


def _group_resource_numbers(
    grouped_servers: Sequence[ServerState],
    resource: str,
    *,
    show_used: bool,
) -> tuple[int, int, dict[str, float], int, int]:
    count = 0
    total = 0
    occupied = 0
    unavailable = 0
    partition_amounts: dict[str, float] = {}
    for server in grouped_servers:
        (
            server_count,
            server_total,
            server_partitions,
            server_occupied,
            server_unavailable,
        ) = _resource_numbers(
            server,
            resource,
            show_used=show_used,
        )
        count += server_count
        total += server_total
        occupied += server_occupied
        unavailable += server_unavailable
        for partition, amount in server_partitions.items():
            partition_amounts[partition] = (
                partition_amounts.get(partition, 0.0) + amount
            )
    return count, total, partition_amounts, occupied, unavailable


def _add_resource_columns(
    table: Table,
    resource: str,
    *,
    show_used: bool,
    include_bar: bool,
    include_split: bool,
    counts_width: Optional[int] = None,
    bar_width: Optional[int] = None,
    split_width: Optional[int] = None,
) -> None:
    label = _resource_header(
        resource,
        show_used=show_used,
    )
    table.add_column(
        label,
        header_style="bold white",
        justify="right",
        no_wrap=True,
        width=counts_width,
    )
    if include_bar:
        table.add_column("", no_wrap=True, width=bar_width)
    if include_split:
        table.add_column("", no_wrap=True, width=split_width)


def _resource_label(resource: str) -> str:
    return "Memory" if resource == "mem" else resource.upper()


def _resource_header(
    resource: str,
    *,
    show_used: bool,
) -> str:
    status = "used" if show_used else "free"
    return f"{_resource_label(resource)} {status}"


def _resource_row(
    values: Mapping[str, tuple[int, int, Mapping[str, float], int, int]],
    *,
    resources: Sequence[str],
    show_used: bool,
    show_total: bool,
    include_bar: bool,
    include_split: bool,
    bar_widths: Mapping[str, int] = BAR_WIDTHS,
) -> list[Text]:
    cells: list[Text] = []
    for resource in resources:
        count, total, partitions, occupied, unavailable = values[resource]
        cells.append(
            _build_counts(
                count,
                total,
                show_used=show_used,
                show_total=show_total,
            )
        )
        if include_bar:
            bar_partitions = dict(partitions)
            attributed = int(round(sum(bar_partitions.values())))
            unattributed = max(occupied - attributed, 0)
            if unattributed:
                bar_partitions["other"] = (
                    bar_partitions.get("other", 0) + unattributed
                )
            cells.append(
                _build_bar(
                    bar_partitions,
                    free=max(total - occupied - unavailable, 0),
                    unavailable=unavailable,
                    width=bar_widths[resource],
                )
            )
        if include_split:
            cells.append(_build_split(partitions))
    return cells


def _capacity_summary(
    count: int,
    total: int,
    *,
    resource: str,
    show_used: bool,
) -> Text:
    summary = Text()
    summary.append(
        str(count),
        style=f"bold {_availability_style(count, total, show_used=show_used)}",
    )
    if not show_used:
        summary.append("/", style="dim")
        summary.append(str(total), style="white")
    summary.append(
        f" {_pluralize(count if show_used else total, resource)} "
        f"{'used' if show_used else 'free'}"
    )
    return summary


def _cluster_overview(
    servers: Sequence[ServerState],
    *,
    show_used: bool,
    overview_title: str,
) -> Text:
    counts = [
        _resource_numbers(
            server,
            "gpu",
            show_used=show_used,
        )
        for server in servers
    ]
    count = sum(item[0] for item in counts)
    total = sum(item[1] for item in counts)
    return _title_line(
        overview_title,
        _capacity_summary(
            count,
            total,
            resource="GPU",
            show_used=show_used,
        ),
    )


def _title_line(title: str, summary: Text) -> Text:
    return Text.assemble((title, "bold cyan"), "  ", summary, no_wrap=True)


def _summary_table(
    groups: Sequence[tuple[str, Sequence[ServerState]]],
    *,
    show_used: bool,
    width: Optional[int],
) -> Table:
    rows = [
        (
            gpu_type,
            _group_resource_numbers(
                servers,
                "gpu",
                show_used=show_used,
            ),
            len(servers),
        )
        for gpu_type, servers in groups
    ]
    gpu_values = [values for _, values, _ in rows]
    include_bar, include_split = _detail_policy(
        width,
        required_widths=[
            _max_width("GPU Type", [gpu_type for gpu_type, _, _ in rows]),
            _max_width(
                _resource_header(
                    "gpu",
                    show_used=show_used,
                ),
                [f"{count}/{total}" for count, total, _, _, _ in gpu_values],
            ),
            _max_width("Nodes", [str(count) for _, _, count in rows]),
        ],
        bar_widths=[BAR_WIDTHS["gpu"] + 2],
        split_widths=[
            max(
                [
                    1,
                    *(
                        len(_build_split(parts).plain)
                        for _, _, parts, _, _ in gpu_values
                    ),
                ]
            )
        ],
    )

    table = _data_table(show_header=True)
    table.add_column("GPU Type", header_style="bold white", no_wrap=True)
    _add_resource_columns(
        table,
        "gpu",
        show_used=show_used,
        include_bar=include_bar,
        include_split=include_split,
    )
    table.add_column(
        "Nodes",
        header_style="bold white",
        justify="right",
        no_wrap=True,
    )
    for gpu_type, gpu_values, node_count in rows:
        table.add_row(
            Text(gpu_type, style="bold cyan"),
            *_resource_row(
                {"gpu": gpu_values},
                resources=("gpu",),
                show_used=show_used,
                show_total=not show_used,
                include_bar=include_bar,
                include_split=include_split,
            ),
            Text(str(node_count), style=NODE_COUNT_COLOR),
        )
    return table


_ResourceValues = dict[str, tuple[int, int, dict[str, float], int, int]]
_NODE_RESOURCES = ("gpu", "cpu", "mem")


def _node_values(
    server: ServerState,
    *,
    show_used: bool,
) -> _ResourceValues:
    return {
        resource: _resource_numbers(
            server,
            resource,
            show_used=show_used,
        )
        for resource in _NODE_RESOURCES
    }


@dataclass(frozen=True)
class _NodeColumns:
    node_width: int
    count_widths: Mapping[str, int]
    split_widths: Mapping[str, int]


def _node_columns(
    values_by_node: Mapping[str, _ResourceValues],
    *,
    show_used: bool,
) -> _NodeColumns:
    measured_values = list(values_by_node.values())
    return _NodeColumns(
        node_width=_max_width("Node", list(values_by_node)),
        count_widths={
            resource: _max_width(
                _resource_header(
                    resource,
                    show_used=show_used,
                ),
                [
                    f"{values[resource][0]}/{values[resource][1]}"
                    for values in measured_values
                ],
            )
            for resource in _NODE_RESOURCES
        },
        split_widths={
            resource: max(
                [
                    1,
                    *(
                        len(_build_split(values[resource][2]).plain)
                        for values in measured_values
                    ),
                ]
            )
            for resource in _NODE_RESOURCES
        },
    )


def _nodes_table(
    rows: Sequence[tuple[ServerState, _ResourceValues]],
    *,
    columns: _NodeColumns,
    show_used: bool,
    width: Optional[int],
    show_header: bool,
) -> Any:
    resources = _NODE_RESOURCES
    node_width = columns.node_width
    count_widths = columns.count_widths
    split_widths = columns.split_widths
    required_widths = [
        node_width,
        *(count_widths[resource] for resource in resources),
    ]
    if not _fits_width(width, required_widths):
        cards: list[Text] = []
        for server, values in rows:
            card = Text(server.name, style="bright_black")
            for resource in resources:
                count, total, _, _, _ = values[resource]
                card.append(
                    (
                        f"\n{_resource_header(resource, show_used=show_used)}: "
                    ),
                    style="bold white",
                )
                card.append_text(
                    _build_counts(
                        count,
                        total,
                        show_used=show_used,
                    )
                )
            cards.append(card)
        return Group(*cards)
    include_bar, include_split = _detail_policy(
        width,
        required_widths=required_widths,
        bar_widths=[BAR_WIDTHS[resource] + 2 for resource in resources],
        split_widths=[split_widths[resource] for resource in resources],
    )
    bar_widths = BAR_WIDTHS
    if not include_bar:
        for compact_width in range(min(BAR_WIDTHS.values()), 2, -1):
            compact_widths = {resource: compact_width for resource in resources}
            if _fits_width(
                width,
                [
                    *required_widths,
                    *(compact_widths[resource] + 2 for resource in resources),
                ],
            ):
                bar_widths = compact_widths
                include_bar = True
                break

    widths = [node_width]
    right = [False]
    headers = ["Node"]
    for resource in resources:
        widths.append(count_widths[resource])
        right.append(True)
        headers.append(_resource_header(resource, show_used=show_used))
        if include_bar:
            widths.append(bar_widths[resource] + 2)
            right.append(False)
            headers.append("")
        if include_split:
            widths.append(split_widths[resource])
            right.append(False)
            headers.append("")
    return _FixedRows(
        widths=widths,
        right=right,
        rows=[
            [
                Text(server.name, style="bright_black"),
                *_resource_row(
                    values,
                    resources=resources,
                    show_used=show_used,
                    show_total=True,
                    include_bar=include_bar,
                    include_split=include_split,
                    bar_widths=bar_widths,
                ),
            ]
            for server, values in rows
        ],
        headers=headers if show_header else None,
    )


@dataclass(frozen=True)
class _FixedRows:
    """Rows whose column widths are already known, laid out like a borderless
    rich Table without measuring every cell again (one table per GPU type made
    that the slowest part of the nodes view)."""

    widths: Sequence[int]
    right: Sequence[bool]
    rows: Sequence[Sequence[Text]]
    headers: Optional[Sequence[str]]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # rich's default header style ("bold") applies to unlabeled columns.
        header = (
            [
                [
                    Text(header, style="bold white" if header else "bold")
                    for header in self.headers
                ]
            ]
            if self.headers is not None
            else []
        )
        for row_index, cells in enumerate([*header, *self.rows]):
            line = Text(no_wrap=True, overflow="crop", end="")
            for index, (cell, width, right) in enumerate(
                zip(cells, self.widths, self.right)
            ):
                if index:
                    # Like rich, a header gap takes the style of the header before it.
                    gap_style = cells[index - 1].style if header and row_index == 0 else ""
                    line.append(" ", style=gap_style)
                padded = cell.copy()
                padding = max(width - cell.cell_len, 0)
                if right:
                    padded.pad_left(padding)
                else:
                    padded.pad_right(padding)
                line.append_text(padded)
            yield line
            yield Segment.line()


def _group_header(
    gpu_type: str,
    servers: Sequence[ServerState],
    *,
    show_used: bool,
) -> Text:
    count, total, _, _, _ = _group_resource_numbers(
        servers,
        "gpu",
        show_used=show_used,
    )
    return _title_line(
        gpu_type,
        _capacity_summary(
            count,
            total,
            resource="GPU",
            show_used=show_used,
        ),
    )


def render_table(
    servers: Sequence[ServerState],
    *,
    width: Optional[int] = None,
    show_used: bool = False,
    overview_title: str = "Cluster Overview",
    verbose: bool = False,
) -> Any:
    if show_used and not verbose:
        servers = [
            server
            for server in servers
            if _resource_numbers(
                server,
                "gpu",
                show_used=True,
            )[0]
            > 0
        ]
    groups = _group_servers(
        servers,
        show_used=show_used,
    )
    renderables: list[Any] = [
        _cluster_overview(
            servers,
            show_used=show_used,
            overview_title=overview_title,
        )
    ]
    if not verbose:
        if groups:
            renderables.append(
                _summary_table(
                    groups,
                    show_used=show_used,
                    width=width,
                )
            )
        return Group(*renderables)

    values_by_node = {
        server.name: _node_values(
            server,
            show_used=show_used,
        )
        for _, grouped_servers in groups
        for server in grouped_servers
    }
    columns = _node_columns(
        values_by_node,
        show_used=show_used,
    )
    renderables.append(
        _nodes_table(
            [],
            columns=columns,
            show_used=show_used,
            width=width,
            show_header=True,
        )
    )
    for index, (gpu_type, grouped_servers) in enumerate(groups):
        if index:
            renderables.append(Text(""))
        renderables.append(
            _group_header(
                gpu_type,
                grouped_servers,
                show_used=show_used,
            )
        )
        renderables.append(
            _nodes_table(
                [(server, values_by_node[server.name]) for server in grouped_servers],
                columns=columns,
                show_used=show_used,
                width=width,
                show_header=False,
            )
        )

    return Group(*renderables)

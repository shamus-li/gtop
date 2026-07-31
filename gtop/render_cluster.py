from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from rich.console import Group
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
    _resource_split,
    _usage_partitions,
)

BAR_WIDTHS = {"gpu": 8, "cpu": 12, "mem": 10}
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


def visible_servers(
    servers: Mapping[str, ServerState],
    *,
    target_users: Optional[set[str]],
    show_shards: bool = False,
) -> list[ServerState]:
    visible = sorted(
        servers.values(),
        key=lambda server: (
            ",".join(sorted(server.features)),
            -(
                server.gpu.capacity(show_shards)
                - server.gpu.occupied(show_shards)
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
    show_shards: bool,
    show_used: bool = False,
) -> tuple[int, int, dict[str, float], int]:
    partition_amounts = _usage_partitions(
        _resource_split(server, resource, show_shards)
    )
    used = sum(int(round(amount)) for amount in partition_amounts.values())
    if resource == "gpu":
        total = server.gpu.capacity(show_shards)
        occupied = min(server.gpu.occupied(show_shards), total)
        count = used if show_used else max(total - occupied, 0)
        return count, total, partition_amounts, occupied
    if resource == "cpu":
        total = server.cpu.total
        occupied = min(max(total - server.cpu.idle, 0), total)
        return (
            used if show_used else server.cpu.idle,
            total,
            partition_amounts,
            occupied,
        )

    total = int(round(server.mem.total / 1024.0))
    free = int(round(server.mem.idle / 1024.0))
    occupied = min(max(total - free, 0), total)
    return (used if show_used else free), total, partition_amounts, occupied


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
    show_shards: bool,
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
            show_shards=show_shards,
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
    return sorted(
        groups,
        key=lambda item: (
            *_gpu_capability_rank(item[0]),
            -sum(count(server, "gpu") for server in item[1]),
        ),
    )


def _group_resource_numbers(
    grouped_servers: Sequence[ServerState],
    resource: str,
    *,
    show_shards: bool,
    show_used: bool,
) -> tuple[int, int, dict[str, float], int]:
    count = 0
    total = 0
    occupied = 0
    partition_amounts: dict[str, float] = {}
    for server in grouped_servers:
        server_count, server_total, server_partitions, server_occupied = _resource_numbers(
            server,
            resource,
            show_shards=show_shards,
            show_used=show_used,
        )
        count += server_count
        total += server_total
        occupied += server_occupied
        for partition, amount in server_partitions.items():
            partition_amounts[partition] = (
                partition_amounts.get(partition, 0.0) + amount
            )
    return count, total, partition_amounts, occupied


def _add_resource_columns(
    table: Table,
    resource: str,
    *,
    show_shards: bool,
    show_used: bool,
    include_bar: bool,
    include_split: bool,
    counts_width: Optional[int] = None,
    split_width: Optional[int] = None,
) -> None:
    label = _resource_header(
        resource,
        show_shards=show_shards,
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
        table.add_column("", no_wrap=True)
    if include_split:
        table.add_column("", no_wrap=True, width=split_width)


def _resource_label(resource: str, *, show_shards: bool) -> str:
    if resource == "gpu" and show_shards:
        return "Shard"
    return "Memory" if resource == "mem" else resource.upper()


def _resource_header(
    resource: str,
    *,
    show_shards: bool,
    show_used: bool,
) -> str:
    status = "used" if show_used else "free"
    return f"{_resource_label(resource, show_shards=show_shards)} {status}"


def _resource_row(
    values: Mapping[str, tuple[int, int, Mapping[str, float], int]],
    *,
    resources: Sequence[str],
    show_used: bool,
    show_total: bool,
    include_bar: bool,
    include_split: bool,
) -> list[Text]:
    cells: list[Text] = []
    for resource in resources:
        count, total, partitions, occupied = values[resource]
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
            attributed = sum(
                int(round(amount))
                for amount in bar_partitions.values()
            )
            unattributed = max(occupied - attributed, 0)
            if unattributed:
                bar_partitions["other"] = (
                    bar_partitions.get("other", 0) + unattributed
                )
            cells.append(
                _build_bar(
                    bar_partitions,
                    free=max(total - occupied, 0),
                    width=BAR_WIDTHS[resource],
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
        f" {_pluralize(count, resource)} {'used' if show_used else 'free'}"
    )
    return summary


def _cluster_overview(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool,
    overview_title: str,
) -> Table:
    counts = [
        _resource_numbers(
            server,
            "gpu",
            show_shards=show_shards,
            show_used=show_used,
        )
        for server in servers
    ]
    count = sum(item[0] for item in counts)
    total = sum(item[1] for item in counts)
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_row(
        Text(overview_title, style="bold cyan"),
        _capacity_summary(
            count,
            total,
            resource="shard" if show_shards else "GPU",
            show_used=show_used,
        ),
    )
    return table


def _summary_table(
    groups: Sequence[tuple[str, Sequence[ServerState]]],
    *,
    show_shards: bool,
    show_used: bool,
    width: Optional[int],
) -> Table:
    rows = [
        (
            gpu_type,
            _group_resource_numbers(
                servers,
                "gpu",
                show_shards=show_shards,
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
                    show_shards=show_shards,
                    show_used=show_used,
                ),
                [f"{count}/{total}" for count, total, _, _ in gpu_values],
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
                        for _, _, parts, _ in gpu_values
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
        show_shards=show_shards,
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


def _nodes_table(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool,
    width: Optional[int],
    show_header: bool,
    measurement_servers: Optional[Sequence[ServerState]] = None,
) -> Any:
    resources = ("gpu", "cpu", "mem")
    measured_servers = (
        list(measurement_servers)
        if measurement_servers is not None
        else list(servers)
    )
    rows = [
        (
            server,
            {
                resource: _resource_numbers(
                    server,
                    resource,
                    show_shards=show_shards,
                    show_used=show_used,
                )
                for resource in resources
            },
        )
        for server in servers
    ]
    measured_values = [
        {
            resource: _resource_numbers(
                server,
                resource,
                show_shards=show_shards,
                show_used=show_used,
            )
            for resource in resources
        }
        for server in measured_servers
    ]
    count_widths = {
        resource: _max_width(
            _resource_header(
                resource,
                show_shards=show_shards,
                show_used=show_used,
            ),
            [
                f"{values[resource][0]}/{values[resource][1]}"
                for values in measured_values
            ],
        )
        for resource in resources
    }
    split_widths = {
        resource: max(
            [
                1,
                *(
                    len(_build_split(values[resource][2]).plain)
                    for values in measured_values
                ),
            ]
        )
        for resource in resources
    }
    required_widths = [
        _max_width("Node", [server.name for server in measured_servers]),
        *(count_widths[resource] for resource in resources),
    ]
    if not _fits_width(width, required_widths):
        cards: list[Text] = []
        for server, values in rows:
            card = Text(server.name, style="bright_black")
            for resource in resources:
                count, total, _, _ = values[resource]
                card.append(
                    (
                        f"\n{_resource_header(resource, show_shards=show_shards, show_used=show_used)}: "
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

    table = _data_table(show_header=show_header)
    table.add_column(
        "Node",
        header_style="bold white",
        no_wrap=True,
        width=_max_width("Node", [server.name for server in measured_servers]),
    )
    for resource in resources:
        _add_resource_columns(
            table,
            resource,
            show_shards=show_shards,
            show_used=show_used,
            include_bar=include_bar,
            include_split=include_split,
            counts_width=count_widths[resource],
            split_width=split_widths[resource] if include_split else None,
        )
    for server, values in rows:
        table.add_row(
            Text(server.name, style="bright_black"),
            *_resource_row(
                values,
                resources=resources,
                show_used=show_used,
                show_total=True,
                include_bar=include_bar,
                include_split=include_split,
            ),
        )
    return table


def _group_header(
    gpu_type: str,
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
    show_used: bool,
) -> Table:
    count, total, _, _ = _group_resource_numbers(
        servers,
        "gpu",
        show_shards=show_shards,
        show_used=show_used,
    )
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_row(
        Text(gpu_type, style="bold cyan"),
        _capacity_summary(
            count,
            total,
            resource="shard" if show_shards else "GPU",
            show_used=show_used,
        ),
    )
    return table


def render_table(
    servers: Sequence[ServerState],
    *,
    show_shards: bool = False,
    width: Optional[int] = None,
    show_used: bool = False,
    overview_title: str = "Cluster Overview",
    verbose: bool = False,
) -> Any:
    groups = _group_servers(
        servers,
        show_shards=show_shards,
        show_used=show_used,
    )
    renderables: list[Any] = [
        _cluster_overview(
            servers,
            show_shards=show_shards,
            show_used=show_used,
            overview_title=overview_title,
        )
    ]
    if not verbose:
        renderables.append(
            _summary_table(
                groups,
                show_shards=show_shards,
                show_used=show_used,
                width=width,
            )
        )
        return Group(*renderables)

    all_grouped_servers = [
        server
        for _, grouped_servers in groups
        for server in grouped_servers
    ]
    renderables.append(
        _nodes_table(
            [],
            show_shards=show_shards,
            show_used=show_used,
            width=width,
            show_header=True,
            measurement_servers=all_grouped_servers,
        )
    )
    for index, (gpu_type, grouped_servers) in enumerate(groups):
        if index:
            renderables.append(Text(""))
        renderables.append(
            _group_header(
                gpu_type,
                grouped_servers,
                show_shards=show_shards,
                show_used=show_used,
            )
        )
        renderables.append(
            _nodes_table(
                grouped_servers,
                show_shards=show_shards,
                show_used=show_used,
                width=width,
                show_header=False,
                measurement_servers=all_grouped_servers,
            )
        )

    return Group(*renderables)

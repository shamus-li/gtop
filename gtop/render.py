from __future__ import annotations

from functools import lru_cache
import math
import pwd
import re
from typing import Any, Mapping, Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import (
    ResourceUsageSplit,
    ServerState,
    UserUsage,
)
from .partitions import SEMANTIC_PARTITIONS, partition_bucket

TOP_USER_BAR_WIDTH = 12
SEMANTIC_PALETTE = {
    "priority": "#c764f4",
    "gpu": "#4fd3a1",
    "default": "#88b4ff",
}
OTHER_PARTITION_COLOR = "#a4b0be"


def help_legend() -> Text:
    legend = Text("\nLegend:\n")
    for partition, color in SEMANTIC_PALETTE.items():
        legend.append("  ")
        legend.append("████", style=f"dim {color}")
        legend.append(f" = {partition}\n")
    legend.append("  ")
    legend.append("████", style=f"dim {OTHER_PARTITION_COLOR}")
    legend.append(" = unattributed usage\n")
    legend.append("  ")
    legend.append("××××", style="dim red")
    legend.append(" = down or draining\n")
    legend.append("  counts after bars: priority / gpu / default")
    return legend
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


def _top_user_label(user: str, *, include_full_name: bool = True) -> str:
    if not include_full_name:
        return user
    full_name = _lookup_full_name(user)
    if not full_name or full_name == user:
        return user
    return f"{user} ({full_name})"


def _partition_color(partition: str) -> str:
    bucket = partition_bucket(partition)
    return OTHER_PARTITION_COLOR if bucket is None else SEMANTIC_PALETTE[bucket]


def _usage_partitions(usage_info: ResourceUsageSplit) -> dict[str, float]:
    return {
        partition: amount
        for partition, amount in usage_info.partitions.items()
        if amount > 0
    }


def _summary_partitions(summary: UserUsage) -> dict[str, float]:
    return {
        partition: float(amount)
        for partition, amount in summary.usage_by_partition.items()
    }


def _semantic_partition_totals(
    partition_amounts: Mapping[str, int | float],
) -> dict[str, int]:
    totals = {partition: 0 for partition in SEMANTIC_PARTITIONS}
    for partition, amount in partition_amounts.items():
        bucket = partition_bucket(partition)
        if bucket is not None:
            totals[bucket] += int(round(amount))
    return totals


def _partition_segments(
    partition_amounts: Mapping[str, float],
) -> list[tuple[str, int, str]]:
    normalized = {
        partition: int(round(amount))
        for partition, amount in partition_amounts.items()
        if int(round(amount)) > 0
    }
    totals = _semantic_partition_totals(normalized)
    segments = [
        (partition, totals[partition], _partition_color(partition))
        for partition in SEMANTIC_PARTITIONS
    ]
    other_total = normalized.get("other", 0)
    if other_total:
        segments.append(("other", other_total, OTHER_PARTITION_COLOR))
    return segments


def _format_gpu_name(gpu_type: str) -> str:
    for source in (
        "nvidia_geforce_",
        "nvidia_rtx_",
        "nvidia_",
        "tesla_",
        "geforce_",
    ):
        gpu_type = gpu_type.replace(source, "")
    gpu_type = gpu_type.replace("_generation", "")
    gpu_type = gpu_type.replace("_workstation_edition", "")
    words = re.split(r"[\s_-]+", gpu_type)
    known_tokens = {
        "rtx": "RTX",
        "gtx": "GTX",
        "gpu": "GPU",
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
        if not word:
            continue
        lower_word = word.lower()
        if lower_word in known_tokens:
            compact_words.append(known_tokens[lower_word])
        elif any(char.isdigit() for char in word):
            compact_words.append(word.upper().replace("PCIE", "PCIe"))
        else:
            compact_words.append(word.title())
    compact = " ".join(compact_words)
    compact = compact.replace("Rtx ", "").replace("Geforce ", "").replace("Tesla ", "")
    return compact.replace(" TI", " Ti").replace("Max Q", "Max-Q") or "GPU"


def _display_gpu_type(server: ServerState) -> str:
    gpu_type = server.gpu.type
    if gpu_type.startswith("Shard(") and gpu_type.endswith(")"):
        gpu_type = gpu_type[6:-1]
    if gpu_type.startswith("(") and gpu_type.endswith(")"):
        gpu_type = gpu_type[1:-1]
    return " + ".join(
        _format_gpu_name(member.strip())
        for member in gpu_type.split("|")
        if member.strip()
    )


def _pluralize(count: int, singular: str) -> str:
    return singular if count == 1 else f"{singular}s"


def _split_segments(values: Sequence[int], width: int) -> list[int]:
    total = sum(values)
    if width <= 0:
        return [0 for _ in values]
    if total <= 0:
        result = [0 for _ in values]
        result[-1] = width
        return result

    positive = sum(value > 0 for value in values)
    minimum = 1 if positive <= width else 0
    distributable = width - minimum * positive
    raw = [value / total * distributable for value in values]
    base = [
        (minimum if source > 0 else 0) + math.floor(value)
        for source, value in zip(values, raw)
    ]
    remainder = width - sum(base)
    order = sorted(
        range(len(values)),
        key=lambda index: (raw[index] - base[index], raw[index]),
        reverse=True,
    )
    for index in order:
        if remainder <= 0:
            break
        if values[index] > 0:
            base[index] += 1
            remainder -= 1
    if remainder > 0:
        base[-1] += remainder
    return base


def _build_bar(
    partition_amounts: Mapping[str, float],
    *,
    free: int,
    width: int,
    unavailable: int = 0,
) -> Text:
    segments = _partition_segments(partition_amounts)
    lengths = _split_segments(
        [count for _, count, _ in segments] + [unavailable, free],
        width,
    )
    bar = Text()
    bar.append("[", style="dim")
    for (_, _, color), segment_length in zip(segments, lengths[:-2]):
        if segment_length:
            bar.append("█" * segment_length, style=f"dim {color}")
    if lengths[-2]:
        bar.append("×" * lengths[-2], style="dim red")
    if lengths[-1]:
        bar.append("·" * lengths[-1], style="dim")
    bar.append("]", style="dim")
    return bar


def _build_split(
    partition_amounts: Mapping[str, float],
) -> Text:
    segments = _partition_segments(partition_amounts)
    split = Text()
    by_name = {name: count for name, count, _ in segments}
    for index, partition in enumerate(SEMANTIC_PARTITIONS):
        count = by_name.get(partition, 0)
        if index:
            split.append("/", style="dim")
        split.append(
            str(count),
            style=(
                f"dim {_partition_color(partition)}"
                if count == 0
                else _partition_color(partition)
            ),
        )
    return split


def _availability_style(
    count: int,
    total: int,
    *,
    show_used: bool = False,
) -> str:
    if total <= 0:
        return "white"
    if count <= 0:
        return "bright_black" if show_used else "red"
    if count >= total:
        return "bright_white" if show_used else "green"
    return "white" if show_used else "yellow"


def _build_counts(
    count: int,
    total: int,
    *,
    show_used: bool,
    show_total: bool = True,
) -> Text:
    counts = Text()
    counts.append(
        str(count),
        style=_availability_style(count, total, show_used=show_used),
    )
    if show_total:
        counts.append("/", style="dim")
        counts.append(str(total), style="white")
    return counts


def _fits_width(width: Optional[int], column_widths: Sequence[int]) -> bool:
    return (
        width is None
        or sum(column_widths) + max(len(column_widths) - 1, 0) * 2 <= width
    )


def _max_width(header: str, values: Sequence[str]) -> int:
    return max([len(header), *(len(value) for value in values)])


def _data_table(*, show_header: bool) -> Table:
    return Table(
        box=None,
        show_header=show_header,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        padding=(0, 1),
    )


def _detail_policy(
    width: Optional[int],
    *,
    required_widths: Sequence[int],
    bar_widths: Sequence[int],
    split_widths: Sequence[int],
) -> tuple[bool, bool]:
    if _fits_width(width, [*required_widths, *bar_widths, *split_widths]):
        return True, True
    if _fits_width(width, [*required_widths, *bar_widths]):
        return True, False
    return False, False


def _build_top_users_table(
    users: Sequence[UserUsage],
    *,
    unit: str,
    width: Optional[int],
) -> Table:
    include_full_name = True
    include_bar = True
    include_split = True

    def column_widths() -> list[int]:
        labels = [
            _top_user_label(stats.user, include_full_name=include_full_name)
            for stats in users
        ]
        widths = [
            _max_width("User", labels),
            _max_width(
                _pluralize(2, unit),
                [str(stats.total_usage()) for stats in users],
            ),
            _max_width(
                "Nodes",
                [str(len(stats.nodes)) for stats in users],
            ),
        ]
        if include_bar:
            widths.append(TOP_USER_BAR_WIDTH + 2)
        if include_split:
            widths.append(
                max(
                    len(_build_split(_summary_partitions(stats)).plain)
                    for stats in users
                )
            )
        return widths

    for detail in ("split", "full_name", "bar"):
        if _fits_width(width, column_widths()):
            break
        if detail == "split":
            include_split = False
        elif detail == "full_name":
            include_full_name = False
        else:
            include_bar = False

    table = _data_table(show_header=True)
    table.add_column("User", header_style="bold white", no_wrap=True)
    table.add_column(
        _pluralize(2, unit),
        header_style="bold white",
        justify="right",
        no_wrap=True,
    )
    table.add_column(
        "Nodes",
        header_style="bold white",
        justify="right",
        no_wrap=True,
    )
    if include_bar:
        table.add_column("", no_wrap=True)
    if include_split:
        table.add_column("", no_wrap=True)

    for stats in users:
        partition_amounts = _summary_partitions(stats)
        row: list[Text] = [
            Text(
                _top_user_label(
                    stats.user,
                    include_full_name=include_full_name,
                ),
                style="cyan",
            ),
            Text(str(stats.total_usage()), style="bold yellow"),
            Text(str(len(stats.nodes)), style=NODE_COUNT_COLOR),
        ]
        if include_bar:
            row.append(
                _build_bar(
                    partition_amounts,
                    free=0,
                    width=TOP_USER_BAR_WIDTH,
                )
            )
        if include_split:
            row.append(_build_split(partition_amounts))
        table.add_row(*row)
    return table


def print_top_users(
    users: Sequence[UserUsage],
    *,
    unit: str,
    console: Optional[Any] = None,
) -> None:
    active_console = console or Console()
    active_console.print(Text("Top Users", style="bold cyan"))
    active_console.print(
        _build_top_users_table(
            users,
            unit=unit,
            width=getattr(active_console, "width", None),
        )
    )
    active_console.print()


def print_filtered_users(
    users: Mapping[str, UserUsage],
    *,
    unit: str,
    console: Optional[Any] = None,
) -> None:
    active_console = console or Console()
    visible_users = [
        (user, stats)
        for user, stats in sorted(users.items())
        if stats.total_usage() > 0
    ]
    if not visible_users:
        return

    width = getattr(active_console, "width", None)
    include_full_name = True
    include_partitions = True

    def column_widths() -> list[int]:
        widths = [
            max(
                len(_top_user_label(user, include_full_name=include_full_name))
                for user, _ in visible_users
            ),
            max(
                len(f"{stats.total_usage()} {_pluralize(stats.total_usage(), unit)}")
                for _, stats in visible_users
            ),
            max(
                len(f"{len(stats.nodes)} {_pluralize(len(stats.nodes), 'node')}")
                for _, stats in visible_users
            ),
        ]
        if include_partitions:
            widths.append(
                max(
                    len(
                        "  ".join(
                            f"{partition}:{count}"
                            for partition, count in sorted(
                                stats.usage_by_partition.items()
                            )
                        )
                    )
                    for _, stats in visible_users
                )
            )
        return widths

    if not _fits_width(width, column_widths()):
        include_partitions = False
    if not _fits_width(width, column_widths()):
        include_full_name = False

    table = _data_table(show_header=False)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    if include_partitions:
        table.add_column(no_wrap=True)

    for user, stats in visible_users:
        total_usage = stats.total_usage()
        node_count = len(stats.nodes)
        row = [
            Text(
                _top_user_label(
                    user,
                    include_full_name=include_full_name,
                ),
                style="cyan",
            ),
            Text.assemble(
                (str(total_usage), "bold yellow"),
                f" {_pluralize(total_usage, unit)}",
            ),
            Text.assemble(
                (str(node_count), f"bold {NODE_COUNT_COLOR}"),
                f" {_pluralize(node_count, 'node')}",
            ),
        ]
        if include_partitions:
            breakdown = Text()
            for index, (partition, count) in enumerate(
                sorted(stats.usage_by_partition.items())
            ):
                if index:
                    breakdown.append("  ")
                breakdown.append(f"{partition}:", style="white")
                breakdown.append(
                    str(count), style=f"bold {_partition_color(partition)}"
                )
            row.append(breakdown)
        table.add_row(*row)

    active_console.print(
        Text(_pluralize(len(visible_users), "User"), style="bold cyan")
    )
    active_console.print(table)
    active_console.print()


def print_usage_history(
    gpu_hours: Mapping[str, float],
    accounts: Mapping[str, Sequence[str]],
    *,
    title: str,
    limit: int,
    console: Optional[Any] = None,
) -> None:
    active_console = console or Console()
    totals = sorted(gpu_hours.items(), key=lambda item: (-item[1], item[0]))
    grand_total = sum(gpu_hours.values())
    top = [(user, total) for user, total in totals[:limit] if round(total) > 0]
    labs = {user: ",".join(accounts.get(user, ())) for user, _ in top}
    include_full_name = _fits_width(
        getattr(active_console, "width", None),
        [
            max(len(_top_user_label(user)) for user, _ in top),
            9,
            5,
            max(len(lab) for lab in labs.values()),
        ],
    )

    table = _data_table(show_header=True)
    table.add_column("User", header_style="bold white", no_wrap=True)
    table.add_column("GPU-hours", header_style="bold white", justify="right", no_wrap=True)
    table.add_column("Share", header_style="bold white", justify="right", no_wrap=True)
    table.add_column("Lab", header_style="bold white", no_wrap=True)
    for user, total in top:
        table.add_row(
            Text(_top_user_label(user, include_full_name=include_full_name), style="cyan"),
            Text(f"{round(total):,}", style="bold yellow"),
            Text(f"{total / grand_total:.0%}", style=NODE_COUNT_COLOR),
            Text(labs[user], style=NODE_COUNT_COLOR),
        )
    active_console.print(
        Text.assemble(
            (title, "bold cyan"),
            (f"  {round(grand_total):,} GPU-hours, {len(totals)} users", "dim"),
        )
    )
    active_console.print(table)

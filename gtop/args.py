from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Optional, Sequence

from .constants import MIN_JOB_CPUS, MIN_JOB_MEM_GB
from .partitions import partition_names

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
        # Imported here so parsing arguments stays fast; only help needs rich.
        from rich.console import Console

        from .render import help_legend

        Console(file=file).print(help_legend())


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


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
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

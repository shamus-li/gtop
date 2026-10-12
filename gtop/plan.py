from __future__ import annotations

import argparse
import getpass
from typing import NamedTuple, Optional

from .command_options import set_value_option
from .constants import (
    JOBS_SACCT_COMMAND,
    NODES_COMMAND,
    PARTITION_COMMAND,
    SACCT_COMMAND,
    SQUEUE_COMMAND,
)
from .runner import Command


class ViewCommands(NamedTuple):
    """The SLURM commands a view runs. main() starts them before loading the
    rest of gtop, and the view requests these same commands."""

    sacct: Command
    squeue: Optional[Command] = None
    partitions: bool = False

    def all(self) -> list[Command]:
        return [
            NODES_COMMAND,
            self.sacct,
            *((self.squeue,) if self.squeue else ()),
            *((PARTITION_COMMAND,) if self.partitions else ()),
        ]


def target_users(args: argparse.Namespace) -> Optional[set[str]]:
    users: set[str] = set(getattr(args, "users", None) or ())
    if getattr(args, "me", False):
        users.add(getpass.getuser())
    return users or None


def sacct_command(
    command: Command,
    *,
    states: Optional[tuple[str, ...]],
    users: Optional[set[str]] = None,
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


def view_commands(args: argparse.Namespace) -> Optional[ViewCommands]:
    """None for the --week/--month/--year view, which reads a cache instead."""
    if args.command == "users" and args.window:
        return None
    if args.command == "nodes" and args.names:
        # Pending jobs have no node yet, so fetch the whole active queue.
        return ViewCommands(
            sacct=JOBS_SACCT_COMMAND, squeue=SQUEUE_COMMAND, partitions=True
        )
    users = target_users(args)
    if args.command == "jobs":
        # sacct misses requeued and array-range pending jobs that squeue lists.
        return ViewCommands(
            sacct=sacct_command(JOBS_SACCT_COMMAND, states=None, users=users),
            squeue=set_value_option(
                SQUEUE_COMMAND,
                ("-u", "--user"),
                ",".join(sorted(users)) if users else None,
            ),
        )
    return ViewCommands(
        sacct=sacct_command(SACCT_COMMAND, states=("RUNNING",), users=users)
        if users
        else SACCT_COMMAND,
        partitions=args.command == "available",
    )

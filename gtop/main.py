from __future__ import annotations

import sys

from .args import parse_args
from .constants import EXIT_SUCCESS
from .plan import view_commands
from .runner import PrefetchRunner


def main() -> None:
    """Starts the view's SLURM commands before importing the rest of gtop, so
    slurmctld answers while rich and the renderers load."""
    args = parse_args(None)
    commands = view_commands(args)
    runner = PrefetchRunner(commands.all() if commands is not None else ())
    try:
        from .cli import run

        raise SystemExit(run(args, runner=runner))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        raise SystemExit(EXIT_SUCCESS)
    finally:
        runner.close()

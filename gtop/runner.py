from __future__ import annotations

from subprocess import PIPE, Popen, TimeoutExpired
from typing import Iterable, NamedTuple, Protocol, Union

Command = tuple[str, ...]


class CommandResult(NamedTuple):
    command: Command
    stdout: str
    stderr: str
    returncode: int


class CommandRunner(Protocol):
    def run(self, command: Command, timeout: int) -> CommandResult:
        ...


def _start(command: Command) -> Union[Popen[str], CommandResult]:
    try:
        return Popen(command, stdout=PIPE, stderr=PIPE, text=True, encoding="utf-8")
    except OSError as error:
        return CommandResult(command, "", str(error), -1)


def _finish(
    started: Union[Popen[str], CommandResult], command: Command, timeout: int
) -> CommandResult:
    if isinstance(started, CommandResult):
        return started
    try:
        stdout, stderr = started.communicate(timeout=timeout)
    except TimeoutExpired:
        started.kill()
        stdout, stderr = started.communicate()
        note = f"Timed out after {timeout} seconds"
        return CommandResult(command, stdout, f"{stderr}\n{note}" if stderr else note, -1)
    return CommandResult(command, stdout, stderr, started.returncode)


class SubprocessRunner:
    def run(self, command: Command, timeout: int) -> CommandResult:
        return _finish(_start(command), command, timeout)


class PrefetchRunner:
    """Starts commands as soon as it is created, so they run while gtop loads.

    run() returns a started command's result and runs any other command
    normally, so a command missing from the prefetch only costs time.
    """

    def __init__(self, commands: Iterable[Command]):
        self._started = {command: _start(command) for command in dict.fromkeys(commands)}

    def run(self, command: Command, timeout: int) -> CommandResult:
        started = self._started.pop(command, None)
        if started is None:
            return SubprocessRunner().run(command, timeout)
        return _finish(started, command, timeout)

    def close(self) -> None:
        """Stops commands that were started but never collected."""
        for started in self._started.values():
            if isinstance(started, Popen):
                started.kill()
                started.communicate()
        self._started.clear()

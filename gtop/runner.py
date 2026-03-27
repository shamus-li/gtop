from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Dict, Mapping, Optional, Protocol


@dataclass(frozen=True)
class CommandResult:
    command: str
    stdout: str
    stderr: str
    returncode: int


class CommandRunner(Protocol):
    def run(self, command: str, timeout: int) -> CommandResult:
        ...


class SubprocessRunner:
    def run(self, command: str, timeout: int) -> CommandResult:
        try:
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
            return CommandResult(
                command=command,
                stdout=stdout_bytes.decode("utf-8"),
                stderr=stderr_bytes.decode("utf-8"),
                returncode=process.returncode,
            )
        except TimeoutExpired:
            process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
            stderr = stderr_bytes.decode("utf-8")
            if stderr:
                stderr = f"{stderr}\nTimed out after {timeout} seconds"
            else:
                stderr = f"Timed out after {timeout} seconds"
            return CommandResult(
                command=command,
                stdout=stdout_bytes.decode("utf-8"),
                stderr=stderr,
                returncode=-1,
            )
        except OSError as error:
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(error),
                returncode=-1,
            )


def run_commands(
    commands: Mapping[str, str],
    *,
    timeout: int,
    runner: Optional[CommandRunner] = None,
    parallel: bool = True,
) -> Dict[str, CommandResult]:
    active_runner = runner or SubprocessRunner()

    if not parallel:
        return {
            name: active_runner.run(command, timeout)
            for name, command in commands.items()
        }

    results: Dict[str, CommandResult] = {}
    with ThreadPoolExecutor(max_workers=min(len(commands), 4)) as executor:
        futures = {
            executor.submit(active_runner.run, command, timeout): name
            for name, command in commands.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results

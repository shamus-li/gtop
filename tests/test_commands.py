from subprocess import PIPE, TimeoutExpired
from unittest.mock import Mock, patch

from gtop.command_options import set_value_option
from gtop.runner import SubprocessRunner


def test_subprocess_runner_executes_argv_directly():
    command = ("sinfo", "-N", "--format=%N")
    process = Mock()
    process.communicate.return_value = ("node-a\n", "")
    process.returncode = 0

    with patch("gtop.runner.Popen", return_value=process) as popen:
        result = SubprocessRunner().run(command, timeout=5)

    popen.assert_called_once_with(
        command,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        encoding="utf-8",
    )
    assert result.stdout == "node-a\n"
    assert result.returncode == 0


def test_subprocess_runner_kills_timed_out_command():
    command = ("sacct", "-a")
    process = Mock()
    process.communicate.side_effect = [
        TimeoutExpired(command, 2),
        ("partial output", "partial error"),
    ]

    with patch("gtop.runner.Popen", return_value=process):
        result = SubprocessRunner().run(command, timeout=2)

    process.kill.assert_called_once_with()
    assert result.stdout == "partial output"
    assert result.stderr == "partial error\nTimed out after 2 seconds"
    assert result.returncode == -1


def test_set_value_option_replaces_short_and_attached_forms():
    command = (
        "sacct",
        "-a",
        "-s",
        "RUNNING",
        "--state=PENDING",
        "-P",
    )

    assert set_value_option(
        command,
        ("--state", "-s"),
        "REQUEUED",
    ) == ("sacct", "-a", "-P", "--state", "REQUEUED")


def test_set_value_option_removes_option_when_value_is_none():
    command = ("sinfo", "-N", "--partition=gpu", "-h")

    assert set_value_option(
        command,
        ("-p", "--partition", "--partitions"),
        None,
    ) == ("sinfo", "-N", "-h")

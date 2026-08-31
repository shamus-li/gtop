#!/usr/bin/env python3

import io
import json
from unittest.mock import patch

import pytest
from rich.console import Console

from gtop.cli import (
    View,
    _sacct_command,
    _write_json_output,
    build_parser,
    cli_main,
    main,
)
from gtop.collector import CollectionOptions, collect_cluster_state
from gtop.constants import (
    DEFAULT_TIMEOUT,
    EXIT_COMMAND_ERROR,
    EXIT_NO_MATCHES,
    EXIT_PARSE_ERROR,
    EXIT_SUCCESS,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
)
from gtop.constraints import compile_constraint
from gtop.runner import Command, CommandResult


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def run(self, command: Command, timeout: int) -> CommandResult:
        self.calls.append((command, timeout))
        return self.responses[command]


class RecordingConsole:
    def __init__(self):
        self.calls = []

    def print(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def filtered_collect_command(*users: str) -> Command:
    return _sacct_command(
        SACCT_COMMAND,
        states=("RUNNING",),
        users=set(users),
    )


def make_result(
    command: Command,
    stdout: str,
    returncode: int = 0,
    stderr: str = "",
) -> CommandResult:
    return CommandResult(
        command=command,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def make_small_cluster_outputs():
    sinfo_output = "\n".join(
        [
            "node-a|gpu,gpu-high|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a100:2(S:0)|gpu:a100:2(IDX:0-1)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|priority_partition|node-a|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "bob|default_partition|node-b|RUNNING|billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|102|",
        ]
    )
    return sinfo_output, sacct_output


def test_collect_cluster_state_returns_typed_state():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )

    state = collect_cluster_state(runner=runner, options=CollectionOptions())

    assert "node-a" in state.servers
    assert state.servers["node-a"].gpu.num == 4
    assert state.servers["node-a"].usage["gpu"].partitions == {"priority_partition": 1}
    assert len(state.jobs) == 2


def test_collect_cluster_state_can_skip_user_allocations():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )

    state = collect_cluster_state(
        runner=runner,
        options=CollectionOptions(store_allocations=False),
    )

    assert state.servers["node-a"].usage["gpu"].partitions == {"priority_partition": 1}
    assert state.servers["node-b"].usage["gpu"].partitions == {"default_partition": 2}
    assert state.servers["node-a"].allocations == {}
    assert state.servers["node-b"].allocations == {}


def test_collect_cluster_state_rejects_unparseable_sacct_output():
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(
                SINFO_COMMAND,
                "node-a|gpu|gpu:a100:4|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
            SACCT_COMMAND: make_result(
                SACCT_COMMAND,
                "fatal|error|from|sacct|detail=bad|output",
            ),
        }
    )

    code = cli_main(
        [],
        runner=runner,
        console=RecordingConsole(),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_PARSE_ERROR


def test_cli_summary_json_has_capacity_and_gpu_types_only():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stdout = RecordingConsole()
    stderr = RecordingConsole()

    code = cli_main(
        ["--json"],
        runner=runner,
        console=stdout,
        stderr_console=stderr,
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["view"] == "summary"
    assert payload["capacity"] == {
        "total": 6,
        "used": 3,
        "free": 3,
        "unit": "GPU",
    }
    assert payload["gpu_types"][0]["type"] == "A100"
    assert payload["gpu_types"][0]["usage"] == {
        "partitions": {
            "default_partition": 2,
            "priority_partition": 1,
        }
    }
    assert set(payload) == {"view", "capacity", "gpu_types"}
    assert "allocations" not in json.dumps(payload)
    assert "utilization" not in json.dumps(payload)


def test_cli_json_output_excludes_cpu_only_nodes():
    sinfo_output = "\n".join(
        [
            "gpu-node|gpu|gpu:a100:4(S:0-1)|gpu:a100:0(IDX:N/A)|0/0/0/0|0|0",
            "cpu-node|cpu|(null)|(null)|0/0/0/16|0|0",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--nodes", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["view"] == "nodes"
    assert [node["name"] for node in payload["nodes"]] == ["gpu-node"]
    assert payload["nodes"][0]["usage"] == {"partitions": {}}
    assert "allocations" not in json.dumps(payload)


def test_cli_top_users_json_contains_only_ranked_users():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--top-users", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert set(payload) == {"view", "unit", "users"}
    assert payload["view"] == "top-users"
    assert [(user["rank"], user["user"]) for user in payload["users"]] == [
        (1, "bob"),
        (2, "alice"),
    ]
    assert payload["users"][0]["partitions"] == {"default_partition": 2}
    assert payload["users"][0]["total"] == 2
    assert "capacity" not in payload


def test_cli_filtered_nodes_json_separates_capacity_from_usage():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--nodes", "--json", "--users", "alice"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    node = payload["nodes"][0]
    assert node["gpu"] == {
        "total": 4,
        "used": 1,
        "free": 3,
        "unit": "GPU",
    }
    assert node["usage"]["partitions"]["priority_partition"]["gpu"] == 1


def test_cli_partition_scope_applies_to_summary_mode():
    _, sacct_output = make_small_cluster_outputs()
    partition_sinfo_command = (*SINFO_COMMAND, "-p", "cornell")
    partition_sacct_command = SACCT_COMMAND
    runner = FakeRunner(
        {
            partition_sinfo_command: make_result(
                partition_sinfo_command,
                "node-a|gpu,gpu-high|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
            partition_sacct_command: make_result(
                partition_sacct_command,
                sacct_output,
            ),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--json", "--partition", "cornell", "--no-parallel"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["capacity"]["total"] == 4
    assert payload["gpu_types"][0]["type"] == "A100"
    assert payload["gpu_types"][0]["usage"]["partitions"] == {
        "priority_partition": 1
    }
    assert runner.calls == [
        (partition_sinfo_command, DEFAULT_TIMEOUT),
        (partition_sacct_command, DEFAULT_TIMEOUT),
    ]


def test_cli_top_users_includes_shared_partition_usage_on_scoped_nodes():
    sacct_output = "\n".join(
        [
            "alice|cornell|node-a|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "bob|other|node-a|RUNNING|billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|102|",
            "charlie|cornell|node-b|RUNNING|billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|103|",
        ]
    )
    partition_sinfo_command = (*SINFO_COMMAND, "-p", "cornell")
    partition_sacct_command = SACCT_COMMAND
    runner = FakeRunner(
        {
            partition_sinfo_command: make_result(
                partition_sinfo_command,
                "node-a|gpu,gpu-high|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
            partition_sacct_command: make_result(
                partition_sacct_command,
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["-U", "--partition", "cornell", "--no-parallel"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Partition scope: cornell" in output
    assert "alice" in output
    assert "bob" in output
    assert "charlie" not in output
    assert runner.calls == [
        (partition_sinfo_command, DEFAULT_TIMEOUT),
        (partition_sacct_command, DEFAULT_TIMEOUT),
    ]


def test_cli_no_matches_exit_code():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )

    code = cli_main(["--constraint", "missing", "--no-parallel"], runner=runner)

    assert code == EXIT_NO_MATCHES
    assert runner.calls == [
        (SINFO_COMMAND, DEFAULT_TIMEOUT),
        (SACCT_COMMAND, DEFAULT_TIMEOUT),
    ]


def test_cli_filtered_human_no_match_prints_one_message():
    sinfo_command = (*SINFO_COMMAND, "-p", "gpu")
    sacct_command = _sacct_command(
        SACCT_COMMAND,
        states=("RUNNING",),
        users={"nobody"},
    )
    runner = FakeRunner(
        {
            sinfo_command: make_result(
                sinfo_command,
                "node-a|gpu|gpu:a100:4|gpu:a100:0(IDX:N/A)|0/0/0/0|0|0",
            ),
            sacct_command: make_result(sacct_command, ""),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["-u", "nobody", "-p", "gpu"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_NO_MATCHES
    assert len(stdout.calls) == 1
    assert "No usage found matching the criteria." in str(stdout.calls[0][0][0])


def test_cli_top_users_human_no_match_prints_one_message():
    sinfo_output, _ = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["-U"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_NO_MATCHES
    assert len(stdout.calls) == 1
    assert "No users found matching the criteria." in str(stdout.calls[0][0][0])


def test_cli_json_no_match_keeps_stdout_empty():
    sinfo_output, _ = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )
    stdout = RecordingConsole()
    stderr = RecordingConsole()

    code = cli_main(
        ["-U", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=stderr,
    )

    assert code == EXIT_NO_MATCHES
    assert stdout.calls == []
    assert "No users found matching the criteria." in str(stderr.calls[0][0][0])


def test_cli_rejects_boolean_constraint_syntax_before_collection():
    runner = FakeRunner({})

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--constraint", "gpu&"], runner=runner)

    assert excinfo.value.code == 2
    assert runner.calls == []


def test_cli_rejects_empty_constraint_feature():
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["-C", ""])

    assert excinfo.value.code == 2


def test_cli_nodes_constraint_filters_collected_nodes():
    sinfo_output = "\n".join(
        [
            "node-a|gpu,gpu-high|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "alice|priority_partition|node-a|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    with patch(
        "gtop.collector.compile_constraint",
        wraps=compile_constraint,
    ) as compile_mock:
        code = cli_main(
            ["-v", "--constraint", "gpu-high"],
            runner=runner,
            console=console,
            stderr_console=RecordingConsole(),
        )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert compile_mock.call_count == 1
    assert "node-a" in output
    assert "node-b" not in output
    assert {call[0] for call in runner.calls} == {SINFO_COMMAND, SACCT_COMMAND}


def test_cli_constraint_features_use_and_semantics():
    sinfo_output = "\n".join(
        [
            "node-a|gpu,gpu-high|gpu:a100:4|gpu:a100:0(IDX:N/A)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a40:2|gpu:a40:0(IDX:N/A)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--nodes", "--json", "-C", "gpu", "gpu-high"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert [node["name"] for node in payload["nodes"]] == ["node-a"]


def test_cli_command_failure_exit_code():
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, "", returncode=1, stderr="boom"),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )

    code = cli_main(["--json"], runner=runner)

    assert code == EXIT_COMMAND_ERROR


def test_cli_unknown_user_is_a_concise_no_match():
    user = "missing-user"
    sacct_command = _sacct_command(
        SACCT_COMMAND,
        states=("RUNNING",),
        users={user},
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, make_small_cluster_outputs()[0]),
            sacct_command: make_result(
                sacct_command,
                "",
                returncode=1,
                stderr=f"sacct: error: Invalid user id: {user}",
            ),
        }
    )
    stdout = RecordingConsole()
    stderr = RecordingConsole()

    code = cli_main(
        ["--users", user],
        runner=runner,
        console=stdout,
        stderr_console=stderr,
    )

    assert code == EXIT_NO_MATCHES
    assert str(stdout.calls[0][0][0]) == f"Unknown user: {user}."
    assert not stderr.calls


def test_cli_parse_failure_exit_code():
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, "garbage"),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )

    code = cli_main(["--json"], runner=runner)

    assert code == EXIT_PARSE_ERROR


def test_cli_empty_partition_scope_is_a_no_match():
    partition = "missing"
    sinfo_command = (*SINFO_COMMAND, "-p", partition)
    sacct_command = SACCT_COMMAND
    runner = FakeRunner(
        {
            sinfo_command: make_result(sinfo_command, ""),
            sacct_command: make_result(sacct_command, ""),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--partition", partition],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_NO_MATCHES
    assert str(stdout.calls[0][0][0]) == (
        f"No servers found in partition scope: {partition}."
    )


def test_cli_command_overrides_and_timeout_are_forwarded():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    sinfo_command = ("custom-sinfo", "--label", "two words")
    sacct_command = ("custom-sacct",)
    runner = FakeRunner(
        {
            sinfo_command: make_result(sinfo_command, sinfo_output),
            sacct_command: make_result(sacct_command, sacct_output),
        }
    )

    code = cli_main(
        [
            "--json",
            "--no-parallel",
            "--timeout",
            "5",
            "--sinfo-command",
            "custom-sinfo --label 'two words'",
            "--sacct-command",
            "custom-sacct",
        ],
        runner=runner,
        console=RecordingConsole(),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert runner.calls == [(sinfo_command, 5), (sacct_command, 5)]


def test_cli_me_filters_to_current_user():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stdout = RecordingConsole()

    with patch("gtop.cli.getpass.getuser", return_value="alice"):
        code = cli_main(
            ["--json", "--me"],
            runner=runner,
            console=stdout,
            stderr_console=RecordingConsole(),
        )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["view"] == "summary"
    assert payload["capacity"]["used"] == 1
    assert payload["gpu_types"][0]["usage"] == {
        "partitions": {"priority_partition": 1}
    }
    assert "target_users" not in json.dumps(payload)


def test_cli_me_respects_constraint():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stdout = RecordingConsole()

    with patch("gtop.cli.getpass.getuser", return_value="alice"):
        code = cli_main(
            ["--json", "--me", "--constraint", "gpu-high"],
            runner=runner,
            console=stdout,
            stderr_console=RecordingConsole(),
        )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["capacity"]["total"] == 4
    assert payload["gpu_types"][0]["usage"]["partitions"] == {
        "priority_partition": 1
    }


def test_cli_does_not_print_top_users_by_default():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        [],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    printed_strings = [
        args[0] for args, _ in stdout.calls if args and isinstance(args[0], str)
    ]
    assert not any("Top Users" in text for text in printed_strings)


def test_cli_default_table_uses_compact_resource_schema():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        [],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Cluster Overview  3/6 GPUs free" in output
    assert "GPU Type" in output
    assert "Nodes" in output
    assert "Memory" not in output
    assert "CPU" not in output
    assert "A100" in output
    assert "3/6 GPUs free" in output
    assert "3/6" in output
    assert "0/2" in output
    assert "node-a" not in output
    assert "node-b" not in output
    header_line = next(line for line in output.splitlines() if "GPU Type" in line)
    assert header_line.index("Nodes") > header_line.index("GPU")


def test_cli_text_and_json_use_sinfo_occupancy_for_availability():
    sinfo_output = "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:3(IDX:0-2)|0/0/0/0|0|0"
    sacct_output = (
        "alice|priority_partition|node-a|RUNNING|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    responses = {
        SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
        SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
    }
    stream = io.StringIO()

    text_code = cli_main(
        [],
        runner=FakeRunner(responses),
        console=Console(file=stream, width=140, force_terminal=False),
        stderr_console=RecordingConsole(),
    )
    json_output = RecordingConsole()
    json_code = cli_main(
        ["--json"],
        runner=FakeRunner(responses),
        console=json_output,
        stderr_console=RecordingConsole(),
    )

    payload = json.loads(json_output.calls[0][0][0])
    assert text_code == json_code == EXIT_SUCCESS
    assert "Cluster Overview  1/4 GPU free" in stream.getvalue()
    assert "[██████··]" in stream.getvalue()
    assert payload["capacity"]["used"] == 3
    assert payload["capacity"]["total"] == 4
    assert payload["gpu_types"][0]["usage"]["partitions"] == {
        "priority_partition": 1
    }


def test_cli_help_contains_display_legend():
    parser = build_parser()
    stream = io.StringIO()

    parser.print_help(stream)
    help_text = stream.getvalue()

    assert "Core options:" in help_text
    assert "Debug options:" in help_text
    assert help_text.count("████") == 4
    assert "████ = priority" in help_text
    assert "████ = gpu" in help_text
    assert "████ = default" in help_text
    assert "████ = unattributed usage" in help_text
    assert "dots = free" not in help_text
    assert "counts after bars: priority / gpu / default" in help_text


@pytest.mark.parametrize("width", [80, 140])
def test_cli_verbose_table_aligns_bars_across_rows(width: int):
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=False)

    code = cli_main(
        ["-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    row_lines = [line for line in output if "node-a" in line or "node-b" in line]
    assert code == EXIT_SUCCESS
    assert max(len(line) for line in output) <= width
    assert len(row_lines) == 2
    assert row_lines[0].index("[") == row_lines[1].index("[")
    assert row_lines[0].index("[", row_lines[0].index("[") + 1) == row_lines[1].index(
        "[", row_lines[1].index("[") + 1
    )
    assert row_lines[0].index(
        "[", row_lines[0].index("[", row_lines[0].index("[") + 1) + 1
    ) == row_lines[1].index(
        "[", row_lines[1].index("[", row_lines[1].index("[") + 1) + 1
    )


def test_cli_verbose_table_aligns_bars_with_long_names_on_narrow_terminal():
    sinfo_output = "\n".join(
        [
            "node-with-a-very-long-name-a|gpu|gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4(S:0)|gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:1(IDX:0)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a100:4(S:0)|gpu:a100:2(IDX:0-1)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|priority_partition|node-with-a-very-long-name-a|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "bob|default_partition|node-b|RUNNING|billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|102|",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        ["-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    row_lines = [
        line
        for line in output
        if "node-with-a-very-long-name-a" in line or "node-b" in line
    ]
    assert code == EXIT_SUCCESS
    assert len(row_lines) == 2
    assert row_lines[0].index("[") == row_lines[1].index("[")
    assert row_lines[0].index("[", row_lines[0].index("[") + 1) == row_lines[1].index(
        "[", row_lines[1].index("[") + 1
    )
    assert row_lines[0].index(
        "[", row_lines[0].index("[", row_lines[0].index("[") + 1) + 1
    ) == row_lines[1].index(
        "[", row_lines[1].index("[", row_lines[1].index("[") + 1) + 1
    )


def test_cli_default_summary_does_not_truncate_gpu_type_names():
    sinfo_output = (
        "node-a|gpu|"
        "gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:2(S:0)|"
        "gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:1(IDX:0)|"
        "0/0/0/0|0|0"
    )
    sacct_output = (
        "alice|default_partition|node-a|RUNNING|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        [],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Pro 6000 Blackwell Max-Q" in output
    assert "…" not in output


def test_cli_default_table_shows_gpu_type_once_per_group():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        [],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert output.count("A100") == 1
    assert "A100" in output and "3/6 GPUs free" in output
    assert "node-a" not in output
    assert "node-b" not in output


def test_cli_verbose_shows_node_by_node_breakdown():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        ["-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "node-a" in output
    assert "node-b" in output


def test_cli_default_text_view_hides_cpu_only_nodes():
    sinfo_output = "\n".join(
        [
            "gpu-node|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            "cpu-node|cpu|(null)|(null)|0/0/0/0|0|0",
        ]
    )
    sacct_output = (
        "alice|priority_partition|gpu-node|RUNNING|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        [],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "A100" in output
    assert "gpu-node" not in output
    assert "cpu-node" not in output


def test_cli_shard_mode_only_shows_sharded_nodes():
    sinfo_output = "\n".join(
        [
            "shard-node|gpu|gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)|gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:0(0/200)|0/0/0/0|0|0",
            "gpu-node|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|gpu|shard-node|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "bob|gpu|gpu-node|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|102|",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        ["-s"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "A40" in output
    assert "Shard" in output
    assert "200/400 shards free" in output
    assert "shard-node" not in output
    assert "gpu-node" not in output
    assert "200/400" in output
    assert "0/200" in output


def test_cli_group_headers_sort_by_gpu_capability():
    sinfo_output = "\n".join(
        [
            "node-t4|gpu|gpu:nvidia_t4:2(S:0)|gpu:nvidia_t4:0(IDX:N/A)|0/0/0/0|0|0",
            "node-2080|gpu|gpu:nvidia_geforce_rtx_2080_ti:2(S:0)|gpu:nvidia_geforce_rtx_2080_ti:0(IDX:N/A)|0/0/0/0|0|0",
            "node-a100|gpu|gpu:nvidia_a100:2(S:0)|gpu:nvidia_a100:0(IDX:N/A)|0/0/0/0|0|0",
            "node-h100|gpu|gpu:nvidia_h100_nvl:2(S:0)|gpu:nvidia_h100_nvl:0(IDX:N/A)|0/0/0/0|0|0",
            "node-b200|gpu|gpu:nvidia_b200:2(S:0)|gpu:nvidia_b200:0(IDX:N/A)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=200, force_terminal=False)

    code = cli_main(
        [],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    group_lines = [
        line
        for line in output.splitlines()
        if any(
            name in line for name in ("T4", "RTX 2080 Ti", "A100", "H100 NVL", "B200")
        )
    ]
    assert group_lines == [
        next(line for line in group_lines if "T4" in line),
        next(line for line in group_lines if "RTX 2080 Ti" in line),
        next(line for line in group_lines if "A100" in line),
        next(line for line in group_lines if "H100 NVL" in line),
        next(line for line in group_lines if "B200" in line),
    ]


def test_cli_shard_json_only_includes_sharded_nodes():
    sinfo_output = "\n".join(
        [
            "shard-node|gpu|gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)|gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:0(0/200)|0/0/0/0|0|0",
            "gpu-node|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|gpu|shard-node|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "bob|gpu|gpu-node|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|102|",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["-s", "--nodes", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert [node["name"] for node in payload["nodes"]] == ["shard-node"]
    assert payload["nodes"][0]["gpu"]["total"] == 400
    assert payload["nodes"][0]["usage"]["partitions"]["gpu"]["gpu"] == 200


def test_cli_top_users_include_full_name_when_available():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    with patch("gtop.render._lookup_full_name", return_value="Alice [Example]"):
        code = cli_main(
            ["-U"],
            runner=runner,
            console=console,
            stderr_console=RecordingConsole(),
        )

    assert code == EXIT_SUCCESS
    assert "alice (Alice [Example])" in stream.getvalue()


def test_cli_user_summary_treats_brackets_as_plain_text():
    sinfo_output = "node-a|gpu|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0"
    sacct_output = (
        "alice[lab]|priority[queue]|node-a|RUNNING|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice[lab]"): make_result(
                filtered_collect_command("alice[lab]"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        ["--users", "alice[lab]"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert output.splitlines()[0] == "User"
    assert "alice[lab]" in output
    assert "priority[queue]" in output
    assert "Summary of Resources Used by Specified Users" not in output


def test_cli_filtered_users_include_full_name_when_available():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    with patch("gtop.render._lookup_full_name", return_value="Alice [Example]"):
        code = cli_main(
            ["--users", "alice"],
            runner=runner,
            console=console,
            stderr_console=RecordingConsole(),
        )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "alice (Alice [Example])" in output


def test_cli_filtered_user_view_shows_user_usage_not_cluster_free():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Usage  1 GPU used" in output
    assert "1 GPU used" in output
    assert "A100" in output
    assert "node-a" not in output
    assert "1/0/0" in output
    assert "node-b" not in output
    assert "3/4 GPUs free" not in output


def test_cli_filtered_summary_hides_cpu_only_gpu_nodes():
    sinfo_output = (
        "node-a|gpu|gpu:a6000:4(S:0)|gpu:a6000:4(IDX:0-3)|0/0/0/0|0|0"
    )
    sacct_output = (
        "alice|gpu|node-a|RUNNING|billing=16,cpu=16,mem=32G,node=1|101|"
    )
    command = filtered_collect_command("alice")
    responses = {
        SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
        command: make_result(command, sacct_output),
    }
    stream = io.StringIO()

    code = cli_main(
        ["--users", "alice"],
        runner=FakeRunner(responses),
        console=Console(file=stream, width=100, force_terminal=False),
        stderr_console=RecordingConsole(),
    )
    json_output = RecordingConsole()
    json_code = cli_main(
        ["--json", "--users", "alice"],
        runner=FakeRunner(responses),
        console=json_output,
        stderr_console=RecordingConsole(),
    )

    payload = json.loads(json_output.calls[0][0][0])
    assert code == json_code == EXIT_SUCCESS
    assert "Usage  0 GPUs used" in stream.getvalue()
    assert "GPU Type" not in stream.getvalue()
    assert "A6000" not in stream.getvalue()
    assert payload["capacity"] == {
        "free": 0,
        "total": 0,
        "unit": "GPU",
        "used": 0,
    }
    assert payload["gpu_types"] == []


def test_cli_three_way_partition_split_distinguishes_gpu_partition():
    sinfo_output = "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0"
    sacct_output = (
        "alice|gpu|node-a|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "1 GPU used" in output
    assert "0/1/0" in output


def test_cli_filtered_user_summary_counts_shard_usage_as_gpu_occupancy():
    sinfo_output = (
        "dgx-spark|gpu|gpu:nvidia_gb10:1(S:0-19),shard:nvidia_gb10:80(S:0-19)|"
        "gpu:nvidia_gb10:0(IDX:N/A),shard:nvidia_gb10:40(0/80)|0/0/0/0|0|0"
    )
    sacct_output = (
        "alice|spark|dgx-spark|RUNNING|"
        "billing=10,cpu=10,gres/shard:nvidia_gb10=40,gres/shard=40,mem=50G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Usage  1 GPU used" in output
    assert "GB10" in output
    assert "1" in output


def test_cli_filtered_user_summary_does_not_double_count_full_gpu_jobs_on_sharded_nodes():
    sinfo_output = (
        "shard-node|gpu|gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)|"
        "gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:0(0/200)|0/0/0/0|0|0"
    )
    sacct_output = (
        "alice|gpu|shard-node|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"), sacct_output
            ),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--json", "--users", "alice"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["gpu_types"][0]["usage"]["partitions"]["gpu"] == 1
    assert payload["capacity"]["used"] == 1


def test_cli_filtered_node_view_has_no_jobs_appendix():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"),
                sacct_output,
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["--users", "alice", "-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "node-a" in output
    assert "node-b" not in output
    assert "My Jobs" not in output
    assert "Filtered Jobs" not in output


def test_cli_filtered_user_summary_aligns_bars_for_group_totals():
    sinfo_output = "\n".join(
        [
            "node-a|gpu|gpu:a100:6(S:0)|gpu:a100:5(IDX:0-4)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a100:6(S:0)|gpu:a100:5(IDX:0-4)|0/0/0/0|0|0",
            "node-c|gpu|gpu:b200:2(S:0)|gpu:b200:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|priority_partition|node-a|RUNNING|billing=8,cpu=8,gres/gpu=5,mem=32G,node=1|101|",
            "alice|gpu|node-b|RUNNING|billing=8,cpu=8,gres/gpu=5,mem=32G,node=1|102|",
            "alice|default_partition|node-c|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|103|",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(
                filtered_collect_command("alice"), sacct_output
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    summary_lines = [line for line in output if "A100" in line or "B200" in line]
    assert code == EXIT_SUCCESS
    assert len(summary_lines) == 2
    assert "10" in summary_lines[0]
    assert "1" in summary_lines[1]
    assert "10/12" not in summary_lines[0]
    assert "1/2" not in summary_lines[1]
    assert summary_lines[0].index("[") == summary_lines[1].index("[")


def test_cli_top_users_only_mode_hides_other_sections():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["-U"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    printed_strings = [str(args[0]) for args, _ in stdout.calls if args]
    assert code == EXIT_SUCCESS
    assert any("Top Users" in text for text in printed_strings)
    assert not any("Cluster GPU Overview" in text for text in printed_strings)
    assert not any(
        "Summary of Resources Used by Specified Users" in text
        for text in printed_strings
    )


def test_cli_top_users_is_a_boolean_view():
    parser = build_parser()

    args = parser.parse_args(["-U"])

    assert args.view is View.TOP_USERS


@pytest.mark.parametrize("flag", ["-v", "--nodes"])
def test_cli_node_view_aliases(flag):
    args = build_parser().parse_args([flag])

    assert args.view is View.NODES


@pytest.mark.parametrize(
    "argv",
    [
        ["--nodes", "--jobs"],
        ["--nodes", "--top-users"],
        ["--jobs", "--top-users"],
    ],
)
def test_cli_views_are_mutually_exclusive(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(argv)

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--jobs", "--shard"],
        ["--top-users", "--users", "alice"],
        ["--top-users", "--me"],
    ],
)
def test_cli_rejects_options_irrelevant_to_view(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(argv)

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--verbose"],
        ["--states", "RUNNING"],
        ["--sort", "name"],
        ["--my-partitions"],
        ["--mine"],
        ["-U", "5"],
    ],
)
def test_cli_rejects_removed_options(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(argv)

    assert excinfo.value.code == 2


def test_sacct_command_removes_state_and_keeps_user_filter():
    command = ("sacct", "-a", "-s", "RUNNING", "-P")

    assert _sacct_command(
        command,
        states=None,
        users={"alice"},
    ) == (
        "sacct",
        "-a",
        "-P",
        "--user",
        "alice",
    )


def test_sacct_state_pushdown_is_summary_only():
    assert "--state=RUNNING" in SACCT_COMMAND
    assert not any(
        token == "--state"
        or token == "-s"
        or token.startswith("--state=")
        for token in JOBS_SACCT_COMMAND
    )


def test_cli_me_short_alias():
    args = build_parser().parse_args(["-m"])

    assert args.me


def test_cli_me_and_users_are_mutually_exclusive():
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["-m", "-u", "alice"])

    assert excinfo.value.code == 2


def test_main_exits_cleanly_on_keyboard_interrupt():
    with patch("gtop.cli.cli_main", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as excinfo:
            main()

    assert excinfo.value.code == 130


def test_main_exits_cleanly_on_broken_pipe():
    with patch("gtop.cli.cli_main", side_effect=BrokenPipeError):
        with patch("gtop.cli.sys.stdout"):
            with pytest.raises(SystemExit) as excinfo:
                main()

    assert excinfo.value.code == EXIT_SUCCESS


def test_json_output_flushes_piped_stdout():
    stream = io.StringIO()

    with patch("gtop.cli.sys.stdout", stream):
        _write_json_output("{}", None)

    assert stream.getvalue() == "{}\n"


def test_cli_top_users_uses_shared_collection_pipeline():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        ["-U"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert {call[0] for call in runner.calls} == {SINFO_COMMAND, SACCT_COMMAND}
    assert "Top Users" in stream.getvalue()


def test_cli_top_users_uses_table_with_full_split_header():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["-U"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "1/0/0" in output
    assert "0/0/2" in output
    assert "Breakdown" not in output
    assert "priority/gpu/default-partition" not in output
    assert "P:" not in output
    assert " D:" not in output
    header_line = next(
        line for line in output.splitlines() if "User" in line and "Nodes" in line
    )
    assert header_line.index("Nodes") > header_line.index("GPU")


def test_cli_top_users_respects_constraint():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["-U", "--constraint", "gpu-high"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "alice" in output
    assert "bob" not in output
    assert {call[0] for call in runner.calls} == {SINFO_COMMAND, SACCT_COMMAND}


def test_cli_top_users_includes_shard_only_usage():
    sacct_output = (
        "alice|101|spark_run|RUNNING|spark|dgx-spark|"
        "billing=10,cpu=10,gres/shard:nvidia_gb10=40,gres/shard=40,mem=50G,node=1|15-00:00:00|"
    )
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "dgx-spark|gpu|gpu:nvidia_gb10:1(S:0-19),shard:nvidia_gb10:80(S:0-19)|gpu:nvidia_gb10:0(IDX:N/A),shard:nvidia_gb10:40(0/80)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=160, force_terminal=False)

    code = cli_main(
        ["-U"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "alice" in output
    assert "1" in output
    assert {call[0] for call in runner.calls} == {SINFO_COMMAND, SACCT_COMMAND}


def test_cli_top_users_collects_all_nodes_once():
    sacct_output = "\n".join(
        [
            "alice|101|spark_a|RUNNING|spark|dgx-spark|billing=10,cpu=10,gres/shard:nvidia_gb10=40,gres/shard=40,mem=50G,node=1|15-00:00:00|",
            "alice|102|spark_b|RUNNING|spark-interactive|dgx-spark-02|billing=5,cpu=5,gres/shard:nvidia_gb10=20,gres/shard=20,mem=20G,node=1|2-00:00:00|",
        ]
    )
    full_sinfo_output = "\n".join(
        [
            "dgx-spark|gpu|gpu:nvidia_gb10:1(S:0-19),shard:nvidia_gb10:80(S:0-19)|gpu:nvidia_gb10:0(IDX:N/A),shard:nvidia_gb10:40(0/80)|0/0/0/0|0|0",
            "dgx-spark-02|gpu|gpu:nvidia_gb10:1(S:0-1),shard:nvidia_gb10:80(S:0-1)|gpu:nvidia_gb10:0(IDX:N/A),shard:nvidia_gb10:20(20/80)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
            SINFO_COMMAND: make_result(SINFO_COMMAND, full_sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["-U"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "alice" in output
    assert "  2 " in output
    assert {call[0] for call in runner.calls} == {SINFO_COMMAND, SACCT_COMMAND}


def test_cli_json_output_is_not_wrapped_by_rich_console():
    sinfo_output = (
        "node-a|gpu,gpu-high|"
        "gpu:nvidia_rtx_6000_ada_generation:4(S:0-15),"
        "gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:2(S:8-15)|"
        "gpu:nvidia_rtx_6000_ada_generation:2(IDX:4-5),"
        "gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:0(IDX:N/A)|"
        "0/0/0/0|0|0"
    )
    sacct_output = (
        "alice|priority_partition|node-a|RUNNING|"
        "billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=40, force_terminal=False)

    code = cli_main(
        ["--json"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stream.getvalue())
    assert payload["gpu_types"][0]["type"] == ("6000 Ada + Pro 6000 Blackwell Max-Q")


def test_cli_jobs_output_includes_job_rows():
    _, sacct_output = make_small_cluster_outputs()
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "node-a|gpu|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Jobs" in output
    assert "node-a" in output
    assert "1/4 GPUs used" in output
    assert "101" in output
    assert "alice" in output
    assert "ID" in output
    assert "GPU" in output


def test_cli_jobs_json_uses_fixed_active_states_and_jobs_shape():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|wait_b|PENDING|gpu||gpu-high&ssd|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|104|retry_d|REQUEUED|gpu||billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|103|done_c|COMPLETED|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            JOBS_SACCT_COMMAND: make_result(JOBS_SACCT_COMMAND, sacct_output),
            SINFO_COMMAND: make_result(
                SINFO_COMMAND,
                "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--jobs", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["view"] == "jobs"
    assert [(job["job_id"], job["state"]) for job in payload["jobs"]] == [
        ("102", "PENDING"),
        ("104", "REQUEUED"),
        ("101", "RUNNING"),
    ]
    assert set(payload) == {"view", "jobs"}
    assert payload["jobs"][0]["constraints"] == ["gpu-high", "ssd"]
    assert set(payload["jobs"][0]["resources"]) == {
        "gpu",
        "shards",
        "cpu",
        "memory_gib",
    }


def test_cli_jobs_multi_node_headers_show_aggregate_capacity_and_gpu_type():
    sacct_output = "alice|101|run_a|RUNNING|gpu|node-[1-2]|billing=16,cpu=16,gres/gpu=2,mem=64G,node=2|7-00:00:00|"
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    sinfo_output = "\n".join(
        [
            "node-1|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            "node-2|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(jobs_sinfo_command, sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=220, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "node-[1-2]" in output
    assert "A100" in output
    assert "2/8 GPUs used" in output


def test_cli_jobs_does_not_print_overview_before_sinfo_succeeds():
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(
                jobs_command,
                "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            ),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "",
                returncode=1,
                stderr="boom",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_COMMAND_ERROR
    assert "Jobs" not in stream.getvalue()


def test_cli_jobs_output_uses_compact_parsed_columns():
    sacct_output = (
        "alice|101|train_run|RUNNING|priority_partition|node-a|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|"
    )
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Partition" in output
    assert "GPU" in output
    assert "CPU" in output
    assert "MEM" in output
    assert "Time" in output
    assert "train_run" in output
    assert "7-00:00:00" in output
    assert "AllocTRES" not in output
    assert "NodeList" not in output


def test_cli_jobs_output_uses_shard_capacity_for_sharded_nodes():
    sacct_output = (
        "alice|101|spark_run|RUNNING|spark|dgx-spark|"
        "billing=10,cpu=10,gres/shard:gb10=40,gres/shard=40,mem=50G,node=1|15-00:00:00|"
    )
    jobs_command = JOBS_SACCT_COMMAND
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "dgx-spark|gpu|gpu:gb10:1(S:0),shard:gb10:80(S:0)|gpu:gb10:0(IDX:N/A),shard:gb10:40(0/80)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "dgx-spark" in output
    assert "GB10" in output
    assert "40/80 shards used" in output
    assert "40s" in output


def test_cli_jobs_output_converts_full_gpu_jobs_on_sharded_nodes():
    sacct_output = (
        "alice|101|full_gpu_run|RUNNING|gpu|shard-node|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|"
    )
    jobs_command = JOBS_SACCT_COMMAND
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "shard-node|gpu|gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)|gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:0(0/200)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "shard-node" in output
    assert "A40" in output
    assert "200/400 shards used" in output
    assert "full_gpu_run" in output


def test_cli_jobs_mode_filters_partition_and_active_states():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|gpu|node-b|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|103|done_c|COMPLETED|monakhova|node-c|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = _sacct_command(
        JOBS_SACCT_COMMAND,
        states=None,
        users={"alice"},
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            (*SINFO_COMMAND, "-p", "monakhova"): make_result(
                (*SINFO_COMMAND, "-p", "monakhova"),
                "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "--partition", "monakhova"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "run_a" in output
    assert "pend_b" not in output
    assert "done_c" not in output


def test_cli_partition_short_flag_parses_multiple_values():
    parser = build_parser()

    args = parser.parse_args(["-p", "monakhova", "gpu"])

    assert args.partition == ["monakhova", "gpu"]


@pytest.mark.parametrize(
    ("user_args", "users", "expected_ids"),
    [
        ([], set(), {"101", "103", "104", "105", "107"}),
        (["--me"], {"alice"}, {"101", "105", "107"}),
        (["--users", "bob"], {"bob"}, {"103", "104"}),
    ],
)
def test_cli_jobs_partition_scope_includes_all_usage_on_nodes(
    user_args, users, expected_ids
):
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova|monakhova-compute-01|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|other-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "bob|103|interactive|RUNNING|monakhova-interactive|monakhova-compute-01|cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "bob|104|shared|RUNNING|gpu|monakhova-compute-01|cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|105|pending_here|PENDING|monakhova|None assigned||7-00:00:00|",
            "alice|106|pending_elsewhere|PENDING|gpu|None assigned||7-00:00:00|",
            "alice|107|shared|RUNNING|gpu|monakhova-compute-01|cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "bob|108|done|COMPLETED|monakhova-interactive|monakhova-compute-01|cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = _sacct_command(
        JOBS_SACCT_COMMAND,
        states=None,
        users=users,
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            (*SINFO_COMMAND, "-p", "monakhova"): make_result(
                (*SINFO_COMMAND, "-p", "monakhova"),
                "monakhova-compute-01|gpu|gpu:a6000:8(S:0)|gpu:a6000:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stdout = RecordingConsole()

    with patch("gtop.cli.getpass.getuser", return_value="alice"):
        code = cli_main(
            ["-j", "--json", "-p", "monakhova", *user_args],
            runner=runner,
            console=stdout,
            stderr_console=RecordingConsole(),
        )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert {job["job_id"] for job in payload["jobs"]} == expected_ids


def test_cli_jobs_mode_flattens_comma_separated_partitions():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova*|monakhova-compute-01|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|scavenge|other-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|103|run_c|RUNNING|debug|third-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = _sacct_command(
        JOBS_SACCT_COMMAND,
        states=None,
        users={"alice"},
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            (*SINFO_COMMAND, "-p", "monakhova,scavenge"): make_result(
                (*SINFO_COMMAND, "-p", "monakhova,scavenge"),
                "\n".join(
                    [
                        "monakhova-compute-01|gpu|gpu:a6000:8(S:0)|gpu:a6000:1(IDX:0)|0/0/0/0|0|0",
                        "other-node|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0",
                    ]
                ),
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "-p", "monakhova,scavenge"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "run_a" in output
    assert "run_b" in output
    assert "run_c" not in output


def test_cli_jobs_mode_partition_filter_avoids_node_substring_false_positive():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|cpu|alphagpu01|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = _sacct_command(
        JOBS_SACCT_COMMAND,
        states=None,
        users={"alice"},
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            (*SINFO_COMMAND, "-p", "gpu"): make_result(
                (*SINFO_COMMAND, "-p", "gpu"),
                "gpu-node|gpu|gpu:a100:4(S:0)|gpu:a100:0(IDX:N/A)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "--partition", "gpu"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_NO_MATCHES
    assert "No jobs found matching the criteria." in output


def test_cli_jobs_mode_constraint_filters_jobs_by_node_features():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|node-b|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    sinfo_output = "\n".join(
        [
            "node-a|gpu,gpu-high|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            "node-b|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(jobs_sinfo_command, sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "--constraint", "gpu-high"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Jobs  1 running" in output
    assert "run_a" in output
    assert "run_b" not in output


def test_cli_jobs_mode_constraint_can_yield_no_matches():
    sacct_output = "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|"
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    sinfo_output = "node-a|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0"
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(jobs_sinfo_command, sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "--constraint", "gpu-high"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_NO_MATCHES
    assert "No jobs found matching the criteria." in stream.getvalue()


def test_cli_jobs_mode_constraint_keeps_matching_pending_jobs_without_nodes():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|monakhova-interactive|None assigned|gpu-high|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|10:00:00|",
        ]
    )
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    sinfo_output = "node-a|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0"
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(jobs_sinfo_command, sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice", "--constraint", "gpu-high"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "run_a" not in output
    assert "pend_b" in output


def test_cli_jobs_mode_constraint_rejects_nonmatching_pending_jobs():
    sacct_output = (
        "alice|102|pend_b|PENDING|gpu|None assigned|gpu-low|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|10:00:00|"
    )
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            SINFO_COMMAND: make_result(
                SINFO_COMMAND,
                "node-a|gpu|gpu:a40:2|gpu:a40:0(IDX:N/A)|0/0/0/0|0|0",
            ),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["-j", "-u", "alice", "-C", "gpu-high"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_NO_MATCHES
    assert "No jobs found matching the criteria." in str(stdout.calls[0][0][0])


def test_cli_jobs_partition_scope_matches_pending_partition_choices():
    jobs_command = _sacct_command(
        JOBS_SACCT_COMMAND,
        states=None,
        users={"alice"},
    )
    scoped_sinfo = (*SINFO_COMMAND, "-p", "gpu")
    runner = FakeRunner(
        {
            jobs_command: make_result(
                jobs_command,
                "alice|102|pend_b|PENDING|gpu,default_partition|None assigned|||10:00:00|",
            ),
            scoped_sinfo: make_result(
                scoped_sinfo,
                "node-a|gpu|gpu:a100:4|gpu:a100:0(IDX:N/A)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()

    code = cli_main(
        ["--jobs", "--users", "alice", "--partition", "gpu"],
        runner=runner,
        console=Console(file=stream, width=120, force_terminal=False),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert "pend_b" in stream.getvalue()


def test_cli_jobs_can_show_pending_jobs_without_nodes():
    runner = FakeRunner(
        {
            JOBS_SACCT_COMMAND: make_result(
                JOBS_SACCT_COMMAND,
                "alice|102|pend_b|PENDING|gpu|None assigned|||10:00:00|",
            ),
            SINFO_COMMAND: make_result(SINFO_COMMAND, ""),
        }
    )
    stream = io.StringIO()

    code = cli_main(
        ["--jobs"],
        runner=runner,
        console=Console(file=stream, width=120, force_terminal=False),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert "pend_b" in stream.getvalue()


def test_cli_jobs_mode_uses_shared_collection_pipeline():
    _, sacct_output = make_small_cluster_outputs()
    jobs_command = JOBS_SACCT_COMMAND
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "\n".join(
                    [
                        "node-a|gpu|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
                        "node-b|gpu|gpu:a100:2(S:0)|gpu:a100:2(IDX:0-1)|0/0/0/0|0|0",
                    ]
                ),
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=140, force_terminal=False)

    code = cli_main(
        ["--jobs"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert {call[0] for call in runner.calls} == {
        jobs_command,
        jobs_sinfo_command,
    }


def test_cli_jobs_view_keeps_pending_and_running_rows_in_one_table():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|gpu||billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            (*JOBS_SACCT_COMMAND, "--user", "alice"): make_result(
                (*JOBS_SACCT_COMMAND, "--user", "alice"),
                sacct_output,
            ),
            SINFO_COMMAND: make_result(
                SINFO_COMMAND,
                "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Pending / Unassigned" not in output
    assert "pend_b" in output
    assert "1/4 GPUs used" in output
    assert sum(1 for line in output.splitlines() if line.startswith("ID")) == 1


def test_cli_jobs_mode_pushes_user_filter_into_sacct_command():
    _, sacct_output = make_small_cluster_outputs()
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(
                jobs_sinfo_command,
                "node-a|gpu|gpu:a100:4(S:0-1)|gpu:a100:1(IDX:0)|0/0/0/0|0|0",
            ),
        }
    )

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=RecordingConsole(),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert {call[0] for call in runner.calls} == {
        jobs_command,
        jobs_sinfo_command,
    }


def test_cli_jobs_headers_align_gpu_type_column():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|n1|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|very-long-node-name|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = (*JOBS_SACCT_COMMAND, "--user", "alice")
    jobs_sinfo_command = SINFO_COMMAND
    sinfo_output = "\n".join(
        [
            "n1|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0",
            "very-long-node-name|gpu|gpu:b200:8(S:0)|gpu:b200:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    runner = FakeRunner(
        {
            jobs_command: make_result(jobs_command, sacct_output),
            jobs_sinfo_command: make_result(jobs_sinfo_command, sinfo_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=220, force_terminal=False)

    code = cli_main(
        ["--jobs", "--users", "alice"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    header_lines = {
        "n1": next(line for line in output if "GPUs used" in line and "n1" in line),
        "very-long-node-name": next(
            line
            for line in output
            if "GPUs used" in line and "very-long-node-name" in line
        ),
    }
    assert code == EXIT_SUCCESS
    assert header_lines["n1"].index("A40") == header_lines["very-long-node-name"].index(
        "B200"
    )

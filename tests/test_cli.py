#!/usr/bin/env python3

import io
import json
from unittest.mock import patch

import pytest
from rich.console import Console

from gtop import (
    CollectionOptions,
    CommandResult,
    DEFAULT_TIMEOUT,
    EXIT_COMMAND_ERROR,
    EXIT_NO_MATCHES,
    EXIT_PARSE_ERROR,
    EXIT_SUCCESS,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
    cli_main,
    collect_cluster_state,
)
from gtop.cli import _jobs_sacct_command, build_parser, main


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def run(self, command: str, timeout: int) -> CommandResult:
        self.calls.append((command, timeout))
        return self.responses[command]


class RecordingConsole:
    def __init__(self):
        self.calls = []

    def print(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def filtered_collect_command(*users: str) -> str:
    return _jobs_sacct_command(
        SACCT_COMMAND,
        states=("RUNNING",),
        users=set(users),
    )


def make_result(command: str, stdout: str, returncode: int = 0, stderr: str = ""):
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
    assert state.servers["node-a"].usage["gpu"].priority == 1
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
        options=CollectionOptions(store_users=False),
    )

    assert state.servers["node-a"].usage["gpu"].priority == 1
    assert state.servers["node-b"].usage["gpu"].default == 2
    assert state.servers["node-a"].users == {}
    assert state.servers["node-b"].users == {}


def test_cli_json_output_respects_sort_mode():
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
        ["--json", "--sort", "free-gpu"],
        runner=runner,
        console=stdout,
        stderr_console=stderr,
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert payload["servers"][0]["name"] == "node-a"
    assert payload["summary"]["gpu_total"] == 6


def test_cli_no_matches_exit_code():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            SACCT_COMMAND: make_result(SACCT_COMMAND, sacct_output),
        }
    )

    code = cli_main(["--constraint", "missing"], runner=runner)

    assert code == EXIT_NO_MATCHES


def test_cli_command_failure_exit_code():
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, "", returncode=1, stderr="boom"),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )

    code = cli_main(["--json"], runner=runner)

    assert code == EXIT_COMMAND_ERROR


def test_cli_parse_failure_exit_code():
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, "garbage"),
            SACCT_COMMAND: make_result(SACCT_COMMAND, ""),
        }
    )

    code = cli_main(["--json"], runner=runner)

    assert code == EXIT_PARSE_ERROR


def test_cli_command_overrides_and_timeout_are_forwarded():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            "custom-sinfo": make_result("custom-sinfo", sinfo_output),
            "custom-sacct": make_result("custom-sacct", sacct_output),
        }
    )

    code = cli_main(
        [
            "--json",
            "--no-parallel",
            "--timeout",
            "5",
            "--sinfo-command",
            "custom-sinfo",
            "--sacct-command",
            "custom-sacct",
        ],
        runner=runner,
        console=RecordingConsole(),
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    assert runner.calls == [("custom-sinfo", 5), ("custom-sacct", 5)]


def test_cli_me_filters_to_current_user():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
    assert payload["summary"]["target_users"]["alice"]["priority_usage"] == 1
    assert payload["summary"]["target_users"]["alice"]["default_usage"] == 0


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
    printed_strings = [args[0] for args, _ in stdout.calls if args and isinstance(args[0], str)]
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
    assert "magenta = priority   cyan = gpu   blue = default   dim = free" not in output
    header_line = next(line for line in output.splitlines() if "GPU Type" in line)
    assert header_line.index("Nodes") > header_line.index("GPU")


def test_cli_help_contains_display_legend():
    parser = build_parser()

    help_text = parser.format_help()

    assert "Core options:" in help_text
    assert "Debug options:" in help_text
    assert "magenta = priority   cyan = gpu   blue = default   dim = free" in help_text
    assert "counts are free/total, then priority/gpu/default" in help_text


def test_cli_verbose_table_aligns_bars_across_rows():
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
        ["-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    row_lines = [line for line in output if "node-a" in line or "node-b" in line]
    assert code == EXIT_SUCCESS
    assert len(row_lines) == 2
    assert row_lines[0].index("[") == row_lines[1].index("[")
    assert row_lines[0].index("[", row_lines[0].index("[") + 1) == row_lines[1].index("[", row_lines[1].index("[") + 1)
    assert row_lines[0].index("[", row_lines[0].index("[", row_lines[0].index("[") + 1) + 1) == row_lines[1].index("[", row_lines[1].index("[", row_lines[1].index("[") + 1) + 1)


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
    assert row_lines[0].index("[", row_lines[0].index("[") + 1) == row_lines[1].index("[", row_lines[1].index("[") + 1)
    assert row_lines[0].index("[", row_lines[0].index("[", row_lines[0].index("[") + 1) + 1) == row_lines[1].index("[", row_lines[1].index("[", row_lines[1].index("[") + 1) + 1)


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
        if any(name in line for name in ("T4", "RTX 2080 Ti", "A100", "H100 NVL", "B200"))
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
        ["-s", "--json"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert [server["name"] for server in payload["servers"]] == ["shard-node"]
    assert payload["servers"][0]["usage"]["shard"]["gpu"] == 200


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
    assert "Filtered Users" in output
    assert "alice[lab]" in output
    assert "priority[queue]" in output
    assert "Summary of Resources Used by Specified Users" not in output


def test_cli_filtered_users_include_full_name_when_available():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
    assert "Filtered Usage  1/4 GPU used" in output
    assert "1/4 GPU used" in output
    assert "A100" in output
    assert "node-a" not in output
    assert "1/0/0" in output
    assert "node-b" not in output
    assert "3/4 GPUs free" not in output


def test_cli_three_way_partition_split_distinguishes_gpu_partition():
    sinfo_output = "node-a|gpu|gpu:a100:4(S:0)|gpu:a100:1(IDX:0)|0/0/0/0|0|0"
    sacct_output = (
        "alice|gpu|node-a|RUNNING|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|"
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
    assert "1/4 GPU used" in output
    assert "0/1/0" in output


def test_cli_verbose_filtered_user_view_shows_nodes():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
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
    assert "10/12" in summary_lines[0]
    assert "1/2" in summary_lines[1]
    assert summary_lines[0].index("[") == summary_lines[1].index("[")


def test_cli_me_verbose_shows_filtered_job_tables():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=180, force_terminal=False)

    with patch("gtop.cli.getpass.getuser", return_value="alice"):
        code = cli_main(
            ["--me", "-v"],
            runner=runner,
            console=console,
            stderr_console=RecordingConsole(),
        )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "node-a" in output
    assert "101" in output
    assert "alice" in output
    assert "node-b" not in output


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
    assert not any("Summary of Resources Used by Specified Users" in text for text in printed_strings)


def test_cli_top_users_default_count_is_25():
    parser = build_parser()

    args = parser.parse_args(["-U"])

    assert args.top_users == 25


def test_main_exits_cleanly_on_keyboard_interrupt():
    with patch("gtop.cli.cli_main", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as excinfo:
            main()

    assert excinfo.value.code == 130


def test_cli_top_users_only_mode_skips_sinfo():
    _, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
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
    assert runner.calls == [(SACCT_COMMAND, DEFAULT_TIMEOUT)]
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


def test_cli_users_filter_limits_top_users_summary():
    sinfo_output, sacct_output = make_small_cluster_outputs()
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
        }
    )
    stdout = RecordingConsole()

    code = cli_main(
        ["--json", "-u", "alice", "-U", "5"],
        runner=runner,
        console=stdout,
        stderr_console=RecordingConsole(),
    )

    assert code == EXIT_SUCCESS
    payload = json.loads(stdout.calls[0][0][0])
    assert [item["user"] for item in payload["summary"]["top_users"]] == ["alice"]


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
    assert payload["servers"][0]["gpu"]["type"].startswith("(")


def test_cli_jobs_output_includes_job_rows():
    _, sacct_output = make_small_cluster_outputs()
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a"
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


def test_cli_jobs_output_uses_compact_parsed_columns():
    sacct_output = (
        "alice|101|train_run|RUNNING|priority_partition|node-a|"
        "billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|"
    )
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a"
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


def test_cli_jobs_mode_filters_partition_and_active_states():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|gpu|node-b|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|103|done_c|COMPLETED|monakhova|node-c|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            f"{JOBS_SACCT_COMMAND} --user=alice": make_result(
                f"{JOBS_SACCT_COMMAND} --user=alice",
                sacct_output,
            ),
            f"{SINFO_COMMAND} -n=node-a": make_result(
                f"{SINFO_COMMAND} -n=node-a",
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


def test_cli_jobs_mode_partition_filter_matches_nodelist_too():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|monakhova-compute-01|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|other-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            f"{JOBS_SACCT_COMMAND} --user=alice": make_result(
                f"{JOBS_SACCT_COMMAND} --user=alice",
                sacct_output,
            ),
            f"{SINFO_COMMAND} -n=monakhova-compute-01": make_result(
                f"{SINFO_COMMAND} -n=monakhova-compute-01",
                "monakhova-compute-01|gpu|gpu:a6000:8(S:0)|gpu:a6000:1(IDX:0)|0/0/0/0|0|0",
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
    assert "run_b" not in output


def test_cli_jobs_mode_partition_filter_matches_any_requested_partition():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|monakhova-compute-01|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|scavenge|other-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|103|run_c|RUNNING|debug|third-node|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            f"{JOBS_SACCT_COMMAND} --user=alice": make_result(
                f"{JOBS_SACCT_COMMAND} --user=alice",
                sacct_output,
            ),
            f"{SINFO_COMMAND} -n=monakhova-compute-01,other-node": make_result(
                f"{SINFO_COMMAND} -n=monakhova-compute-01,other-node",
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
        ["--jobs", "--users", "alice", "-p", "monakhova", "scavenge"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "run_a" in output
    assert "run_b" in output
    assert "run_c" not in output


def test_cli_jobs_mode_constraint_filters_jobs_by_node_features():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|node-b|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a,node-b"
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
    assert "run_a" in output
    assert "run_b" not in output


def test_cli_jobs_mode_constraint_can_yield_no_matches():
    sacct_output = "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|"
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a"
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


def test_cli_jobs_mode_constraint_keeps_pending_jobs_without_assigned_nodes():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|monakhova-interactive||billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|10:00:00|",
        ]
    )
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a"
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


def test_cli_jobs_mode_skips_sinfo():
    _, sacct_output = make_small_cluster_outputs()
    jobs_command = JOBS_SACCT_COMMAND
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a,node-b"
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
    assert runner.calls == [
        (jobs_command, DEFAULT_TIMEOUT),
        (jobs_sinfo_command, DEFAULT_TIMEOUT),
    ]


def test_cli_jobs_view_keeps_pending_rows_without_unassigned_section():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|monakhova|node-a|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|pend_b|PENDING|gpu||billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    runner = FakeRunner(
        {
            f"{JOBS_SACCT_COMMAND} --user=alice": make_result(
                f"{JOBS_SACCT_COMMAND} --user=alice",
                sacct_output,
            ),
            f"{SINFO_COMMAND} -n=node-a": make_result(
                f"{SINFO_COMMAND} -n=node-a",
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
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=node-a"
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
    assert runner.calls == [
        (jobs_command, DEFAULT_TIMEOUT),
        (jobs_sinfo_command, DEFAULT_TIMEOUT),
    ]


def test_cli_jobs_headers_align_gpu_type_column():
    sacct_output = "\n".join(
        [
            "alice|101|run_a|RUNNING|gpu|n1|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
            "alice|102|run_b|RUNNING|gpu|very-long-node-name|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|7-00:00:00|",
        ]
    )
    jobs_command = f"{JOBS_SACCT_COMMAND} --user=alice"
    jobs_sinfo_command = f"{SINFO_COMMAND} -n=n1,very-long-node-name"
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
            line for line in output if "GPUs used" in line and "very-long-node-name" in line
        ),
    }
    assert code == EXIT_SUCCESS
    assert header_lines["n1"].index("A40") == header_lines["very-long-node-name"].index("B200")


def test_cli_me_verbose_job_headers_align_gpu_type_column():
    sinfo_output = "\n".join(
        [
            "n1|gpu|gpu:a40:2(S:0)|gpu:a40:1(IDX:0)|0/0/0/0|0|0",
            "very-long-node-name|gpu|gpu:b200:8(S:0)|gpu:b200:1(IDX:0)|0/0/0/0|0|0",
        ]
    )
    sacct_output = "\n".join(
        [
            "alice|priority_partition|n1|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|101|",
            "alice|gpu|very-long-node-name|RUNNING|billing=8,cpu=8,gres/gpu=1,mem=32G,node=1|102|",
        ]
    )
    runner = FakeRunner(
        {
            SINFO_COMMAND: make_result(SINFO_COMMAND, sinfo_output),
            filtered_collect_command("alice"): make_result(filtered_collect_command("alice"), sacct_output),
        }
    )
    stream = io.StringIO()
    console = Console(file=stream, width=220, force_terminal=False)

    code = cli_main(
        ["--users", "alice", "-v"],
        runner=runner,
        console=console,
        stderr_console=RecordingConsole(),
    )

    output = stream.getvalue().splitlines()
    header_lines = {
        "n1": next(line for line in output if "GPUs used" in line and "n1" in line),
        "very-long-node-name": next(
            line for line in output if "GPUs used" in line and "very-long-node-name" in line
        ),
    }
    assert code == EXIT_SUCCESS
    assert header_lines["n1"].index("A40") == header_lines["very-long-node-name"].index("B200")

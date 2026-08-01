import io
import json
from pathlib import Path

from rich.console import Console

from gtop.accounting import process_jobs, summarize_users
from gtop.cli import cli_main
from gtop.constants import (
    EXIT_SUCCESS,
    JOBS_SACCT_COMMAND,
    SACCT_COMMAND,
    SINFO_COMMAND,
    SINFO_FIELD_WIDTHS,
)
from gtop.render import _display_gpu_type
from gtop.runner import Command, CommandResult
from gtop.slurm import parse_jobs, parse_sinfo

FIXTURE_DIR = Path(__file__).resolve().parent


class FixtureRunner:
    def __init__(self, responses: dict[Command, CommandResult]):
        self.responses = responses

    def run(self, command: Command, timeout: int) -> CommandResult:
        return self.responses[command]


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def _result(command: Command, stdout: str) -> CommandResult:
    return CommandResult(command=command, stdout=stdout, stderr="", returncode=0)


def _runner(*, jobs_view: bool = False) -> FixtureRunner:
    sacct_command = JOBS_SACCT_COMMAND if jobs_view else SACCT_COMMAND
    return FixtureRunner(
        {
            SINFO_COMMAND: _result(
                SINFO_COMMAND,
                _fixture("sinfo-output-empire.txt"),
            ),
            sacct_command: _result(
                sacct_command,
                _fixture("sacct-output-empire.txt"),
            ),
        }
    )


def test_empire_snapshot_parses_repeated_nodes_and_generic_gres():
    servers = parse_sinfo(_fixture("sinfo-output-empire.txt"))
    jobs = parse_jobs(_fixture("sacct-output-empire.txt"))

    assert set(servers) == {"alphacpu01", "alphagh01", "alphagpu01", "alphagpu24"}
    assert sum(server.gpu.num for server in servers.values()) == 17
    assert sum(server.gpu.used for server in servers.values()) == 8
    assert _display_gpu_type(servers["alphagh01"]) == "GPU"
    assert _display_gpu_type(servers["alphagpu01"]) == "H100 80GB HBM3"

    gpu_servers = {name: server for name, server in servers.items() if server.gpu.num}
    process_jobs(jobs, gpu_servers)
    users = summarize_users(gpu_servers)

    assert users["user_a"].total_usage() == 8
    assert "user_d" not in users


def test_empire_fixed_width_sinfo_layout_parses():
    fields = (
        "alphagpu24",
        "location=local,nvidia_h200,xeon_8568y",
        "gpu:nvidia_h200:8(S:0-1)",
        "gpu:nvidia_h200:0(IDX:N/A)",
        "0/96/0/96",
        "0",
        "1907348",
    )
    line = "".join(
        value.ljust(width) for value, width in zip(fields, SINFO_FIELD_WIDTHS)
    )

    server = parse_sinfo(line)["alphagpu24"]

    assert server.gpu.num == 8
    assert server.gpu.used == 0
    assert server.cpu.idle == 96
    assert server.mem.total == 1907348


def test_empire_summary_and_json_report_gpu_nodes_only():
    stream = io.StringIO()
    code = cli_main(
        [],
        runner=_runner(),
        console=Console(file=stream, width=140, force_terminal=False),
        stderr_console=Console(file=io.StringIO(), force_terminal=False),
    )

    assert code == EXIT_SUCCESS
    assert "Cluster Overview  9/17 GPUs free" in stream.getvalue()
    assert "H100 80GB HBM3" in stream.getvalue()
    assert "H200" in stream.getvalue()
    assert "GPU" in stream.getvalue()
    assert "Null" not in stream.getvalue()

    json_stream = io.StringIO()
    json_code = cli_main(
        ["--json"],
        runner=_runner(),
        console=Console(file=json_stream, width=140, force_terminal=False),
        stderr_console=Console(file=io.StringIO(), force_terminal=False),
    )
    payload = json.loads(json_stream.getvalue())

    assert json_code == EXIT_SUCCESS
    assert payload["capacity"] == {
        "free": 9,
        "total": 17,
        "unit": "GPU",
        "used": 8,
    }
    assert [row["type"] for row in payload["gpu_types"]] == [
        "GPU",
        "H100 80GB HBM3",
        "H200",
    ]

    nodes_stream = io.StringIO()
    nodes_code = cli_main(
        ["--nodes", "--json"],
        runner=_runner(),
        console=Console(file=nodes_stream, width=140, force_terminal=False),
        stderr_console=Console(file=io.StringIO(), force_terminal=False),
    )
    nodes_payload = json.loads(nodes_stream.getvalue())

    assert nodes_code == EXIT_SUCCESS
    assert {node["name"] for node in nodes_payload["nodes"]} == {
        "alphagh01",
        "alphagpu01",
        "alphagpu24",
    }


def test_empire_jobs_exclude_assigned_cpu_nodes_but_keep_pending_jobs():
    stream = io.StringIO()

    code = cli_main(
        ["--jobs"],
        runner=_runner(jobs_view=True),
        console=Console(file=stream, width=200, force_terminal=False),
        stderr_console=Console(file=io.StringIO(), force_terminal=False),
    )

    output = stream.getvalue()
    assert code == EXIT_SUCCESS
    assert "Jobs  1 running  2 pending" in output
    assert "gpu_training" in output
    assert "gpu_pending" in output
    assert "multi_partition_pending" in output
    assert "cpu_editor" not in output
    assert "alphacpu01" not in output
    assert sum(line.startswith("ID") for line in output.splitlines()) == 1

    json_stream = io.StringIO()
    json_code = cli_main(
        ["--jobs", "--json"],
        runner=_runner(jobs_view=True),
        console=Console(file=json_stream, width=200, force_terminal=False),
        stderr_console=Console(file=io.StringIO(), force_terminal=False),
    )
    jobs = json.loads(json_stream.getvalue())["jobs"]

    assert json_code == EXIT_SUCCESS
    assert {job["job_name"] for job in jobs} == {
        "gpu_training",
        "gpu_pending",
        "multi_partition_pending",
    }
    running = next(job for job in jobs if job["job_name"] == "gpu_training")
    assert running["resources"] == {
        "cpu": 96,
        "gpu": 8,
        "memory_gib": 0,
        "shards": 0,
    }

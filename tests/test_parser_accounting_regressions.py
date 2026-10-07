from __future__ import annotations

import math

import pytest

from gtop.accounting import (
    process_jobs,
    project_servers_for_users,
    summarize_users,
)
from gtop.collector import (
    ClusterParseError,
    CollectionOptions,
    collect_cluster_state,
)
from gtop.constants import SACCT_COMMAND, SINFO_COMMAND, SQUEUE_COMMAND
from gtop.resources import parse_gpu, parse_usage
from gtop.runner import Command, CommandResult
from gtop.slurm import parse_jobs, parse_nodelist, parse_sinfo
from sacct_rows import sacct_row


class FakeRunner:
    def __init__(self, responses: dict[Command, CommandResult]):
        self.responses = responses

    def run(self, command: Command, timeout: int) -> CommandResult:
        return self.responses[command]


def result(command: Command, stdout: str) -> CommandResult:
    return CommandResult(command=command, stdout=stdout, stderr="", returncode=0)


def test_gpu_and_shard_requests_are_additive():
    servers = parse_sinfo(
        "node-1|gpu|gpu:a40:2,shard:a40:400|"
        "gpu:a40:1(IDX:0),shard:a40:40(0/200,40/200)|"
        "0/8/0/8|0|65536",
    )
    jobs = parse_jobs(sacct_row("cpu=1,gres/gpu=1,gres/shard=40", name="mixed"))

    process_jobs(jobs, servers)
    gpu_users = summarize_users(servers)
    projected = project_servers_for_users(
        list(servers.values()),
        target_users={"alice"},
    )[0]

    assert servers["node-1"].allocations["1"].usage.shard == 240
    assert servers["node-1"].usage["shard"].partitions == {"gpu": 240}
    assert gpu_users["alice"].total_usage() == 2
    assert projected.usage["gpu"].partitions == {"gpu": 2}
    assert projected.usage["shard"].partitions == {"gpu": 240}


def test_parse_nodelist_expands_multiple_brackets_and_suffixes():
    assert parse_nodelist("rack[1-2]n[01-02]-gpu") == [
        "rack1n01-gpu",
        "rack1n02-gpu",
        "rack2n01-gpu",
        "rack2n02-gpu",
    ]


def test_parse_nodelist_rejects_empty_host():
    with pytest.raises(ValueError, match="Invalid hostlist"):
        parse_nodelist("node-1,")


def test_parse_jobs_keeps_historical_rows_with_duplicate_job_ids():
    jobs = parse_jobs(
        "\n".join(
            [
                sacct_row("gres/gpu=1", job_id="639182", state="RESIZING"),
                sacct_row("gres/gpu=1", job_id="639182", state="PREEMPTED"),
            ]
        )
    )

    assert [job.state for job in jobs] == ["RESIZING", "PREEMPTED"]


def test_collector_deduplicates_active_jobs_last_record_wins():
    runner = FakeRunner(
        {
            SINFO_COMMAND: result(
                SINFO_COMMAND,
                "node-1|gpu|gpu:a100:4|gpu:a100:2(IDX:0-1)|0/8/0/8|0|65536",
            ),
            SACCT_COMMAND: result(
                SACCT_COMMAND,
                "\n".join(
                    [
                        sacct_row("gres/gpu=1", name="first"),
                        sacct_row("gres/gpu=2", name="last"),
                    ]
                ),
            ),
        }
    )

    state = collect_cluster_state(
        runner=runner,
        options=CollectionOptions(),
    )

    assert len(state.jobs) == 1
    assert state.jobs[0].job_name == "last"
    assert state.servers["node-1"].usage["gpu"].partitions == {"gpu": 2}


def test_collector_merges_squeue_jobs_over_sacct_records():
    runner = FakeRunner(
        {
            SINFO_COMMAND: result(
                SINFO_COMMAND,
                "node-1|gpu|gpu:a100:4|gpu:a100:1(IDX:0)|0/8/0/8|0|65536",
            ),
            SACCT_COMMAND: result(
                SACCT_COMMAND,
                "\n".join(
                    [
                        sacct_row("gres/gpu=1", job_id="1"),
                        sacct_row("gres/gpu=1", job_id="2_7", state="REQUEUED"),
                        sacct_row("gres/gpu=1", job_id="9", user="bob"),
                    ]
                ),
            ),
            SQUEUE_COMMAND: result(
                SQUEUE_COMMAND,
                "\n".join(
                    [
                        sacct_row("gres/gpu=1", job_id="1", constraints="(null)"),
                        sacct_row("gres/gpu=1", job_id="2_7", state="PENDING", nodelist=""),
                        sacct_row(
                            "gres/gpu=1", job_id="3_[1-4,9]", state="PENDING", nodelist=""
                        ),
                    ]
                ),
            ),
        }
    )

    state = collect_cluster_state(
        runner=runner,
        options=CollectionOptions(squeue_command=SQUEUE_COMMAND),
    )

    assert {job.job_id: job.state for job in state.jobs} == {
        "1": "RUNNING",
        "2_7": "PENDING",
        "9": "RUNNING",
        "3_[1-4,9]": "PENDING",
    }
    assert all(not job.constraints for job in state.jobs)
    assert state.servers["node-1"].usage["gpu"].partitions == {"gpu": 2}


def test_parse_jobs_canonicalizes_decorated_running_state():
    jobs = parse_jobs(sacct_row("gres/gpu=1", state="RUNNING+"))
    servers = parse_sinfo(
        "node-1|gpu|gpu:a100:2|gpu:a100:1(IDX:0)|0/8/0/8|0|65536",
    )

    process_jobs(jobs, servers)

    assert jobs[0].state == "RUNNING"
    assert servers["node-1"].usage["gpu"].partitions == {"gpu": 1}


def test_parse_jobs_skips_inactive_rows_before_resource_parsing():
    output = "\n".join(
        [
            sacct_row("gres/gpu=1"),
            sacct_row("gres/gpu=invalid", job_id="2", name="done", state="COMPLETED"),
        ]
    )

    jobs = parse_jobs(output, states={"RUNNING", "PENDING", "REQUEUED"})

    assert [job.job_id for job in jobs] == ["1"]
    with pytest.raises(ValueError, match="Invalid resource value"):
        parse_jobs(output)


def test_parse_jobs_keeps_pipes_inside_job_names_and_constraints():
    jobs = parse_jobs(
        sacct_row("cpu=4,gres/gpu=2", name="seed-1|lr-0.1", constraints="a100|h100")
    )

    assert jobs[0].job_name == "seed-1|lr-0.1"
    assert jobs[0].constraints == frozenset({"a100", "h100"})
    assert jobs[0].usage.gpu == 2


def test_parse_jobs_drops_running_records_long_past_their_time_limit():
    output = "\n".join(
        [
            sacct_row("gres/gpu=1", job_id="1", time_limit="01:00:00", elapsed="13-22:47:58"),
            sacct_row("gres/gpu=1", job_id="2", time_limit="01:00:00", elapsed="01:02:30"),
            sacct_row("gres/gpu=1", job_id="3", time_limit="UNLIMITED", elapsed="30-00:00:00"),
        ]
    )

    assert [job.job_id for job in parse_jobs(output)] == ["2", "3"]


def test_parse_jobs_rejects_empty_state():
    with pytest.raises(ValueError, match="Malformed sacct record"):
        parse_jobs(sacct_row("gres/gpu=1", state=""))


def test_parse_sinfo_rejects_any_malformed_nonblank_row():
    output = "\n".join(
        [
            "node-1|gpu|gpu:a100:2|gpu:a100:1(IDX:0)|0/8/0/8|0|65536",
            "node-2|gpu|gpu:a100:2|gpu:a100:1(IDX:0)|0/8/0/8|0",
        ]
    )

    with pytest.raises(ValueError, match="Malformed sinfo record on line 2"):
        parse_sinfo(output)


def test_collector_surfaces_malformed_sinfo_as_cluster_parse_error():
    runner = FakeRunner(
        {
            SINFO_COMMAND: result(
                SINFO_COMMAND,
                "node-1|gpu|gpu:a100:2|gpu:a100:1|0/8/0/8|0",
            ),
            SACCT_COMMAND: result(SACCT_COMMAND, ""),
        }
    )

    with pytest.raises(ClusterParseError, match="Malformed sinfo record on line 1"):
        collect_cluster_state(
            runner=runner,
            options=CollectionOptions(),
        )


@pytest.mark.parametrize(
    "value",
    [
        "cpu=-1",
        "mem=-1G",
        "gres/gpu=-1",
        "gres/shard=-1",
        "cpu=1e309",
        "mem=1e309G",
        "gres/gpu=1e309",
        "gres/shard=1e309",
    ],
)
def test_parse_usage_rejects_negative_and_nonfinite_resources(value: str):
    with pytest.raises(ValueError, match="non-negative finite"):
        parse_usage(value)


@pytest.mark.parametrize(
    ("gres", "gres_used"),
    [
        ("gpu:a100:2", "gpu:a100:3"),
        ("gpu:a100:2,shard:a100:200", "gpu:a100:0,shard:a100:201"),
        (
            "gpu:a100:2,shard:a100:200",
            "gpu:a100:2,shard:a100:1(0/100,1/100)",
        ),
        ("(null)", "gpu:a100:1"),
    ],
)
def test_parse_gpu_rejects_observed_usage_above_capacity(
    gres: str,
    gres_used: str,
):
    with pytest.raises(ValueError, match="exceeds"):
        parse_gpu(gres, gres_used)


def test_expanded_sinfo_nodes_do_not_share_gpu_state():
    servers = parse_sinfo(
        "node-[1-2]|gpu|gpu:a100:2|gpu:a100:0(IDX:N/A)|0/8/0/8|0|65536",
    )

    assert servers["node-1"].gpu is not servers["node-2"].gpu
    servers["node-1"].gpu.used = 1
    assert servers["node-2"].gpu.used == 0


def test_multi_node_user_total_is_conserved_before_rounding():
    servers = parse_sinfo(
        "\n".join(
            [
                "node-1|gpu|gpu:a100:1|gpu:a100:0(IDX:N/A)|0/8/0/8|0|65536",
                "node-2|gpu|gpu:a100:1|gpu:a100:0(IDX:N/A)|0/8/0/8|0|65536",
            ]
        ),
    )
    jobs = parse_jobs(
        sacct_row("cpu=8,gres/gpu=1,mem=16G", name="multi", nodelist="node-[1-2]")
    )

    process_jobs(jobs, servers)
    users = summarize_users(servers)

    assert (
        math.fsum(server.allocations["1"].usage.gpu for server in servers.values()) == 1
    )
    assert users["alice"].total_usage() == 1


def test_multi_node_shard_occupancy_is_rounded_once_for_user_total():
    servers = parse_sinfo(
        "\n".join(
            [
                "node-1|gpu|gpu:a100:1,shard:a100:100|none|0/8/0/8|0|65536",
                "node-2|gpu|gpu:a100:1,shard:a100:100|none|0/8/0/8|0|65536",
            ]
        ),
    )
    jobs = parse_jobs(
        sacct_row("cpu=8,gres/shard=20,mem=16G", name="multi", nodelist="node-[1-2]")
    )

    process_jobs(jobs, servers)
    users = summarize_users(servers)

    assert users["alice"].total_usage() == 1

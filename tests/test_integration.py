#!/usr/bin/env python3
"""Integration-oriented tests for gtop parsing helpers."""

from pathlib import Path

import pytest


from gtop.constants import NODES_COMMAND, SACCT_COMMAND
from gtop.accounting import process_jobs, summarize_users
from gtop.render_cluster import visible_servers
from gtop.resources import parse_gpu
from gtop.slurm import (
    array_task_count,
    expand_range,
    parse_features_field,
    parse_jobs,
    parse_nodes,
)
from slurm_rows import node_row, sacct_row

FIXTURE_DIR = Path(__file__).resolve().parent


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def test_klara_regular_gpu():
    """Test parsing klara node with regular A6000 GPUs"""

    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:8(IDX:0-7)"
    result = parse_gpu(gres, gres_used)

    assert result.type == "nvidia_rtx_a6000"  # Should not be sharded
    assert result.num == 8
    assert result.used == 8


def test_dutta_sharded_gpu():
    """Test parsing dutta-compute-01 node with true sharded H100 GPUs"""
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    result = parse_gpu(gres, gres_used)

    assert "Shard" in result.type  # Should be sharded due to shard: entry
    assert result.num == 2
    assert result.shards == 48
    assert result.used == 1


def test_snavely_range_pattern():
    """Test parsing snavely-compute-09 with range pattern (should NOT be sharded)"""
    gres = "gpu:nvidia_geforce_gtx_titan_x:4(S:0-1)"
    gres_used = "gpu:nvidia_geforce_gtx_titan_x:2(IDX:0,2)"
    result = parse_gpu(gres, gres_used)

    assert (
        "Shard" not in result.type
    )  # Should NOT be sharded (only shard: entries matter)
    assert result.num == 4
    assert result.used == 2


def test_unicorn_mixed_gpu_types():
    """Test parsing unicorn-compute-01 with mixed GPU types"""
    gres = "gpu:nvidia_geforce_rtx_2080_ti:2(S:0),gpu:nvidia_a40:2(S:1)"
    gres_used = "gpu:nvidia_geforce_rtx_2080_ti:2(IDX:0-1),gpu:nvidia_a40:2(IDX:2-3)"
    result = parse_gpu(gres, gres_used)

    assert result.num == 4
    assert result.used == 4
    assert "|" in result.type  # Should show mixed types


def test_cpu_only_node():
    """Test parsing CPU-only nodes"""
    gres = "(null)"
    result = parse_gpu(gres)

    assert result.type == "null"
    assert result.num == 0


def test_sinfo_parsing_sample():
    """Test parsing a sample of the sinfo data structure"""

    # Sample sinfo line format (space-separated)
    sample_line = "dutta-compute-01 gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23) gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24) 98/286/0/384 1516436 1547600"

    # Mock the parsing (since we'd need the full parse_nodes function)
    parts = sample_line.split()
    if len(parts) >= 6:
        node_name, gres, gres_used = parts[0], parts[1], parts[2]
        gpu_result = parse_gpu(gres, gres_used)

        assert node_name == "dutta-compute-01"
        assert gpu_result.num == 2
        assert gpu_result.shards == 48


def test_parse_nodes_unicorn_nodes():
    """Ensure unicorn sinfo fixture is parsed correctly."""

    servers = parse_nodes(read_fixture("nodes-unicorn.txt"))

    assert "klara" in servers
    assert servers["klara"].gpu.num == 8
    assert "unicorn-compute-01" in servers
    assert servers["unicorn-compute-01"].gpu.type.count("|") == 1
    assert "gpu" in servers["unicorn-compute-01"].features
    assert "gpu-high" in servers["klara"].features


def test_parse_nodes_g2():
    servers = parse_nodes(read_fixture("nodes-g2.txt"))

    # Parsing preserves CPU-only nodes; view policy filters them later.
    assert "g2-cpu-28" in servers
    assert servers["g2-cpu-28"].gpu.num == 0

    # GPU nodes preserve their GPU counts
    assert "badfellow" in servers
    assert servers["badfellow"].gpu.num == 4
    assert "gpu-high" in servers["badfellow"].features


def test_process_jobs_applies_gpu_usage_split():
    """Processing sacct output should attribute GPU usage to nodes."""

    sinfo = read_fixture("nodes-g2.txt")
    servers = parse_nodes(sinfo)
    gtop_output = read_fixture("gtop-output-g2.txt")

    process_jobs(parse_jobs(gtop_output), servers)

    badfellow = servers["badfellow"].usage["gpu"]
    assert badfellow.partitions == {
        "cuvl": 2.0,
        "default_partition": 2.0,
        "gpu": 1.0,
    }

    # CPU nodes should keep zero GPU usage even after processing
    assert sum(servers["g2-cpu-29"].usage["gpu"].partitions.values()) == 0


def test_parse_features_field_strips_multipliers():
    """Feature fields with SLURM multipliers should normalize to base names."""

    result = parse_features_field("gpu-high*2, gpu-low*3, avx512")

    assert "gpu-high" in result
    assert "gpu-low" in result
    assert "avx512" in result
    assert all("*" not in feature for feature in result)


def test_visible_servers_preserves_feature_and_free_capacity_order():
    servers = parse_nodes(
        "\n".join(
            [
                node_row("node-b", "gpu:test:2", "gpu:test:1(IDX:0)", features="gpu-low"),
                node_row("node-d", "gpu:test:4", "gpu:test:3(IDX:0-2)", features="gpu-high"),
                node_row("node-a", "gpu:test:4", "gpu:test:1(IDX:0)", features="gpu-high"),
            ]
        )
    )

    assert [
        server.name
        for server in visible_servers(servers, target_users=None)
    ] == ["node-a", "node-d", "node-b"]


def test_parse_nodes_reads_state_partitions_and_a_reason_with_spaces():
    servers = parse_nodes(
        "NodeName=demo-node Arch=x86_64 CPUAlloc=10 CPUEfctv=32 CPUTot=32 "
        "AvailableFeatures=gpu,gpu-high Gres=gpu:nvidia_a100:4(S:0-1) "
        "GresUsed=gpu:nvidia_a100:2(IDX:0-1) OS=Linux 6.8.0 #139-Ubuntu SMP "
        "RealMemory=65536 AllocMem=2048 State=MIXED+DRAIN "
        "Partitions=gpu,lab Reason=disk swap | fan [root@2026-10-09T09:21:17]"
    )

    node = servers["demo-node"]
    assert node.gpu.num == 4
    assert (node.cpu.idle, node.cpu.other, node.cpu.total) == (0, 22, 32)
    assert node.partitions == {"gpu", "lab"}
    assert not node.accepts_jobs
    assert node.reason == "disk swap | fan [root@2026-10-09T09:21:17]"


def test_process_jobs_counts_shard_jobs_as_gpu_occupancy():
    servers = parse_nodes(read_fixture("nodes-unicorn.txt"))
    output = sacct_row(
        "billing=8,cpu=8,gres/shard:nvidia_h100_nvl=12,gres/shard=12,mem=64G,node=1",
        job_id="12345",
        partition="default_partition",
        nodelist="dutta-compute-01",
    )

    process_jobs(parse_jobs(output), servers)

    dutta = servers["dutta-compute-01"]
    assert dutta.usage["shard"].partitions == {"default_partition": 12}
    assert dutta.usage["gpu"].partitions == {"default_partition": 1}


def test_parse_jobs_preserves_an_empty_time_limit_column():
    jobs = parse_jobs(sacct_row("gres/gpu=1", job_id="123", time_limit=""))

    assert jobs[0].job_id == "123"
    assert jobs[0].usage.gpu == 1
    assert jobs[0].time_limit == ""


def test_parse_jobs_reads_constraints():
    jobs = parse_jobs(
        sacct_row(
            "gres/gpu=1",
            state="PENDING",
            nodelist="None assigned",
            constraints="gpu-high&ssd",
        )
    )

    assert jobs[0].constraints == frozenset({"gpu-high", "ssd"})


def test_process_jobs_converts_gpu_usage_to_shards_on_sharded_nodes():
    servers = parse_nodes(read_fixture("nodes-unicorn.txt"))
    jobs = parse_jobs(
        sacct_row(
            "billing=8,cpu=8,gres/gpu:nvidia_h100_nvl=2,gres/gpu=2,mem=64G,node=1",
            job_id="12345",
            nodelist="dutta-compute-01",
        )
    )
    process_jobs(jobs, servers)

    dutta = servers["dutta-compute-01"]
    assert dutta.usage["gpu"].partitions["gpu"] == 2
    assert dutta.usage["shard"].partitions["gpu"] == 48
    allocation = dutta.allocations["12345"]
    assert allocation.job is jobs[0]
    assert allocation.usage.gpu == 2
    assert allocation.usage.shard == 48
    assert jobs[0].usage.gpu == 2
    assert not hasattr(jobs[0], "usage_str")

    users = summarize_users(servers)
    assert users["alice"].total_usage() == 2


def test_process_jobs_does_not_reallocate_filtered_nodes():
    servers = parse_nodes(
        "\n".join(
            [
                node_row("node-1", "gpu:a100:4", "gpu:a100:0(IDX:N/A)", features="gpu-high", cpus=8, memory=65536),
                node_row("node-2", "gpu:a100:4", "gpu:a100:0(IDX:N/A)", features="gpu", cpus=8, memory=65536),
            ]
        ),
    )
    jobs = parse_jobs(sacct_row("cpu=8,gres/gpu=2,mem=32G", nodelist="node-[1-2]"))

    process_jobs(jobs, {"node-1": servers["node-1"]})

    assert servers["node-1"].usage["gpu"].partitions == {"gpu": 1}
    assert servers["node-1"].usage["cpu"].partitions == {"gpu": 4}


def test_nodes_command_includes_hidden_nodes_and_gres_usage():
    assert {"-a", "-d"} <= set(NODES_COMMAND)


def test_sacct_command_requests_all_users():
    assert "-a" in SACCT_COMMAND


def test_expand_range_preserves_original_width():
    assert expand_range("1-3") == ["1", "2", "3"]
    assert expand_range("01-03") == ["01", "02", "03"]


def test_cached_week_history_expires_after_15_minutes_but_year_waits_for_refresh(
    tmp_path, monkeypatch
):
    import json
    import time

    from gtop import history

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def write(window: str, age_seconds: float) -> None:
        path = tmp_path / "gtop" / f"usage-{window}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"generated_at": time.time() - age_seconds, "gpu_hours": {}, "accounts": {}}
            )
        )

    write("week", 14 * 60)
    assert history.cached_history("week") is not None
    write("week", 16 * 60)
    assert history.cached_history("week") is None
    write("year", 30 * 86400)
    assert history.cached_history("year") is not None


@pytest.mark.parametrize(
    ("job_id", "tasks"),
    [
        ("123", 1),
        ("123_4", 1),
        ("135000_[269-999%25]", 731),
        ("108203_[12-14,16-35,37-41%12]", 28),
        ("77_[1-9:2]", 5),
    ],
)
def test_array_task_count_counts_pending_array_ranges(job_id: str, tasks: int):
    assert array_task_count(job_id) == tasks

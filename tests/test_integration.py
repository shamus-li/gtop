#!/usr/bin/env python3
"""Integration-oriented tests for gtop parsing helpers."""

from pathlib import Path

from gtop import (
    CpuInfo,
    MemoryInfo,
    ResourceUsageSplit,
    SINFO_COMMAND,
    ServerState,
    sort_server_names,
    SINFO_FIELD_WIDTHS,
)
from gtop.accounting import process_jobs
from gtop.constraints import matches_constraint
from gtop.resources import parse_gpu
from gtop.slurm import expand_range, parse_features_field, parse_sinfo

FIXTURE_DIR = Path(__file__).resolve().parent


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def test_matches_constraint_basic_boolean_logic():
    features = {"gpu", "gpu-high", "intel"}

    assert matches_constraint(features, "gpu")
    assert matches_constraint(features, "gpu&gpu-high")
    assert matches_constraint(features, "gpu|amd")
    assert matches_constraint(features, "[gpu-high|gpu-low]")
    assert matches_constraint(features, "gpu-high*2")
    assert not matches_constraint(features, "amd")
    assert not matches_constraint(features, "gpu-high&amd")


def test_matches_constraint_parentheses_respected():
    features = {"gpu", "gpu-legacy", "amd"}

    assert matches_constraint(features, "gpu|(cpu&amd)")
    assert not matches_constraint(features, "(cpu&amd)|gpu-high")


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

    # Mock the parsing (since we'd need the full parse_sinfo function)
    parts = sample_line.split()
    if len(parts) >= 6:
        node_name, gres, gres_used = parts[0], parts[1], parts[2]
        gpu_result = parse_gpu(gres, gres_used)

        assert node_name == "dutta-compute-01"
        assert gpu_result.num == 2
        assert gpu_result.shards == 48


def test_parse_sinfo_unicorn_nodes():
    """Ensure unicorn sinfo fixture is parsed correctly."""

    servers = parse_sinfo(read_fixture("sinfo-output-unicorn.txt"), gpu_only=False)

    assert "klara" in servers
    assert servers["klara"].gpu.num == 8
    assert "unicorn-compute-01" in servers
    assert servers["unicorn-compute-01"].gpu.type.count("|") == 1
    assert "gpu" in servers["unicorn-compute-01"].features
    assert "gpu-high" in servers["klara"].features


def test_parse_sinfo_g2_fixed_width():
    """g2 sinfo output uses fixed-width columns without whitespace delimiters."""

    servers = parse_sinfo(read_fixture("sinfo-output-g2.txt"), gpu_only=False)

    # CPU-only nodes should still be present when gpu_only is False
    assert "g2-cpu-28" in servers
    assert servers["g2-cpu-28"].gpu.num == 0

    # GPU nodes preserve their GPU counts
    assert "badfellow" in servers
    assert servers["badfellow"].gpu.num == 4
    assert "gpu-high" in servers["badfellow"].features


def test_parse_sinfo_g2_gpu_filters_cpu_nodes():
    """The gpu flag should remove nodes with null GRES entries."""

    servers = parse_sinfo(read_fixture("sinfo-output-g2.txt"), gpu_only=True)
    assert "g2-cpu-28" not in servers
    assert any(info.gpu.num > 0 for info in servers.values())


def test_constraint_filtering_with_fixture():
    """Constraint expressions should narrow the server list as expected."""

    servers = parse_sinfo(read_fixture("sinfo-output-g2.txt"), gpu_only=False)

    matching = [
        name for name, info in servers.items() if matches_constraint(info.features, "gpu-high")
    ]

    assert "sun-compute-01" in matching
    assert "ma-compute-01" in matching
    assert "g2-cpu-28" not in matching


def test_process_jobs_applies_gpu_usage_split():
    """Processing sacct output should attribute GPU usage to nodes."""

    sinfo = read_fixture("sinfo-output-g2.txt")
    servers = parse_sinfo(sinfo, gpu_only=False)
    gtop_output = read_fixture("gtop-output-g2.txt")

    process_jobs(gtop_output, servers)

    badfellow = servers["badfellow"].usage["gpu"]
    assert badfellow.priority == 2.0
    assert badfellow.gpu == 1.0
    assert badfellow.default == 2.0

    # CPU nodes should keep zero GPU usage even after processing
    assert servers["g2-cpu-29"].usage["gpu"].default == 0


def test_parse_features_field_strips_multipliers():
    """Feature fields with SLURM multipliers should normalize to base names."""

    result = parse_features_field("gpu-high*2, gpu-low*3, avx512")

    assert "gpu-high" in result
    assert "gpu-low" in result
    assert "avx512" in result
    assert all("*" not in feature for feature in result)


def make_server(
    name: str,
    *,
    features: set[str],
    gpu_num: int,
    gpu_priority: float = 0,
    gpu_default: float = 0,
    shard_count: int = 0,
    shard_priority: float = 0,
    shard_default: float = 0,
) -> ServerState:
    server = ServerState(
        name=name,
        features=features,
        gpu=parse_gpu(f"gpu:test:{gpu_num}") if gpu_num > 0 else parse_gpu("(null)"),
        cpu=CpuInfo(),
        mem=MemoryInfo(),
    )
    server.gpu.shards = shard_count
    server.usage["gpu"] = ResourceUsageSplit(priority=gpu_priority, default=gpu_default)
    server.usage["shard"] = ResourceUsageSplit(
        priority=shard_priority,
        default=shard_default,
    )
    return server


def test_sort_server_names_by_feature_representation():
    servers = {
        "node-b": make_server(
            "node-b",
            features={"gpu-low"},
            gpu_num=2,
            gpu_priority=1,
            shard_count=20,
            shard_priority=4,
        ),
        "node-c": make_server("node-c", features=set(), gpu_num=0),
        "node-a": make_server(
            "node-a",
            features={"gpu-high"},
            gpu_num=4,
            gpu_priority=1,
        ),
        "node-d": make_server(
            "node-d",
            features={"gpu-high"},
            gpu_num=4,
            gpu_priority=3,
        ),
    }

    assert sort_server_names(servers, "feature") == ["node-c", "node-a", "node-d", "node-b"]
    assert sort_server_names(servers, "name") == ["node-a", "node-b", "node-c", "node-d"]
    assert sort_server_names(servers, "free-gpu") == ["node-a", "node-b", "node-d", "node-c"]
    assert sort_server_names(servers, "used-gpu") == ["node-d", "node-a", "node-b", "node-c"]
    assert sort_server_names(servers, "free-shard", show_shards=True) == [
        "node-b",
        "node-a",
        "node-c",
        "node-d",
    ]


def test_parse_sinfo_fixed_width_line():
    fields = [
        "demo-node",
        "gpu,gpu-high",
        "gpu:nvidia_a100:4(S:0-1)",
        "gpu:nvidia_a100:2(IDX:0-1)",
        "10/22/0/32",
        "2048",
        "65536",
    ]

    segments = [
        value.ljust(width) for value, width in zip(fields, SINFO_FIELD_WIDTHS)
    ]
    fixed_width_line = "".join(segments)

    servers = parse_sinfo(fixed_width_line, gpu_only=False)
    assert "demo-node" in servers
    assert servers["demo-node"].gpu.num == 4
    assert servers["demo-node"].mem.total == 65536


def test_process_jobs_accepts_pipe_delimited_sacct_output():
    servers = parse_sinfo(read_fixture("sinfo-output-unicorn.txt"), gpu_only=False)
    pipe_output = (
        "alice|default_partition|dutta-compute-01|RUNNING|"
        "billing=8,cpu=8,gres/shard:nvidia_h100_nvl=12,gres/shard=12,mem=64G,node=1|12345|"
    )

    process_jobs(pipe_output, servers)

    dutta = servers["dutta-compute-01"]
    assert dutta.usage["shard"].default == 12
    assert dutta.usage["gpu"].default == 0


def test_process_jobs_converts_gpu_usage_to_shards_on_sharded_nodes():
    servers = parse_sinfo(read_fixture("sinfo-output-unicorn.txt"), gpu_only=False)
    pipe_output = (
        "alice|gpu|dutta-compute-01|RUNNING|"
        "billing=8,cpu=8,gres/gpu:nvidia_h100_nvl=2,gres/gpu=2,mem=64G,node=1|12345|"
    )

    process_jobs(pipe_output, servers)

    dutta = servers["dutta-compute-01"]
    assert dutta.usage["gpu"].gpu == 2
    assert dutta.usage["shard"].gpu == 48
    assert dutta.users["12345"].gpu == 2
    assert dutta.users["12345"].shard == 48


def test_sinfo_command_requests_per_node_output():
    assert " -N " in f" {SINFO_COMMAND} "


def test_sacct_command_requests_all_users():
    from gtop import SACCT_COMMAND

    assert " -a " in f" {SACCT_COMMAND} "


def test_expand_range_preserves_original_width():
    assert expand_range("1-3") == ["1", "2", "3"]
    assert expand_range("01-03") == ["01", "02", "03"]

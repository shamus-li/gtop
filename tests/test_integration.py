#!/usr/bin/env python3
"""
Integration test for gtop using real sinfo and gtop output data
"""

from gtop import parse_gpu


def test_klara_regular_gpu():
    """Test parsing klara node with regular A6000 GPUs"""

    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:8(IDX:0-7)"
    result = parse_gpu(gres, gres_used)

    assert result["type"] == "nvidia_rtx_a6000"  # Should not be sharded
    assert result["num"] == 8
    assert result["used"] == 8


def test_dutta_sharded_gpu():
    """Test parsing dutta-compute-01 node with true sharded H100 GPUs"""
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    result = parse_gpu(gres, gres_used)

    assert "Shard" in result["type"]  # Should be sharded due to shard: entry
    assert result["num"] == 2
    assert result["shards"] == 48
    assert result["used"] == 1


def test_snavely_range_pattern():
    """Test parsing snavely-compute-09 with range pattern (should NOT be sharded)"""
    gres = "gpu:nvidia_geforce_gtx_titan_x:4(S:0-1)"
    gres_used = "gpu:nvidia_geforce_gtx_titan_x:2(IDX:0,2)"
    result = parse_gpu(gres, gres_used)

    assert (
        "Shard" not in result["type"]
    )  # Should NOT be sharded (only shard: entries matter)
    assert result["num"] == 4
    assert result["used"] == 2


def test_unicorn_mixed_gpu_types():
    """Test parsing unicorn-compute-01 with mixed GPU types"""
    gres = "gpu:nvidia_geforce_rtx_2080_ti:2(S:0),gpu:nvidia_a40:2(S:1)"
    gres_used = "gpu:nvidia_geforce_rtx_2080_ti:2(IDX:0-1),gpu:nvidia_a40:2(IDX:2-3)"
    result = parse_gpu(gres, gres_used)

    assert result["num"] == 4
    assert result["used"] == 4
    assert "|" in result["type"]  # Should show mixed types


def test_cpu_only_node():
    """Test parsing CPU-only nodes"""
    gres = "(null)"
    result = parse_gpu(gres)

    assert result["type"] == "null"
    assert result["num"] == 0


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
        assert gpu_result["num"] == 2
        assert gpu_result["shards"] == 48

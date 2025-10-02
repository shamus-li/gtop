#!/usr/bin/env python3
"""
Test suite for gtop GPU parsing functionality
"""

from gtop import parse_gpu, parse_usage


def test_regular_gpu_parsing():
    """Test parsing of regular (non-sharded) GPU allocations"""

    # Test case 1: Regular A6000 GPU (klara)
    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:8(IDX:0-7)"
    result = parse_gpu(gres, gres_used)

    # This should NOT be marked as sharded - (S:0) likely means single shard mode
    assert result["num"] == 8
    assert result["used"] == 8
    assert "Shard" not in result["type"]


def test_mixed_gpu_types():
    """Test parsing of mixed GPU types"""
    gres = "gpu:nvidia_geforce_rtx_2080_ti:2(S:0),gpu:nvidia_a40:2(S:1)"
    gres_used = "gpu:nvidia_geforce_rtx_2080_ti:2(IDX:0-1),gpu:nvidia_a40:2(IDX:2-3)"
    result = parse_gpu(gres, gres_used)

    assert result["num"] == 4
    assert result["used"] == 4
    assert "|" in result["type"]


def test_sharded_gpu_parsing():
    """Test parsing of actually sharded GPU allocations"""

    # Test case 1: True sharded GPUs (dutta-compute-01)
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    result = parse_gpu(gres, gres_used)

    assert result["num"] == 2
    assert result["shards"] > 0
    assert result["used"] == 1
    assert "Shard" in result["type"]


def test_large_shard_count():
    """Test parsing of GPUs with large shard counts"""
    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "gpu:nvidia_a40:2(IDX:0-1),shard:nvidia_a40:0(0/200,0/200)"
    result = parse_gpu(gres, gres_used)

    assert result["num"] == 2
    assert result["shards"] == 400


def test_cpu_only_job():
    """Test parsing of CPU-only job TRES allocations"""

    tres = "billing=4,cpu=4,mem=1G,node=1"
    result = parse_usage(tres)

    assert result["cpu"] == 4
    assert result["mem"] == 1
    assert result["gpu"] == 0


def test_gpu_job_no_double_counting():
    """Test parsing of GPU job TRES without double counting"""
    tres = "billing=8,cpu=8,gres/gpu:nvidia_rtx_a6000=1,gres/gpu=1,mem=39.06G,node=1"
    result = parse_usage(tres)

    assert result["cpu"] == 8
    assert abs(result["mem"] - 39.06) < 0.1
    assert result["gpu"] == 1  # Should not double count


def test_multi_gpu_job():
    """Test parsing of multi-GPU job TRES"""
    tres = "billing=32,cpu=32,gres/gpu:nvidia_rtx_a6000=4,gres/gpu=4,mem=64G,node=1"
    result = parse_usage(tres)

    assert result["cpu"] == 32
    assert result["mem"] == 64
    assert result["gpu"] == 4  # Should not double count


def test_null_gres():
    """Test null/empty GRES handling"""

    result = parse_gpu("(null)")
    assert result["type"] == "null"
    assert result["num"] == 0


def test_none_gres():
    """Test 'none' GRES values are treated as empty."""

    result = parse_gpu("none")
    assert result["type"] == "null"
    assert result["num"] == 0


def test_complex_shard_range():
    """Test complex shard ranges should not be considered sharded"""
    gres = "gpu:nvidia_geforce_gtx_titan_x:2(S:0,3)"
    result = parse_gpu(gres)
    assert result["num"] == 2
    assert result["shards"] == 0  # Comma pattern should not be sharded


def test_range_pattern_not_sharded():
    """Test range patterns should not be considered sharded"""
    gres = "gpu:nvidia_geforce_gtx_titan_x:4(S:0-1)"
    result = parse_gpu(gres)
    assert result["num"] == 4
    assert "Shard" not in result["type"]
    assert result["shards"] == 0

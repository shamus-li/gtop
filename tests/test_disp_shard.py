import io

from rich.console import Console

from gtop.models import CpuInfo, MemoryInfo, ServerState
from gtop.render_cluster import render_table
from gtop.resources import parse_gpu


def make_server(
    gpu_gres: str,
    gpu_used: str,
) -> ServerState:
    return ServerState(
        name="demo-node",
        features=set(),
        gpu=parse_gpu(gpu_gres, gpu_used),
        cpu=CpuInfo(),
        mem=MemoryInfo(),
        state="mixed",
        reason="",
    )


def render(server: ServerState) -> str:
    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False)
    console.print(render_table([server], width=120))
    return stream.getvalue()


def test_default_shows_gpu_count_not_shards():
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    gres_used = "gpu:nvidia_h100_nvl:1(IDX:0),shard:nvidia_h100_nvl:0(0/24,0/24)"
    server = make_server(gres, gres_used)

    assert server.gpu.num == 2
    assert server.gpu.shards == 48
    assert server.gpu.occupied() == 1
    assert "1/2 GPUs free" in render(server)


def test_disp_shard_with_usage():
    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "gpu:nvidia_a40:1(IDX:0),shard:nvidia_a40:100(0/200)"
    server = make_server(gres, gres_used)

    assert server.gpu.occupied() == 2
    assert "0/2 GPUs free" in render(server)


def test_non_sharded_gpu_unchanged():
    gres = "gpu:nvidia_rtx_a6000:8(S:0)"
    gres_used = "gpu:nvidia_rtx_a6000:2(IDX:0-1)"
    server = make_server(gres, gres_used)

    assert server.gpu.num == 8
    assert server.gpu.shards == 0
    assert server.gpu.occupied() == 2
    assert "6/8 GPUs free" in render(server)


def test_regular_gpu_type_display():
    gres = "gpu:nvidia_h100_nvl:2(S:12-23),shard:nvidia_h100_nvl:48(S:12-23)"
    result = parse_gpu(gres)

    assert result.type.startswith("Shard(")
    assert "nvidia_h100_nvl" in result.type


def test_integration_default_vs_disp_shard():
    gres = "gpu:nvidia_a40:2(S:1),shard:nvidia_a40:400(S:1)"
    gres_used = "none"
    server = make_server(gres, gres_used)

    assert server.gpu.num == 2
    assert server.gpu.occupied() == 0

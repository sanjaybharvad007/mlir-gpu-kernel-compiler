"""CUDA backend: codegen is checked on every platform by compiling the generated
kernels as host C++ against csrc/cuda_host_stub.h; execution tests run on a GPU."""
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import tensorjit as tj
import tensorjit.compiler as C
from tensorjit.backends.cuda_backend import generate
from tensorjit.ir import DType, TensorType

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "csrc" / "cuda_host_stub.h"
CXX = shutil.which("g++") or shutil.which("clang++")

CASES = {
    "mlp": (lambda x, g, w1, b1, w2, b2: x + tj.linear(tj.gelu(tj.linear(tj.layer_norm(x, g), w1, b1)), w2, b2),
            [(64, 128), (128,), (256, 128), (256,), (128, 256), (128,)]),
    "rows": (lambda x, w: (tj.softmax(x), tj.rms_norm(x, w), tj.sum(x * x), tj.max(x, keepdim=True)), [(8, 100), (100,)]),
    "bmm": (lambda a, b: tj.relu(tj.matmul(a, b)).sum(axis=2), [(2, 3, 8, 16), (3, 16, 8)]),
    "pointwise": (lambda a, b: tj.where(a, tj.maximum(a, b), tj.minimum(a, b)) + tj.exp(a) ** 2.0, [(4, 5, 6), (6,)]),
    "movement": (lambda t: tj.exp(tj.permute(t, 2, 0, 1)).reshape(5, 12) + 1.0, [(3, 4, 5)]),
    "wmma": (lambda x, w, b: tj.gelu(tj.linear(x, w, b)), [(64, 128), (256, 128), (256,)]),
}


def _source(name, dtype):
    fn, shapes = CASES[name]
    g = tj.trace(fn, [TensorType(s, dtype) for s in shapes])
    return generate(C.lower(g, "cuda"))


@pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("dtype", [DType.f32, DType.f16, DType.bf16])
@pytest.mark.parametrize("name", list(CASES))
def test_generated_cuda_typechecks_as_host_cpp(name, dtype, tmp_path):
    src, launches = _source(name, dtype)
    assert len(launches) >= 1 and all(l.block[0] <= 1024 for l in launches)
    path = tmp_path / f"{name}_{dtype.value}.cu"
    path.write_text(src)
    r = subprocess.run([CXX, "-fsyntax-only", "-x", "c++", "-std=c++17", "-DTJ_HOST_STUB", "-include", str(STUB), str(path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:2000]


def test_wmma_selected_only_for_aligned_half_precision():
    src16, _ = _source("wmma", DType.f16)
    src32, _ = _source("wmma", DType.f32)
    assert "wmma::mma_sync" in src16 and "wmma::mma_sync" not in src32
    # unaligned shapes fall back to the bounds-checked SIMT kernel
    g = tj.trace(lambda x, w: tj.matmul(x, w), [TensorType((50, 128), DType.f16), TensorType((128, 70), DType.f16)])
    src, _ = generate(C.lower(g, "cuda"))
    assert "wmma::mma_sync" not in src and "acc[4][4]" in src


def test_launch_geometry():
    _, launches = _source("mlp", DType.f32)
    kinds = [k.kind for k in C.lower(tj.trace(CASES["mlp"][0], [TensorType(s) for s in CASES["mlp"][1]]), "cuda").kernels]
    assert kinds == ["row", "matmul", "matmul"]
    assert launches[0].grid == (64, 1, 1) and launches[0].block == (128, 1, 1)      # one block per row, 128 >= N
    assert launches[1].grid == (4, 1, 1) and launches[2].grid == (2, 1, 1)         # 256/64 and 128/64 column tiles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU execution")
@pytest.mark.parametrize("use_graph", [False, True])
def test_cuda_backend_matches_eager_on_gpu(use_graph, monkeypatch):
    monkeypatch.setenv("TENSORJIT_CUDA_GRAPH", "1" if use_graph else "0")
    torch.manual_seed(0)
    dev = "cuda"
    x, g = torch.randn(64, 128, device=dev), torch.ones(128, device=dev)
    w1, b1 = torch.randn(256, 128, device=dev), torch.randn(256, device=dev)
    w2, b2 = torch.randn(128, 256, device=dev), torch.randn(128, device=dev)
    f = tj.jit(CASES["mlp"][0], backend="cuda")
    out = f(x, g, w1, b1, w2, b2)
    ref = x + F.linear(F.gelu(F.linear(F.layer_norm(x, (128,), g), w1, b1)), w2, b2)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    out2 = f(x + 1, g, w1, b1, w2, b2)                         # second call: graph replay with new inputs
    ref2 = (x + 1) + F.linear(F.gelu(F.linear(F.layer_norm(x + 1, (128,), g), w1, b1)), w2, b2)
    torch.testing.assert_close(out2, ref2, atol=2e-3, rtol=2e-3)
    cp = f.compiled(x, g, w1, b1, w2, b2)
    assert cp.stats["graph_replays"] == (2 if use_graph else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU execution")
def test_cuda_wmma_matches_eager_on_gpu():
    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    w, b = torch.randn(256, 128, device="cuda", dtype=torch.float16), torch.randn(256, device="cuda", dtype=torch.float16)
    f = tj.jit(CASES["wmma"][0], backend="cuda")
    out = f(x, w, b)
    ref = F.gelu(F.linear(x.float(), w.float(), b.float()))
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)

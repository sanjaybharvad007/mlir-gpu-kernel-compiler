import os
import shutil

import pytest
import torch
import torch.nn.functional as F

import tensorjit as tj
from tensorjit.backends.mlir_backend import MLIR_PIPELINE, emit_mlir, run_pipeline, verify
from tensorjit.ir import TensorType
from tensorjit.passes import optimize


def _mlir(fn, *shapes):
    return emit_mlir(optimize(tj.trace(fn, [TensorType(s) for s in shapes])))


def test_mlir_emission_structure():
    s = _mlir(lambda x, w, b: tj.softmax(tj.gelu(tj.matmul(x, w) + b)), (4, 8), (8, 6), (6,))
    assert "func.func @main" in s and "llvm.emit_c_interface" in s
    assert "linalg.matmul" in s and "linalg.generic" in s and "math.erf" in s
    assert '"reduction"' in s and "arith.maximumf" in s
    assert "affine_map<(d0, d1) -> (d1)>" in s  # bias broadcast map


def test_mlir_batch_matmul_named_vs_generic():
    s1 = _mlir(lambda a, b: tj.matmul(a, b), (3, 4, 5), (3, 5, 6))
    assert "linalg.batch_matmul" in s1
    s2 = _mlir(lambda a, b: tj.matmul(a, b), (2, 3, 4, 5), (3, 5, 6))
    assert "linalg.batch_matmul" not in s2 and '"reduction"' in s2


def test_mlir_movement_ops():
    s = _mlir(lambda x: tj.exp(tj.permute(x, 1, 0)).reshape(-1), (3, 4))
    assert "linalg.transpose" in s and "permutation = [1, 0]" in s and "tensor.reshape" in s


@pytest.mark.skipif(shutil.which("mlir-opt") is None, reason="needs an MLIR toolchain on PATH")
def test_mlir_verifies_and_lowers():
    s = _mlir(lambda x, w, b: tj.layer_norm(tj.gelu(tj.matmul(x, w) + b)), (4, 8), (8, 6), (6,))
    verify(s)
    llvm = run_pipeline(s, MLIR_PIPELINE)
    assert "llvm.func" in llvm


@pytest.mark.skipif(shutil.which("mlir-opt") is None or shutil.which("clang") is None, reason="MLIR toolchain")
def test_mlir_backend_executes():
    x, w, b = torch.randn(4, 8), torch.randn(8, 6), torch.randn(6)
    f = tj.jit(lambda x, w, b: tj.softmax(tj.matmul(x, w) + b), backend="mlir")
    torch.testing.assert_close(f(x, w, b), torch.softmax(x @ w + b, -1), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- torch.compile


class Block(torch.nn.Module):
    def __init__(self, d=32, h=64):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, h)
        self.fc2 = torch.nn.Linear(h, d)
        self.drop = torch.nn.Dropout(0.0)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(self.ln(x))))
        y = x + self.fc2(h)
        return torch.softmax(y, dim=-1) * 2 + y.mean(dim=-1, keepdim=True)


def test_torch_compile_backend_matches_eager(monkeypatch):
    monkeypatch.setenv("TENSORJIT_STRICT", "1")
    monkeypatch.setenv("TENSORJIT_NO_AUTOTUNE", "1")
    torch.manual_seed(0)
    torch._dynamo.reset()
    m = Block().eval()
    x = torch.randn(16, 32)
    c = torch.compile(m, backend=tj.torch_compile_backend)
    with torch.no_grad():
        torch.testing.assert_close(c(x), m(x), atol=1e-4, rtol=1e-4)


def test_torch_compile_backend_falls_back_on_unsupported(monkeypatch):
    monkeypatch.delenv("TENSORJIT_STRICT", raising=False)
    torch._dynamo.reset()

    def f(x):
        return torch.cumsum(x, 0) + 1  # not supported by tensorjit

    x = torch.randn(8)
    c = torch.compile(f, backend=tj.torch_compile_backend)
    torch.testing.assert_close(c(x), f(x))


def test_torch_compile_functional_graph(monkeypatch):
    monkeypatch.setenv("TENSORJIT_STRICT", "1")
    monkeypatch.setenv("TENSORJIT_NO_AUTOTUNE", "1")
    torch._dynamo.reset()

    def f(x, w):
        return F.silu(x @ w.t()).sum(-1)

    x, w = torch.randn(5, 8), torch.randn(3, 8)
    c = torch.compile(f, backend=tj.torch_compile_backend)
    torch.testing.assert_close(c(x, w), f(x, w), atol=1e-4, rtol=1e-4)
    assert os.environ.get("TENSORJIT_STRICT") == "1"

"""Memory planner and the native (C++) runtime."""
import os

import pytest
import torch
import torch.nn.functional as F

import tensorjit as tj
import tensorjit.compiler as C
from tensorjit.ir import TensorType
from tensorjit.passes.memory_planner import ALIGN, plan_memory


def _prog(fn, *shapes, backend="c"):
    return C.lower(tj.trace(fn, [TensorType(s) for s in shapes]), backend)


def test_planner_shares_storage_between_disjoint_lifetimes():
    def chain(x):
        y = x
        for _ in range(6):
            y = tj.matmul(tj.softmax(tj.gelu(y * 1.1 + 0.1)), x)
        return y

    plan = plan_memory(_prog(chain, (256, 256)))
    assert plan.naive_bytes > 2 * plan.arena_bytes          # real reuse
    assert plan.savings > 0.5
    assert all(b.offset % ALIGN == 0 for b in plan.buffers.values())


def test_planner_never_overlaps_live_buffers():
    def mlp(x, g, w1, b1, w2, b2):
        h = tj.layer_norm(x, g)
        return x + tj.linear(tj.gelu(tj.linear(h, w1, b1)), w2, b2)

    plan = plan_memory(_prog(mlp, (512, 256), (256,), (1024, 256), (1024,), (256, 1024), (256,)))
    bufs = list(plan.buffers.values())
    for i, a in enumerate(bufs):
        for b in bufs[i + 1:]:
            alive_together = not (a.last < b.first or b.last < a.first)
            overlap = not (a.offset + a.size <= b.offset or b.offset + b.size <= a.offset)
            assert not (alive_together and overlap)
    # the LayerNorm output and the GELU output are both live at the second matmul
    assert plan.arena_bytes == sum(b.size for b in bufs)


def test_planner_excludes_graph_outputs():
    plan = plan_memory(_prog(lambda x: (tj.exp(x), tj.exp(x) + 1.0), (64,)))
    names = {b.node.name for b in plan.buffers.values()}
    # exp(x) is returned -> must not be in the arena; only intermediates are
    assert all("add" not in n for n in names)


@pytest.fixture
def native_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORJIT_NATIVE_RUNTIME", "1")
    monkeypatch.setenv("TENSORJIT_NO_AUTOTUNE", "1")
    monkeypatch.setenv("TENSORJIT_CACHE_DIR", str(tmp_path / "cache"))


def test_native_runtime_matches_eager(native_env):
    from tensorjit.runtime.native import NativeProgram

    torch.manual_seed(0)
    x, g = torch.randn(64, 128), torch.ones(128)
    w1, b1, w2, b2 = torch.randn(256, 128), torch.randn(256), torch.randn(128, 256), torch.randn(128)

    def mlp(x, g, w1, b1, w2, b2):
        return x + tj.linear(tj.gelu(tj.linear(tj.layer_norm(x, g), w1, b1)), w2, b2)

    f = tj.jit(mlp, backend="c")
    out = f(x, g, w1, b1, w2, b2)
    cp = f.compiled(x, g, w1, b1, w2, b2)
    assert isinstance(cp, NativeProgram) and cp.num_kernels == 3
    ref = x + F.linear(F.gelu(F.linear(F.layer_norm(x, (128,), g), w1, b1)), w2, b2)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    assert cp.native.calls == 1 and cp.native.arena_bytes == cp.plan.arena_bytes


def test_native_runtime_reshape_and_const_outputs(native_env):
    x = torch.randn(3, 4, 5)
    f = tj.jit(lambda t: (tj.exp(t).reshape(5, 12), tj.sum(t * 0.0 + 2.0)), backend="c")
    a, b = f(x)
    torch.testing.assert_close(a, torch.exp(x).reshape(5, 12))
    torch.testing.assert_close(b, torch.full((3, 4), 10.0))


def test_native_runtime_rejects_bad_inputs(native_env):
    f = tj.jit(lambda t: tj.exp(t), backend="c")
    f(torch.randn(8))
    with pytest.raises(TypeError):
        f.compiled(torch.randn(8))(torch.randn(8), torch.randn(8))


def test_native_runtime_lower_overhead_than_python(native_env):
    import time

    def chain(x, w):
        y = x
        for _ in range(8):
            y = tj.softmax(tj.matmul(y, w) + 0.5)
        return y

    x, w = torch.randn(16, 32), torch.randn(32, 32)
    fast = tj.jit(chain, backend="c")
    fast(x, w)                                   # compile now, while the native runtime is enabled
    os.environ["TENSORJIT_NATIVE_RUNTIME"] = "0"
    slow = tj.jit(chain, backend="c")
    slow(x, w)                                   # ... and this one with the Python executor
    os.environ["TENSORJIT_NATIVE_RUNTIME"] = "1"
    from tensorjit.runtime.native import NativeProgram
    assert isinstance(fast.compiled(x, w), NativeProgram) and not isinstance(slow.compiled(x, w), NativeProgram)
    for f in (fast, slow):
        for _ in range(50):
            f(x, w)
    torch.testing.assert_close(fast(x, w), slow(x, w))

    def t(f):
        s = time.perf_counter()
        for _ in range(300):
            f(x, w)
        return time.perf_counter() - s

    assert t(fast) < t(slow)

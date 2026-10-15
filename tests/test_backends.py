"""End-to-end numerical tests for every backend against PyTorch eager."""
import os

import pytest
import torch
import torch.nn.functional as F

import tensorjit as tj

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"

BACKENDS = ["c"]
if CUDA or INTERP:
    BACKENDS.append("triton")
try:
    from tensorjit.backends.mlir_backend import toolchain_available

    if toolchain_available():
        BACKENDS.append("mlir")
except Exception:  # pragma: no cover
    pass

DEV = "cuda" if CUDA else "cpu"


def _close(a, b, tol=1e-4):
    torch.testing.assert_close(a.cpu().float(), b.cpu().float(), atol=tol, rtol=tol)


@pytest.fixture(autouse=True)
def _no_autotune(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORJIT_NO_AUTOTUNE", "1")
    monkeypatch.setenv("TENSORJIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(17,), (8, 33), (2, 3, 5)])
def test_elementwise_chain(backend, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV)
    y = torch.randn(shape[-1], device=DEV)
    f = tj.jit(lambda a, b: tj.exp(tj.relu(a * 2.0 + b)) / (1.0 + tj.sigmoid(a)) - tj.tanh(b), backend=backend)
    ref = torch.exp(torch.relu(x * 2 + y)) / (1 + torch.sigmoid(x)) - torch.tanh(y)
    _close(f(x, y), ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", ["gelu", "silu", "log", "sqrt", "rsqrt", "abs", "sigmoid"])
def test_unary_ops(backend, op):
    torch.manual_seed(0)
    x = torch.rand(64, device=DEV) + 0.5
    f = tj.jit(getattr(tj, op), backend=backend)
    ref = {"gelu": F.gelu, "silu": F.silu, "log": torch.log, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt,
           "abs": torch.abs, "sigmoid": torch.sigmoid}[op](x)
    _close(f(x), ref, tol=2e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_where_maximum_pow(backend):
    torch.manual_seed(0)
    x, y = torch.randn(40, device=DEV), torch.randn(40, device=DEV)
    f = tj.jit(lambda a, b: tj.where(a, tj.maximum(a, b), tj.minimum(a, b)) + tj.abs(a) ** 2.0, backend=backend)
    ref = torch.where(x != 0, torch.maximum(x, y), torch.minimum(x, y)) + x.abs() ** 2
    _close(f(x, y), ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (7, 13, 5), (64, 64, 64), (33, 128, 65)])
def test_matmul_with_epilogue(backend, m, k, n):
    torch.manual_seed(0)
    a, b, bias = torch.randn(m, k, device=DEV), torch.randn(k, n, device=DEV), torch.randn(n, device=DEV)
    f = tj.jit(lambda x, y, z: tj.gelu(tj.matmul(x, y) + z), backend=backend)
    _close(f(a, b, bias), F.gelu(a @ b + bias), tol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_matmul_broadcast(backend):
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 8, 16, device=DEV), torch.randn(3, 16, 8, device=DEV)
    f = tj.jit(lambda x, y: tj.relu(tj.matmul(x, y)), backend=backend)
    _close(f(a, b), torch.relu(a @ b), tol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_transposed_weight(backend):
    torch.manual_seed(0)
    x, w, b = torch.randn(10, 32, device=DEV), torch.randn(48, 32, device=DEV), torch.randn(48, device=DEV)
    f = tj.jit(tj.linear, backend=backend)
    _close(f(x, w, b), F.linear(x, w, b), tol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 17, 100, 1000])
def test_reductions(backend, n):
    torch.manual_seed(0)
    x = torch.randn(6, n, device=DEV)
    f = tj.jit(lambda t: (tj.sum(t), tj.max(t), tj.mean(t, keepdim=True)), backend=backend)
    s, m, mu = f(x)
    _close(s, x.sum(-1), tol=1e-3)
    _close(m, x.amax(-1))
    _close(mu, x.mean(-1, keepdim=True))


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_non_last_axis_via_transpose(backend):
    torch.manual_seed(0)
    x = torch.randn(4, 5, 6, device=DEV)
    f = tj.jit(lambda t: tj.sum(t, axis=1), backend=backend)
    _close(f(x), x.sum(1), tol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_softmax_layernorm_rmsnorm(backend):
    torch.manual_seed(0)
    x = torch.randn(9, 70, device=DEV)
    g, b = torch.randn(70, device=DEV), torch.randn(70, device=DEV)
    f = tj.jit(lambda t, w, c: (tj.softmax(t), tj.layer_norm(t, w, c), tj.rms_norm(t, w)), backend=backend)
    sm, ln, rms = f(x, g, b)
    _close(sm, torch.softmax(x, -1))
    _close(ln, F.layer_norm(x, (70,), g, b), tol=1e-3)
    _close(rms, x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g, tol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transpose_and_reshape(backend):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, device=DEV)
    f = tj.jit(lambda t: tj.exp(tj.permute(t, 2, 0, 1)).reshape(5, 12) + 1.0, backend=backend)
    _close(f(x), torch.exp(x.permute(2, 0, 1)).reshape(5, 12) + 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transformer_block(backend):
    """A full pre-norm transformer MLP block + attention-score softmax."""
    torch.manual_seed(0)
    T_, D, H = 12, 32, 64
    x = torch.randn(T_, D, device=DEV)
    ln_g = torch.randn(D, device=DEV)
    w1, b1 = torch.randn(H, D, device=DEV), torch.randn(H, device=DEV)
    w2, b2 = torch.randn(D, H, device=DEV), torch.randn(D, device=DEV)
    wq, wk = torch.randn(D, D, device=DEV), torch.randn(D, D, device=DEV)

    def block(x, ln_g, w1, b1, w2, b2, wq, wk):
        h = tj.layer_norm(x, ln_g)
        q, k = tj.linear(h, wq), tj.linear(h, wk)
        scores = tj.softmax(tj.matmul(q, k.T) * (1.0 / 8.0))
        mlp = tj.linear(tj.gelu(tj.linear(h, w1, b1)), w2, b2)
        return x + mlp, scores

    def ref(x, ln_g, w1, b1, w2, b2, wq, wk):
        h = F.layer_norm(x, (D,), ln_g)
        q, k = F.linear(h, wq), F.linear(h, wk)
        scores = torch.softmax(q @ k.T / 8.0, -1)
        return x + F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2), scores

    f = tj.jit(block, backend=backend)
    out, sc = f(x, ln_g, w1, b1, w2, b2, wq, wk)
    r_out, r_sc = ref(x, ln_g, w1, b1, w2, b2, wq, wk)
    _close(out, r_out, tol=2e-3)
    _close(sc, r_sc, tol=1e-3)
    assert f.compiled(x, ln_g, w1, b1, w2, b2, wq, wk).num_kernels < 20


def test_specialisation_cache_hits_and_recompiles():
    f = tj.jit(lambda a: tj.exp(a) + 1.0, backend="c")
    x = torch.randn(8)
    f(x)
    f(x)
    assert f.stats.compiles == 1 and f.stats.hits == 1
    f(torch.randn(9))
    assert f.stats.compiles == 2


@pytest.mark.skipif(not (CUDA or INTERP), reason="Triton path")
def test_generated_triton_source_is_specialised():
    x = torch.randn(5, 7, device=DEV)
    f = tj.jit(lambda a: tj.exp(a) * 3.0, backend="triton")
    src = f.compiled(x).source
    assert "@triton.jit" in src and "35" in src  # numel baked in


def test_generated_c_source_is_specialised():
    x = torch.randn(5, 7)
    f = tj.jit(lambda a: tj.exp(a) * 3.0, backend="c")
    src = f.compiled(x).source
    assert "#pragma omp simd" in src and "< 7;" in src and "expf(" in src


def test_output_buffers_reused_only_when_dropped():
    f = tj.jit(lambda a: tj.exp(a), backend="c")
    x = torch.randn(256, 512)
    a = f(x)
    view = a[:3]
    b = f(x + 1)
    assert a.data_ptr() != b.data_ptr()          # `a` (and a view of it) are alive
    del view
    c = f(x + 2)
    assert c.data_ptr() not in (a.data_ptr(), b.data_ptr())
    pa = a.data_ptr()
    del a
    d = f(x + 3)                                  # `a` was dropped: its buffer is recycled
    assert d.data_ptr() == pa
    torch.testing.assert_close(b, torch.exp(x + 1))
    torch.testing.assert_close(c, torch.exp(x + 2))
    torch.testing.assert_close(d, torch.exp(x + 3))

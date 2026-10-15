"""Python tracing frontend.

``tensorjit.jit`` runs the decorated function once per input signature with
``Tensor`` proxies that record every op into a ``Graph``; the graph is then
optimised, fused, and compiled for the device of the inputs.
"""
from __future__ import annotations

import threading
from typing import Callable, Sequence

import torch

from ..ir import DType, Graph, Node, TensorType
from ..runtime import FROM_TORCH

_tls = threading.local()


def _graph() -> Graph:
    g = getattr(_tls, "graph", None)
    if g is None:
        raise RuntimeError("tensorjit ops must be used inside a traced (@tensorjit.jit) function")
    return g


class Tensor:
    """Symbolic tensor; every operation appends a node to the active graph."""

    __array_priority__ = 1000  # make numpy/torch defer to our reflected ops

    def __init__(self, node: Node):
        self.node = node

    # -- metadata ------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.node.shape

    @property
    def dtype(self) -> DType:
        return self.node.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"tj.Tensor({self.node.name}: {self.node.type})"

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o): return _bin("add", self, o)
    def __radd__(self, o): return _bin("add", o, self)
    def __sub__(self, o): return _bin("sub", self, o)
    def __rsub__(self, o): return _bin("sub", o, self)
    def __mul__(self, o): return _bin("mul", self, o)
    def __rmul__(self, o): return _bin("mul", o, self)
    def __truediv__(self, o): return _bin("div", self, o)
    def __rtruediv__(self, o): return _bin("div", o, self)
    def __pow__(self, o): return _bin("pow", self, o)
    def __neg__(self): return _un("neg", self)
    def __matmul__(self, o): return matmul(self, o)

    # -- methods -------------------------------------------------------------
    def exp(self): return _un("exp", self)
    def log(self): return _un("log", self)
    def sqrt(self): return _un("sqrt", self)
    def rsqrt(self): return _un("rsqrt", self)
    def tanh(self): return _un("tanh", self)
    def abs(self): return _un("abs", self)
    def relu(self): return _un("relu", self)
    def sigmoid(self): return _un("sigmoid", self)
    def sum(self, axis=-1, keepdim=False): return reduce("sum", self, axis, keepdim)
    def max(self, axis=-1, keepdim=False): return reduce("max", self, axis, keepdim)
    def mean(self, axis=-1, keepdim=False): return reduce("mean", self, axis, keepdim)
    def reshape(self, *shape): return reshape(self, *shape)
    def view(self, *shape): return reshape(self, *shape)
    def transpose(self, a: int, b: int): return transpose(self, a, b)
    def permute(self, *perm): return permute(self, *perm)

    @property
    def T(self):
        assert self.ndim == 2
        return permute(self, 1, 0)


def _wrap(x, like: Tensor | None = None) -> Node:
    if isinstance(x, Tensor):
        return x.node
    if isinstance(x, (int, float, bool)):
        return _graph().const(float(x), like.dtype if like is not None else DType.f32)
    raise TypeError(f"unsupported operand {type(x)}")


def _bin(op: str, a, b) -> Tensor:
    like = a if isinstance(a, Tensor) else b
    return Tensor(_graph().create(op, [_wrap(a, like), _wrap(b, like)]))


def _un(op: str, a: Tensor) -> Tensor:
    return Tensor(_graph().create(op, [a.node]))


# -- functional API -------------------------------------------------------------
def exp(x): return _un("exp", x)
def log(x): return _un("log", x)
def sqrt(x): return _un("sqrt", x)
def rsqrt(x): return _un("rsqrt", x)
def tanh(x): return _un("tanh", x)
def abs(x): return _un("abs", x)  # noqa: A001
def relu(x): return _un("relu", x)
def sigmoid(x): return _un("sigmoid", x)
def gelu(x): return _un("gelu", x)
def silu(x): return _un("silu", x)
def maximum(a, b): return _bin("maximum", a, b)
def minimum(a, b): return _bin("minimum", a, b)


def where(c, a, b):
    like = next(t for t in (c, a, b) if isinstance(t, Tensor))
    return Tensor(_graph().create("where", [_wrap(c, like), _wrap(a, like), _wrap(b, like)]))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(_graph().create("matmul", [a.node, b.node]))


def reduce(op: str, x: Tensor, axis: int = -1, keepdim: bool = False) -> Tensor:
    if axis < 0:
        axis += x.ndim
    if axis != x.ndim - 1:
        # reduce over a non-last axis by moving it last (transpose kernel + reduce)
        perm = [d for d in range(x.ndim) if d != axis] + [axis]
        x = permute(x, *perm)
    return Tensor(_graph().create(op, [x.node], axis=-1, keepdim=keepdim))


def sum(x, axis=-1, keepdim=False): return reduce("sum", x, axis, keepdim)  # noqa: A001
def max(x, axis=-1, keepdim=False): return reduce("max", x, axis, keepdim)  # noqa: A001
def mean(x, axis=-1, keepdim=False): return reduce("mean", x, axis, keepdim)


def reshape(x: Tensor, *shape) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(_graph().create("reshape", [x.node], shape=tuple(int(s) for s in shape)))


def permute(x: Tensor, *perm) -> Tensor:
    if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
        perm = tuple(perm[0])
    perm = tuple(p % x.ndim for p in perm)
    if perm == tuple(range(x.ndim)):
        return x
    return Tensor(_graph().create("transpose", [x.node], perm=perm))


def transpose(x: Tensor, a: int, b: int) -> Tensor:
    perm = list(range(x.ndim))
    perm[a % x.ndim], perm[b % x.ndim] = perm[b % x.ndim], perm[a % x.ndim]
    return permute(x, *perm)


def broadcast_to(x: Tensor, shape) -> Tensor:
    return Tensor(_graph().create("broadcast_to", [x.node], shape=tuple(shape)))


# -- composite ops (expressed in primitives; fusion takes care of the rest) ----
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    assert axis in (-1, x.ndim - 1), "softmax over last axis"
    m = reduce("max", x, -1, keepdim=True)
    e = exp(x - m)
    return e / reduce("sum", e, -1, keepdim=True)


def layer_norm(x: Tensor, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    mu = reduce("mean", x, -1, keepdim=True)
    d = x - mu
    var = reduce("mean", d * d, -1, keepdim=True)
    y = d * rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    ms = reduce("mean", x * x, -1, keepdim=True)
    y = x * rsqrt(ms + eps)
    return y * weight if weight is not None else y


def linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
    """x @ w.T + b, PyTorch nn.Linear convention (w: [out, in])."""
    y = matmul(x, w.T)
    return y + b if b is not None else y


# -- tracing ----------------------------------------------------------------------
def trace(fn: Callable, arg_types: Sequence[TensorType], name: str = "main") -> Graph:
    g = Graph(name)
    prev = getattr(_tls, "graph", None)
    _tls.graph = g
    try:
        args = [Tensor(g.add_input(t)) for t in arg_types]
        out = fn(*args)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if not isinstance(o, Tensor):
                raise TypeError("jit functions must return tensors (or a tuple of them)")
        g.set_outputs([o.node for o in outs])
    finally:
        _tls.graph = prev
    g.toposort()
    return g


def tensor_type_of(t: torch.Tensor) -> TensorType:
    return TensorType(tuple(t.shape), FROM_TORCH[t.dtype])

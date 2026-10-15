"""``torch.compile`` backend.

    compiled = torch.compile(model, backend=tensorjit.torch_compile_backend)

TorchDynamo hands us an FX ``GraphModule`` of torch-level ops plus example
inputs.  We translate the supported subset into tensorjit IR, compile it, and
return a callable; if any node is unsupported the graph falls back to eager
(``torch.compile`` semantics: never wrong, just not accelerated).  Set
``TENSORJIT_STRICT=1`` to raise instead of falling back.
"""
from __future__ import annotations

import logging
import operator
import os
from typing import Callable

import torch
import torch.fx as fx
import torch.nn.functional as F

from ..compiler import compile_graph, pick_backend
from ..frontend import tracer as T
from ..ir import Graph
from .tracer import Tensor, tensor_type_of

log = logging.getLogger("tensorjit")


class Unsupported(Exception):
    pass


def _unsupported(what: str):
    raise Unsupported(what)


def _as_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (int, float, bool)):
        return x
    raise Unsupported(f"non-tensor argument {type(x)}")


def _reduce_dim(x: Tensor, dim, keepdim: bool, op: str):
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            raise Unsupported("multi-dim reductions")
        dim = dim[0]
    if dim is None:
        x = T.reshape(x, -1)
        dim = -1
    return T.reduce(op, x, dim, keepdim)


def _softmax(x, dim=-1, *_, **__):
    if dim not in (-1, x.ndim - 1):
        raise Unsupported("softmax over non-last dim")
    return T.softmax(x)


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, **_):
    if tuple(normalized_shape) != (x.shape[-1],):
        raise Unsupported("layer_norm over multiple dims")
    return T.layer_norm(x, weight, bias, eps)


def _linear(x, w, b=None):
    return T.linear(x, w, b)


def _transpose(x, d0, d1):
    return T.transpose(x, d0, d1)


def _permute(x, *dims):
    return T.permute(x, *dims)


def _view(x, *shape):
    return T.reshape(x, *shape)


def _expand(x, *shape):
    shape = tuple(s if s != -1 else x.shape[i] for i, s in enumerate(shape))
    return T.broadcast_to(x, shape)


def _pow(x, e):
    return x ** e


def _mean(x, dim=None, keepdim=False, **_):
    return _reduce_dim(x, dim, keepdim, "mean")


def _sum(x, dim=None, keepdim=False, **_):
    return _reduce_dim(x, dim, keepdim, "sum")


def _amax(x, dim=-1, keepdim=False, **_):
    return _reduce_dim(x, dim, keepdim, "max")


def _max(x, dim=None, keepdim=False, **_):
    if dim is None:
        return _reduce_dim(x, None, False, "max")
    raise Unsupported("torch.max with dim returns (values, indices)")


def _clamp_min(x, lo):
    return T.maximum(x, lo)


FUNCTION_MAP: dict[object, Callable] = {
    operator.add: lambda a, b: a + b, torch.add: lambda a, b, alpha=1: a + (b * alpha if alpha != 1 else b),
    operator.sub: lambda a, b: a - b, torch.sub: lambda a, b, alpha=1: a - (b * alpha if alpha != 1 else b),
    operator.mul: lambda a, b: a * b, torch.mul: lambda a, b: a * b,
    operator.truediv: lambda a, b: a / b, torch.div: lambda a, b: a / b,
    operator.neg: lambda a: -a, torch.neg: lambda a: -a,
    operator.pow: _pow, torch.pow: _pow,
    operator.matmul: T.matmul, torch.matmul: T.matmul, torch.mm: T.matmul, torch.bmm: T.matmul,
    torch.exp: T.exp, torch.log: T.log, torch.sqrt: T.sqrt, torch.rsqrt: T.rsqrt, torch.tanh: T.tanh,
    torch.abs: T.abs, torch.sigmoid: T.sigmoid, torch.relu: T.relu, torch.maximum: T.maximum,
    torch.minimum: T.minimum, torch.where: T.where, torch.clamp_min: _clamp_min,
    F.relu: T.relu, F.gelu: lambda x, approximate="none": T.gelu(x), F.silu: T.silu, F.sigmoid: T.sigmoid,
    F.tanh: T.tanh, F.softmax: _softmax, torch.softmax: _softmax, F.layer_norm: _layer_norm, F.linear: _linear,
    torch.sum: _sum, torch.mean: _mean, torch.amax: _amax, torch.max: _max,
    F.dropout: lambda x, p=0.5, training=False, inplace=False: x if not training else _unsupported("dropout(train)"),
    torch.dropout: lambda x, p, train: x if not train else _unsupported("dropout(train)"),
    torch.transpose: _transpose, torch.permute: _permute, torch.reshape: _view, torch.t: lambda x: x.T,
}

METHOD_MAP: dict[str, Callable] = {
    "add": FUNCTION_MAP[torch.add], "sub": FUNCTION_MAP[torch.sub], "mul": FUNCTION_MAP[torch.mul],
    "div": FUNCTION_MAP[torch.div], "neg": FUNCTION_MAP[torch.neg], "pow": _pow, "matmul": T.matmul,
    "mm": T.matmul, "bmm": T.matmul, "exp": T.exp, "log": T.log, "sqrt": T.sqrt, "rsqrt": T.rsqrt,
    "tanh": T.tanh, "abs": T.abs, "sigmoid": T.sigmoid, "relu": T.relu, "softmax": _softmax,
    "sum": _sum, "mean": _mean, "amax": _amax, "max": _max, "transpose": _transpose, "permute": _permute,
    "view": _view, "reshape": _view, "expand": _expand, "t": lambda x: x.T, "contiguous": lambda x: x,
    "clone": lambda x: x, "float": lambda x: x, "clamp_min": _clamp_min, "size": None, "dim": None,
}


def fx_to_graph(gm: fx.GraphModule, example_inputs) -> tuple[Graph, list[int]]:
    """Translate an FX graph.  Returns (graph, indices of placeholders that are tensors)."""
    env: dict[str, object] = {}
    tensor_placeholders: list[int] = []
    params = dict(gm.named_parameters())
    params.update(dict(gm.named_buffers()))

    def build(g: Graph):
        nonlocal env
        env = {}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for i, (n, ex) in enumerate(zip(placeholders, example_inputs)):
            if isinstance(ex, torch.Tensor):
                env[n.name] = Tensor(g.add_input(tensor_type_of(ex), name=f"%{n.name}"))
                tensor_placeholders.append(i)
            else:
                env[n.name] = ex
        outputs = None
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                continue
            args = fx.node.map_arg(n.args, lambda a: env[a.name])
            kwargs = fx.node.map_arg(n.kwargs, lambda a: env[a.name])
            if n.op == "get_attr":
                p = params.get(n.target)
                if p is None:
                    raise Unsupported(f"get_attr {n.target}")
                env[n.name] = Tensor(g.add_input(tensor_type_of(p), name=f"%{n.target.replace('.', '_')}"))
                tensor_placeholders.append(("attr", n.target))  # type: ignore[arg-type]
            elif n.op == "call_function":
                fn = FUNCTION_MAP.get(n.target)
                if fn is None:
                    if n.target is getattr:
                        raise Unsupported("getattr")
                    raise Unsupported(f"function {n.target}")
                env[n.name] = fn(*args, **kwargs)
            elif n.op == "call_method":
                fn = METHOD_MAP.get(n.target)
                if fn is None:
                    raise Unsupported(f"method {n.target}")
                env[n.name] = fn(*args, **kwargs)
            elif n.op == "call_module":
                mod = gm.get_submodule(n.target)
                if isinstance(mod, torch.nn.Linear):
                    w = Tensor(g.add_input(tensor_type_of(mod.weight), name=f"%{n.target}_w"))
                    tensor_placeholders.append(("attr", f"{n.target}.weight"))  # type: ignore[arg-type]
                    b = None
                    if mod.bias is not None:
                        b = Tensor(g.add_input(tensor_type_of(mod.bias), name=f"%{n.target}_b"))
                        tensor_placeholders.append(("attr", f"{n.target}.bias"))  # type: ignore[arg-type]
                    env[n.name] = T.linear(args[0], w, b)
                elif isinstance(mod, torch.nn.LayerNorm):
                    w = b = None
                    if mod.weight is not None:
                        w = Tensor(g.add_input(tensor_type_of(mod.weight), name=f"%{n.target}_w"))
                        tensor_placeholders.append(("attr", f"{n.target}.weight"))  # type: ignore[arg-type]
                    if mod.bias is not None:
                        b = Tensor(g.add_input(tensor_type_of(mod.bias), name=f"%{n.target}_b"))
                        tensor_placeholders.append(("attr", f"{n.target}.bias"))  # type: ignore[arg-type]
                    env[n.name] = _layer_norm(args[0], mod.normalized_shape, w, b, mod.eps)
                elif isinstance(mod, (torch.nn.GELU, torch.nn.ReLU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                    f = {torch.nn.GELU: T.gelu, torch.nn.ReLU: T.relu, torch.nn.SiLU: T.silu,
                         torch.nn.Sigmoid: T.sigmoid, torch.nn.Tanh: T.tanh}[type(mod)]
                    env[n.name] = f(args[0])
                elif isinstance(mod, torch.nn.Dropout):
                    env[n.name] = args[0]  # inference
                else:
                    raise Unsupported(f"module {type(mod).__name__}")
            elif n.op == "output":
                outputs = args[0]
            else:
                raise Unsupported(n.op)
        outs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        if not all(isinstance(o, Tensor) for o in outs):
            raise Unsupported("non-tensor output")
        return [o.node for o in outs]

    g = Graph(name="fx_graph")
    prev = getattr(T._tls, "graph", None)
    T._tls.graph = g
    try:
        g.set_outputs(build(g))
    finally:
        T._tls.graph = prev
    g.toposort()
    return g, tensor_placeholders


def torch_compile_backend(gm: fx.GraphModule, example_inputs: list):
    """Entry point for ``torch.compile(..., backend=tensorjit.torch_compile_backend)``."""
    strict = os.environ.get("TENSORJIT_STRICT", "0") == "1"
    try:
        graph, sources = fx_to_graph(gm, example_inputs)
        dev = next((e.device.type for e in example_inputs if isinstance(e, torch.Tensor)), "cpu")
        backend = pick_backend(dev)
        compiled = compile_graph(graph, backend, device=dev)
    except (Unsupported, NotImplementedError, AssertionError, ValueError) as e:
        if strict:
            raise
        log.warning("tensorjit: falling back to eager for this graph (%s)", e)
        return gm.forward

    params = dict(gm.named_parameters())
    params.update(dict(gm.named_buffers()))
    out_is_list = isinstance(
        next(n for n in gm.graph.nodes if n.op == "output").args[0], (tuple, list)
    )

    def run(*args):
        ins = []
        for s in sources:
            ins.append(args[s] if isinstance(s, int) else params[s[1]])
        outs = compiled(*ins)
        return list(outs) if out_is_list else outs[0]

    run.tensorjit_compiled = compiled  # type: ignore[attr-defined]
    return run

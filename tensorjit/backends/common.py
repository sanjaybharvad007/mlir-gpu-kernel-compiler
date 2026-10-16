"""Shared codegen helpers: strides for broadcasting/transposes, index
expressions, and per-op expression templates for each target language."""
from __future__ import annotations

from ..ir import Node, OpKind

Shape = tuple[int, ...]


def contiguous_strides(shape: Shape) -> tuple[int, ...]:
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


def broadcast_strides(in_shape: Shape, iter_shape: Shape) -> tuple[int, ...]:
    """Strides to read a (contiguous) ``in_shape`` tensor from every point of
    ``iter_shape``; broadcast dims get stride 0."""
    cs = contiguous_strides(in_shape)
    pad = len(iter_shape) - len(in_shape)
    out = [0] * pad
    for d in range(len(in_shape)):
        out.append(0 if in_shape[d] == 1 and iter_shape[pad + d] != 1 else cs[d])
    return tuple(out)


def access_strides(inp: Node, consumer: Node, iter_shape: Shape) -> tuple[int, ...]:
    """How ``consumer`` reads ``inp`` when iterating over ``iter_shape``."""
    if consumer.op == "transpose":
        perm = consumer.attrs["perm"]
        cs = contiguous_strides(inp.shape)
        return tuple(cs[p] for p in perm)
    return broadcast_strides(inp.shape, iter_shape)


def index_expr(idx_vars: list[str], strides: tuple[int, ...]) -> str:
    terms = [f"{v}*{s}" if s != 1 else v for v, s in zip(idx_vars, strides) if s != 0]
    return " + ".join(terms) if terms else "0"


def decompose(linear: str, shape: Shape, prefix: str, lang: str = "c") -> tuple[list[str], list[str]]:
    """Statements that split a linear index into a multi-index over ``shape``."""
    stmts, names = [], []
    rem = linear
    cs = contiguous_strides(shape)
    for d, (s, st) in enumerate(zip(shape, cs)):
        name = f"{prefix}{d}"
        names.append(name)
        if d == len(shape) - 1:
            stmts.append(f"{name} = {rem}" if lang == "triton" else f"const long {name} = {rem};")
        else:
            if lang == "triton":
                stmts.append(f"{name} = ({rem}) // {st}")
                stmts.append(f"{prefix}r{d} = ({rem}) % {st}")
                rem = f"{prefix}r{d}"
            else:
                stmts.append(f"const long {name} = ({rem}) / {st};")
                stmts.append(f"const long {prefix}r{d} = ({rem}) % {st};")
                rem = f"{prefix}r{d}"
    return stmts, names


C_EXPR = {
    "neg": "(-{0})",
    "exp": "expf({0})",
    "log": "logf({0})",
    "sqrt": "sqrtf({0})",
    "rsqrt": "(1.0f / sqrtf({0}))",
    "tanh": "tanhf({0})",
    "relu": "fmaxf({0}, 0.0f)",
    "sigmoid": "(1.0f / (1.0f + expf(-({0}))))",
    "gelu": "(0.5f * {0} * (1.0f + tj_erff({0} * 0.70710678118654752f)))",
    "silu": "({0} / (1.0f + expf(-({0}))))",
    "abs": "fabsf({0})",
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "div": "({0} / {1})",
    "pow": "powf({0}, {1})",
    "maximum": "fmaxf({0}, {1})",
    "minimum": "fminf({0}, {1})",
    "where": "(({0}) != 0.0f ? ({1}) : ({2}))",
}

# CUDA device code: same as C but with the hardware erf (no vectorisation concern).
CUDA_EXPR = dict(C_EXPR, gelu="(0.5f * {0} * (1.0f + erff({0} * 0.70710678118654752f)))")

TRITON_EXPR = {
    "neg": "(-{0})",
    "exp": "tl.exp({0})",
    "log": "tl.log({0})",
    "sqrt": "tl.sqrt({0})",
    "rsqrt": "tl.rsqrt({0})",
    "tanh": "(1.0 - 2.0 / (tl.exp(2.0 * ({0})) + 1.0))",
    "relu": "tl.maximum({0}, 0.0)",
    "sigmoid": "tl.sigmoid({0})",
    "gelu": "(0.5 * {0} * (1.0 + tl.erf({0} * 0.7071067811865476)))",
    "silu": "({0} * tl.sigmoid({0}))",
    "abs": "tl.abs({0})",
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "div": "({0} / {1})",
    "pow": "tl.exp(tl.log({0}) * ({1}))",
    "maximum": "tl.maximum({0}, {1})",
    "minimum": "tl.minimum({0}, {1})",
    "where": "tl.where({0} != 0, {1}, {2})",
}


def scalar_literal(v: float, lang: str) -> str:
    if v != v:  # nan
        return "NAN" if lang in ("c", "cuda") else "float('nan')"
    if v in (float("inf"), float("-inf")):
        s = "INFINITY" if lang in ("c", "cuda") else "float('inf')"
        return f"(-{s})" if v < 0 else s
    return f"{v!r}f" if lang in ("c", "cuda") else repr(float(v))


def emit_expr(node: Node, operand: dict[int, str], lang: str) -> str:
    """Expression for one elementwise node given operand names."""
    table = {"c": C_EXPR, "cuda": CUDA_EXPR, "triton": TRITON_EXPR}[lang]
    args = []
    for i in node.inputs:
        if i.op == "const" and i.shape == ():
            args.append(scalar_literal(i.attrs["value"], lang))
        else:
            args.append(operand[id(i)])
    assert node.kind == OpKind.ELEMENTWISE, node.op
    return table[node.op].format(*args)

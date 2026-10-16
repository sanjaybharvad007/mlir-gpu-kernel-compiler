"""Graph-level cleanups: DCE, CSE, algebraic simplification, constant folding,
broadcast folding.  These run to a fixed point before fusion."""
from __future__ import annotations

import math

from ..ir import Graph, Node, OpKind


def dce(g: Graph) -> bool:
    live = set(id(o) for o in g.outputs) | set(id(i) for i in g.inputs)
    changed = False
    for n in reversed(list(g.nodes)):
        if id(n) in live:
            continue
        if not n.users:
            g.erase(n)
            changed = True
    return changed


def _key(n: Node) -> tuple:
    return (n.op, tuple(id(i) for i in n.inputs), n.type, tuple(sorted(n.attrs.items())))


def cse(g: Graph) -> bool:
    seen: dict[tuple, Node] = {}
    changed = False
    for n in list(g.nodes):
        if n.op == "input":
            continue
        k = _key(n)
        if k in seen:
            n.replace_all_uses_with(seen[k])
            g.outputs = [seen[k] if o is n else o for o in g.outputs]
            g.erase(n)
            changed = True
        else:
            seen[k] = n
    return changed


_UNARY = {
    "neg": lambda x: -x,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "rsqrt": lambda x: 1.0 / math.sqrt(x),
    "tanh": math.tanh,
    "relu": lambda x: max(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "abs": abs,
    "gelu": lambda x: 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))),
    "silu": lambda x: x / (1.0 + math.exp(-x)),
}
_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "maximum": max,
    "minimum": min,
}


def _is_const(n: Node, value=None) -> bool:
    return n.op == "const" and n.shape == () and (value is None or n.attrs["value"] == value)


def _replace(g: Graph, old: Node, new: Node) -> None:
    old.replace_all_uses_with(new)
    g.outputs = [new if o is old else o for o in g.outputs]
    if not old.users and old not in g.outputs:
        g.erase(old)


def algebraic_simplify(g: Graph) -> bool:
    """x*1 -> x, x+0 -> x, x*0 -> 0, const(const) folding, --x -> x, exp(log x) -> x."""
    changed = False
    for n in list(g.nodes):
        if n.op not in _UNARY and n.op not in _BINARY:
            continue
        ins = n.inputs
        # constant folding (scalar consts only; shape-preserving)
        if all(_is_const(i) for i in ins):
            f = _UNARY.get(n.op) or _BINARY.get(n.op)
            try:
                val = f(*(i.attrs["value"] for i in ins))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            _replace(g, n, g.const(val, n.dtype, n.shape))
            changed = True
            continue
        if n.op in _BINARY:
            a, b = ins
            new = None
            if n.op in ("add", "sub") and _is_const(b, 0.0) and a.shape == n.shape:
                new = a
            elif n.op == "add" and _is_const(a, 0.0) and b.shape == n.shape:
                new = b
            elif n.op in ("mul", "div") and _is_const(b, 1.0) and a.shape == n.shape:
                new = a
            elif n.op == "mul" and _is_const(a, 1.0) and b.shape == n.shape:
                new = b
            elif n.op == "mul" and (_is_const(a, 0.0) or _is_const(b, 0.0)) and n.shape == ():
                new = g.const(0.0, n.dtype, n.shape)   # non-scalar zeros stay a kernel: backends take scalar consts only
            if new is not None:
                _replace(g, n, new)
                changed = True
                continue
        if n.op == "neg" and ins[0].op == "neg":
            _replace(g, n, ins[0].inputs[0])
            changed = True
        elif n.op == "exp" and ins[0].op == "log":
            _replace(g, n, ins[0].inputs[0])
            changed = True
    return changed


def fold_broadcast(g: Graph) -> bool:
    """broadcast_to feeding only elementwise users is implicit (numpy broadcasting)."""
    changed = False
    for n in list(g.nodes):
        if n.op != "broadcast_to" or n in g.outputs:
            continue
        if all(u.kind == OpKind.ELEMENTWISE for u in n.users):
            _replace(g, n, n.inputs[0])
            changed = True
    return changed


def fold_reshape_chains(g: Graph) -> bool:
    """reshape(reshape(x)) -> reshape(x); reshape to the same shape -> x."""
    changed = False
    for n in list(g.nodes):
        if n.op != "reshape":
            continue
        src = n.inputs[0]
        if src.shape == n.shape:
            _replace(g, n, src)
            changed = True
        elif src.op == "reshape" and len(src.users) == 1:
            new = g.create("reshape", [src.inputs[0]], shape=n.shape)
            _replace(g, n, new)
            changed = True
    return changed


def fold_transpose_into_matmul(g: Graph) -> bool:
    """matmul(a, transpose(b, swap last two)) -> matmul(a, b) {trans_b}.

    This is the nn.Linear layout (weight stored [out, in]); reading B with
    swapped strides is free, materialising the transpose is a full copy.
    """
    changed = False
    for n in list(g.nodes):
        if n.op != "matmul" or n.attrs.get("trans_b"):
            continue
        b = n.inputs[1]
        r = b.type.rank
        if b.op != "transpose" or tuple(b.attrs["perm"]) != tuple(range(r - 2)) + (r - 1, r - 2):
            continue
        src = b.inputs[0]
        b.users = [u for u in b.users if u is not n]
        n.inputs[1] = src
        src.users.append(n)
        n.attrs["trans_b"] = True
        changed = True
    return changed

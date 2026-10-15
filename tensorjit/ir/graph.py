"""tensorjit IR: a small SSA tensor-level graph.

Each ``Node`` is one tensor-valued operation with a static ``TensorType``
(shape + dtype).  Shapes are concrete at compile time: the JIT specialises a
graph per (shape, dtype) signature, which is what lets the backends bake
sizes/strides into the generated code and unroll/vectorise aggressively.

Op categories (``OpKind``) drive the fusion pass:
  * ELEMENTWISE  - pointwise, numpy broadcasting
  * MATMUL       - 2-D or batched (leading dims broadcast) matrix product
  * REDUCE       - reduction over the last axis (sum / max / mean)
  * MOVEMENT     - reshape / transpose / broadcast (metadata-only when possible)
  * CONST        - scalar or tensor constant
"""
from __future__ import annotations

import enum
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

Shape = tuple[int, ...]


class DType(str, enum.Enum):
    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"
    i32 = "i32"

    @property
    def c_type(self) -> str:
        return {"f32": "float", "f16": "_Float16", "bf16": "unsigned short", "i32": "int"}[self.value]

    @property
    def itemsize(self) -> int:
        return {"f32": 4, "f16": 2, "bf16": 2, "i32": 4}[self.value]

    @property
    def mlir(self) -> str:
        return {"f32": "f32", "f16": "f16", "bf16": "bf16", "i32": "i32"}[self.value]


@dataclass(frozen=True)
class TensorType:
    shape: Shape
    dtype: DType = DType.f32

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __str__(self) -> str:
        return f"tensor<{'x'.join(map(str, self.shape))}x{self.dtype.value}>"


class OpKind(enum.Enum):
    ELEMENTWISE = "elementwise"
    MATMUL = "matmul"
    REDUCE = "reduce"
    MOVEMENT = "movement"
    CONST = "const"
    INPUT = "input"


# name -> (kind, arity, C expression template, mlir arith op)
# C templates use {0}, {1}; "a" is the accumulator for reductions.
OPS: dict[str, tuple[OpKind, int]] = {
    "input": (OpKind.INPUT, 0),
    "const": (OpKind.CONST, 0),
    # unary elementwise
    "neg": (OpKind.ELEMENTWISE, 1),
    "exp": (OpKind.ELEMENTWISE, 1),
    "log": (OpKind.ELEMENTWISE, 1),
    "sqrt": (OpKind.ELEMENTWISE, 1),
    "rsqrt": (OpKind.ELEMENTWISE, 1),
    "tanh": (OpKind.ELEMENTWISE, 1),
    "relu": (OpKind.ELEMENTWISE, 1),
    "sigmoid": (OpKind.ELEMENTWISE, 1),
    "gelu": (OpKind.ELEMENTWISE, 1),
    "silu": (OpKind.ELEMENTWISE, 1),
    "abs": (OpKind.ELEMENTWISE, 1),
    # binary elementwise (broadcasting)
    "add": (OpKind.ELEMENTWISE, 2),
    "sub": (OpKind.ELEMENTWISE, 2),
    "mul": (OpKind.ELEMENTWISE, 2),
    "div": (OpKind.ELEMENTWISE, 2),
    "pow": (OpKind.ELEMENTWISE, 2),
    "maximum": (OpKind.ELEMENTWISE, 2),
    "minimum": (OpKind.ELEMENTWISE, 2),
    # ternary
    "where": (OpKind.ELEMENTWISE, 3),
    # contractions / reductions
    "matmul": (OpKind.MATMUL, 2),
    "sum": (OpKind.REDUCE, 1),
    "max": (OpKind.REDUCE, 1),
    "mean": (OpKind.REDUCE, 1),
    # movement
    "reshape": (OpKind.MOVEMENT, 1),
    "transpose": (OpKind.MOVEMENT, 1),
    "broadcast_to": (OpKind.MOVEMENT, 1),
}

_ids = itertools.count()


@dataclass(eq=False)
class Node:
    op: str
    inputs: list["Node"]
    type: TensorType
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    users: list["Node"] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = f"%{self.op}{next(_ids)}"
        for i in self.inputs:
            i.users.append(self)

    @property
    def kind(self) -> OpKind:
        return OPS[self.op][0]

    @property
    def shape(self) -> Shape:
        return self.type.shape

    @property
    def dtype(self) -> DType:
        return self.type.dtype

    def replace_all_uses_with(self, other: "Node") -> None:
        for u in list(self.users):
            u.inputs = [other if x is self else x for x in u.inputs]
            other.users.append(u)
        self.users = []

    def __repr__(self) -> str:
        return f"{self.name} = {self.op}({', '.join(i.name for i in self.inputs)}) : {self.type}"


def broadcast_shapes(*shapes: Shape) -> Shape:
    rank = max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        dim = 1
        for s in shapes:
            d = s[len(s) - rank + i] if len(s) - rank + i >= 0 else 1
            if d != 1:
                if dim != 1 and dim != d:
                    raise ValueError(f"incompatible shapes for broadcasting: {shapes}")
                dim = d
        out.append(dim)
    return tuple(out)


class Graph:
    """SSA graph with explicit inputs and outputs, in topological order."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: list[Node] = []
        self.nodes: list[Node] = []
        self.outputs: list[Node] = []

    # -- building ----------------------------------------------------------
    def add_input(self, type: TensorType, name: Optional[str] = None) -> Node:
        n = Node("input", [], type, name=name or f"%arg{len(self.inputs)}")
        self.inputs.append(n)
        self.nodes.append(n)
        return n

    def const(self, value: float, dtype: DType = DType.f32, shape: Shape = ()) -> Node:
        n = Node("const", [], TensorType(shape, dtype), attrs={"value": float(value)})
        self.nodes.append(n)
        return n

    def create(self, op: str, inputs: Sequence[Node], **attrs) -> Node:
        kind, arity = OPS[op]
        if arity and len(inputs) != arity:
            raise ValueError(f"{op} expects {arity} inputs, got {len(inputs)}")
        typ = infer_type(op, list(inputs), attrs)
        n = Node(op, list(inputs), typ, attrs=dict(attrs))
        self.nodes.append(n)
        return n

    def set_outputs(self, outs: Sequence[Node]) -> None:
        self.outputs = list(outs)

    # -- maintenance -------------------------------------------------------
    def erase(self, node: Node) -> None:
        assert not node.users, f"cannot erase {node.name}: still has users"
        for i in node.inputs:
            i.users = [u for u in i.users if u is not node]
        self.nodes.remove(node)

    def toposort(self) -> None:
        seen: set[int] = set()
        order: list[Node] = []

        def visit(n: Node):
            if id(n) in seen:
                return
            seen.add(id(n))
            for i in n.inputs:
                visit(i)
            order.append(n)

        for n in self.nodes:
            visit(n)
        self.nodes = order

    def signature(self) -> str:
        """Hash of structure + types: the kernel-cache key."""
        h = hashlib.sha256()
        idx = {id(n): i for i, n in enumerate(self.nodes)}
        for n in self.nodes:
            h.update(f"{n.op}|{[idx[id(i)] for i in n.inputs]}|{n.type}|{sorted(n.attrs.items())}\n".encode())
        h.update(f"out|{[idx[id(o)] for o in self.outputs]}".encode())
        return h.hexdigest()[:16]

    def __str__(self) -> str:
        from .printer import print_graph

        return print_graph(self)


def infer_type(op: str, inputs: list[Node], attrs: dict) -> TensorType:
    kind = OPS[op][0]
    if kind == OpKind.ELEMENTWISE:
        shape = broadcast_shapes(*(i.shape for i in inputs))
        dtypes = {i.dtype for i in inputs if i.op != "const"} or {inputs[0].dtype}
        if len(dtypes) != 1:
            raise TypeError(f"{op}: mixed dtypes {dtypes}")
        return TensorType(shape, next(iter(dtypes)))
    if kind == OpKind.MATMUL:
        a, b = inputs
        kb, n = (b.shape[-1], b.shape[-2]) if attrs.get("trans_b") else (b.shape[-2], b.shape[-1])
        if a.type.rank < 2 or b.type.rank < 2 or a.shape[-1] != kb:
            raise ValueError(f"matmul shape mismatch {a.shape} @ {b.shape} (trans_b={attrs.get('trans_b', False)})")
        batch = broadcast_shapes(a.shape[:-2], b.shape[:-2])
        return TensorType(batch + (a.shape[-2], n), a.dtype)
    if kind == OpKind.REDUCE:
        (x,) = inputs
        axis = attrs.get("axis", -1)
        if axis != -1 and axis != x.type.rank - 1:
            raise NotImplementedError("reductions are supported over the last axis only")
        keepdim = attrs.get("keepdim", False)
        shape = x.shape[:-1] + ((1,) if keepdim else ())
        return TensorType(shape, x.dtype)
    if kind == OpKind.MOVEMENT:
        (x,) = inputs
        if op == "reshape":
            shape = tuple(attrs["shape"])
            if -1 in shape:
                known = -1
                for s in shape:
                    known *= s
                shape = tuple(x.type.numel // known if s == -1 else s for s in shape)
            assert TensorType(shape).numel == x.type.numel, "reshape numel mismatch"
            return TensorType(shape, x.dtype)
        if op == "transpose":
            perm = tuple(attrs["perm"])
            return TensorType(tuple(x.shape[p] for p in perm), x.dtype)
        if op == "broadcast_to":
            return TensorType(tuple(attrs["shape"]), x.dtype)
    raise NotImplementedError(op)

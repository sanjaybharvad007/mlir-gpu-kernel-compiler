"""Operator fusion: partitions the graph into kernels.

Three kernel shapes come out of this pass:

* ``elementwise`` - a chain/DAG of pointwise ops sharing one iteration space
  (broadcast inputs are read with stride 0).  Transposes are single-node
  elementwise kernels with permuted input strides.
* ``matmul``      - a matrix product plus any pointwise *epilogue* (bias, GELU,
  residual add, ...) that consumes it in the same iteration space.
* ``reduce``      - a last-axis reduction plus a pointwise *prologue* whose
  only consumer is the reduction (e.g. ``mean((x - mu)**2)``).

Fusion is greedy over the topological order and never duplicates compute: a
value needed both inside and outside a kernel is simply written out as an
additional kernel output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Graph, Node, OpKind


@dataclass
class Kernel:
    kind: str                       # "elementwise" | "matmul" | "reduce" | "copy"
    nodes: list[Node] = field(default_factory=list)   # compute nodes, topo order
    anchor: Node | None = None      # the matmul / reduce node
    inputs: list[Node] = field(default_factory=list)  # values read from outside
    outputs: list[Node] = field(default_factory=list) # values that escape

    @property
    def iter_shape(self):
        """Iteration space: the anchor's output shape, or the elementwise shape."""
        if self.kind == "reduce":
            return self.anchor.inputs[0].shape
        if self.kind == "row":
            return tuple(self.rows) + (self.n,)  # type: ignore[attr-defined]
        return (self.anchor or self.nodes[-1]).shape

    def name(self) -> str:
        return f"{self.kind}_{abs(hash(tuple(n.name for n in self.nodes))) % 10**8:08x}"

    def __repr__(self) -> str:
        return f"Kernel<{self.kind} {[n.op for n in self.nodes]} -> {[o.name for o in self.outputs]}>"


@dataclass
class Program:
    graph: Graph
    kernels: list[Kernel]
    aliases: dict[int, tuple[Node, tuple]]  # reshape node id -> (source, shape)

    def __str__(self) -> str:
        lines = [f"program {self.graph.name}: {len(self.kernels)} kernels"]
        for k in self.kernels:
            lines.append(f"  {k.kind:12s} {[n.op for n in self.nodes_of(k)]:} -> {[o.name for o in k.outputs]}")
        return "\n".join(lines)

    def nodes_of(self, k: Kernel):
        return k.nodes


def _finalize(k: Kernel, g: Graph) -> None:
    inside = {id(n) for n in k.nodes}
    outs = {id(o) for o in g.outputs}
    k.inputs, k.outputs = [], []
    seen = set()
    for n in k.nodes:
        for i in n.inputs:
            if id(i) not in inside and i.op != "const" and id(i) not in seen:
                seen.add(id(i))
                k.inputs.append(i)
        if id(n) in outs or any(id(u) not in inside for u in n.users):
            k.outputs.append(n)


def fuse(g: Graph) -> Program:
    g.toposort()
    producer: dict[int, Kernel] = {}     # node id -> kernel that computes it
    aliases: dict[int, tuple[Node, tuple]] = {}
    kernels: list[Kernel] = []
    graph_outs = {id(o) for o in g.outputs}

    def resolve(n: Node) -> Node:
        """Follow reshape aliases to the materialised value."""
        while id(n) in aliases:
            n = aliases[id(n)][0]
        return n

    for n in g.nodes:
        if n.kind in (OpKind.INPUT, OpKind.CONST):
            continue

        if n.op == "reshape":
            src = n.inputs[0]
            # reshape of a materialised contiguous value is a free view
            aliases[id(n)] = (src, n.shape)
            if id(src) in producer and src not in producer[id(src)].outputs:
                producer[id(src)].outputs.append(src)
            continue

        if n.kind == OpKind.MOVEMENT:  # transpose / broadcast_to -> copy kernel
            k = Kernel("copy", [n], anchor=n)
            kernels.append(k)
            producer[id(n)] = k
            continue

        if n.kind == OpKind.MATMUL:
            k = Kernel("matmul", [n], anchor=n)
            kernels.append(k)
            producer[id(n)] = k
            continue

        if n.kind == OpKind.REDUCE:
            k = Kernel("reduce", [n], anchor=n)
            src = n.inputs[0]
            pk = producer.get(id(src))
            # prologue fusion: pull in a pure elementwise producer kernel whose
            # every output is consumed only by this reduction.
            if (
                pk is not None
                and pk.kind == "elementwise"
                and all(all(u is n for u in o.users) and id(o) not in graph_outs for o in _escaping(pk))
                and all(m.shape == src.shape for m in pk.nodes)
            ):
                kernels.remove(pk)
                k.nodes = pk.nodes + [n]
                for m in pk.nodes:
                    producer[id(m)] = k
            kernels.append(k)
            producer[id(n)] = k
            continue

        assert n.kind == OpKind.ELEMENTWISE
        # Candidate producer kernels we could join.
        target: Kernel | None = None
        for i in n.inputs:
            pk = producer.get(id(resolve(i)))
            if pk is None or id(i) in aliases:
                continue  # inputs / consts / reshaped values are read, not fused through
            if pk.kind == "elementwise" and pk.iter_shape == n.shape:
                target = pk
                break
            if pk.kind == "matmul" and pk.anchor.shape == n.shape and _epilogue_ok(pk, n):
                target = pk
                break
            if pk.kind == "reduce" and pk.anchor.shape == n.shape and _epilogue_ok(pk, n):
                target = pk
                break
        if target is not None and _no_cycle(target, n, producer, resolve):
            target.nodes.append(n)
            producer[id(n)] = target
        else:
            k = Kernel("elementwise", [n])
            kernels.append(k)
            producer[id(n)] = k

    for k in kernels:
        # Kernel inputs may be reshape nodes: they keep the shape the consumer
        # sees, and the executor resolves them to views of the source buffer.
        _finalize(k, g)
        _dedupe(k)
    return Program(g, _schedule(kernels, producer, resolve), aliases)


def _schedule(kernels: list[Kernel], producer: dict, resolve) -> list[Kernel]:
    """Topological order of kernels by data dependence (a node may join a
    kernel created earlier than one of its producers, so creation order is
    not a valid schedule)."""
    deps: dict[int, set[int]] = {}
    by_id = {id(k): k for k in kernels}
    for k in kernels:
        deps[id(k)] = {id(producer[id(resolve(i))]) for i in k.inputs if id(resolve(i)) in producer} - {id(k)}
    order, done = [], set()
    remaining = list(kernels)
    while remaining:
        progress = False
        for k in list(remaining):
            if deps[id(k)] <= done:
                order.append(k)
                done.add(id(k))
                remaining.remove(k)
                progress = True
        if not progress:  # pragma: no cover
            raise RuntimeError("cyclic kernel dependency after fusion: " + ", ".join(repr(by_id[i]) for i in deps))
    return order


def _escaping(k: Kernel) -> list[Node]:
    inside = {id(n) for n in k.nodes}
    return [n for n in k.nodes if any(id(u) not in inside for u in n.users)]


def _epilogue_ok(pk: Kernel, n: Node) -> bool:
    """A matmul/reduce epilogue may only grow along the anchor's output shape."""
    return all(i.shape == () or len(i.shape) <= len(pk.anchor.shape) for i in n.inputs)


def _no_cycle(target: Kernel, n: Node, producer: dict, resolve) -> bool:
    """Joining n to target must not make target depend (transitively) on a
    kernel that itself reads from target."""
    inside = {id(m) for m in target.nodes}
    stack = [i for i in n.inputs if id(i) not in inside]
    seen = set()
    while stack:
        x = resolve(stack.pop())
        if id(x) in seen:
            continue
        seen.add(id(x))
        pk = producer.get(id(x))
        if pk is target:
            return False
        if pk is not None:
            stack.extend(pk.inputs or [i for m in pk.nodes for i in m.inputs])
    return True


def _dedupe(k: Kernel) -> None:
    seen, out = set(), []
    for i in k.inputs:
        if id(i) not in seen:
            seen.add(id(i))
            out.append(i)
    k.inputs = out


# ----------------------------------------------------------------------------
# Row fusion: LayerNorm / softmax / RMSNorm as one kernel
# ----------------------------------------------------------------------------


def _row_shape_class(k: Kernel):
    """(rows, N, cls) if the kernel is a per-row computation, else None.

    cls is "vec" for an elementwise kernel over [rows..., N], "red" for a
    last-axis reduction of a [rows..., N] value, "scalar" for elementwise work
    on per-row values ([rows...] or [rows..., 1])."""
    if k.kind == "reduce":
        src = k.anchor.inputs[0].shape
        if len(src) < 1:
            return None
        return src[:-1], src[-1], "red"
    if k.kind == "elementwise":
        s = k.iter_shape
        return (s[:-1], s[-1], "vec") if len(s) >= 1 else None
    return None


def row_fuse(program: Program, max_n: int | None = None) -> Program:
    """Merge chains of per-row kernels (same leading dims, last-axis
    reductions, broadcasts of reduced values) into single ``row`` kernels so the
    row is read from memory once and reductions never round-trip through HBM."""
    kernels = program.kernels
    producer: dict[int, Kernel] = {}
    for k in kernels:
        for o in k.outputs:
            producer[id(o)] = k
    aliases = program.aliases

    def resolve(n: Node) -> Node:
        while id(n) in aliases:
            n = aliases[id(n)][0]
        return n

    # transitive kernel dependencies (for cycle avoidance)
    deps: dict[int, set[int]] = {}
    for k in kernels:
        d = set()
        for i in k.inputs:
            p = producer.get(id(resolve(i)))
            if p is not None:
                d.add(id(p))
                d |= deps.get(id(p), set())
        deps[id(k)] = d

    groups: list[list[Kernel]] = []
    group_of: dict[int, list[Kernel]] = {}
    for k in kernels:
        info = _row_shape_class(k)
        joined = False
        if info is not None and (max_n is None or info[1] <= max_n):
            rows, n, cls = info
            # candidate groups: producers of our inputs
            for i in k.inputs:
                p = producer.get(id(resolve(i)))
                if p is None or id(p) not in group_of or id(i) in aliases:
                    continue
                grp = group_of[id(p)]
                g_rows, g_n = grp[0]._row
                if g_rows != rows or g_n != n:
                    continue
                # scalar-class kernels must be [rows] or [rows, 1]
                if cls == "vec" and k.iter_shape not in (rows + (n,), rows + (1,), rows):
                    continue
                # every kernel-produced input must come from the group or from a
                # kernel that does not depend on the group (no cycles)
                ok = True
                for j in k.inputs:
                    pj = producer.get(id(resolve(j)))
                    if pj is None or pj in grp:
                        continue
                    if any(id(m) in deps[id(pj)] for m in grp):
                        ok = False
                        break
                if not ok:
                    continue
                grp.append(k)
                group_of[id(k)] = grp
                joined = True
                break
        if not joined:
            if info is not None and (max_n is None or info[1] <= max_n):
                k._row = (info[0], info[1])  # type: ignore[attr-defined]
                grp = [k]
                groups.append(grp)
                group_of[id(k)] = grp
            else:
                groups.append([k])

    out: list[Kernel] = []
    for grp in groups:
        if len(grp) == 1:
            out.append(grp[0])
            continue
        rk = Kernel("row", nodes=[n for k in grp for n in k.nodes])
        rk.stages = grp  # type: ignore[attr-defined]
        rk.rows, rk.n = grp[0]._row  # type: ignore[attr-defined]
        _finalize(rk, program.graph)
        _dedupe(rk)
        out.append(rk)
    prod2 = {id(o): k for k in out for o in k.outputs}
    return Program(program.graph, _schedule(out, prod2, resolve), aliases)

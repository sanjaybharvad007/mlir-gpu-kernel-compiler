"""Static memory planning for intermediate buffers.

After fusion the program is a straight-line sequence of kernels, so every
intermediate's lifetime is a simple interval [def_kernel, last_use_kernel].
That makes buffer assignment an interval-graph colouring problem: two
intermediates can share storage iff their intervals do not overlap.

We allocate one arena per program and assign each intermediate an offset in
it with a greedy best-fit over free *ranges* (not just whole buffers), the
same scheme TVM, XLA and Inductor use.  Benefits over per-call allocation:

* zero allocator calls at run time (one arena, reused across calls);
* no first-touch page faults in the hot path;
* peak memory is known at compile time and typically 40-70% below the sum of
  intermediate sizes for transformer blocks (see ``plan.savings``).

Graph outputs are *not* placed in the arena: they are handed to the caller and
must not be overwritten by the next call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..ir import Node
from .fusion import Program

ALIGN = 64  # bytes; cache-line and AVX-512 friendly, also satisfies cudaMalloc alignment rules


@dataclass
class Buffer:
    node: Node
    size: int               # bytes (aligned)
    first: int              # index of the kernel that defines it
    last: int               # index of the last kernel that reads it
    offset: int = -1        # byte offset in the arena

    @property
    def shape(self):
        return self.node.shape


@dataclass
class MemoryPlan:
    buffers: dict[int, Buffer] = field(default_factory=dict)   # node id -> Buffer
    arena_bytes: int = 0
    naive_bytes: int = 0                                         # sum of sizes without sharing

    @property
    def savings(self) -> float:
        return 1.0 - self.arena_bytes / self.naive_bytes if self.naive_bytes else 0.0

    def offset(self, node: Node) -> int | None:
        b = self.buffers.get(id(node))
        return None if b is None else b.offset

    def __str__(self) -> str:
        lines = [f"arena {self.arena_bytes} B (naive {self.naive_bytes} B, {self.savings:.0%} saved)"]
        for b in sorted(self.buffers.values(), key=lambda b: b.offset):
            lines.append(f"  [{b.offset:8d}, +{b.size:8d})  k{b.first}..k{b.last}  {b.node.name} {b.shape}")
        return "\n".join(lines)


def _aligned(nbytes: int) -> int:
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def plan_memory(program: Program) -> MemoryPlan:
    g = program.graph
    aliases = program.aliases

    def resolve(n: Node) -> Node:
        while id(n) in aliases:
            n = aliases[id(n)][0]
        return n

    # Values returned to the caller (directly or through a reshape) live outside the arena.
    escaping = {id(resolve(o)) for o in g.outputs}

    # 1. lifetimes
    bufs: dict[int, Buffer] = {}
    for ki, k in enumerate(program.kernels):
        for o in k.outputs:
            if id(o) in escaping:
                continue
            nbytes = math.prod(o.shape) * o.dtype.itemsize if o.shape else o.dtype.itemsize
            bufs[id(o)] = Buffer(o, _aligned(max(nbytes, 1)), first=ki, last=ki)
        for i in k.inputs:
            src = resolve(i)
            if id(src) in bufs:
                bufs[id(src)].last = ki

    # 2. greedy best-fit over free ranges, processing buffers in order of definition
    #    (sizes descending within a kernel helps packing).
    order = sorted(bufs.values(), key=lambda b: (b.first, -b.size))
    live: list[Buffer] = []       # placed buffers, used to compute free ranges at each step
    arena = 0
    for b in order:
        # buffers whose lifetime ended before b starts are free again
        live = [x for x in live if x.last >= b.first]
        occupied = sorted((x.offset, x.offset + x.size) for x in live)
        # free ranges between occupied intervals, choose the smallest that fits
        best, best_gap = None, None
        cursor = 0
        for start, end in occupied + [(arena, arena)]:
            gap = start - cursor
            if gap >= b.size and (best_gap is None or gap < best_gap):
                best, best_gap = cursor, gap
            cursor = max(cursor, end)
        if best is None:
            best = cursor
        b.offset = best
        arena = max(arena, b.offset + b.size)
        live.append(b)

    plan = MemoryPlan(bufs, arena_bytes=arena, naive_bytes=sum(b.size for b in bufs.values()))
    _verify(plan)
    return plan


def _verify(plan: MemoryPlan) -> None:
    """No two live-at-the-same-time buffers may overlap."""
    bs = list(plan.buffers.values())
    for i, a in enumerate(bs):
        for b in bs[i + 1:]:
            overlap_time = not (a.last < b.first or b.last < a.first)
            overlap_mem = not (a.offset + a.size <= b.offset or b.offset + b.size <= a.offset)
            assert not (overlap_time and overlap_mem), f"memory plan conflict: {a.node.name} vs {b.node.name}"

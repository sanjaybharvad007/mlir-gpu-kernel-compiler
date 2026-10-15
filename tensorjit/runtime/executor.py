"""Runs a fused ``Program``: allocates intermediates, resolves reshape aliases,
and invokes one compiled callable per kernel in order."""
from __future__ import annotations

import sys
from typing import Callable, Sequence

import torch

from ..ir import DType, Node
from ..passes import Program

KernelFn = Callable[[list[torch.Tensor], list[torch.Tensor]], None]

TORCH_DTYPE = {DType.f32: torch.float32, DType.f16: torch.float16, DType.bf16: torch.bfloat16, DType.i32: torch.int32}
FROM_TORCH = {v: k for k, v in TORCH_DTYPE.items()}


class BufferPool:
    """Reuses intermediate buffers across calls.

    Fresh large allocations are served by mmap and every first-touch page
    faults inside the kernel, which for a 512 KB activation costs more than the
    kernel itself.  Intermediates (values that never leave the program) are
    therefore recycled; graph outputs are always freshly allocated because they
    are handed to the caller.
    """

    MAX_HELD = 8

    def __init__(self):
        self._free: dict[tuple, list[torch.Tensor]] = {}
        self._handed_out: dict[tuple, list[torch.Tensor]] = {}

    def take(self, shape, dtype, device) -> torch.Tensor:
        lst = self._free.get((tuple(shape), dtype, device))
        if lst:
            return lst.pop()
        return torch.empty(shape, dtype=dtype, device=device)

    def give(self, t: torch.Tensor) -> None:
        self._free.setdefault((tuple(t.shape), t.dtype, t.device.type), []).append(t)

    _BASELINE: int | None = None  # refcount of a tensor referenced only by the scan loop

    def _scan(self, held: list[torch.Tensor], record: bool = False):
        """Index of the first buffer nobody else references, else None.

        Written so the calibration probe and the real scan share one code path:
        CPython's refcount at this point depends on the exact loop/call shape.
        """
        for i, t in enumerate(held):
            rc = sys.getrefcount(t)
            if record:
                return rc
            if (
                rc == self._BASELINE
                and t._use_count() == 1                       # no view holds this impl
                and torch._C._storage_Use_Count(t.untyped_storage()._cdata) == 2  # +1 for the probe handle
            ):
                return i
        return None

    def take_output(self, shape, dtype, device) -> torch.Tensor:
        """Buffer for a value returned to the caller.  A previously returned
        buffer is reused only once the caller has dropped every reference to it."""
        if BufferPool._BASELINE is None:
            BufferPool._BASELINE = self._scan([torch.empty(1)], record=True)
        key = (tuple(shape), dtype, device)
        held = self._handed_out.setdefault(key, [])
        i = self._scan(held)
        if i is not None:
            held.append(held.pop(i))
            return held[-1]
        t = torch.empty(shape, dtype=dtype, device=device)
        held.append(t)
        if len(held) > self.MAX_HELD:
            held.pop(0)
        return t


class CompiledProgram:
    def __init__(self, program: Program, kernel_fns: Sequence[KernelFn], device: str, backend: str):
        self.program = program
        self.kernel_fns = list(kernel_fns)
        self.device = device
        self.backend = backend
        self.pool = BufferPool()
        # Values handed to the caller (directly or through a reshape view) must
        # not be recycled; everything else is an intermediate.
        self._escaping: set[int] = set()
        for o in program.graph.outputs:
            n = o
            while id(n) in program.aliases:
                n = program.aliases[id(n)][0]
            self._escaping.add(id(n))
        assert len(self.kernel_fns) == len(program.kernels)

    @property
    def num_kernels(self) -> int:
        return len(self.kernel_fns)

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        g = self.program.graph
        assert len(inputs) == len(g.inputs), f"expected {len(g.inputs)} inputs, got {len(inputs)}"
        env: dict[int, torch.Tensor] = {}
        recycled: list[torch.Tensor] = []
        for node, t in zip(g.inputs, inputs):
            assert tuple(t.shape) == node.shape, f"{node.name}: expected {node.shape}, got {tuple(t.shape)}"
            env[id(node)] = t.contiguous()

        aliases = self.program.aliases
        device = self.device

        def get(n: Node) -> torch.Tensor:
            # Non-recursive on purpose: a self-referencing closure forms a GC
            # cycle that keeps `env` (and every buffer in it) alive until the
            # next collection, defeating buffer reuse.
            t = env.get(id(n))
            if t is not None:
                return t
            chain = []
            m = n
            while id(m) not in env and id(m) in aliases:
                chain.append(m)
                m = aliases[id(m)][0]
            if id(m) in env:
                t = env[id(m)]
            elif m.op == "const":
                t = torch.full(m.shape, m.attrs["value"], dtype=TORCH_DTYPE[m.dtype], device=device)
                env[id(m)] = t
            else:
                raise KeyError(f"value {m.name} was never computed")
            for r in reversed(chain):
                t = t.reshape(aliases[id(r)][1])
                env[id(r)] = t
            return t

        for k, fn in zip(self.program.kernels, self.kernel_fns):
            ins = [get(i) for i in k.inputs]
            outs = []
            for o in k.outputs:
                dt = TORCH_DTYPE[o.dtype]
                escapes = id(o) in self._escaping
                t = self.pool.take_output(o.shape, dt, self.device) if escapes else self.pool.take(o.shape, dt, self.device)
                if not escapes:
                    recycled.append(t)
                outs.append(t)
            fn(ins, outs)
            for o, t in zip(k.outputs, outs):
                env[id(o)] = t
        result = [get(o) for o in g.outputs]
        for t in recycled:
            self.pool.give(t)
        env.clear()
        return result

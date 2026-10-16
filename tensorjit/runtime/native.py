"""Native (C++) executor: builds ``csrc/runtime.cpp`` on first use and runs a
C-backend program through it with a statically planned arena.

Compared with ``CompiledProgram`` (the Python executor):
  * one C++ call per program instead of one Python round-trip per kernel;
  * intermediates live in one planned arena (``passes.memory_planner``);
  * only caller-visible outputs are allocated per call.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch

from ..ir import Node
from ..passes import Program
from ..passes.memory_planner import MemoryPlan, plan_memory
from .cache import cache_dir, content_hash
from .executor import TORCH_DTYPE, BufferPool

_SRC = Path(__file__).resolve().parents[2] / "csrc" / "runtime.cpp"
_module = None


def build_runtime():
    """Compile and import ``tensorjit._runtime`` (cached by source hash)."""
    global _module
    if _module is not None:
        return _module
    import pybind11

    src = _SRC.read_text()
    tag = content_hash(src, sys.version, "rt-v1")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = cache_dir() / f"_runtime_{tag}{ext}"
    if not out.exists():
        cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
        if cxx is None:
            raise RuntimeError("no C++ compiler found (set CXX)")
        cmd = [
            cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden",
            f"-I{pybind11.get_include()}", f"-I{sysconfig.get_paths()['include']}",
            str(_SRC), "-o", str(out) + ".tmp", "-ldl",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"runtime build failed:\n{r.stderr}")
        os.replace(str(out) + ".tmp", out)
    spec = importlib.util.spec_from_file_location("tensorjit._runtime", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _module = mod
    return mod


class NativeProgram:
    """Drop-in replacement for ``CompiledProgram`` backed by the C++ runtime."""

    def __init__(self, program: Program, so_path: str, symbols: list[str], plan: MemoryPlan | None = None):
        rt = build_runtime()
        self.program = program
        self.plan = plan or plan_memory(program)
        g = program.graph
        aliases = program.aliases

        def resolve(n: Node) -> Node:
            while id(n) in aliases:
                n = aliases[id(n)][0]
            return n

        self._input_index = {id(n): i for i, n in enumerate(g.inputs)}
        # caller-visible outputs: every escaping kernel output, in kernel order
        escaping = {id(resolve(o)) for o in g.outputs}
        self._out_nodes: list[Node] = []
        for k in program.kernels:
            for o in k.outputs:
                if id(o) in escaping and o not in self._out_nodes:
                    self._out_nodes.append(o)
        out_index = {id(o): j for j, o in enumerate(self._out_nodes)}

        def ref(n: Node) -> tuple[int, int]:
            src = resolve(n)
            if id(src) in self._input_index:
                return rt.INPUT, self._input_index[id(src)]
            if id(src) in out_index:
                return rt.OUTPUT, out_index[id(src)]
            off = self.plan.offset(src)
            if off is None:
                raise RuntimeError(f"{n.name} has no storage (not an input, output or planned intermediate)")
            return rt.ARENA, off

        args = [[ref(i) for i in k.inputs] + [ref(o) for o in k.outputs] for k in program.kernels]
        self.native = rt.Program(so_path, symbols, args, self.plan.arena_bytes, len(g.inputs), len(self._out_nodes))
        self.device = "cpu"
        self.backend = "c-native"
        self.so_path = so_path
        self.symbols = list(symbols)
        self.source = ""            # filled in by compile_graph from the C backend
        self.matmul_configs = {}
        self.pool = BufferPool()
        self._resolve = resolve

    @property
    def num_kernels(self) -> int:
        return self.native.num_kernels

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        g = self.program.graph
        if len(inputs) != len(g.inputs):
            raise TypeError(f"expected {len(g.inputs)} inputs, got {len(inputs)}")
        ins = [t.contiguous() for t in inputs]
        outs = [self.pool.take_output(o.shape, TORCH_DTYPE[o.dtype], "cpu") for o in self._out_nodes]
        self.native.run([t.data_ptr() for t in ins], [t.data_ptr() for t in outs])
        env = {id(o): t for o, t in zip(self._out_nodes, outs)}
        env.update({id(n): t for n, t in zip(g.inputs, ins)})
        result = []
        for o in g.outputs:
            src = self._resolve(o)
            if id(src) in env:
                t = env[id(src)]
            elif src.op == "const":
                t = torch.full(src.shape, src.attrs["value"], dtype=TORCH_DTYPE[src.dtype])
            else:
                raise RuntimeError(f"output {o.name} was not produced")
            result.append(t.reshape(o.shape) if tuple(t.shape) != tuple(o.shape) else t)
        return result

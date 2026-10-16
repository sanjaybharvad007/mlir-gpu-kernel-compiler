"""Top-level pipeline: trace -> optimise -> fuse -> backend codegen -> cached callable."""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .frontend.tracer import tensor_type_of, trace
from .ir import Graph
from .passes import Program, fuse, optimize, row_fuse
from .runtime import CompiledProgram

log = logging.getLogger("tensorjit")

BACKENDS = ("c", "triton", "mlir", "cuda")


def pick_backend(device: str, requested: str | None = None) -> str:
    requested = requested or os.environ.get("TENSORJIT_BACKEND")
    if requested:
        assert requested in BACKENDS, f"unknown backend {requested}"
        return requested
    if device == "cuda" or os.environ.get("TRITON_INTERPRET") == "1":
        return "triton"
    return "c"


ROW_FUSION_MAX_N = {"c": None, "triton": 8192, "cuda": 8192, "mlir": None}


def lower(graph: Graph, backend: str = "c") -> Program:
    """IR -> optimised IR -> fused kernel program."""
    program = fuse(optimize(graph))
    if os.environ.get("TENSORJIT_NO_ROW_FUSION") != "1":
        program = row_fuse(program, max_n=ROW_FUSION_MAX_N.get(backend))
    return program


def compile_graph(graph: Graph, backend: str, device: str = "cpu", tune: bool = True) -> CompiledProgram:
    program = lower(graph, backend)
    if backend == "c":
        from .backends import c_backend

        cp = c_backend.compile_program(program, tune=tune)
        if os.environ.get("TENSORJIT_NATIVE_RUNTIME", "1") == "1":
            try:
                from .runtime.native import NativeProgram

                native = NativeProgram(program, cp.so_path, cp.symbols)
                native.source, native.matmul_configs = cp.source, cp.matmul_configs
                return native
            except Exception as e:  # no C++ compiler / pybind11: keep the Python executor
                log.warning("native runtime unavailable, using Python executor (%s)", e)
        return cp
    if backend == "triton":
        from .backends import triton_backend

        return triton_backend.compile_program(program, device=device, tune=tune)
    if backend == "cuda":
        from .backends import cuda_backend

        return cuda_backend.compile_program(program, device=device, tune=tune)
    if backend == "mlir":
        from .backends import mlir_backend

        return mlir_backend.compile_program(program)
    raise ValueError(backend)


@dataclass
class CompileStats:
    compiles: int = 0
    hits: int = 0
    compile_time_s: float = 0.0
    kernels: dict[str, int] = field(default_factory=dict)


class JitFunction:
    """Shape/dtype-specialising JIT wrapper around a Python function."""

    def __init__(self, fn: Callable, backend: str | None = None, tune: bool = True):
        self.fn = fn
        self.backend = backend
        self.tune = tune
        self.cache: dict[tuple, CompiledProgram] = {}
        self.stats = CompileStats()
        self._lock = threading.Lock()
        functools.update_wrapper(self, fn)

    def _key(self, args: Sequence[torch.Tensor]) -> tuple:
        dev = args[0].device.type
        return (tuple((tuple(a.shape), a.dtype) for a in args), dev, pick_backend(dev, self.backend))

    def trace(self, *args: torch.Tensor) -> Graph:
        return trace(self.fn, [tensor_type_of(a) for a in args], name=getattr(self.fn, "__name__", "main"))

    def program(self, *args: torch.Tensor) -> Program:
        return lower(self.trace(*args), self._key(args)[2])

    def compiled(self, *args: torch.Tensor) -> CompiledProgram:
        key = self._key(args)
        with self._lock:
            cp = self.cache.get(key)
            if cp is not None:
                self.stats.hits += 1
                return cp
            t0 = time.perf_counter()
            graph = self.trace(*args)
            backend = key[2]
            cp = compile_graph(graph, backend, device=key[1], tune=self.tune)
            self.stats.compiles += 1
            self.stats.compile_time_s += time.perf_counter() - t0
            self.stats.kernels[str(key)] = cp.num_kernels
            self.cache[key] = cp
            return cp

    def __call__(self, *args: torch.Tensor):
        if not args or not all(isinstance(a, torch.Tensor) for a in args):
            raise TypeError("jit functions take torch.Tensor positional arguments only")
        outs = self.compiled(*args)(*args)
        return outs[0] if len(outs) == 1 else tuple(outs)


def jit(fn: Callable | None = None, *, backend: str | None = None, tune: bool = True):
    """Decorator: ``@tensorjit.jit`` or ``@tensorjit.jit(backend="triton")``."""
    if fn is None:
        return lambda f: JitFunction(f, backend=backend, tune=tune)
    return JitFunction(fn, backend=backend, tune=tune)

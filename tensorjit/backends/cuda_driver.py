"""Minimal ctypes bindings to NVRTC and the CUDA driver API.

Enough to: compile a CUDA C++ source string to PTX (NVRTC), load it as a
module, launch kernels on PyTorch's current stream, and capture a sequence of
launches into a CUDA graph.  No build step and no dependency beyond the CUDA
toolkit's shared libraries (``libnvrtc`` for compilation, ``libcuda`` which is
always present with the driver).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Sequence

import torch

_nvrtc = None
_cuda = None


class CudaError(RuntimeError):
    pass


def _load(name: str, candidates: Sequence[str]):
    for c in candidates:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    found = ctypes.util.find_library(name)
    if found:
        return ctypes.CDLL(found)
    raise CudaError(f"could not load lib{name}; is the CUDA toolkit installed?")


def nvrtc():
    global _nvrtc
    if _nvrtc is None:
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        _nvrtc = _load("nvrtc", ["libnvrtc.so", f"{home}/lib64/libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13"])
    return _nvrtc


def cuda():
    global _cuda
    if _cuda is None:
        _cuda = _load("cuda", ["libcuda.so.1", "libcuda.so"])
        torch.cuda.init()  # make sure a primary context exists before we use the driver API
    return _cuda


def _check_nvrtc(res: int, prog=None):
    if res != 0:
        msg = nvrtc().nvrtcGetErrorString(res)
        nvrtc().nvrtcGetErrorString.restype = ctypes.c_char_p
        detail = ""
        if prog is not None:
            n = ctypes.c_size_t()
            nvrtc().nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value)
            nvrtc().nvrtcGetProgramLog(prog, buf)
            detail = "\n" + buf.value.decode(errors="replace")
        raise CudaError(f"NVRTC error {res}: {msg}{detail}")


def _check_cu(res: int, what: str = ""):
    if res != 0:
        s = ctypes.c_char_p()
        cuda().cuGetErrorString(res, ctypes.byref(s))
        raise CudaError(f"CUDA driver error {res} ({s.value.decode() if s.value else '?'}) {what}")


def compute_capability(device: int = 0) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def compile_to_ptx(source: str, name: str = "tensorjit.cu", extra_opts: Sequence[str] = ()) -> bytes:
    lib = nvrtc()
    prog = ctypes.c_void_p()
    _check_nvrtc(lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None))
    major, minor = compute_capability()
    opts = [f"--gpu-architecture=compute_{major}{minor}", "--std=c++17", "-default-device", "--use_fast_math",
            "-DTJ_NVRTC=1", *extra_opts]
    if os.environ.get("TENSORJIT_CUDA_DEBUG") == "1":
        opts += ["-lineinfo"]
    c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    res = lib.nvrtcCompileProgram(prog, len(opts), c_opts)
    _check_nvrtc(res, prog)
    n = ctypes.c_size_t()
    _check_nvrtc(lib.nvrtcGetPTXSize(prog, ctypes.byref(n)))
    buf = ctypes.create_string_buffer(n.value)
    _check_nvrtc(lib.nvrtcGetPTX(prog, buf))
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


class Module:
    """A loaded PTX module; kernels are launched on PyTorch's current stream."""

    def __init__(self, ptx: bytes):
        lib = cuda()
        self.handle = ctypes.c_void_p()
        _check_cu(lib.cuModuleLoadData(ctypes.byref(self.handle), ptx), "cuModuleLoadData")
        self._fns: dict[str, ctypes.c_void_p] = {}

    def function(self, name: str) -> ctypes.c_void_p:
        if name not in self._fns:
            fn = ctypes.c_void_p()
            _check_cu(cuda().cuModuleGetFunction(ctypes.byref(fn), self.handle, name.encode()), f"cuModuleGetFunction {name}")
            self._fns[name] = fn
        return self._fns[name]

    def launch(self, name: str, grid: tuple[int, int, int], block: tuple[int, int, int], args: Sequence[int],
               shared_bytes: int = 0, stream: int | None = None) -> None:
        """args: device pointers / integers, all passed as 64-bit values."""
        fn = self.function(name)
        cargs = [ctypes.c_uint64(int(a)) for a in args]
        ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(a) for a in cargs])
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        _check_cu(cuda().cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_bytes,
                                        ctypes.c_void_p(stream), ptrs, None), f"cuLaunchKernel {name}")

    def __del__(self):
        try:
            if self.handle and cuda():
                cuda().cuModuleUnload(self.handle)
        except Exception:
            pass


class Graph:
    """Captures a callable's launches on a side stream and replays them with one launch."""

    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 2

    def __init__(self):
        self.graph = ctypes.c_void_p()
        self.exec = ctypes.c_void_p()
        self.stream = torch.cuda.Stream()

    def capture(self, fn) -> None:
        lib = cuda()
        s = ctypes.c_void_p(self.stream.cuda_stream)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            _check_cu(lib.cuStreamBeginCapture(s, self.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture")
            try:
                fn(self.stream.cuda_stream)
            finally:
                _check_cu(lib.cuStreamEndCapture(s, ctypes.byref(self.graph)), "cuStreamEndCapture")
        _check_cu(lib.cuGraphInstantiateWithFlags(ctypes.byref(self.exec), self.graph, 0), "cuGraphInstantiate")

    def launch(self, stream: int | None = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        _check_cu(cuda().cuGraphLaunch(self.exec, ctypes.c_void_p(stream)), "cuGraphLaunch")

    def __del__(self):
        try:
            if self.exec:
                cuda().cuGraphExecDestroy(self.exec)
            if self.graph:
                cuda().cuGraphDestroy(self.graph)
        except Exception:
            pass


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        nvrtc()
        cuda()
        return True
    except CudaError:
        return False

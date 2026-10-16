"""Per-call overhead: Python executor vs native C++ runtime (CPU) and, on a GPU,
eager launches vs CUDA-graph replay for the CUDA backend.

Small graphs with many kernels are where framework overhead dominates; this is
the regime CUDA graphs and native runtimes exist for.

    python benchmarks/bench_overhead.py            # CPU part always runs
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tensorjit as tj  # noqa: E402


def chain(x, w):
    y = x
    for _ in range(8):
        y = tj.softmax(tj.matmul(y, w) + 0.5)
    return y


def timeit(f, *args, iters=500):
    for _ in range(50):
        f(*args)
    if torch.cuda.is_available() and args[0].is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        f(*args)
    if torch.cuda.is_available() and args[0].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    x, w = torch.randn(16, 32), torch.randn(32, 32)
    os.environ["TENSORJIT_NATIVE_RUNTIME"] = "0"
    py = tj.jit(chain, backend="c")
    py(x, w)
    os.environ["TENSORJIT_NATIVE_RUNTIME"] = "1"
    nat = tj.jit(chain, backend="c")
    nat(x, w)
    n = nat.compiled(x, w).num_kernels
    print(f"CPU, {n} kernels per call:")
    print(f"  python executor : {timeit(py, x, w):7.1f} us/call")
    print(f"  native runtime  : {timeit(nat, x, w):7.1f} us/call  (C++ side {nat.compiled(x, w).native.mean_us:.1f} us)")
    eager = lambda x, w: __import__("functools").reduce(lambda y, _: torch.softmax(y @ w + 0.5, -1), range(8), x)
    print(f"  torch eager     : {timeit(eager, x, w):7.1f} us/call")

    if torch.cuda.is_available():
        xg, wg = x.cuda(), w.cuda()
        os.environ["TENSORJIT_CUDA_GRAPH"] = "0"
        launches = tj.jit(chain, backend="cuda")
        launches(xg, wg)
        os.environ["TENSORJIT_CUDA_GRAPH"] = "1"
        graph = tj.jit(chain, backend="cuda")
        graph(xg, wg)
        print(f"GPU, {n} kernels per call:")
        print(f"  cuda backend, per-kernel launches : {timeit(launches, xg, wg):7.1f} us/call")
        print(f"  cuda backend, CUDA graph replay   : {timeit(graph, xg, wg):7.1f} us/call")
        print(f"  torch eager                       : {timeit(eager, xg, wg):7.1f} us/call")
        triton = tj.jit(chain, backend="triton")
        triton(xg, wg)
        print(f"  triton backend                    : {timeit(triton, xg, wg):7.1f} us/call")


if __name__ == "__main__":
    main()

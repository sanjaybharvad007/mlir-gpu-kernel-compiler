"""tensorjit vs PyTorch eager vs torch.compile (inductor) on representative
transformer-block shapes.

    python benchmarks/bench.py --device cuda --dtype fp32 --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tensorjit as tj  # noqa: E402

WORKLOADS = {
    # name: (tensorjit fn, torch fn, shape factory)
    "bias_gelu": (
        lambda x, b: tj.gelu(x + b),
        lambda x, b: F.gelu(x + b),
        lambda T, D: [(T, 4 * D), (4 * D,)],
    ),
    "softmax": (
        lambda s: tj.softmax(s),
        lambda s: torch.softmax(s, -1),
        lambda T, D: [(32, T, T)],
    ),
    "layernorm": (
        lambda x, g, b: tj.layer_norm(x, g, b),
        lambda x, g, b: F.layer_norm(x, (x.shape[-1],), g, b),
        lambda T, D: [(T, D), (D,), (D,)],
    ),
    "rmsnorm": (
        lambda x, g: tj.rms_norm(x, g),
        lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g,
        lambda T, D: [(T, D), (D,)],
    ),
    "linear_gelu": (
        lambda x, w, b: tj.gelu(tj.linear(x, w, b)),
        lambda x, w, b: F.gelu(F.linear(x, w, b)),
        lambda T, D: [(T, D), (4 * D, D), (4 * D,)],
    ),
    "mlp_block": (
        lambda x, g, w1, b1, w2, b2: x + tj.linear(tj.gelu(tj.linear(tj.layer_norm(x, g), w1, b1)), w2, b2),
        lambda x, g, w1, b1, w2, b2: x + F.linear(F.gelu(F.linear(F.layer_norm(x, (x.shape[-1],), g), w1, b1)), w2, b2),
        lambda T, D: [(T, D), (D,), (4 * D, D), (4 * D,), (D, 4 * D), (D,)],
    ),
}


def timeit(fn, device, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--seq", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    p.add_argument("--no-inductor", action="store_true")
    p.add_argument("--backend", default=None, help="tensorjit backend: c | triton | cuda (default: auto)")
    p.add_argument("--out", default="tensorjit_bench.csv")
    a = p.parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    rows = []
    for name in a.workloads:
        tj_fn, torch_fn, shapes = WORKLOADS[name]
        jf = tj.jit(tj_fn, backend=a.backend)
        ind = None if a.no_inductor else torch.compile(torch_fn)
        for T in a.seq:
            args = [torch.randn(*s, device=a.device, dtype=dtype) for s in shapes(T, a.hidden)]
            if name == "softmax":
                args = [torch.randn(8, min(T, 2048), min(T, 2048), device=a.device, dtype=dtype)]
            out = jf(*args)
            ref = torch_fn(*args)
            err = (out.float() - ref.float()).abs().max().item()
            t_eager = timeit(lambda: torch_fn(*args), a.device)
            t_tj = timeit(lambda: jf(*args), a.device)
            t_ind = timeit(lambda: ind(*args), a.device) if ind is not None else float("nan")
            nk = jf.compiled(*args).num_kernels
            rows.append(dict(workload=name, seq=T, eager_ms=t_eager, inductor_ms=t_ind, tensorjit_ms=t_tj,
                             speedup_vs_eager=t_eager / t_tj, kernels=nk, max_abs_err=err))
            print(f"{name:12s} T={T:5d}  eager {t_eager:8.3f} ms  inductor {t_ind:8.3f} ms  "
                  f"tensorjit {t_tj:8.3f} ms  ({t_eager / t_tj:4.2f}x, {nk} kernels, err {err:.1e})")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Cost-model-guided autotuner for matmul tile parameters.

1. A roofline-style analytic cost model ranks candidate tile configs: it
   estimates bytes moved from the next level of the memory hierarchy per flop
   (the arithmetic intensity of an (TM x TN x TK) tile) and penalises register
   pressure and tiles that don't divide the problem.
2. Only the top ``k`` candidates are compiled and timed on real buffers.
3. The winner is persisted so later processes warm-start without tuning.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .cache import AutotuneStore


@dataclass(frozen=True)
class MatmulConfig:
    TM: int
    TN: int
    TK: int = 0        # 0 = full K (C backend); Triton uses a real TK
    num_warps: int = 4

    def as_dict(self) -> dict:
        return {"TM": self.TM, "TN": self.TN, "TK": self.TK, "num_warps": self.num_warps}

    @staticmethod
    def from_dict(d: dict) -> "MatmulConfig":
        return MatmulConfig(d["TM"], d["TN"], d.get("TK", 0), d.get("num_warps", 4))


C_CANDIDATES: list[MatmulConfig] = [
    MatmulConfig(4, 16), MatmulConfig(4, 32), MatmulConfig(8, 16), MatmulConfig(8, 32),
    MatmulConfig(4, 64), MatmulConfig(8, 64), MatmulConfig(16, 32), MatmulConfig(2, 64),
    MatmulConfig(4, 4), MatmulConfig(4, 8), MatmulConfig(6, 4), MatmulConfig(8, 4), MatmulConfig(2, 8),
]

TRITON_CANDIDATES: list[MatmulConfig] = [
    MatmulConfig(64, 64, 32, 4), MatmulConfig(128, 64, 32, 4), MatmulConfig(64, 128, 32, 4),
    MatmulConfig(128, 128, 32, 8), MatmulConfig(128, 128, 64, 8), MatmulConfig(64, 64, 64, 4),
    MatmulConfig(32, 32, 32, 2), MatmulConfig(128, 256, 64, 8), MatmulConfig(256, 128, 64, 8),
]


def cost_model(cfg: MatmulConfig, M: int, N: int, K: int, backend: str) -> float:
    """Lower is better.  Units are arbitrary; only the ranking matters."""
    tm, tn = cfg.TM, cfg.TN
    tk = cfg.TK or K
    flops_per_tile = 2.0 * tm * tn * tk
    bytes_per_tile = 4.0 * (tm * tk + tk * tn)          # A and B tile loads
    intensity = flops_per_tile / bytes_per_tile          # flop/byte
    cost = 1.0 / intensity
    # waste from partial tiles at the edges
    eff = (M / (-(-M // tm) * tm)) * (N / (-(-N // tn) * tn))
    cost /= eff
    if backend == "c_dot":
        # register-blocked dot microkernel: TM*TN vector accumulators + TM A vectors
        regs = tm * tn + tm
        if regs > 14:  # AVX2 has 16 ymm registers; spills are catastrophic here
            cost *= (regs / 14.0) ** 2
    elif backend == "c":
        # broadcast-FMA kernel: TN/8 vectors per row of the accumulator tile
        regs = tm * tn / 8.0
        if regs > 12:
            cost *= 1.0 + (regs - 12) / 4.0
        # too few rows per thread => poor parallel efficiency
    if backend in ("c", "c_dot"):
        if M / tm < 2 * (os.cpu_count() or 1):
            cost *= 1.3
    else:
        # occupancy: huge tiles starve the SM for small problems
        tiles = -(-M // tm) * -(-N // tn)
        if tiles < 64:
            cost *= 1.0 + (64 - tiles) / 64.0
        if tm * tn * 4 / 1024 > 256:  # acc bigger than ~256KB of registers
            cost *= 2.0
    return cost


def autotune(
    key: str,
    candidates: Sequence[MatmulConfig],
    build: Callable[[MatmulConfig], Callable[[], None]],
    shape: tuple[int, int, int],
    backend: str,
    top_k: int = 4,
    store: AutotuneStore | None = None,
    warmup: int = 2,
    iters: int = 5,
) -> MatmulConfig:
    """Pick the fastest config; ``build(cfg)`` returns a zero-arg runner."""
    store = store or AutotuneStore()
    if os.environ.get("TENSORJIT_NO_AUTOTUNE") == "1":
        return min(candidates, key=lambda c: cost_model(c, *shape, backend))
    cached = store.get(key)
    if cached is not None:
        return MatmulConfig.from_dict(cached["config"])

    ranked = sorted(candidates, key=lambda c: cost_model(c, *shape, backend))[:top_k]
    best, best_ms = ranked[0], float("inf")
    for cfg in ranked:
        try:
            run = build(cfg)
            for _ in range(warmup):
                run()
            t = time.perf_counter()
            for _ in range(iters):
                run()
            ms = (time.perf_counter() - t) / iters * 1e3
        except Exception:  # config invalid for this backend/shape
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    store.put(key, best.as_dict(), best_ms)
    return best

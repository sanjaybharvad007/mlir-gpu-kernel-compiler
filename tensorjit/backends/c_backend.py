"""C / OpenMP backend: fused kernels -> specialised C -> gcc -O3 -> ctypes.

Every shape and stride is a compile-time constant in the generated code, so
the C compiler can fully unroll inner loops and vectorise them.  Kernel source
is hashed and compiled shared objects are cached on disk.
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import torch

from ..ir import DType, Node
from ..passes import Kernel, Program
from ..runtime import (
    C_CANDIDATES,
    AutotuneStore,
    CompiledProgram,
    MatmulConfig,
    autotune,
    cache_dir,
    content_hash,
    cost_model,
)
from .common import access_strides, contiguous_strides, decompose, emit_expr, index_expr

PRELUDE = """\
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define TJ_MIN(a, b) ((a) < (b) ? (a) : (b))
#define TJ_EXPORT __attribute__((visibility("default")))
/* Vectorisable erf (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).  glibc's erff is a
   scalar call that defeats SIMD; expf is vectorised through libmvec with -ffast-math. */
static inline float tj_erff(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  /* |x| >= 4 is erf == +-1 to fp32 precision; clamping also keeps expf(-x*x) out of
     the underflow range, where libmvec falls back to a slow scalar path. */
  x = x < 0.0f ? -x : x;
  x = x > 4.0f ? 4.0f : x;
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t
                          + 0.254829592f) * t * expf(-x * x);
  return sign * y;
}
"""


@dataclass
class _Gen:
    """Codegen state for one kernel."""

    k: Kernel
    idx: int
    names: dict[int, str]  # node id -> C variable

    def arg_decls(self) -> list[str]:
        lines = []
        for i, n in enumerate(self.k.inputs):
            lines.append(f"  const float* __restrict__ {self.names[id(n)]}_p = (const float*)args[{i}];")
        for j, n in enumerate(self.k.outputs):
            lines.append(f"  float* __restrict__ out{j}_p = (float*)args[{len(self.k.inputs) + j}];")
        return lines


def _check_dtype(k: Kernel) -> None:
    for n in k.nodes + k.inputs:
        if n.dtype != DType.f32:
            raise NotImplementedError(f"C backend supports f32 only (got {n.dtype} at {n.name})")


def _loads(g: _Gen, consumer_nodes: list[Node], iter_shape, idx_vars: list[str]) -> list[str]:
    """Load every kernel input at the current multi-index."""
    lines = []
    for n in g.k.inputs:
        consumer = next((c for c in consumer_nodes if n in c.inputs), consumer_nodes[0])
        st = access_strides(n, consumer, iter_shape)
        lines.append(f"const float {g.names[id(n)]} = {g.names[id(n)]}_p[{index_expr(idx_vars, st)}];")
    return lines


def _body(g: _Gen, nodes: list[Node]) -> list[str]:
    lines = []
    for n in nodes:
        if n.op in ("transpose", "broadcast_to"):
            lines.append(f"const float {g.names[id(n)]} = {g.names[id(n.inputs[0])]};")
        else:
            lines.append(f"const float {g.names[id(n)]} = {emit_expr(n, g.names, 'c')};")
    return lines


def _stores(g: _Gen, idx_vars: list[str], iter_shape) -> list[str]:
    cs = contiguous_strides(iter_shape)
    return [f"out{j}_p[{index_expr(idx_vars, cs)}] = {g.names[id(o)]};" for j, o in enumerate(g.k.outputs)]


def _nested_loops(iter_shape, inner: list[str], parallel: bool = True) -> list[str]:
    rank = len(iter_shape)
    idx = [f"i{d}" for d in range(rank)]
    lines = []
    outer = max(rank - 1, 0)
    if parallel and outer >= 1 and iter_shape[0] * (iter_shape[1] if rank > 2 else 1) >= 4:
        lines.append(f"  #pragma omp parallel for collapse({min(outer, 2)}) schedule(static)")
    for d in range(rank):
        if d == rank - 1:
            lines.append("  " * (d + 1) + "#pragma omp simd")
        lines.append("  " * (d + 1) + f"for (long {idx[d]} = 0; {idx[d]} < {iter_shape[d]}; ++{idx[d]}) {{")
    lines += ["  " * (rank + 1) + s for s in inner]
    for d in reversed(range(rank)):
        lines.append("  " * (d + 1) + "}")
    return lines, idx


def gen_elementwise(g: _Gen) -> str:
    shape = g.k.iter_shape
    idx = [f"i{d}" for d in range(len(shape))]
    inner = _loads(g, g.k.nodes, shape, idx) + _body(g, g.k.nodes) + _stores(g, idx, shape)
    loops, _ = _nested_loops(shape, inner)
    return "\n".join([f"TJ_EXPORT void kernel_{g.idx}(void** args) {{", *g.arg_decls(), *loops, "}"])


def gen_reduce(g: _Gen) -> str:
    k = g.k
    red = k.anchor
    x = red.inputs[0]
    in_shape = x.shape
    N = in_shape[-1]
    rows = math.prod(in_shape[:-1]) if len(in_shape) > 1 else 1
    prologue = [n for n in k.nodes if n is not red and k.nodes.index(n) < k.nodes.index(red)]
    epilogue = [n for n in k.nodes if k.nodes.index(n) > k.nodes.index(red)]
    acc_init = {"sum": "0.0f", "mean": "0.0f", "max": "-INFINITY"}[red.op]
    omp_red = {"sum": "+", "mean": "+", "max": "max"}[red.op]
    update = {"sum": "acc += {v};", "mean": "acc += {v};", "max": "acc = fmaxf(acc, {v});"}[red.op]

    dec_stmts, row_idx = decompose("r", in_shape[:-1], "i") if len(in_shape) > 1 else ([], [])
    idx_in = row_idx + ["n"]
    # loads for the prologue (inputs read inside the N loop)
    pro_inputs = [i for i in k.inputs if any(i in n.inputs for n in prologue + [red])]
    epi_inputs = [i for i in k.inputs if i not in pro_inputs]
    inner = []
    for n in pro_inputs:
        consumer = next((c for c in prologue if n in c.inputs), red)
        st = access_strides(n, consumer, in_shape)
        inner.append(f"const float {g.names[id(n)]} = {g.names[id(n)]}_p[{index_expr(idx_in, st)}];")
    inner += _body(g, prologue)
    inner.append(update.format(v=g.names[id(x)]))

    out_shape = red.shape
    out_idx = row_idx + (["0"] if red.attrs.get("keepdim") else [])
    tail = []
    for n in epi_inputs:
        consumer = next((c for c in epilogue if n in c.inputs), epilogue[0])
        st = access_strides(n, consumer, out_shape)
        tail.append(f"const float {g.names[id(n)]} = {g.names[id(n)]}_p[{index_expr(out_idx, st)}];")
    tail += _body(g, epilogue)
    stores = [f"out{j}_p[r] = {g.names[id(o)]};" for j, o in enumerate(k.outputs)]

    lines = [f"TJ_EXPORT void kernel_{g.idx}(void** args) {{", *g.arg_decls()]
    lines.append("  #pragma omp parallel for schedule(static)")
    lines.append(f"  for (long r = 0; r < {rows}; ++r) {{")
    lines += ["    " + s for s in dec_stmts]
    lines.append(f"    float acc = {acc_init};")
    lines.append(f"    #pragma omp simd reduction({omp_red}:acc)")
    lines.append(f"    for (long n = 0; n < {N}; ++n) {{")
    lines += ["      " + s for s in inner]
    lines.append("    }")
    if red.op == "mean":
        lines.append(f"    acc = acc / {float(N)!r}f;")
    lines.append(f"    const float {g.names[id(red)]} = acc;")
    lines += ["    " + s for s in tail + stores]
    lines += ["  }", "}"]
    return "\n".join(lines)


def gen_matmul(g: _Gen, cfg: MatmulConfig, suffix: str = "") -> str:
    k = g.k
    mm = k.anchor
    a, b = mm.inputs
    M, K = a.shape[-2], a.shape[-1]
    tb = bool(mm.attrs.get("trans_b"))
    N = b.shape[-2] if tb else b.shape[-1]
    out_shape = mm.shape
    batch_shape = out_shape[:-2]
    batch = 1
    for s in batch_shape:
        batch *= s
    TM, TN = cfg.TM, cfg.TN
    epilogue = [n for n in k.nodes if n is not mm]

    # batch strides (0 for broadcast batch dims)
    def bstrides(t: Node):
        cs = contiguous_strides(t.shape)
        pad = len(batch_shape) - (len(t.shape) - 2)
        st = [0] * pad
        for d in range(len(t.shape) - 2):
            st.append(0 if t.shape[d] == 1 and batch_shape[pad + d] != 1 else cs[d])
        return tuple(st)

    dec, bidx = decompose("bi", batch_shape, "b") if batch_shape else ([], [])
    a_off = index_expr(bidx, bstrides(a)) if batch_shape else "0"
    b_off = index_expr(bidx, bstrides(b)) if batch_shape else "0"
    out_idx = bidx + ["i", "j"]
    A, B = g.names[id(a)], g.names[id(b)]
    epi_inputs = [i for i in k.inputs if i is not a and i is not b]
    tail = []
    for n in epi_inputs:
        consumer = next((c for c in epilogue if n in c.inputs), epilogue[0])
        st = access_strides(n, consumer, out_shape)
        tail.append(f"const float {g.names[id(n)]} = {g.names[id(n)]}_p[{index_expr(out_idx, st)}];")
    tail += _body(g, epilogue)
    cs = contiguous_strides(out_shape)
    stores = [f"out{jj}_p[{index_expr(out_idx, cs)}] = {g.names[id(o)]};" for jj, o in enumerate(k.outputs)]

    L = [f"TJ_EXPORT void kernel_{g.idx}{suffix}(void** args) {{"]
    L.append("  #pragma omp parallel")
    L.append("  {")
    # Pointer declarations live *inside* the parallel region: GCC outlines the
    # region into a separate function and shared variables lose `restrict`,
    # which blocks vectorisation of the epilogue.
    L += g.arg_decls()
    L.append(f"  float* __restrict__ scratch = (float*)malloc(sizeof(float) * {TM} * {N});")
    L.append("  #pragma omp for collapse(2) schedule(static)")
    L.append(f"  for (long bi = 0; bi < {batch}; ++bi)")
    L.append(f"  for (long i0 = 0; i0 < {M}; i0 += {TM}) {{")
    L += ["    " + s for s in dec]
    L.append(f"    const float* __restrict__ Ab = {A}_p + ({a_off});")
    L.append(f"    const float* __restrict__ Bb = {B}_p + ({b_off});")
    L.append(f"    const long imax = TJ_MIN({TM}, {M} - i0);")
    L.append(f"    for (long j0 = 0; j0 < {N}; j0 += {TN}) {{")
    L.append(f"      const long jmax = TJ_MIN({TN}, {N} - j0);")
    L.append(f"      float acc[{TM}][{TN}];")
    L.append("      memset(acc, 0, sizeof(acc));")
    if tb:
        # B is [N, K] (K contiguous), the nn.Linear layout.  Register-blocked
        # dot-product microkernel: TM x TN independent VEC-wide accumulators
        # march down K together, so each A/B vector load is reused TN/TM times
        # and the horizontal reduction happens once per output element.
        VEC = 8
        L.append(f"      float accv[{TM}][{TN}][{VEC}];")
        L.append("      memset(accv, 0, sizeof(accv));")
        L.append(f"      if (jmax == {TN} && imax == {TM}) {{")
        L.append(f"        for (long kk = 0; kk + {VEC} <= {K}; kk += {VEC}) {{")
        L.append(f"          for (long ii = 0; ii < {TM}; ++ii) {{")
        L.append(f"            const float* __restrict__ ap = Ab + (i0 + ii) * {K} + kk;")
        L.append(f"            for (long jj = 0; jj < {TN}; ++jj) {{")
        L.append(f"              const float* __restrict__ bp = Bb + (j0 + jj) * {K} + kk;")
        L.append("              #pragma omp simd")
        L.append(f"              for (long l = 0; l < {VEC}; ++l) accv[ii][jj][l] += ap[l] * bp[l];")
        L.append("            }")
        L.append("          }")
        L.append("        }")
        L.append("      } else {")
        L.append(f"        for (long kk = 0; kk + {VEC} <= {K}; kk += {VEC})")
        L.append("          for (long ii = 0; ii < imax; ++ii)")
        L.append("            for (long jj = 0; jj < jmax; ++jj)")
        L.append(f"              for (long l = 0; l < {VEC}; ++l) accv[ii][jj][l] += Ab[(i0 + ii) * {K} + kk + l] * Bb[(j0 + jj) * {K} + kk + l];")
        L.append("      }")
        L.append("      for (long ii = 0; ii < imax; ++ii)")
        L.append("        for (long jj = 0; jj < jmax; ++jj) {")
        L.append("          float s = 0.0f;")
        L.append(f"          for (long l = 0; l < {VEC}; ++l) s += accv[ii][jj][l];")
        L.append(f"          for (long kk = {K} - {K} % {VEC}; kk < {K}; ++kk) s += Ab[(i0 + ii) * {K} + kk] * Bb[(j0 + jj) * {K} + kk];")
        L.append("          acc[ii][jj] = s;")
        L.append("        }")
    else:
        L.append(f"      if (jmax == {TN} && imax == {TM}) {{")
        L.append(f"        for (long kk = 0; kk < {K}; ++kk) {{")
        L.append(f"          const float* __restrict__ brow = Bb + kk * {N} + j0;")
        L.append(f"          for (long ii = 0; ii < {TM}; ++ii) {{")
        L.append(f"            const float av = Ab[(i0 + ii) * {K} + kk];")
        L.append("            #pragma omp simd")
        L.append(f"            for (long jj = 0; jj < {TN}; ++jj) acc[ii][jj] += av * brow[jj];")
        L.append("          }")
        L.append("        }")
        L.append("      } else {")
        L.append(f"        for (long kk = 0; kk < {K}; ++kk)")
        L.append("          for (long ii = 0; ii < imax; ++ii) {")
        L.append(f"            const float av = Ab[(i0 + ii) * {K} + kk];")
        L.append(f"            for (long jj = 0; jj < jmax; ++jj) acc[ii][jj] += av * Bb[kk * {N} + j0 + jj];")
        L.append("          }")
        L.append("      }")
    L.append("      for (long ii = 0; ii < imax; ++ii)")
    L.append(f"        for (long jj = 0; jj < jmax; ++jj) scratch[ii * {N} + j0 + jj] = acc[ii][jj];")
    L.append("    }")
    # Epilogue: one contiguous, SIMD pass per output row (math calls vectorise here).
    L.append("    for (long ii = 0; ii < imax; ++ii) {")
    L.append("      const long i = i0 + ii;")
    L.append("      #pragma omp simd")
    L.append(f"      for (long j = 0; j < {N}; ++j) {{")
    L.append(f"        const float {g.names[id(mm)]} = scratch[ii * {N} + j];")
    L += ["        " + s for s in tail + stores]
    L += ["      }", "    }", "  }", "  free(scratch);", "  }", "}"]
    return "\n".join(L)


def _compile_so(source: str) -> ctypes.CDLL:
    h = content_hash(source, "c-v1")
    so = cache_dir() / f"tj_{h}.so"
    if not so.exists():
        cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            raise RuntimeError("no C compiler found (set CC)")
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "kernels.c")
            with open(src, "w") as f:
                f.write(source)
            flags = ["-O3", "-shared", "-fPIC", "-ffast-math", "-march=native", "-fopenmp", "-std=c11"]
            if sys.platform == "darwin":
                flags = [f for f in flags if f != "-fopenmp"]
            tmp_so = os.path.join(td, "k.so")
            try:
                subprocess.run([cc, *flags, src, "-o", tmp_so, "-lm"], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"C compile failed:\n{e.stderr}\n--- source ---\n{source}") from e
            shutil.move(tmp_so, so)
    return ctypes.CDLL(str(so))


def _make_fn(lib: ctypes.CDLL, name: str):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = None

    def run(ins: list[torch.Tensor], outs: list[torch.Tensor]) -> None:
        ptrs = (ctypes.c_void_p * (len(ins) + len(outs)))(
            *[t.contiguous().data_ptr() for t in ins], *[t.data_ptr() for t in outs]
        )
        fn(ptrs)

    return run


def compile_program(program: Program, tune: bool = True) -> CompiledProgram:
    gens: list[_Gen] = []
    for idx, k in enumerate(program.kernels):
        _check_dtype(k)
        names = {}
        for j, n in enumerate(k.inputs):
            names[id(n)] = f"in{j}"
        for j, n in enumerate(k.nodes):
            names[id(n)] = f"v{j}"
        gens.append(_Gen(k, idx, names))

    store = AutotuneStore()
    chosen: dict[int, MatmulConfig] = {}
    sources = [PRELUDE]
    tune_jobs = []
    for g in gens:
        k = g.k
        if k.kind in ("elementwise", "copy"):
            sources.append(gen_elementwise(g))
        elif k.kind == "reduce":
            sources.append(gen_reduce(g))
        elif k.kind == "row":
            sources.append(gen_row(g))
        elif k.kind == "matmul":
            a, b = k.anchor.inputs
            M, K = a.shape[-2], a.shape[-1]
            N = b.shape[-2] if k.anchor.attrs.get("trans_b") else b.shape[-1]
            key = f"c|matmul|{M}x{N}x{K}|tb{int(bool(k.anchor.attrs.get('trans_b')))}|epi{len(k.nodes) - 1}"
            cached = store.get(key)
            if cached is not None:
                cfg = MatmulConfig.from_dict(cached["config"])          # warm start
            elif not tune or os.environ.get("TENSORJIT_NO_AUTOTUNE") == "1":
                kind = "c_dot" if k.anchor.attrs.get("trans_b") else "c"
                cfg = min(C_CANDIDATES, key=lambda c: cost_model(c, M, N, K, kind))  # analytic pick
            else:
                cfg = None
            if cfg is not None:
                chosen[g.idx] = cfg
                sources.append(gen_matmul(g, cfg))
            else:
                for vi, cfg in enumerate(C_CANDIDATES):
                    sources.append(gen_matmul(g, cfg, suffix=f"_v{vi}"))
                tune_jobs.append((g, key, (M, N, K), "c_dot" if k.anchor.attrs.get("trans_b") else "c"))
        else:
            raise NotImplementedError(k.kind)

    lib = _compile_so("\n\n".join(sources))

    # Autotune matmuls on real buffers, then bind the winner.
    fns: dict[int, object] = {}
    for g, key, (M, N, K), kind in tune_jobs:
        k = g.k
        ins = [torch.randn(i.shape) for i in k.inputs]
        outs = [torch.empty(o.shape) for o in k.outputs]

        def build(cfg, g=g, ins=ins, outs=outs):
            vi = C_CANDIDATES.index(cfg)
            f = _make_fn(lib, f"kernel_{g.idx}_v{vi}")
            return lambda: f(ins, outs)

        cfg = autotune(key, C_CANDIDATES, build, (M, N, K), kind, store=store)
        chosen[g.idx] = cfg
        fns[g.idx] = _make_fn(lib, f"kernel_{g.idx}_v{C_CANDIDATES.index(cfg)}")

    kernel_fns = [fns.get(g.idx) or _make_fn(lib, f"kernel_{g.idx}") for g in gens]
    cp = CompiledProgram(program, kernel_fns, "cpu", "c")
    cp.source = "\n\n".join(sources)  # type: ignore[attr-defined]
    cp.matmul_configs = chosen  # type: ignore[attr-defined]
    return cp


# ----------------------------------------------------------------------------
# Row kernels (LayerNorm / softmax / RMSNorm fused into one pass per row)
# ----------------------------------------------------------------------------


def gen_row(g: _Gen) -> str:
    k = g.k
    stages: list[Kernel] = k.stages  # type: ignore[attr-defined]
    rows, N = tuple(k.rows), k.n  # type: ignore[attr-defined]
    nrows = math.prod(rows) if rows else 1
    inside = {id(n) for n in k.nodes}
    escaping = {id(o): j for j, o in enumerate(k.outputs)}
    stage_of: dict[int, int] = {id(n): si for si, st in enumerate(stages) for n in st.nodes}
    vec_shape = rows + (N,)

    def is_vec(n: Node) -> bool:
        return n.shape == vec_shape

    # N-wide values needed by a later stage get a per-thread temp (or live in
    # their output buffer when they escape anyway).
    temps: list[Node] = []
    for st_i, st in enumerate(stages):
        for n in st.nodes:
            if is_vec(n) and id(n) not in escaping and any(
                stage_of.get(id(u), st_i) != st_i for u in n.users if id(u) in inside
            ):
                temps.append(n)
    temp_idx = {id(n): i for i, n in enumerate(temps)}

    dec, row_idx = decompose("r", rows, "i") if rows else ([], [])

    def vec_elem(n: Node, nvar: str = "n") -> str:
        """Element n of an N-wide value produced by an earlier stage."""
        if id(n) in escaping:
            return f"out{escaping[id(n)]}_p[r * {N} + {nvar}]"
        return f"tmp{temp_idx[id(n)]}[{nvar}]"

    def ext_load(n: Node, consumer: Node, iter_shape, idx_vars) -> str:
        st = access_strides(n, consumer, iter_shape)
        return f"{g.names[id(n)]}_p[{index_expr(idx_vars, st)}]"

    L = [f"TJ_EXPORT void kernel_{g.idx}(void** args) {{", "  #pragma omp parallel", "  {"]
    L += g.arg_decls()
    for i in range(len(temps)):
        L.append(f"  float* __restrict__ tmp{i} = (float*)malloc(sizeof(float) * {N});")
    L.append("  #pragma omp for schedule(static)")
    L.append(f"  for (long r = 0; r < {nrows}; ++r) {{")
    L += ["    " + s for s in dec]
    scalars = [n for n in k.nodes if not is_vec(n)]
    if scalars:
        L.append("    float " + ", ".join(g.names[id(n)] for n in scalars) + ";")
    idx_vec = row_idx + ["n"]

    for st_i, st in enumerate(stages):
        L.append(f"    /* stage {st_i}: {st.kind} {[n.op for n in st.nodes]} */")
        L.append("    {")
        if st.kind == "elementwise" and st.iter_shape == vec_shape:
            body = []
            for n in st.inputs:
                consumer = next((c for c in st.nodes if n in c.inputs), st.nodes[0])
                if id(n) in inside:
                    src = vec_elem(n) if is_vec(n) else g.names[id(n)]
                    body.append(f"const float {g.names[id(n)]}_l = {src};")
                else:
                    body.append(f"const float {g.names[id(n)]}_l = {ext_load(n, consumer, vec_shape, idx_vec)};")
            local = {id(n): f"{g.names[id(n)]}_l" for n in st.inputs}
            for n in st.nodes:
                local[id(n)] = g.names[id(n)]
                body.append(f"const float {g.names[id(n)]} = {emit_expr(n, local, 'c')};")
                if id(n) in escaping:
                    body.append(f"out{escaping[id(n)]}_p[r * {N} + n] = {g.names[id(n)]};")
                elif id(n) in temp_idx:
                    body.append(f"tmp{temp_idx[id(n)]}[n] = {g.names[id(n)]};")
            L.append("    #pragma omp simd")
            L.append(f"    for (long n = 0; n < {N}; ++n) {{")
            L += ["      " + s for s in body]
            L.append("    }")
        elif st.kind == "elementwise":  # per-row scalar work: [rows] or [rows, 1]
            idx = row_idx + (["0"] if st.iter_shape == rows + (1,) else [])
            local = {}
            for n in st.inputs:
                consumer = next((c for c in st.nodes if n in c.inputs), st.nodes[0])
                if id(n) in inside:
                    local[id(n)] = g.names[id(n)]
                else:
                    L.append(f"    const float {g.names[id(n)]}_s = {ext_load(n, consumer, st.iter_shape, idx)};")
                    local[id(n)] = f"{g.names[id(n)]}_s"
            for n in st.nodes:
                local[id(n)] = g.names[id(n)]
                L.append(f"    {g.names[id(n)]} = {emit_expr(n, local, 'c')};")
                if id(n) in escaping:
                    L.append(f"    out{escaping[id(n)]}_p[r] = {g.names[id(n)]};")
        elif st.kind == "reduce":
            red = st.anchor
            pos = st.nodes.index(red)
            prologue, epilogue = st.nodes[:pos], st.nodes[pos + 1:]
            x = red.inputs[0]
            init = {"sum": "0.0f", "mean": "0.0f", "max": "-INFINITY"}[red.op]
            omp_red = {"sum": "+", "mean": "+", "max": "max"}[red.op]
            upd = {"sum": "acc += {v};", "mean": "acc += {v};", "max": "acc = fmaxf(acc, {v});"}[red.op]
            body, local = [], {}
            pro_inputs = [i for i in st.inputs if any(i in m.inputs for m in prologue + [red])]
            for n in pro_inputs:
                consumer = next((c for c in prologue if n in c.inputs), red)
                if id(n) in inside:
                    src = vec_elem(n) if is_vec(n) else g.names[id(n)]
                else:
                    src = ext_load(n, consumer, vec_shape, idx_vec)
                body.append(f"const float {g.names[id(n)]}_l = {src};")
                local[id(n)] = f"{g.names[id(n)]}_l"
            for n in prologue:
                local[id(n)] = g.names[id(n)]
                body.append(f"const float {g.names[id(n)]} = {emit_expr(n, local, 'c')};")
            body.append(upd.format(v=local[id(x)]))
            L.append(f"    float acc = {init};")
            L.append(f"    #pragma omp simd reduction({omp_red}:acc)")
            L.append(f"    for (long n = 0; n < {N}; ++n) {{")
            L += ["      " + s for s in body]
            L.append("    }")
            if red.op == "mean":
                L.append(f"    acc = acc / {float(N)!r}f;")
            L.append(f"    {g.names[id(red)]} = acc;")
            if id(red) in escaping:
                L.append(f"    out{escaping[id(red)]}_p[r] = {g.names[id(red)]};")
            idx = row_idx + (["0"] if red.attrs.get("keepdim") else [])
            local = {id(red): g.names[id(red)]}
            for n in [i for i in st.inputs if i not in pro_inputs]:
                consumer = next((c for c in epilogue if n in c.inputs), epilogue[0])
                if id(n) in inside:
                    local[id(n)] = g.names[id(n)]
                else:
                    L.append(f"    const float {g.names[id(n)]}_e = {ext_load(n, consumer, red.shape, idx)};")
                    local[id(n)] = f"{g.names[id(n)]}_e"
            for n in epilogue:
                local[id(n)] = g.names[id(n)]
                L.append(f"    {g.names[id(n)]} = {emit_expr(n, local, 'c')};")
                if id(n) in escaping:
                    L.append(f"    out{escaping[id(n)]}_p[r] = {g.names[id(n)]};")
        else:
            raise NotImplementedError(f"row stage kind {st.kind}")
        L.append("    }")
    L.append("  }")
    for i in range(len(temps)):
        L.append(f"  free(tmp{i});")
    L += ["  }", "}"]
    return "\n".join(L)

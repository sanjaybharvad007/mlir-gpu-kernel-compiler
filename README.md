# mlir-gpu-kernel-compiler (tensorjit)

A JIT tensor compiler, written to understand what `torch.compile` / Inductor /
XLA / TensorRT do under the hood. Traces tensor programs into an SSA IR,
optimises and **fuses** them, plans memory statically, and generates
**shape-specialised** kernels for four backends, executed by a C++ runtime or
as a single CUDA graph:

| Backend | Target | Codegen | Status |
|---|---|---|---|
| `c` | CPU | C11 + OpenMP + SIMD → `gcc -O3 -march=native`, run by the **C++ runtime** (pybind11, planned arena) | full |
| `cuda` | NVIDIA GPU | generated **CUDA C++** → **NVRTC** → driver API launches, optional **CUDA-graph** replay; SIMT tiled GEMM + **WMMA Tensor-Core** GEMM | codegen type-checked in CI; execution tests need a GPU |
| `triton` | NVIDIA GPU | generated Triton source (`tl.dot` on Tensor Cores) | full; runs under `TRITON_INTERPRET=1` on CPU |
| `mlir` | CPU via LLVM | `linalg` emission → `mlir-opt` → LLVM → `clang`; plus a **C++ `tj` dialect** (TableGen ops, `tj-fuse`, `tj-lower-to-linalg`, `tj-opt`) | emission always; execution needs an MLIR toolchain |

```python
import tensorjit as tj

@tj.jit
def mlp(x, g, w1, b1, w2, b2):
    h = tj.layer_norm(x, g)
    return x + tj.linear(tj.gelu(tj.linear(h, w1, b1)), w2, b2)

y = mlp(x, g, w1, b1, w2, b2)          # torch tensors in / out; compiled once per shape signature
print(mlp.program(x, g, w1, b1, w2, b2))
#   row          ['mean', 'sub', 'mul', 'mean', 'add', 'rsqrt', 'mul']   <- LayerNorm, one pass per row
#   matmul       ['matmul', 'add', 'gelu']                                <- GEMM + fused epilogue
#   matmul       ['matmul', 'add', 'add']                                 <- GEMM + bias + residual
```

Or as a `torch.compile` backend for ordinary `nn.Module`s:

```python
compiled = torch.compile(model, backend=tj.torch_compile_backend)   # falls back to eager on unsupported ops
```

## Pipeline

```
Python tracer ──┐
                ├─► SSA tensor IR ─► cleanup passes ─► fusion ─► row fusion ─► memory planner ─► backend codegen ─► runtime
FX (Dynamo) ────┘        │               │                │                      │                   │                │
                   static shapes     CSE, DCE,       elementwise chains    liveness ->        C / CUDA / Triton   C++ executor,
                   + dtypes          algebraic,      matmul epilogues      one arena,         / MLIR; tiles       CUDA graph,
                                     broadcast &     reduce prologues      offsets, 40-80%    autotuned           kernel cache
                                     transpose       whole-row programs    less peak memory   per shape
                                     folding
```

**IR** (`tensorjit/ir`): nodes are `elementwise | matmul | reduce | movement | const`
with concrete `TensorType(shape, dtype)`. Shapes are static by design: the JIT
specialises per signature, so every backend can bake sizes and strides into the
code and the C compiler / Triton can unroll and vectorise.

**Cleanup passes** (`passes/cleanup.py`): CSE, DCE, constant folding, `x*1 / x+0`
elimination, broadcast folding (numpy broadcasting is implicit in elementwise
kernels, so `broadcast_to` disappears), and **transpose-into-matmul folding**
(`x @ w.T` reads `w` with swapped strides instead of copying it — the
`nn.Linear` layout).

**Fusion** (`passes/fusion.py`):
- elementwise DAGs sharing an iteration space → one loop nest, multiple outputs allowed, no recompute;
- pointwise *epilogues* fused into the matmul that produces their input (bias, GELU, residual);
- pointwise *prologues* fused into last-axis reductions (`mean((x-μ)²)`);
- **row fusion**: chains of per-row stages (reductions over the last axis and the
  elementwise ops that broadcast their results back) collapse into one kernel —
  LayerNorm, RMSNorm and softmax each become a single pass over the row.

**Memory planner** (`passes/memory_planner.py`): after fusion every intermediate
has an interval lifetime over the kernel sequence, so buffers are assigned by
greedy best-fit over free ranges of one arena (interval-graph colouring, as in
XLA/TVM/Inductor). Verified non-overlapping; 82% peak reduction on a chained
transformer block. Graph outputs stay outside the arena.

**Runtime** (`csrc/runtime.cpp`, `runtime/native.py`): a pybind11 module that
dlopens the C backend's shared object, pre-resolves every kernel argument to an
input pointer / arena offset / output pointer, and runs the whole program with
the GIL released — one C++ call per program instead of one Python round-trip per
kernel. 3.3× lower per-call overhead on a 16-kernel graph (112 → 34 µs).

**Backends** (`backends/`):
- *C*: nested loops with constant bounds, `#pragma omp parallel for` over outer
  dims and `#pragma omp simd` on the innermost, register-blocked matmul
  microkernels (broadcast-FMA for `[K,N]` weights, dot-product form for `[N,K]`
  weights), a vectorisable `erf`, and per-thread scratch so matmul epilogues run
  as one contiguous SIMD pass per row.
- *CUDA C++* (`cuda_backend.py`, `cuda_driver.py`): grid-stride elementwise
  kernels; one-block-per-row kernels with warp-shuffle block reductions for
  LayerNorm/softmax/RMSNorm; a 64×64×16 shared-memory tiled GEMM with 4×4
  register micro-tiles and the fused epilogue in the store loop; a **WMMA
  Tensor-Core** GEMM for fp16/bf16. Compiled at run time with **NVRTC** (ctypes
  bindings, no build step), launched through the **driver API** on PyTorch's
  stream, and optionally captured once into a **CUDA graph** with static
  buffers so a call is a single `cuGraphLaunch`. Every generated program is
  type-checked as host C++ in CI via `csrc/cuda_host_stub.h`.
- *Triton*: 1-D flattened elementwise kernels with compile-time index
  decomposition, row kernels where the row lives in registers, and a tiled
  `tl.dot` matmul with the epilogue applied on the accumulator tile. Launch
  grids/`BLOCK` sizes are derived from the specialised shapes.
- *MLIR*: `linalg.matmul` / `linalg.batch_matmul` / `linalg.generic` with
  broadcasting affine maps, `linalg.transpose`, reductions as generics with
  `"reduction"` iterators; pipeline `linalg-fuse-elementwise-ops →
  one-shot-bufferize → convert-linalg-to-loops → … → llvm`, executed through the
  C interface with ctypes memref descriptors. `emit_mlir(graph, dialect="tj")`
  instead emits the high-level ops of the **C++ `tj` dialect** in [`mlir/`](mlir/):
  TableGen-defined `tj.fused_matmul` / `tj.softmax` / `tj.layer_norm` /
  `tj.rms_norm` with verifiers, a `tj-fuse` rewrite pass that folds
  `linalg.matmul + bias + activation` into `tj.fused_matmul`, a
  `tj-lower-to-linalg` pass, a `tj-opt` driver and lit/FileCheck tests.

**Runtime** (`runtime/`): on-disk kernel cache keyed by source hash; a
**cost-model-pruned autotuner** for matmul tiles (ranks candidates by tile
arithmetic intensity, register pressure and edge waste, benchmarks the top-k,
persists the winner to `~/.cache/tensorjit/autotune.json`); and a buffer pool
that recycles intermediates and — using refcounts — output buffers the caller
has dropped, which removes first-touch page-fault cost from every call.

## Results

Per-call overhead on a 16-kernel graph (CPU, `benchmarks/bench_overhead.py`):

| executor | µs / call |
|---|---|
| torch eager | 119 |
| tensorjit, Python executor | 112 |
| tensorjit, **C++ runtime + planned arena** | **34** |

GPU (fill in from `benchmarks/bench.py --device cuda --backend cuda` and `bench_overhead.py`):

| workload | eager | inductor | tensorjit/triton | tensorjit/cuda | tensorjit/cuda + CUDA graph |
|---|---|---|---|---|---|
| mlp_block T=4096 | | | | | |
| softmax 8×2048² | | | | | |

CPU kernels (2 cores, fp32, hidden=512):

| workload | seq | eager | tensorjit | kernels |
|---|---|---|---|---|
| bias + GELU | 1024 | 4.25 ms | **1.16 ms** (3.7×) | 1 |
| softmax (8×1024×1024) | 1024 | 14.1 ms | **4.3 ms** (3.3×) | 1 |
| RMSNorm | 1024 | 0.40 ms | **0.09 ms** (4.5×) | 1 |
| LayerNorm | 1024 | 0.13 ms | 0.23 ms | 1 |
| Linear + GELU (512→2048) | 1024 | 14.2 ms (MKL) | 26.3 ms | 1 |
| MLP block | 1024 | 20.5 ms | 43.6 ms | 3 |

Fused pointwise/row work beats eager by 3–4×; the generated C GEMM reaches
~½ of MKL on this box (register-blocked, no packing). On GPU the Triton backend
uses Tensor Cores through `tl.dot`; run `benchmarks/bench.py --device cuda`
and put the numbers here.

## Install / test

```bash
pip install -e .                     # torch + a C compiler is all the C backend needs
pytest -q                            # C backend + C++ runtime, planner, CUDA codegen (host-checked), MLIR, torch.compile (~85 tests)
TRITON_INTERPRET=1 pytest -q tests/test_backends.py -k triton   # Triton codegen on CPU
pytest -q tests/test_cuda_backend.py                              # on a GPU: NVRTC build, CUDA graph replay, WMMA GEMM
TENSORJIT_DUMP=1 python examples/mlp.py                          # print IR after every pass
```

Env knobs: `TENSORJIT_BACKEND=c|cuda|triton|mlir`, `TENSORJIT_NATIVE_RUNTIME=0`
(Python executor), `TENSORJIT_CUDA_GRAPH=0`, `TENSORJIT_CUDA_WMMA=0`,
`TENSORJIT_NO_AUTOTUNE=1`, `TENSORJIT_NO_ROW_FUSION=1`, `TENSORJIT_STRICT=1`
(torch.compile backend raises instead of falling back), `TENSORJIT_CACHE_DIR`.

## Layout

```
tensorjit/ir         graph.py (IR, type inference, broadcasting)  printer.py
tensorjit/passes     cleanup.py  fusion.py (+ row fusion)  memory_planner.py  pass_manager.py
tensorjit/backends   common.py  c_backend.py  cuda_backend.py  cuda_driver.py (NVRTC/driver ctypes)
                     triton_backend.py  mlir_backend.py (linalg + tj emitters)
tensorjit/runtime    native.py (C++ runtime binding)  executor.py (Python executor, buffer pool)  autotune.py  cache.py
csrc/                runtime.cpp (pybind11 executor)  cuda_host_stub.h (host type-check of generated CUDA)
mlir/                tj dialect: TableGen ops, dialect + passes (C++), tj-opt, CMake, lit tests
tests/  benchmarks/  examples/
```

## Supported ops

Elementwise: add sub mul div pow neg exp log sqrt rsqrt tanh relu sigmoid gelu silu
abs maximum minimum where (numpy broadcasting). Contractions: matmul (2-D and
batched with broadcast batch dims, transposed-B fast path). Reductions: sum max
mean over the last axis (other axes via a fused transpose). Movement: reshape
(free), transpose, broadcast_to. Composites: softmax, layer_norm, rms_norm, linear.

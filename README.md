# mlir-gpu-kernel-compiler (tensorjit)

**A JIT tensor compiler: Python tracer or TorchDynamo in, SSA tensor IR, fusion, static memory planning, and shape-specialised kernels out, for C/OpenMP, CUDA C++ (NVRTC + CUDA graphs), Triton and MLIR, with a C++ `tj` dialect.**

[![CI](https://github.com/sanjaybharvad007/mlir-gpu-kernel-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaybharvad007/mlir-gpu-kernel-compiler/actions/workflows/ci.yml)
[![MLIR build](https://github.com/sanjaybharvad007/mlir-gpu-kernel-compiler/actions/workflows/mlir.yml/badge.svg)](https://github.com/sanjaybharvad007/mlir-gpu-kernel-compiler/actions/workflows/mlir.yml)
![CUDA](https://img.shields.io/badge/CUDA-NVRTC%20%7C%20driver%20API-76B900)
![MLIR](https://img.shields.io/badge/MLIR-linalg%20%7C%20tj%20dialect-262D3A)
![Triton](https://img.shields.io/badge/Triton-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```python
import tensorjit as tj

@tj.jit
def mlp(x, g, w1, b1, w2, b2):
    h = tj.layer_norm(x, g)
    return x + tj.linear(tj.gelu(tj.linear(h, w1, b1)), w2, b2)

y = mlp(x, g, w1, b1, w2, b2)                 # torch tensors in and out; compiled once per shape signature
print(mlp.program(x, g, w1, b1, w2, b2))
#   row      ['mean', 'sub', 'mul', 'mean', 'add', 'rsqrt', 'mul']   LayerNorm: one pass per row
#   matmul   ['matmul', 'add', 'gelu']                                GEMM + fused epilogue
#   matmul   ['matmul', 'add', 'add']                                 GEMM + bias + residual

compiled = torch.compile(model, backend=tj.torch_compile_backend)   # or as a TorchDynamo backend
```

tensorjit exists to make the machinery inside `torch.compile` / Inductor, XLA and TensorRT concrete: every pass is a few hundred lines of readable code, every backend emits source you can print, and the whole pipeline is covered by 85 tests that compare against eager PyTorch.

---

## Contents

- [Highlights](#highlights)
- [Pipeline](#pipeline)
- [Backends](#backends)
- [Installation](#installation)
- [Usage](#usage)
- [Design](#design)
- [Performance](#performance)
- [Testing](#testing)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Supported operators](#supported-operators)
- [Roadmap](#roadmap)
- [References](#references)

---

## Highlights

- **SSA tensor IR** with static shapes and dtypes; the JIT specialises per signature so every backend bakes sizes and strides into code.
- **Optimisation passes**: CSE, DCE, constant and algebraic folding, broadcast folding, transpose-into-matmul folding (`x @ w.T` reads `w` with swapped strides).
- **Fusion**: elementwise DAGs into one loop nest, pointwise epilogues into the producing GEMM, pointwise prologues into reductions, and *row fusion* that turns LayerNorm, RMSNorm and softmax into a single pass per row.
- **Static memory planner**: interval liveness over the kernel sequence, best-fit arena offsets, verified non-overlap; 82% lower peak intermediate memory on a chained transformer block.
- **C++ runtime** (pybind11): one call per program with pre-resolved pointers and the GIL released; 3.3x lower per-call overhead than the Python executor.
- **CUDA C++ backend** compiled at run time with **NVRTC**, launched through the **driver API**, optionally captured into a **CUDA graph**; SIMT tiled GEMM with register micro-tiles and a **WMMA Tensor-Core** GEMM for fp16/bf16.
- **MLIR**: `linalg` emission executed through `mlir-opt`/LLVM, plus a **C++ `tj` dialect** (TableGen ops with verifiers, `tj-fuse`, `tj-lower-to-linalg`, `tj-opt`, lit tests).
- **Cost-model autotuner** for GEMM tiles with an on-disk cache, and a refcount-aware buffer pool that removes first-touch page faults from the hot path.

## Pipeline

```
 Python tracer ──┐
                 ├─► SSA tensor IR ─► cleanup ─► fusion ─► row fusion ─► memory plan ─► codegen ─► runtime
 FX (Dynamo) ────┘        │             │           │            │             │            │           │
                    static shapes   CSE, DCE,   elementwise   LayerNorm /   liveness ->   C / CUDA   C++ executor
                    + dtypes        algebraic,  chains,       softmax /     one arena,    / Triton   CUDA graph
                                    broadcast   GEMM          RMSNorm as    best-fit      / MLIR     kernel cache
                                    + transpose epilogues,    one kernel    offsets       tiles      autotuner
                                    folding     reduce        per row                     autotuned
                                                prologues                                 per shape
```

`TENSORJIT_DUMP=1` prints the IR after every pass; `fn.program(*args)` prints the fused kernel plan and `fn.compiled(*args).source` the generated code.

## Backends

| Backend | Target | Code generation | Execution | Status |
|---|---|---|---|---|
| `c` | CPU | C11 + OpenMP + SIMD pragmas, register-blocked GEMM micro-kernels, vectorisable `erf` | `gcc -O3 -march=native`, run by the C++ runtime over a planned arena | complete |
| `cuda` | NVIDIA GPU | CUDA C++: grid-stride elementwise, block-per-row reductions with warp shuffles, 64x64x16 shared-memory tiled GEMM with 4x4 register tiles, WMMA GEMM for fp16/bf16 | NVRTC -> PTX -> driver API launches on PyTorch's stream; optional one-shot CUDA-graph capture | code generation type-checked in CI as host C++; execution tests run on a GPU |
| `triton` | NVIDIA GPU | 1-D flattened elementwise kernels, register-resident row kernels, tiled `tl.dot` GEMM with the epilogue on the accumulator | Triton JIT; runs on CPU under `TRITON_INTERPRET=1` | complete |
| `mlir` | CPU via LLVM | `linalg.matmul` / `batch_matmul` / `generic` with broadcasting affine maps; or high-level `tj` dialect ops | `mlir-opt` pipeline -> LLVM -> `clang`, called through ctypes memref descriptors | emission always; execution needs an MLIR toolchain (`mlir.yml`, manual) |

## Installation

```bash
git clone https://github.com/sanjaybharvad007/mlir-gpu-kernel-compiler
cd mlir-gpu-kernel-compiler
pip install -e .                       # torch + a C compiler is all the C backend needs
pip install pybind11                   # C++ runtime (built on first use, cached by source hash)
```

Optional: a CUDA 12.x toolkit for the `cuda` backend (NVRTC and `libcuda` are loaded with ctypes, no build step), Triton 3.x for the `triton` backend, and LLVM/MLIR 18+ with CMake for the `mlir` backend and the `tj` dialect (`mlir/README.md`).

## Usage

```python
import torch, tensorjit as tj

# 1. jit a function of torch tensors
@tj.jit(backend="c")                      # c | cuda | triton | mlir; default follows the input device
def rms_norm_linear(x, g, w):
    return tj.linear(tj.rms_norm(x, g), w)

y = rms_norm_linear(x, g, w)
print(rms_norm_linear.program(x, g, w))   # kernel plan after fusion
print(rms_norm_linear.stats)              # compile time, cache hits, kernels per call

# 2. inspect the memory plan and the generated source
from tensorjit.passes.memory_planner import plan_memory
print(plan_memory(rms_norm_linear.program(x, g, w)))
#   arena 1048576 B (naive 5767168 B, 82% saved)
#   [       0, +1048576)  k0..k1  rms_norm_out (4096, 256)
#   ...
print(rms_norm_linear.compiled(x, g, w).source)

# 3. torch.compile backend for an ordinary nn.Module (unsupported ops fall back to eager)
compiled = torch.compile(model, backend=tj.torch_compile_backend)

# 4. emit MLIR from the traced graph
from tensorjit.backends.mlir_backend import emit_mlir
graph = rms_norm_linear.trace(x, g, w)
print(emit_mlir(graph))                   # linalg
print(emit_mlir(graph, dialect="tj"))     # tj.fused_matmul / tj.layer_norm / tj.softmax / tj.rms_norm
```

## Design

### IR

Nodes are `elementwise | matmul | reduce | movement | const` with a concrete `TensorType(shape, dtype)`. Shapes are static by design: the JIT specialises per signature, so the C compiler and Triton can unroll and vectorise, and the planner knows every buffer size at compile time. `Node.meta` carries non-semantic tags (for example which composite a node came from) that the `tj` emitter uses.

### Cleanup passes (`passes/cleanup.py`)

CSE, DCE, scalar constant folding, `x*1` / `x+0` elimination, broadcast folding (numpy broadcasting is implicit in elementwise kernels, so `broadcast_to` disappears) and transpose-into-matmul folding, which turns the `nn.Linear` layout `x @ w.T` into a strided read of `w` instead of a copy.

### Fusion (`passes/fusion.py`)

- Elementwise DAGs that share an iteration space become one loop nest with multiple outputs and no recompute.
- Pointwise *epilogues* fuse into the matmul that produces their input (bias, GELU, residual add).
- Pointwise *prologues* fuse into last-axis reductions (`mean((x - mu)^2)`).
- *Row fusion*: chains of per-row stages (last-axis reductions and the elementwise ops that broadcast their results back) collapse into one kernel, so LayerNorm, RMSNorm and softmax each become a single pass over the row.

### Memory planner (`passes/memory_planner.py`)

After fusion the program is a straight-line kernel sequence, so every intermediate has an interval lifetime `[def, last_use]` and buffer assignment is interval-graph colouring. Buffers are placed by greedy best-fit over the free ranges of one arena (the scheme used by XLA, TVM and Inductor); `_verify` asserts no two simultaneously live buffers overlap. Graph outputs stay outside the arena because they are handed to the caller.

### Runtime (`csrc/runtime.cpp`, `runtime/`)

A pybind11 module dlopens the C backend's shared object, pre-resolves every kernel argument to an input pointer, an arena offset or an output pointer, and runs the whole program with the GIL released: one C++ call per program instead of one Python round-trip per kernel. Around it: an on-disk kernel cache keyed by source hash, a cost-model-pruned GEMM-tile autotuner (ranks candidates by arithmetic intensity, register pressure and edge waste, benchmarks the top-k, persists the winner), and a buffer pool that recycles intermediates and, through refcounts, output buffers the caller has dropped.

### CUDA backend (`backends/cuda_backend.py`, `backends/cuda_driver.py`)

Generated CUDA C++ is compiled with NVRTC and launched through the driver API on PyTorch's current stream. Elementwise programs are grid-stride kernels; row programs are one block per row with warp-shuffle block reductions; GEMM is a 64x64x16 shared-memory tiled kernel with 4x4 register micro-tiles and the fused epilogue applied in the store loop, or a WMMA Tensor-Core kernel when the dtype is fp16/bf16 and the shape is tile-aligned. A program can be captured once into a CUDA graph with static buffers so a call is a single `cuGraphLaunch`. Every generated program is type-checked as host C++ in CI through `csrc/cuda_host_stub.h`.

### MLIR and the `tj` dialect (`backends/mlir_backend.py`, `mlir/`)

The Python emitter produces `linalg` ops with broadcasting affine maps and `"reduction"` iterators, run through `linalg-fuse-elementwise-ops -> one-shot-bufferize -> convert-linalg-to-loops -> ... -> llvm`. With `dialect="tj"` it emits the high-level ops of the C++ dialect in `mlir/`: TableGen-defined `tj.fused_matmul` (with an `#tj<epilogue gelu>` enum attribute), `tj.softmax`, `tj.layer_norm` and `tj.rms_norm` with verifiers; a `tj-fuse` rewrite pass that folds `linalg.matmul + bias + activation` into `tj.fused_matmul`; a `tj-lower-to-linalg` pass; a `tj-opt` driver; and lit/FileCheck tests.

## Performance

**Per-call overhead, 16-kernel graph, CPU** (`benchmarks/bench_overhead.py`)

| executor | us / call |
|---|---:|
| torch eager | 119 |
| tensorjit, Python executor | 112 |
| tensorjit, C++ runtime + planned arena | **34** |

**CPU kernels, 2 cores, fp32, hidden = 512** (`benchmarks/bench.py`)

| workload | seq | eager | tensorjit | kernels |
|---|---:|---:|---:|---:|
| bias + GELU | 1024 | 4.25 ms | **1.16 ms** (3.7x) | 1 |
| softmax (8 x 1024 x 1024) | 1024 | 14.1 ms | **4.3 ms** (3.3x) | 1 |
| RMSNorm | 1024 | 0.40 ms | **0.09 ms** (4.5x) | 1 |
| LayerNorm | 1024 | 0.13 ms | 0.23 ms | 1 |
| Linear + GELU (512 -> 2048) | 1024 | 14.2 ms (MKL) | 26.3 ms | 1 |
| MLP block | 1024 | 20.5 ms | 43.6 ms | 3 |

Fused pointwise and row work beats eager by 3-4x; the generated C GEMM reaches about half of MKL (register-blocked, no packing), which is the expected ceiling without a packed micro-kernel.

**GPU** (`benchmarks/bench.py --device cuda --backend cuda|triton`, `benchmarks/bench_overhead.py --device cuda`): measure and paste.

| workload | eager | inductor | tensorjit/triton | tensorjit/cuda | tensorjit/cuda + CUDA graph |
|---|---:|---:|---:|---:|---:|
| mlp_block T=4096 | | | | | |
| softmax 8 x 2048^2 | | | | | |

## Testing

```bash
pytest -q                                                    # 85 tests: IR/passes, C backend + C++ runtime, planner,
                                                             # CUDA codegen (host type-check), MLIR emission, torch.compile
TRITON_INTERPRET=1 pytest -q tests/test_backends.py -k triton   # Triton codegen on CPU
pytest -q tests/test_cuda_backend.py                          # on a GPU: NVRTC build, CUDA-graph replay, WMMA GEMM
cmake -S mlir -B mlir/build && cmake --build mlir/build && lit mlir/test   # tj dialect (needs LLVM/MLIR)
```

Every backend test compares against eager PyTorch on the same inputs; planner tests check the non-overlap invariant and the arena bound; runtime tests check the native executor returns the same tensors as the Python executor.

## Configuration

| Variable | Effect |
|---|---|
| `TENSORJIT_BACKEND=c\|cuda\|triton\|mlir` | default backend |
| `TENSORJIT_NATIVE_RUNTIME=0` | use the Python executor instead of the C++ runtime |
| `TENSORJIT_CUDA_GRAPH=0` | disable CUDA-graph capture |
| `TENSORJIT_CUDA_WMMA=0` | disable the Tensor-Core GEMM |
| `TENSORJIT_NO_AUTOTUNE=1` | skip GEMM tile autotuning |
| `TENSORJIT_NO_ROW_FUSION=1` | disable row fusion |
| `TENSORJIT_STRICT=1` | torch.compile backend raises instead of falling back to eager |
| `TENSORJIT_DUMP=1` | print the IR after every pass |
| `TENSORJIT_CACHE_DIR` | kernel and autotune cache location |

## Repository layout

```
tensorjit/ir           graph.py (IR, type inference, broadcasting)  printer.py
tensorjit/frontend     tracer.py (Python tracer)  torch_backend.py (TorchDynamo/FX backend)
tensorjit/passes       cleanup.py  fusion.py (+ row fusion)  memory_planner.py  pass_manager.py
tensorjit/backends     common.py  c_backend.py  cuda_backend.py  cuda_driver.py (NVRTC / driver ctypes)
                       triton_backend.py  mlir_backend.py (linalg + tj emitters)
tensorjit/runtime      native.py (C++ runtime binding)  executor.py (Python executor, buffer pool)  autotune.py  cache.py
csrc/                  runtime.cpp (pybind11 executor)  cuda_host_stub.h (host type-check of generated CUDA)
mlir/                  include/TensorJit/*.td, *.h   lib/TensorJitDialect.cpp, Passes.cpp   tools/tj-opt.cpp   test/*.mlir
tests/                 test_ir_and_passes.py  test_backends.py  test_runtime_and_planner.py  test_cuda_backend.py  test_mlir_and_torch.py
benchmarks/            bench.py  bench_overhead.py
examples/              mlp.py
```

## Supported operators

| Class | Operators |
|---|---|
| Elementwise | add sub mul div pow neg exp log sqrt rsqrt tanh relu sigmoid gelu silu abs maximum minimum where (numpy broadcasting) |
| Contraction | matmul (2-D and batched with broadcast batch dims; transposed-B fast path) |
| Reduction | sum max mean over the last axis (other axes via a fused transpose) |
| Movement | reshape (free), transpose, broadcast_to |
| Composite | softmax, layer_norm, rms_norm, linear |

## Roadmap

- Packed GEMM micro-kernel for the C backend (MKL-class throughput)
- Dynamic-shape buckets with guard-free re-dispatch
- Backward pass through the same IR (fused gradients)
- `gpu` dialect lowering for the MLIR backend (SPIR-V / NVVM)

## References

- Lattner, C. et al. *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation.* CGO 2021.
- Ansel, J. et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.* ASPLOS 2024.
- Chen, T. et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI 2018.
- Tillet, P. et al. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.
- Sekiyama, T. et al. *Profile-guided memory optimization for deep neural networks.* 2018 (interval-based buffer planning).

## License

MIT. See [LICENSE](LICENSE).

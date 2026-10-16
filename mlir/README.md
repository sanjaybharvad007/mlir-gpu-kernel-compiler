# tensorjit MLIR dialect (C++)

The Python compiler emits `linalg`; this directory is the same fusion contract
expressed natively in MLIR, the way a production compiler would carry it:

- **`tj` dialect** (TableGen): `tj.fused_matmul` (matmul + bias + relu/gelu/silu,
  `trans_b`), `tj.softmax`, `tj.layer_norm`, `tj.rms_norm`, with verifiers.
- **`tj-fuse`** pass: rewrite patterns that recognise `linalg.matmul` →
  broadcast-add generic → activation generic (and an explicit `linalg.transpose`
  of the weight) and fold them into one `tj.fused_matmul`.
- **`tj-lower-to-linalg`** pass: expands every `tj.*` op back into
  `linalg.generic` / `tensor` / `arith` / `math` so the stock pipeline
  (`one-shot-bufferize` → loops → LLVM) can consume it.
- **`tj-opt`**: `mlir-opt` with the dialect and passes registered.
- **lit tests** in `test/` (`FileCheck`).

```bash
# needs LLVM/MLIR 18+ (e.g. apt.llvm.org packages mlir-18-tools, libmlir-18-dev)
cmake -S mlir -B build-mlir -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
cmake --build build-mlir --target tj-opt check-tj
./build-mlir/tj-opt mlir/test/fuse.mlir --tj-fuse
```

The Python side can target these ops directly: `tensorjit.backends.mlir_backend.emit_mlir(graph, dialect="tj")`
emits `tj.*` for the fused kernels instead of expanded `linalg`.

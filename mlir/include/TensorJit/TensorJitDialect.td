//===- TensorJitDialect.td - tensorjit dialect definition -----*- tablegen -*-===//
#ifndef TENSORJIT_DIALECT
#define TENSORJIT_DIALECT

include "mlir/IR/OpBase.td"

def TensorJit_Dialect : Dialect {
  let name = "tj";
  let cppNamespace = "::mlir::tj";
  let summary = "tensorjit high-level tensor ops (matmul+epilogue, normalisations, softmax)";
  let description = [{
    The `tj` dialect carries the fused, high-level operations that tensorjit's
    Python frontend produces before lowering to `linalg`.  Ops here are the
    units the GPU code generator knows how to emit as single kernels, so the
    dialect doubles as the fusion contract: `tj-fuse` folds `linalg`-level
    patterns into these ops, `tj-lower-to-linalg` expands them back for the
    generic MLIR pipeline.
  }];
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
}

class TensorJit_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorJit_Dialect, mnemonic, traits>;

#endif // TENSORJIT_DIALECT

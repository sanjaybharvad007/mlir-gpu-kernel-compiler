//===- Passes.h - tensorjit passes ---------------------------------*- C++ -*-===//
#ifndef TENSORJIT_PASSES_H
#define TENSORJIT_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::tj {

/// linalg.matmul + (broadcast add) + (relu|gelu|silu generic)  ->  tj.fused_matmul
std::unique_ptr<Pass> createFusePass();
/// tj.* ops -> linalg / tensor / arith / math
std::unique_ptr<Pass> createLowerToLinalgPass();

void registerPasses();

}  // namespace mlir::tj

#endif // TENSORJIT_PASSES_H

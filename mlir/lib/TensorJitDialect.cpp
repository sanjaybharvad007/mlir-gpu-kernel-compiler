//===- TensorJitDialect.cpp - tensorjit dialect implementation ------------===//
#include "TensorJit/TensorJitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tj;

#include "TensorJit/TensorJitDialect.cpp.inc"
#include "TensorJit/TensorJitEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TensorJit/TensorJitAttrs.cpp.inc"

void TensorJitDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorJit/TensorJitOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TensorJit/TensorJitAttrs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "TensorJit/TensorJitOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult FusedMatmulOp::verify() {
  auto a = cast<RankedTensorType>(getLhs().getType());
  auto b = cast<RankedTensorType>(getRhs().getType());
  auto r = cast<RankedTensorType>(getResult().getType());
  if (a.getRank() < 2 || b.getRank() < 2 || r.getRank() != a.getRank())
    return emitOpError("operands must be rank >= 2 and result rank must match lhs");
  const int64_t m = a.getDimSize(a.getRank() - 2), k = a.getDimSize(a.getRank() - 1);
  const int64_t kb = getTransB() ? b.getDimSize(b.getRank() - 1) : b.getDimSize(b.getRank() - 2);
  const int64_t n = getTransB() ? b.getDimSize(b.getRank() - 2) : b.getDimSize(b.getRank() - 1);
  if (k != kb) return emitOpError("contraction dimension mismatch: ") << k << " vs " << kb;
  if (r.getDimSize(r.getRank() - 2) != m || r.getDimSize(r.getRank() - 1) != n)
    return emitOpError("result shape must be [..., ") << m << ", " << n << "]";
  if (getBias()) {
    auto bias = cast<RankedTensorType>(getBias().getType());
    if (bias.getRank() != 1 || bias.getDimSize(0) != n)
      return emitOpError("bias must be a rank-1 tensor of size ") << n;
  }
  return success();
}

LogicalResult LayerNormOp::verify() {
  auto x = cast<RankedTensorType>(getInput().getType());
  const int64_t last = x.getDimSize(x.getRank() - 1);
  for (Value v : {getWeight(), getBias()}) {
    if (!v) continue;
    auto t = cast<RankedTensorType>(v.getType());
    if (t.getRank() != 1 || t.getDimSize(0) != last)
      return emitOpError("affine parameters must be rank-1 tensors of size ") << last;
  }
  if (getResult().getType() != x) return emitOpError("result type must match input type");
  return success();
}

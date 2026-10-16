//===- TensorJitDialect.h - tensorjit dialect ---------------------*- C++ -*-===//
#ifndef TENSORJIT_DIALECT_H
#define TENSORJIT_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TensorJit/TensorJitDialect.h.inc"
#include "TensorJit/TensorJitEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TensorJit/TensorJitAttrs.h.inc"

#define GET_OP_CLASSES
#include "TensorJit/TensorJitOps.h.inc"

#endif // TENSORJIT_DIALECT_H

//===- tj-opt.cpp - mlir-opt with the tensorjit dialect and passes ---------===//
#include "TensorJit/Passes.h"
#include "TensorJit/TensorJitDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tj::registerPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::tj::TensorJitDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "tensorjit optimizer driver\n", registry));
}

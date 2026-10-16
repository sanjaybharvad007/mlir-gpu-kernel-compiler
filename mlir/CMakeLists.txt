cmake_minimum_required(VERSION 3.20)
project(tensorjit-mlir LANGUAGES CXX C)

# Build against an installed LLVM/MLIR (18+):
#   cmake -S mlir -B build-mlir -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm
#   cmake --build build-mlir --target tj-opt check-tj
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

# --- TableGen -----------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS include/TensorJit/TensorJitOps.td)
mlir_tablegen(include/TensorJit/TensorJitOps.h.inc -gen-op-decls)
mlir_tablegen(include/TensorJit/TensorJitOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/TensorJit/TensorJitDialect.h.inc -gen-dialect-decls -dialect=tj)
mlir_tablegen(include/TensorJit/TensorJitDialect.cpp.inc -gen-dialect-defs -dialect=tj)
mlir_tablegen(include/TensorJit/TensorJitEnums.h.inc -gen-enum-decls)
mlir_tablegen(include/TensorJit/TensorJitEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(include/TensorJit/TensorJitAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tj)
mlir_tablegen(include/TensorJit/TensorJitAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tj)
add_public_tablegen_target(TensorJitOpsIncGen)

# --- dialect + passes library ---------------------------------------------------
add_mlir_dialect_library(MLIRTensorJit
  lib/TensorJitDialect.cpp
  lib/Passes.cpp
  DEPENDS TensorJitOpsIncGen
  LINK_LIBS PUBLIC
    MLIRIR MLIRPass MLIRTransforms MLIRSupport
    MLIRLinalgDialect MLIRTensorDialect MLIRArithDialect MLIRMathDialect MLIRFuncDialect
    MLIRInferTypeOpInterface MLIRSideEffectInterfaces)

# --- tj-opt tool ----------------------------------------------------------------
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(tj-opt tools/tj-opt.cpp)
llvm_update_compile_flags(tj-opt)
target_link_libraries(tj-opt PRIVATE MLIRTensorJit MLIROptLib ${dialect_libs} ${conversion_libs})

# --- lit tests ------------------------------------------------------------------
find_program(LIT_EXE NAMES lit llvm-lit)
if(LIT_EXE)
  add_custom_target(check-tj
    COMMAND ${LIT_EXE} -v ${CMAKE_CURRENT_SOURCE_DIR}/test --param tj_opt=$<TARGET_FILE:tj-opt>
    DEPENDS tj-opt COMMENT "running tensorjit MLIR lit tests")
endif()

//===- TensorJitOps.td - tensorjit op definitions --------------*- tablegen -*-===//
#ifndef TENSORJIT_OPS
#define TENSORJIT_OPS

include "TensorJit/TensorJitDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// The epilogue enum is a dialect attribute (`#tj<epilogue gelu>`), not a bare i32.
def TensorJit_Epilogue : I32EnumAttr<"Epilogue", "pointwise epilogue applied to a matmul result",
    [I32EnumAttrCase<"None", 0, "none">,
     I32EnumAttrCase<"Relu", 1, "relu">,
     I32EnumAttrCase<"Gelu", 2, "gelu">,
     I32EnumAttrCase<"Silu", 3, "silu">]> {
  let cppNamespace = "::mlir::tj";
  let genSpecializedAttr = 0;
}
def TensorJit_EpilogueAttr : EnumAttr<TensorJit_Dialect, TensorJit_Epilogue, "epilogue">;

def TensorJit_FusedMatmulOp : TensorJit_Op<"fused_matmul", [Pure]> {
  let summary = "matrix multiply with fused bias and activation epilogue";
  let description = [{
    `%y = tj.fused_matmul %a, %b (bias %bias) {epilogue = gelu, trans_b}`
    computes `epilogue(%a @ %b^T? + %bias)` in one kernel.  Created by `tj-fuse`
    from `linalg.matmul` followed by a broadcast add and/or activation generic.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       Optional<AnyRankedTensor>:$bias,
                       DefaultValuedAttr<TensorJit_EpilogueAttr, "::mlir::tj::Epilogue::None">:$epilogue,
                       DefaultValuedAttr<BoolAttr, "false">:$trans_b);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`bias` $bias^)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TensorJit_SoftmaxOp : TensorJit_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "numerically stable softmax over the last axis";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def TensorJit_LayerNormOp : TensorJit_Op<"layer_norm", [Pure]> {
  let summary = "layer normalisation over the last axis with optional affine parameters";
  let arguments = (ins AnyRankedTensor:$input, Optional<AnyRankedTensor>:$weight,
                       Optional<AnyRankedTensor>:$bias, DefaultValuedAttr<F32Attr, "1e-5f">:$eps);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input (`weight` $weight^)? (`bias` $bias^)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TensorJit_RmsNormOp : TensorJit_Op<"rms_norm", [Pure]> {
  let summary = "RMS normalisation over the last axis";
  let arguments = (ins AnyRankedTensor:$input, Optional<AnyRankedTensor>:$weight,
                       DefaultValuedAttr<F32Attr, "1e-6f">:$eps);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input (`weight` $weight^)? attr-dict `:` functional-type(operands, results)
  }];
}

#endif // TENSORJIT_OPS

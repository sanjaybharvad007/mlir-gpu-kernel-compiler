// RUN: %tj-opt %s --tj-fuse | %FileCheck %s
#id = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

// gelu(x @ w^T + b): the shape the Python emitter produces for tj.linear + tj.gelu.
// CHECK-LABEL: func.func @linear_gelu
// CHECK:       tj.fused_matmul %arg0, %arg1 bias %arg2 {epilogue = #tj<epilogue gelu>, trans_b = true}
// CHECK-NOT:   linalg.matmul
func.func @linear_gelu(%x: tensor<4x8xf32>, %w: tensor<6x8xf32>, %b: tensor<6xf32>) -> tensor<4x6xf32> {
  %init_t = tensor.empty() : tensor<8x6xf32>
  %wt = linalg.transpose ins(%w : tensor<6x8xf32>) outs(%init_t : tensor<8x6xf32>) permutation = [1, 0]
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x6xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>
  %mm = linalg.matmul ins(%x, %wt : tensor<4x8xf32>, tensor<8x6xf32>) outs(%fill : tensor<4x6xf32>) -> tensor<4x6xf32>
  %i1 = tensor.empty() : tensor<4x6xf32>
  %add = linalg.generic {indexing_maps = [#id, #bias, #id], iterator_types = ["parallel", "parallel"]}
      ins(%mm, %b : tensor<4x6xf32>, tensor<6xf32>) outs(%i1 : tensor<4x6xf32>) {
    ^bb0(%a: f32, %bb: f32, %o: f32):
      %s = arith.addf %a, %bb : f32
      linalg.yield %s : f32
  } -> tensor<4x6xf32>
  %i2 = tensor.empty() : tensor<4x6xf32>
  %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%add : tensor<4x6xf32>) outs(%i2 : tensor<4x6xf32>) {
    ^bb0(%a: f32, %o: f32):
      %half = arith.constant 0.5 : f32
      %k = arith.constant 0.70710678 : f32
      %one = arith.constant 1.0 : f32
      %xk = arith.mulf %a, %k : f32
      %er = math.erf %xk : f32
      %p = arith.addf %one, %er : f32
      %hx = arith.mulf %half, %a : f32
      %r = arith.mulf %hx, %p : f32
      linalg.yield %r : f32
  } -> tensor<4x6xf32>
  return %g : tensor<4x6xf32>
}

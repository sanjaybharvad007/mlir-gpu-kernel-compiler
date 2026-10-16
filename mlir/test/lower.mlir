// RUN: %tj-opt %s --tj-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func.func @softmax
// CHECK:       linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]
// CHECK:       arith.maximumf
// CHECK:       math.exp
// CHECK:       arith.divf
// CHECK-NOT:   tj.softmax
func.func @softmax(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %y = tj.softmax %x : tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @layer_norm
// CHECK:       math.rsqrt
// CHECK-NOT:   tj.layer_norm
func.func @layer_norm(%x: tensor<4x8xf32>, %w: tensor<8xf32>) -> tensor<4x8xf32> {
  %y = tj.layer_norm %x weight %w {eps = 1.0e-5 : f32} : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @fused
// CHECK:       linalg.transpose
// CHECK:       linalg.matmul
// CHECK:       math.erf
// CHECK-NOT:   tj.fused_matmul
func.func @fused(%x: tensor<4x8xf32>, %w: tensor<6x8xf32>, %b: tensor<6xf32>) -> tensor<4x6xf32> {
  %y = tj.fused_matmul %x, %w bias %b {epilogue = #tj<epilogue gelu>, trans_b = true}
       : (tensor<4x8xf32>, tensor<6x8xf32>, tensor<6xf32>) -> tensor<4x6xf32>
  return %y : tensor<4x6xf32>
}

// Round trip: fuse then lower must verify.
// RUN: %tj-opt %s --tj-fuse --tj-lower-to-linalg --verify-each

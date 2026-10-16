// Host-side stand-ins for CUDA constructs so generated kernels can be
// syntax/type-checked with an ordinary C++ compiler (no GPU, no nvcc):
//   g++ -fsyntax-only -x c++ -DTJ_HOST_STUB -include csrc/cuda_host_stub.h generated.cu
// Nothing here is meant to run; Tensor-Core (WMMA) sections are excluded by TJ_HOST_STUB.
#pragma once
#include <cmath>
#include <cstdint>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

struct tj_dim3 { unsigned x = 1, y = 1, z = 1; };
static tj_dim3 threadIdx, blockIdx, blockDim, gridDim;

inline void __syncthreads() {}
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline int __shfl_xor_sync(unsigned, int v, int) { return v; }

struct __half { float v; };
struct __nv_bfloat16 { float v; };
typedef __half half;
inline float __half2float(__half h) { return h.v; }
inline __half __float2half(float f) { return __half{f}; }
inline float __bfloat162float(__nv_bfloat16 b) { return b.v; }
inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16{f}; }

// Minimal nvcuda::wmma surface (types + signatures only) so Tensor-Core kernels type-check too.
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {};
struct row_major {}; struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <typename Use, int M, int N, int K, typename T, typename Layout = void> struct fragment { T x[8]; };
template <typename F> inline void fill_fragment(F&, float) {}
template <typename F, typename T> inline void load_matrix_sync(F&, const T*, unsigned) {}
template <typename F> inline void store_matrix_sync(float*, const F&, unsigned, layout_t) {}
template <typename C, typename A, typename B> inline void mma_sync(C&, const A&, const B&, const C&) {}
}}  // namespace nvcuda::wmma

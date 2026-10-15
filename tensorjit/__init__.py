"""tensorjit: a JIT tensor compiler.

    import tensorjit as tj

    @tj.jit
    def mlp(x, w1, b1, w2, b2):
        h = tj.gelu(tj.linear(x, w1, b1))
        return tj.linear(h, w2, b2)

    y = mlp(x, w1, b1, w2, b2)      # torch tensors in, torch tensors out

Pipeline:  trace -> SSA tensor IR -> cleanup passes (CSE/DCE/algebraic/
broadcast folding) -> fusion (elementwise chains, matmul epilogues, reduce
prologues) -> backend codegen (C+OpenMP / Triton / MLIR-linalg) -> cached,
shape-specialised kernels, with autotuned matmul tiles.
"""
from .compiler import JitFunction, compile_graph, jit, lower, pick_backend
from .frontend.tracer import (
    Tensor, abs, broadcast_to, exp, gelu, layer_norm, linear, log, matmul, max, maximum, mean, minimum,
    permute, relu, reshape, rms_norm, rsqrt, sigmoid, silu, softmax, sqrt, sum, tanh, trace, transpose, where,
)
from .ir import DType, Graph, Node, TensorType

__version__ = "0.3.0"

__all__ = [
    "jit", "JitFunction", "compile_graph", "lower", "pick_backend", "trace",
    "Tensor", "Graph", "Node", "TensorType", "DType",
    "abs", "broadcast_to", "exp", "gelu", "layer_norm", "linear", "log", "matmul", "max", "maximum", "mean",
    "minimum", "permute", "relu", "reshape", "rms_norm", "rsqrt", "sigmoid", "silu", "softmax", "sqrt", "sum",
    "tanh", "transpose", "where",
]

from .frontend.torch_backend import torch_compile_backend  # noqa: E402, F401

__all__.append("torch_compile_backend")

from ..ir import Graph
from .cleanup import algebraic_simplify, cse, dce, fold_broadcast, fold_reshape_chains, fold_transpose_into_matmul
from .fusion import Kernel, Program, fuse, row_fuse
from .pass_manager import PassManager

DEFAULT_PASSES = [
    ("fold_broadcast", fold_broadcast),
    ("fold_reshape_chains", fold_reshape_chains),
    ("fold_transpose_into_matmul", fold_transpose_into_matmul),
    ("algebraic_simplify", algebraic_simplify),
    ("cse", cse),
    ("dce", dce),
]


def optimize(g: Graph) -> Graph:
    return PassManager(DEFAULT_PASSES).run(g)


__all__ = ["optimize", "fuse", "row_fuse", "Kernel", "Program", "PassManager", "DEFAULT_PASSES"]

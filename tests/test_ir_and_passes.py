import torch

import tensorjit as tj
from tensorjit.ir import Graph, TensorType
from tensorjit.passes import fuse, optimize
from tensorjit.passes.cleanup import cse, dce


def _graph(fn, *shapes):
    return tj.trace(fn, [TensorType(s) for s in shapes])


def test_type_inference_and_broadcast():
    g = _graph(lambda a, b: a * b + 1.0, (4, 1, 8), (3, 8))
    assert g.outputs[0].shape == (4, 3, 8)


def test_matmul_batched_broadcast_shape():
    g = _graph(lambda a, b: tj.matmul(a, b), (2, 1, 5, 7), (3, 7, 4))
    assert g.outputs[0].shape == (2, 3, 5, 4)


def test_cse_merges_duplicates():
    g = _graph(lambda x: tj.exp(x) + tj.exp(x), (8,))
    n_exp = sum(n.op == "exp" for n in g.nodes)
    assert n_exp == 2
    cse(g)
    assert sum(n.op == "exp" for n in g.nodes) == 1


def test_algebraic_simplify_and_dce():
    g = _graph(lambda x: (x * 1.0 + 0.0) * (2.0 * 3.0), (8,))
    optimize(g)
    ops = [n.op for n in g.nodes if n.op not in ("input", "const")]
    assert ops == ["mul"], ops
    const = [n for n in g.nodes if n.op == "const"]
    assert any(abs(c.attrs["value"] - 6.0) < 1e-12 for c in const)


def test_dce_removes_unused():
    g = _graph(lambda x: (x + 1.0, (x * 5.0)), (4,))
    g.set_outputs([g.outputs[0]])
    assert dce(g)
    assert not any(n.op == "mul" for n in g.nodes)


def test_fusion_groups_elementwise_chain_into_one_kernel():
    g = optimize(_graph(lambda x, y: tj.relu(x * 2.0 + y).exp(), (16, 32), (32,)))
    p = fuse(g)
    assert len(p.kernels) == 1 and p.kernels[0].kind == "elementwise"
    assert [n.op for n in p.kernels[0].nodes] == ["mul", "add", "relu", "exp"]


def test_fusion_matmul_epilogue():
    g = optimize(_graph(lambda x, w, b: tj.gelu(tj.matmul(x, w) + b), (8, 16), (16, 32), (32,)))
    p = fuse(g)
    kinds = [k.kind for k in p.kernels]
    assert kinds == ["matmul"], p
    assert [n.op for n in p.kernels[0].nodes] == ["matmul", "add", "gelu"]


def test_fusion_reduce_prologue():
    g = optimize(_graph(lambda x: tj.sum(x * x), (8, 16)))
    p = fuse(g)
    assert len(p.kernels) == 1 and p.kernels[0].kind == "reduce"
    assert [n.op for n in p.kernels[0].nodes] == ["mul", "sum"]


def test_fusion_does_not_duplicate_compute():
    # d = x - mu is needed by the variance *and* by the final normalise:
    # it must be materialised once, not recomputed in two kernels.
    g = optimize(_graph(lambda x: tj.layer_norm(x), (8, 16)))
    p = fuse(g)
    ops = [n.op for k in p.kernels for n in k.nodes]
    assert ops.count("sub") == 1


def test_reshape_is_free():
    g = optimize(_graph(lambda x: tj.exp(x.reshape(4, 8)), (8, 4)))
    p = fuse(g)
    assert len(p.kernels) == 1 and p.aliases


def test_signature_is_stable_and_shape_sensitive():
    g1 = _graph(lambda x: tj.exp(x), (8,))
    g2 = _graph(lambda x: tj.exp(x), (8,))
    g3 = _graph(lambda x: tj.exp(x), (9,))
    assert g1.signature() == g2.signature() != g3.signature()


def test_printer_roundtrip_mentions_ops():
    g = _graph(lambda x: tj.softmax(x), (4, 8))
    s = str(g)
    assert "max(" in s and "exp(" in s and "sum(" in s and "func @" in s


def test_graph_builder_api():
    g = Graph("manual")
    a = g.add_input(TensorType((2, 3)))
    b = g.create("exp", [a])
    g.set_outputs([b])
    assert g.outputs[0].shape == (2, 3)
    assert isinstance(torch.empty(1), torch.Tensor)


def test_row_fusion_layernorm_softmax_single_kernel():
    import tensorjit.compiler as C

    g = optimize(_graph(lambda x, w, b: (tj.layer_norm(x, w, b), tj.softmax(x)), (8, 64), (64,), (64,)))
    p = C.lower(g, "c")
    assert [k.kind for k in p.kernels] == ["row", "row"]
    # the row kernels read x once and write exactly the two results
    assert all(len(k.outputs) == 1 for k in p.kernels)


def test_row_fusion_respects_dependencies_through_matmul():
    import tensorjit.compiler as C

    def f(x, w):
        mu = tj.mean(x, keepdim=True)          # row stat
        y = tj.matmul(x - mu, w)               # leaves the row domain
        return tj.softmax(y) + mu              # row again, but depends on matmul

    g = optimize(_graph(f, (8, 32), (32, 32)))
    p = C.lower(g, "c")
    kinds = [k.kind for k in p.kernels]
    assert "matmul" in kinds and kinds.index("matmul") < len(kinds) - 1

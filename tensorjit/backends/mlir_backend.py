"""MLIR backend: tensorjit IR -> MLIR (linalg / tensor / arith / math) -> LLVM.

Emission is pure Python and always available (``emit_mlir``).  Running the
pipeline needs an MLIR toolchain on PATH (``mlir-opt``, ``mlir-translate``,
``clang``), e.g. from an LLVM 18+ release; the compiled function is called
through the C interface with memref descriptors built in ctypes.

Lowering pipeline (``MLIR_PIPELINE``):
    linalg (named ops + generics, tensor semantics)
      -> linalg-fuse-elementwise-ops         (producer/consumer fusion)
      -> one-shot-bufferize                  (tensors -> memrefs)
      -> convert-linalg-to-loops, lower-affine, convert-scf-to-cf
      -> arith/math/memref/func -> llvm
      -> mlir-translate --mlir-to-llvmir -> clang -O3 -shared
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import torch

from ..ir import DType, Graph, Node, OpKind
from ..passes import Program
from ..runtime import CompiledProgram, cache_dir, content_hash

MLIR_PIPELINE = (
    "builtin.module("
    "func.func(linalg-fuse-elementwise-ops),"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "func.func(convert-linalg-to-loops),"
    "func.func(lower-affine),"
    "convert-scf-to-cf,"
    "convert-math-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)


def _ty(n: Node) -> str:
    dims = "x".join(str(s) for s in n.shape)
    return f"tensor<{dims + 'x' if dims else ''}{n.dtype.mlir}>"


def _affine_map(in_shape, out_rank: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(out_rank))
    pad = out_rank - len(in_shape)
    res = []
    for d, s in enumerate(in_shape):
        res.append("0" if s == 1 else f"d{pad + d}")
    return f"affine_map<({dims}) -> ({', '.join(res)})>"


def _ident_map(rank: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({dims})>"


_UNARY_BODY = {
    "neg": "arith.negf {0} : {t}",
    "exp": "math.exp {0} : {t}",
    "log": "math.log {0} : {t}",
    "sqrt": "math.sqrt {0} : {t}",
    "rsqrt": "math.rsqrt {0} : {t}",
    "tanh": "math.tanh {0} : {t}",
    "abs": "math.absf {0} : {t}",
}
_BINARY_BODY = {
    "add": "arith.addf {0}, {1} : {t}",
    "sub": "arith.subf {0}, {1} : {t}",
    "mul": "arith.mulf {0}, {1} : {t}",
    "div": "arith.divf {0}, {1} : {t}",
    "pow": "math.powf {0}, {1} : {t}",
    "maximum": "arith.maximumf {0}, {1} : {t}",
    "minimum": "arith.minimumf {0}, {1} : {t}",
}


class _Emitter:
    def __init__(self, g: Graph):
        self.g = g
        self.lines: list[str] = []
        self.names: dict[int, str] = {}
        self.counter = 0
        self.consts: dict[tuple, str] = {}

    def fresh(self, hint: str = "v") -> str:
        self.counter += 1
        return f"%{hint}{self.counter}"

    def emit(self, s: str) -> None:
        self.lines.append("    " + s)

    def scalar_const(self, v: float, t: str = "f32") -> str:
        key = (v, t)
        if key not in self.consts:
            name = self.fresh("c")
            self.emit(f"{name} = arith.constant {v:.9e} : {t}")
            self.consts[key] = name
        return self.consts[key]

    # -- elementwise body for one op (used inside linalg.generic) -----------
    def body_op(self, n: Node, args: list[str], t: str) -> list[str]:
        out, lines = self.fresh("b"), []
        if n.op in _UNARY_BODY:
            lines.append(f"{out} = {_UNARY_BODY[n.op].format(*args, t=t)}")
        elif n.op in _BINARY_BODY:
            lines.append(f"{out} = {_BINARY_BODY[n.op].format(*args, t=t)}")
        elif n.op == "relu":
            z = self.fresh("b")
            lines += [f"{z} = arith.constant 0.0 : {t}", f"{out} = arith.maximumf {args[0]}, {z} : {t}"]
        elif n.op in ("sigmoid", "silu"):
            one, nx, e, d, s = (self.fresh("b") for _ in range(5))
            lines += [
                f"{one} = arith.constant 1.0 : {t}",
                f"{nx} = arith.negf {args[0]} : {t}",
                f"{e} = math.exp {nx} : {t}",
                f"{d} = arith.addf {one}, {e} : {t}",
                f"{s} = arith.divf {one}, {d} : {t}",
            ]
            if n.op == "silu":
                lines.append(f"{out} = arith.mulf {args[0]}, {s} : {t}")
            else:
                out = s
        elif n.op == "gelu":
            half, k, one, xk, er, p, hx = (self.fresh("b") for _ in range(7))
            lines += [
                f"{half} = arith.constant 0.5 : {t}",
                f"{k} = arith.constant 0.70710678118654752 : {t}",
                f"{one} = arith.constant 1.0 : {t}",
                f"{xk} = arith.mulf {args[0]}, {k} : {t}",
                f"{er} = math.erf {xk} : {t}",
                f"{p} = arith.addf {one}, {er} : {t}",
                f"{hx} = arith.mulf {half}, {args[0]} : {t}",
                f"{out} = arith.mulf {hx}, {p} : {t}",
            ]
        elif n.op == "where":
            z, c = self.fresh("b"), self.fresh("b")
            lines += [
                f"{z} = arith.constant 0.0 : {t}",
                f"{c} = arith.cmpf one, {args[0]}, {z} : {t}",
                f"{out} = arith.select {c}, {args[1]}, {args[2]} : {t}",
            ]
        else:
            raise NotImplementedError(n.op)
        return lines + [out]  # statements, then the SSA name of the result

    def elementwise(self, n: Node) -> None:
        t = n.dtype.mlir
        rank = len(n.shape)
        ins, maps, block_args = [], [], []
        args = []
        for i in n.inputs:
            if i.op == "const" and i.shape == ():
                args.append(self.scalar_const(i.attrs["value"], t))
            else:
                ba = self.fresh("a")
                ins.append(f"{self.names[id(i)]}")
                maps.append(_affine_map(i.shape, rank))
                block_args.append(f"{ba}: {t}")
                args.append(ba)
        maps.append(_ident_map(rank))
        init = self.fresh("init")
        self.emit(f"{init} = tensor.empty() : {_ty(n)}")
        out = self.fresh("v")
        iters = ", ".join(['"parallel"'] * rank)
        in_types = ", ".join(_ty(i) for i in n.inputs if not (i.op == "const" and i.shape == ()))
        ins_clause = f"ins({', '.join(ins)} : {in_types}) " if ins else ""
        self.emit(
            f"{out} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = [{iters}]}} "
            f"{ins_clause}outs({init} : {_ty(n)}) {{"
        )
        self.emit(f"  ^bb0({', '.join(block_args + [f'%o: {t}'])}):")
        body = self.body_op(n, args, t)
        for s in body[:-1]:
            self.emit("    " + s)
        self.emit(f"    linalg.yield {body[-1]} : {t}")
        self.emit(f"}} -> {_ty(n)}")
        self.names[id(n)] = out

    def matmul(self, n: Node) -> None:
        a, b = n.inputs
        t = n.dtype.mlir
        if n.attrs.get("trans_b"):
            # Re-materialise the transpose; MLIR's own fusion/bufferization handles it.
            from ..ir import Node as _N
            from ..ir import TensorType as _T
            r = b.type.rank
            perm = tuple(range(r - 2)) + (r - 1, r - 2)
            tmp = _N("transpose", [], _T(tuple(b.shape[p] for p in perm), b.dtype), attrs={"perm": perm})
            tmp.inputs = [b]
            init, out = self.fresh("init"), self.fresh("tr")
            self.emit(f"{init} = tensor.empty() : {_ty(tmp)}")
            self.emit(f"{out} = linalg.transpose ins({self.names[id(b)]} : {_ty(b)}) outs({init} : {_ty(tmp)}) permutation = [{', '.join(map(str, perm))}]")
            self.names[id(tmp)] = out
            b = tmp
        init = self.fresh("init")
        zero = self.scalar_const(0.0, t)
        filled = self.fresh("fill")
        self.emit(f"{init} = tensor.empty() : {_ty(n)}")
        self.emit(f"{filled} = linalg.fill ins({zero} : {t}) outs({init} : {_ty(n)}) -> {_ty(n)}")
        out = self.fresh("mm")
        A, B = self.names[id(a)], self.names[id(b)]
        if len(n.shape) == 2:
            self.emit(f"{out} = linalg.matmul ins({A}, {B} : {_ty(a)}, {_ty(b)}) outs({filled} : {_ty(n)}) -> {_ty(n)}")
        elif len(n.shape) == 3 and a.shape[0] == b.shape[0] == n.shape[0]:
            self.emit(f"{out} = linalg.batch_matmul ins({A}, {B} : {_ty(a)}, {_ty(b)}) outs({filled} : {_ty(n)}) -> {_ty(n)}")
        else:  # broadcasting batch dims: generic contraction
            r = len(n.shape)
            dims = [f"d{i}" for i in range(r + 1)]  # last is the reduction dim k
            bd = dims[: r - 2]
            m, nn, k = dims[r - 2], dims[r - 1], dims[r]

            def bmap(shape):
                pad = (r - 2) - (len(shape) - 2)
                res = ["0" if shape[d] == 1 else bd[pad + d] for d in range(len(shape) - 2)]
                return res

            ma = f"affine_map<({', '.join(dims)}) -> ({', '.join(bmap(a.shape) + [m, k])})>"
            mb = f"affine_map<({', '.join(dims)}) -> ({', '.join(bmap(b.shape) + [k, nn])})>"
            mc = f"affine_map<({', '.join(dims)}) -> ({', '.join(bd + [m, nn])})>"
            iters = ", ".join(['"parallel"'] * r + ['"reduction"'])
            self.emit(
                f"{out} = linalg.generic {{indexing_maps = [{ma}, {mb}, {mc}], iterator_types = [{iters}]}} "
                f"ins({A}, {B} : {_ty(a)}, {_ty(b)}) outs({filled} : {_ty(n)}) {{"
            )
            self.emit(f"  ^bb0(%x: {t}, %y: {t}, %acc: {t}):")
            self.emit(f"    %p = arith.mulf %x, %y : {t}")
            self.emit(f"    %s = arith.addf %acc, %p : {t}")
            self.emit(f"    linalg.yield %s : {t}")
            self.emit(f"}} -> {_ty(n)}")
        self.names[id(n)] = out

    def reduce(self, n: Node) -> None:
        x = n.inputs[0]
        t = n.dtype.mlir
        r = len(x.shape)
        keep = n.attrs.get("keepdim", False)
        red_shape = x.shape[:-1]
        red_ty = f"tensor<{'x'.join(map(str, red_shape)) + 'x' if red_shape else ''}{t}>"
        init_val = "0.0" if n.op in ("sum", "mean") else "-3.40282347e+38"
        init = self.fresh("init")
        fill = self.fresh("fill")
        c = self.scalar_const(float(init_val), t)
        self.emit(f"{init} = tensor.empty() : {red_ty}")
        self.emit(f"{fill} = linalg.fill ins({c} : {t}) outs({init} : {red_ty}) -> {red_ty}")
        dims = ", ".join(f"d{i}" for i in range(r))
        m_in = f"affine_map<({dims}) -> ({dims})>"
        m_out = f"affine_map<({dims}) -> ({', '.join(f'd{i}' for i in range(r - 1))})>"
        iters = ", ".join(['"parallel"'] * (r - 1) + ['"reduction"'])
        red = self.fresh("red")
        op = "arith.addf" if n.op in ("sum", "mean") else "arith.maximumf"
        self.emit(
            f"{red} = linalg.generic {{indexing_maps = [{m_in}, {m_out}], iterator_types = [{iters}]}} "
            f"ins({self.names[id(x)]} : {_ty(x)}) outs({fill} : {red_ty}) {{"
        )
        self.emit(f"  ^bb0(%x: {t}, %acc: {t}):")
        self.emit(f"    %s = {op} %acc, %x : {t}")
        self.emit(f"    linalg.yield %s : {t}")
        self.emit(f"}} -> {red_ty}")
        if n.op == "mean":
            scaled = self.fresh("mean")
            inv = self.scalar_const(1.0 / x.shape[-1], t)
            e = self.fresh("init")
            self.emit(f"{e} = tensor.empty() : {red_ty}")
            rr = r - 1
            idm = _ident_map(rr)
            its = ", ".join(['"parallel"'] * rr)
            self.emit(
                f"{scaled} = linalg.generic {{indexing_maps = [{idm}, {idm}], iterator_types = [{its}]}} "
                f"ins({red} : {red_ty}) outs({e} : {red_ty}) {{"
            )
            self.emit(f"  ^bb0(%x: {t}, %o: {t}):")
            self.emit(f"    %s = arith.mulf %x, {inv} : {t}")
            self.emit(f"    linalg.yield %s : {t}")
            self.emit(f"}} -> {red_ty}")
            red = scaled
        if keep:
            red = self.reshape_to(red, red_ty, n)
        self.names[id(n)] = red

    def reshape_to(self, src: str, src_ty: str, n: Node) -> str:
        shp = self.fresh("shape")
        rank = len(n.shape)
        self.emit(f"{shp} = arith.constant dense<[{', '.join(map(str, n.shape))}]> : tensor<{rank}xi64>")
        out = self.fresh("rs")
        self.emit(f"{out} = tensor.reshape {src}({shp}) : ({src_ty}, tensor<{rank}xi64>) -> {_ty(n)}")
        return out

    def movement(self, n: Node) -> None:
        x = n.inputs[0]
        if n.op == "reshape":
            self.names[id(n)] = self.reshape_to(self.names[id(x)], _ty(x), n)
        elif n.op == "transpose":
            init = self.fresh("init")
            out = self.fresh("tr")
            self.emit(f"{init} = tensor.empty() : {_ty(n)}")
            self.emit(
                f"{out} = linalg.transpose ins({self.names[id(x)]} : {_ty(x)}) outs({init} : {_ty(n)}) "
                f"permutation = [{', '.join(map(str, n.attrs['perm']))}]"
            )
            self.names[id(n)] = out
        elif n.op == "broadcast_to":
            rank = len(n.shape)
            init, out = self.fresh("init"), self.fresh("bc")
            self.emit(f"{init} = tensor.empty() : {_ty(n)}")
            t = n.dtype.mlir
            iters = ", ".join(['"parallel"'] * rank)
            self.emit(
                f"{out} = linalg.generic {{indexing_maps = [{_affine_map(x.shape, rank)}, {_ident_map(rank)}], "
                f"iterator_types = [{iters}]}} ins({self.names[id(x)]} : {_ty(x)}) outs({init} : {_ty(n)}) {{"
            )
            self.emit(f"  ^bb0(%x: {t}, %o: {t}):")
            self.emit(f"    linalg.yield %x : {t}")
            self.emit(f"}} -> {_ty(n)}")
            self.names[id(n)] = out

    def run(self) -> str:
        g = self.g
        for i, n in enumerate(g.inputs):
            self.names[id(n)] = f"%arg{i}"
        for n in g.nodes:
            if n.kind == OpKind.INPUT or n.kind == OpKind.CONST:
                continue
            if n.kind == OpKind.ELEMENTWISE:
                self.elementwise(n)
            elif n.kind == OpKind.MATMUL:
                self.matmul(n)
            elif n.kind == OpKind.REDUCE:
                self.reduce(n)
            elif n.kind == OpKind.MOVEMENT:
                self.movement(n)
        for o in g.outputs:
            if id(o) not in self.names:  # output is a bare constant
                self.names[id(o)] = self.scalar_const(o.attrs["value"], o.dtype.mlir)
        args = ", ".join(f"%arg{i}: {_ty(n)}" for i, n in enumerate(g.inputs))
        rets = ", ".join(_ty(o) for o in g.outputs)
        head = f"  func.func @{g.name}({args}) -> ({rets}) attributes {{llvm.emit_c_interface}} {{"
        ret = f"    return {', '.join(self.names[id(o)] for o in g.outputs)} : {rets}"
        return "\n".join(["module {", head, *self.lines, ret, "  }", "}"]) + "\n"


def emit_mlir(graph: Graph, dialect: str = "linalg") -> str:
    """Lower a tensorjit graph to MLIR text.

    dialect="linalg": every op expanded to linalg/tensor/arith/math (default).
    dialect="tj":     composites (softmax, layer_norm, rms_norm, linear+activation)
                      emitted as `tj.*` ops of the C++ dialect in mlir/.
    """
    if dialect == "tj":
        return _TjEmitter(graph).run()
    return _Emitter(graph).run()


# ----------------------------------------------------------------------------
# Toolchain driver
# ----------------------------------------------------------------------------


def toolchain_available() -> bool:
    return all(shutil.which(t) for t in ("mlir-opt", "mlir-translate", "clang"))


def run_pipeline(mlir_text: str, pipeline: str = MLIR_PIPELINE) -> str:
    """mlir-opt pipeline -> LLVM dialect text (raises with mlir-opt's diagnostics)."""
    r = subprocess.run(["mlir-opt", f"--pass-pipeline={pipeline}"], input=mlir_text, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mlir-opt failed:\n{r.stderr}")
    return r.stdout


def verify(mlir_text: str) -> None:
    r = subprocess.run(["mlir-opt", "--verify-each"], input=mlir_text, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"MLIR verification failed:\n{r.stderr}")


def _build_so(mlir_text: str) -> str:
    h = content_hash(mlir_text, MLIR_PIPELINE, "mlir-v1")
    so = cache_dir() / f"tj_mlir_{h}.so"
    if so.exists():
        return str(so)
    llvm_dialect = run_pipeline(mlir_text)
    with tempfile.TemporaryDirectory() as td:
        ll = os.path.join(td, "k.ll")
        r = subprocess.run(["mlir-translate", "--mlir-to-llvmir"], input=llvm_dialect, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"mlir-translate failed:\n{r.stderr}")
        with open(ll, "w") as f:
            f.write(r.stdout)
        tmp = os.path.join(td, "k.so")
        subprocess.run(["clang", "-O3", "-shared", "-fPIC", "-march=native", ll, "-o", tmp, "-lm"], check=True)
        shutil.move(tmp, so)
    return str(so)


def _memref_struct(rank: int):
    fields = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64)]
    if rank:
        fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]
    return type(f"MemRef{rank}", (ctypes.Structure,), {"_fields_": fields})


def _to_memref(t: torch.Tensor):
    rank = t.dim()
    M = _memref_struct(rank)
    m = M()
    m.allocated = m.aligned = t.data_ptr()
    m.offset = 0
    if rank:
        m.sizes = (ctypes.c_int64 * rank)(*t.shape)
        m.strides = (ctypes.c_int64 * rank)(*t.stride())
    return m


@dataclass
class _MlirRunner:
    lib: ctypes.CDLL
    fn: object
    graph: Graph
    result_struct: type

    def __call__(self, ins: list[torch.Tensor], outs: list[torch.Tensor]) -> None:
        arg_refs = [_to_memref(t.contiguous()) for t in ins]
        res = self.result_struct()
        self.fn(ctypes.byref(res), *[ctypes.byref(a) for a in arg_refs])
        libc = ctypes.CDLL(None)
        for j, (o, node) in enumerate(zip(outs, self.graph.outputs)):
            mr = getattr(res, f"r{j}")
            n = o.numel()
            buf = (ctypes.c_float * n).from_address(mr.aligned)
            o.copy_(torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(o.shape))
            libc.free(ctypes.c_void_p(mr.allocated))


def compile_program(program: Program) -> CompiledProgram:
    """Whole-graph compilation through MLIR (fusion is done by MLIR itself)."""
    if not toolchain_available():
        raise RuntimeError("MLIR toolchain (mlir-opt, mlir-translate, clang) not found on PATH")
    g = program.graph
    if any(n.dtype != DType.f32 for n in g.nodes):
        raise NotImplementedError("MLIR backend currently handles f32 graphs")
    text = emit_mlir(g)
    so = _build_so(text)
    lib = ctypes.CDLL(so)
    fn = getattr(lib, f"_mlir_ciface_{g.name}")
    fn.restype = None
    fields = [(f"r{j}", _memref_struct(len(o.shape))) for j, o in enumerate(g.outputs)]
    Res = type("Result", (ctypes.Structure,), {"_fields_": fields})
    runner = _MlirRunner(lib, fn, g, Res)

    # Present the whole graph as a single "kernel" to the executor.
    from ..passes import Kernel

    k = Kernel("mlir", nodes=[n for n in g.nodes if n.kind not in (OpKind.INPUT, OpKind.CONST)])
    k.inputs, k.outputs = list(g.inputs), list(g.outputs)
    single = Program(g, [k], aliases={})
    cp = CompiledProgram(single, [runner], "cpu", "mlir")
    cp.source = text  # type: ignore[attr-defined]
    return cp


__all__ = ["emit_mlir", "run_pipeline", "verify", "toolchain_available", "compile_program", "MLIR_PIPELINE"]


# ----------------------------------------------------------------------------
# `tj` dialect emission (see mlir/ for the C++ dialect, tj-fuse and tj-lower-to-linalg)
# ----------------------------------------------------------------------------

_ACTIVATIONS = {"gelu": "gelu", "relu": "relu", "silu": "silu"}


def _composite_internals(root: Node, args: list[Node], outputs: set[int] = frozenset()) -> set[int] | None:
    """Node ids strictly inside the composite (reachable from root without
    crossing its arguments).  None if any internal value escapes, in which case
    the composite cannot be collapsed into one op."""
    stop = {id(a) for a in args}
    internal: set[int] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        for i in n.inputs:
            if id(i) in stop or i.op in ("input", "const") or id(i) in internal:
                continue
            internal.add(id(i))
            stack.append(i)
    # every internal node must be used only by internal nodes or the root
    allowed = internal | {id(root)}
    return internal if all(
        id(m) not in outputs and all(id(u) in allowed for u in m.users) for m in _nodes_by_id(root, internal)
    ) else None


def _nodes_by_id(root: Node, ids: set[int]) -> list[Node]:
    out, stack, seen = [], [root], set()
    while stack:
        n = stack.pop()
        for i in n.inputs:
            if id(i) in ids and id(i) not in seen:
                seen.add(id(i))
                out.append(i)
                stack.append(i)
    return out


class _TjEmitter(_Emitter):
    """Emits composite roots as tj.* ops and everything else as linalg."""

    def _plan(self):
        g = self.g
        skip: set[int] = set()
        roots: dict[int, dict] = {}
        outputs = {id(o) for o in g.outputs}
        for n in g.nodes:
            comp = n.meta.get("composite")
            if not comp:
                continue
            internals = _composite_internals(n, comp["args"], outputs)
            if internals is None:
                continue
            info = dict(comp)
            info["root"] = n
            # linear + activation (single use) -> fused_matmul with epilogue
            if comp["op"] == "linear" and len(n.users) == 1 and n.users[0].op in _ACTIVATIONS \
                    and n not in g.outputs:
                act = n.users[0]
                internals = internals | {id(n)}
                info["epilogue"] = _ACTIVATIONS[act.op]
                info["root"] = act
                n = act
            skip |= internals
            roots[id(n)] = info
        return skip, roots

    def tj_op(self, info: dict) -> None:
        n = info["root"]
        args = info["args"]
        t = n.dtype.mlir
        out = self.fresh("tj")
        names = [self.names[id(a)] for a in args]
        types = [_ty(a) for a in args]
        op = info["op"]
        if op == "softmax":
            self.emit(f"{out} = tj.softmax {names[0]} : {_ty(n)}")
        elif op == "layer_norm":
            w = f" weight {names[1]}" if info.get("has_weight") else ""
            b = f" bias {names[2 if info.get('has_weight') else 1]}" if info.get("has_bias") else ""
            self.emit(f"{out} = tj.layer_norm {names[0]}{w}{b} {{eps = {info['eps']:.6e} : {t}}} "
                      f": ({', '.join(types)}) -> {_ty(n)}")
        elif op == "rms_norm":
            w = f" weight {names[1]}" if info.get("has_weight") else ""
            self.emit(f"{out} = tj.rms_norm {names[0]}{w} {{eps = {info['eps']:.6e} : {t}}} "
                      f": ({', '.join(types)}) -> {_ty(n)}")
        elif op == "linear":
            b = f" bias {names[2]}" if info.get("has_bias") else ""
            epi = info.get("epilogue", "none")
            self.emit(f"{out} = tj.fused_matmul {names[0]}, {names[1]}{b} "
                      f"{{epilogue = #tj<epilogue {epi}>, trans_b = true}} : ({', '.join(types)}) -> {_ty(n)}")
        else:
            raise NotImplementedError(op)
        self.names[id(n)] = out

    def run(self) -> str:
        g = self.g
        skip, roots = self._plan()
        for i, n in enumerate(g.inputs):
            self.names[id(n)] = f"%arg{i}"
        for n in g.nodes:
            if n.kind in (OpKind.INPUT, OpKind.CONST) or id(n) in skip:
                continue
            if id(n) in roots:
                self.tj_op(roots[id(n)])
            elif n.kind == OpKind.ELEMENTWISE:
                self.elementwise(n)
            elif n.kind == OpKind.MATMUL:
                self.matmul(n)
            elif n.kind == OpKind.REDUCE:
                self.reduce(n)
            elif n.kind == OpKind.MOVEMENT:
                self.movement(n)
        for o in g.outputs:
            if id(o) not in self.names:
                self.names[id(o)] = self.scalar_const(o.attrs["value"], o.dtype.mlir)
        args = ", ".join(f"%arg{i}: {_ty(n)}" for i, n in enumerate(g.inputs))
        rets = ", ".join(_ty(o) for o in g.outputs)
        head = f"  func.func @{g.name}({args}) -> ({rets}) attributes {{llvm.emit_c_interface}} {{"
        ret = f"    return {', '.join(self.names[id(o)] for o in g.outputs)} : {rets}"
        return "\n".join(["module {", head, *self.lines, ret, "  }", "}"]) + "\n"


def emit_mlir_tj(graph: Graph) -> str:
    """Emit MLIR using the `tj` dialect for composites (softmax, layer_norm,
    rms_norm, linear[+activation]) and linalg for everything else.  Feed the
    result to ``tj-opt --tj-lower-to-linalg`` (see mlir/)."""
    return _TjEmitter(graph).run()

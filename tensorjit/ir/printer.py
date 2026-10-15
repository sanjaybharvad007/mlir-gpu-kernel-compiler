"""Human-readable IR dump (MLIR-flavoured)."""
from __future__ import annotations

from .graph import Graph


def _fmt_attr(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def print_graph(g: Graph) -> str:
    args = ", ".join(f"{i.name}: {i.type}" for i in g.inputs)
    rets = ", ".join(str(o.type) for o in g.outputs)
    lines = [f"func @{g.name}({args}) -> ({rets}) {{"]
    for n in g.nodes:
        if n.op == "input":
            continue
        attrs = ""
        if n.attrs:
            attrs = " {" + ", ".join(f"{k} = {_fmt_attr(v)}" for k, v in n.attrs.items()) + "}"
        lines.append(f"  {n.name} = {n.op}({', '.join(i.name for i in n.inputs)}){attrs} : {n.type}")
    lines.append(f"  return {', '.join(o.name for o in g.outputs)}")
    lines.append("}")
    return "\n".join(lines)

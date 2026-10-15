"""Tiny pass manager with optional IR dumping between passes (TENSORJIT_DUMP=1)."""
from __future__ import annotations

import os
import sys
from typing import Callable

from ..ir import Graph

Pass = Callable[[Graph], bool]  # returns True if the graph changed


class PassManager:
    def __init__(self, passes: list[tuple[str, Pass]]):
        self.passes = passes
        self.dump = os.environ.get("TENSORJIT_DUMP", "0") == "1"

    def run(self, g: Graph, max_iters: int = 8) -> Graph:
        if self.dump:
            print(f"=== before passes ===\n{g}", file=sys.stderr)
        for it in range(max_iters):
            changed = False
            for name, p in self.passes:
                c = p(g)
                changed |= c
                if c:
                    g.toposort()
                if self.dump and c:
                    print(f"=== after {name} (iter {it}) ===\n{g}", file=sys.stderr)
            if not changed:
                break
        g.toposort()
        return g

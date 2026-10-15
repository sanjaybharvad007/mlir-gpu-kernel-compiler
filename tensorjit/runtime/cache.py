"""On-disk caches: compiled shared objects and persisted autotune choices.

Cache root: ``$TENSORJIT_CACHE_DIR`` or ``~/.cache/tensorjit``.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


def cache_dir() -> Path:
    d = Path(os.environ.get("TENSORJIT_CACHE_DIR", Path.home() / ".cache" / "tensorjit"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:20]


class AutotuneStore:
    """JSON-backed key -> best-config store shared across processes."""

    _lock = threading.Lock()

    def __init__(self, name: str = "autotune.json"):
        self.path = cache_dir() / name
        self._data: dict[str, dict] | None = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._load().get(key)

    def put(self, key: str, cfg: dict, ms: float) -> None:
        with self._lock:
            d = self._load()
            d[key] = {"config": cfg, "ms": ms}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
            os.replace(tmp, self.path)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            if self.path.exists():
                self.path.unlink()

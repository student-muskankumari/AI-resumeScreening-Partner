"""Small JSON-file cache so repeated runs make no repeated network calls.

Best effort by design: a cache that cannot be read or written is ignored and
the pipeline simply does the work again.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


class DiskCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def _path(self, namespace: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:150]
        return self.root / namespace / f"{safe}.json"

    def get(self, namespace: str, key: str, max_age_seconds: float | None = None) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        try:
            if max_age_seconds is not None and time.time() - path.stat().st_mtime > max_age_seconds:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, namespace: str, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

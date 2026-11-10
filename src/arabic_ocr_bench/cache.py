"""Per-line prediction resume cache stored as ONE append-only JSONL file.

Why not one JSON file per prediction (the old ``results/_cache/<sha1>.json``)?
The repo lives on an exFAT SSD whose allocation block is 1 MiB, so every tiny
file costs 1 MiB on disk: ~10k cache entries = 19 MB of data but ~10 GB of
disk, and every git worktree re-materialises all of it. One JSONL file costs
only its real size (~20 MB).

Why JSONL and not SQLite? The cache is committed on purpose so that Colab / 3090
runs can send results back through git. Append-only text merges cleanly with
git's ``merge=union`` driver (see ``.gitattributes``); a binary DB would not.

Format: one JSON object per line, ``{"_k": <cache key>, ...row}``. If a key
appears twice the LAST line wins (so a retried error can be superseded).
The legacy ``_cache/<key>.json`` directory is still read as a fallback, so old
Colab notebooks that write it keep working.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

FILENAME = "_cache.jsonl"
LEGACY_DIRNAME = "_cache"


class ResultCache:
    def __init__(self, root: str | Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / FILENAME
        self.legacy_dir = root / LEGACY_DIRNAME
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn write from a crash — ignore that line
                key = obj.pop("_k", None)
                if key:
                    self._rows[key] = obj
        # a crash mid-append can leave the file without a trailing newline;
        # restore it so the next append starts on its own line
        if self.path.stat().st_size and not self._ends_with_newline():
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write("\n")

    def _ends_with_newline(self) -> bool:
        with self.path.open("rb") as fh:
            fh.seek(-1, 2)
            return fh.read(1) == b"\n"

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._rows.get(key)
        if row is not None:
            return dict(row)
        legacy = self.legacy_dir / f"{key}.json"
        if legacy.exists():
            try:
                return json.loads(legacy.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def put(self, key: str, row: dict) -> None:
        line = json.dumps({"_k": key, **row}, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._rows[key] = dict(row)

    def __len__(self) -> int:
        return len(self._rows)

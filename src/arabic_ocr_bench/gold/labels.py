"""Label storage: gold/labels/<labeler>/<tranche>.jsonl, append-only (the newest record per line wins, history stays)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

_LOCK = threading.Lock()


def label_path(root: str | Path, labeler: str, tranche_name: str) -> Path:
    return Path(root) / labeler / f"{tranche_name}.jsonl"


def append_label(root: str | Path, labeler: str, tranche_name: str, rec: dict) -> None:
    p = label_path(root, labeler, tranche_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def load_labels(root: str | Path, labeler: str, tranche_name: str) -> dict[str, dict]:
    """line_id -> newest record for this labeler."""
    p = label_path(root, labeler, tranche_name)
    out: dict[str, dict] = {}
    if p.exists():
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                out[r["line_id"]] = r
    return out


def list_labelers(root: str | Path, tranche_name: str) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / f"{tranche_name}.jsonl").exists())

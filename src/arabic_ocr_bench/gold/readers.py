"""Offline readers for the cached line datasets (HF parquet snapshots, plus the local AraMS zip).

Rows come back in storage order with a stable id ``<name>/<split>/<row:06d>`` (the same ids the Tadqiq scans use,
so toolkit flags can be joined by id). Nothing is downloaded here.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import zipfile
from pathlib import Path
from typing import Iterator

GT_KEYS = ("text", "transcription", "answer", "label", "ground_truth", "gt", "target")
META = {"manuscript": ("manuscript_name", "source"), "page": ("image_name",), "file": ("filename",),
        "region": ("region_type",), "script": ("script",)}

# local (non-HF) line datasets: repo id -> (metadata.csv, images zip)
LOCAL = {"local/AraMS-28k-HTR": ("/Volumes/PortableSSD/data/arams_28k_htr/AraMs-28k-HTR/metadata.csv",
                                 "/Volumes/PortableSSD/data/arams_28k_htr/AraMS-28k-HTR.zip")}


def hub_dir() -> Path:
    return Path(os.environ.get("GOLD_HUB", "/Volumes/PortableSSD/huggingface/hub"))


def source_name(repo: str) -> str:
    return repo.split("/")[-1]


def _snapshot_root(repo: str) -> Path:
    return hub_dir() / f"datasets--{repo.replace('/', '--')}" / "snapshots"


def snapshot_id(repo: str) -> str:
    """Revision folder name of the newest cached snapshot (provenance for the manifest)."""
    if repo in LOCAL:
        return "local:" + Path(LOCAL[repo][1]).name
    snaps = [p for p in _snapshot_root(repo).glob("*") if p.is_dir()]
    return max(snaps, key=lambda p: p.stat().st_mtime).name if snaps else "missing"


def files(repo: str) -> dict[str, list[str]]:
    """split -> parquet shards of the newest snapshot."""
    snap = _snapshot_root(repo)
    fs = [f for f in glob.glob(str(snap / "*" / "**" / "*.parquet"), recursive=True)
          if not os.path.basename(f).startswith("._")]
    if not fs:
        return {}
    newest = max({Path(f).relative_to(snap).parts[0] for f in fs}, key=lambda s: os.path.getmtime(snap / s))
    out: dict[str, list[str]] = {}
    for f in sorted(f for f in fs if Path(f).relative_to(snap).parts[0] == newest):
        m = re.match(r"([A-Za-z_]+?)-\d", os.path.basename(f))
        out.setdefault(m.group(1) if m else Path(f).stem, []).append(f)
    return out


def iter_rows(repo: str, split: str, images: bool = True,
              only_rows: set[int] | None = None) -> Iterator[tuple[int, str, bytes | None, dict]]:
    """Yield (row, gt, raw_image_bytes | None, meta) in storage order.

    ``only_rows`` skips the image decode/read for every other row (gt/meta are still yielded for all rows only when
    only_rows is None; otherwise only the wanted rows are yielded).
    """
    if repo in LOCAL:
        meta_csv, zpath = LOCAL[repo]
        z = zipfile.ZipFile(zpath) if images else None
        i = -1
        for r in csv.DictReader(open(meta_csv, encoding="utf-8-sig")):
            if r["split"] != split:
                continue
            i += 1
            if only_rows is not None and i not in only_rows:
                continue
            meta = {"manuscript": r["book_id"], "page": r["page_id"], "line": r["line_idx"], "region": r["line_type"],
                    "file": os.path.basename(r["img_path"])}
            raw = None
            if z is not None:
                try:
                    raw = z.read(r["img_path"].lstrip("./"))
                except KeyError:
                    raw = None
            yield i, r["gt_text"] or "", raw, meta
        return
    import pyarrow.parquet as pq
    i = -1
    for f in files(repo)[split]:
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        gk = next(k for k in GT_KEYS if k in cols)
        want = [gk] + [c for v in META.values() for c in v if c in cols] + (["image"] if images and "image" in cols else [])
        for b in pf.iter_batches(columns=want, batch_size=256):
            d = b.to_pydict()
            for j in range(b.num_rows):
                i += 1
                if only_rows is not None and i not in only_rows:
                    continue
                meta = {}
                for mk, cands in META.items():
                    for c in cands:
                        if c in d and d[c][j] not in (None, ""):
                            meta[mk] = str(d[c][j])
                            break
                raw = d["image"][j]["bytes"] if images and "image" in d and d["image"][j] else None
                yield i, d[gk][j] or "", raw, meta

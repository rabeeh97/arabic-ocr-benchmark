"""Seeded ranking + image de-duplication of every eligible test line.

For each dataset: every row of the chosen split gets ``rank_key = sha256(seed|dataset|line_id)``; rows whose image is a
duplicate of an earlier kept image (same dataset, or a higher-priority dataset) are set aside; the rest are ranked by
key. "The first 50 / 100 / 150 ..." of a dataset are therefore plain prefixes of one fixed list: any prefix is a
uniform random sample, and lines added to a dataset later never reorder the lines already ranked.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import yaml

from . import images, readers


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    keys = [d["key"] for d in cfg["datasets"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset keys in config")
    return cfg


def line_id(key: str, split: str, row: int) -> str:
    return f"{key}/{split}/{row:06d}"


def rank_key(seed: str, key: str, lid: str) -> str:
    return hashlib.sha256(f"{seed}|{key}|{lid}".encode("utf-8")).hexdigest()


class DupIndex:
    """Exact-hash dict + multi-index buckets over the difference hash (pigeonhole: with <= max_hamming differing bits,
    at least one of ``chunks`` equal-width chunks is identical)."""

    def __init__(self, bits: int, max_hamming: int, min_corr: float, aspect_tol: float):
        self.n = bits * bits
        self.chunks = max_hamming + 1
        self.width = -(-self.n // self.chunks)
        self.max_hamming, self.min_corr, self.aspect_tol = max_hamming, min_corr, aspect_tol
        self.exact: dict[str, str] = {}
        self.buckets: dict[tuple[int, int], list[str]] = {}
        self.sigs: dict[str, dict] = {}

    def _chunks(self, dh: int):
        mask = (1 << self.width) - 1
        for c in range(self.chunks):
            yield c, (dh >> (c * self.width)) & mask

    def find(self, sig: dict) -> tuple[str, str] | None:
        if sig["sha"] in self.exact:
            return self.exact[sig["sha"]], "exact"
        seen: set[str] = set()
        for c, v in self._chunks(sig["dhash"]):
            for other in self.buckets.get((c, v), ()):
                if other in seen:
                    continue
                seen.add(other)
                if images.near_duplicate(sig, self.sigs[other], self.max_hamming, self.min_corr, self.aspect_tol):
                    return other, "near"
        return None

    def add(self, lid: str, sig: dict) -> None:
        self.exact.setdefault(sig["sha"], lid)
        self.sigs[lid] = sig
        for c, v in self._chunks(sig["dhash"]):
            self.buckets.setdefault((c, v), []).append(lid)


def _signatures(spec: dict, bits: int, cache_dir: Path | None, progress) -> list[dict]:
    snap = readers.snapshot_id(spec["repo"])
    cache = cache_dir / f"sig_{spec['key']}_{spec['split']}_{snap}.pkl" if cache_dir else None
    if cache and cache.exists():
        return pickle.load(open(cache, "rb"))
    sigs: list[dict] = []
    for row, _gt, raw, _meta in readers.iter_rows(spec["repo"], spec["split"], images=True):
        if raw is None:
            sigs.append({"sha": f"noimage-{row}", "dhash": 0, "thumb": b"\0" * 576, "w": 1, "h": 1, "missing": True})
        else:
            sigs.append(images.signature(raw, bits))
        if progress and (row + 1) % 2000 == 0:
            progress(f"    {spec['key']}: {row + 1} images hashed")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(sigs, open(cache, "wb"))
    return sigs


def build_pool(cfg: dict, only: list[str] | None = None, cache_dir: str | Path | None = None, progress=print):
    """Return ({key: rows}, meta). Rows: non-duplicates first in rank order, then duplicates (rank None)."""
    dd = cfg["dedup"]
    index = DupIndex(dd["dhash_bits"], dd["max_hamming"], dd["min_corr"], dd["aspect_tol"])
    cache_dir = Path(cache_dir) if cache_dir else None
    specs = sorted(cfg["datasets"], key=lambda d: (d["priority"], d["key"]))
    pool: dict[str, list[dict]] = {}
    meta: dict = {"seed": cfg["seed"], "datasets": {}}
    for spec in specs:
        if only and spec["key"] not in only:
            continue
        key, split = spec["key"], spec["split"]
        progress(f"  {key} ({spec['repo']} / {split})")
        sigs = _signatures(spec, dd["dhash_bits"], cache_dir, progress)
        rows, kinds, cross = [], Counter(), Counter()
        for row, sig in enumerate(sigs):
            lid = line_id(key, split, row)
            rec = {"line_id": lid, "row": row, "rank": None, "rank_key": rank_key(cfg["seed"], key, lid),
                   "dup_of": None, "dup_kind": None, "sha": sig["sha"][:16], "dhash": f"{sig['dhash']:064x}",
                   "w": sig["w"], "h": sig["h"]}
            if sig.get("missing"):
                rec["dup_of"], rec["dup_kind"] = None, "no_image"
            else:
                hit = index.find(sig)
                if hit:
                    rec["dup_of"], rec["dup_kind"] = hit
                    kinds[hit[1]] += 1
                    cross[hit[0].split("/")[0]] += 1
                else:
                    index.add(lid, sig)
            rows.append(rec)
        kept = sorted((r for r in rows if r["dup_of"] is None and r["dup_kind"] != "no_image"),
                      key=lambda r: r["rank_key"])
        for i, r in enumerate(kept, 1):
            r["rank"] = i
        rest = [r for r in rows if r["rank"] is None]
        pool[key] = kept + rest
        meta["datasets"][key] = {
            "repo": spec["repo"], "split": split, "snapshot": readers.snapshot_id(spec["repo"]),
            "rows": len(rows), "eligible": len(kept), "dup_exact": kinds["exact"], "dup_near": kinds["near"],
            "no_image": sum(r["dup_kind"] == "no_image" for r in rows),
            "dups_by_dataset": dict(cross),
        }
        progress(f"    rows {len(rows)}, eligible {len(kept)}, exact dups {kinds['exact']}, near dups {kinds['near']}"
                 + (f", of which in: {dict(cross)}" if cross else ""))
    return pool, meta


def write_pool(pool: dict, meta: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, rows in pool.items():
        with open(out / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = {}
    if (out / "_meta.json").exists():
        old = json.load(open(out / "_meta.json", encoding="utf-8")).get("datasets", {})
    old.update(meta["datasets"])
    json.dump({"seed": meta["seed"], "datasets": old}, open(out / "_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)


def read_pool(pool_dir: str | Path, key: str) -> list[dict]:
    return [json.loads(l) for l in open(Path(pool_dir) / f"{key}.jsonl", encoding="utf-8") if l.strip()]


def tranche_rows(rows: list[dict], tranche: int, size: int) -> list[dict]:
    """Ranks (tranche-1)*size+1 .. tranche*size. Capacity-limited datasets simply return fewer rows."""
    lo, hi = (tranche - 1) * size, tranche * size
    return [r for r in rows if r["rank"] is not None and lo < r["rank"] <= hi]

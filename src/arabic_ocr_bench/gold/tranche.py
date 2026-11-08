"""Export a tranche: ranks (t-1)*size+1 .. t*size of every dataset, with images, tags and toolkit notes."""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from . import pool as poolmod
from . import readers, tags

CALIBRATION_RANKS = 5  # the first 5 lines of every dataset in tranche 1 are labelled by every labeler (agreement check)


def _toolkit_notes(results_dir: Path, toolkit_dir: str | None, wanted: set[str]) -> dict[str, list[dict]]:
    if not toolkit_dir:
        return {}
    f = results_dir / toolkit_dir / "flags.jsonl"
    if not f.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for ln in open(f, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r["id"] in wanted:
            out[r["id"]] = [{"check": x["check"], "level": x.get("level"), "detail": x.get("detail", ""),
                             "score": x.get("score")} for x in r.get("flags", [])]
    return out


def _ext(raw: bytes) -> str:
    fmt = (Image.open(io.BytesIO(raw)).format or "PNG").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)


def export_tranche(cfg: dict, pool_dir: str | Path, out_dir: str | Path, tranche: int, root: str | Path,
                   only: list[str] | None = None, progress=print) -> dict:
    size = cfg["tranche_size"]
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    results_dir = Path(root) / cfg["toolkit_results"]
    lines: list[dict] = []
    manifest = {"seed": cfg["seed"], "tranche": tranche, "size": size, "datasets": {}}
    for spec in sorted(cfg["datasets"], key=lambda d: (d["tier"], d["priority"], d["key"])):
        key = spec["key"]
        if only and key not in only:
            continue
        sel = poolmod.tranche_rows(poolmod.read_pool(pool_dir, key), tranche, size)
        if not sel:
            progress(f"  {key}: no lines at ranks {(tranche - 1) * size + 1}-{tranche * size} (capacity reached)")
            manifest["datasets"][key] = {"line_ids": []}
            continue
        by_row = {r["row"]: r for r in sel}
        name = readers.source_name(spec["repo"])
        wanted_src = {f"{name}/{spec['split']}/{row:06d}" for row in by_row}
        notes = _toolkit_notes(results_dir, spec.get("toolkit_dir"), wanted_src)
        (out / "images" / key).mkdir(exist_ok=True)
        got = []
        for row, gt, raw, meta in readers.iter_rows(spec["repo"], spec["split"], images=True, only_rows=set(by_row)):
            r = by_row[row]
            rel = None
            if raw:
                rel = f"images/{key}/{row:06d}.{_ext(raw)}"
                (out / rel).write_bytes(raw)
            src_id = f"{name}/{spec['split']}/{row:06d}"
            got.append({
                "line_id": r["line_id"], "source_id": src_id, "dataset": key, "tier": spec["tier"],
                "language": spec["language"], "tranche": tranche, "rank": r["rank"],
                "calibration": tranche == 1 and r["rank"] <= CALIBRATION_RANKS,
                "text_original": gt, "image": rel, "tags": tags.auto_tags(spec, gt, raw),
                "toolkit_notes": notes.get(src_id, []),
            })
        got.sort(key=lambda x: x["rank"])
        lines.extend(got)
        manifest["datasets"][key] = {"line_ids": [x["line_id"] for x in got], "snapshot": readers.snapshot_id(spec["repo"])}
        progress(f"  {key}: {len(got)} lines (ranks {got[0]['rank']}-{got[-1]['rank']}), "
                 f"{sum(1 for x in got if x['toolkit_notes'])} with toolkit notes")
    with open(out / "lines.jsonl", "w", encoding="utf-8") as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    manifest["n_lines"] = len(lines)
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return manifest


def load_tranche(tranche_dir: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(Path(tranche_dir) / "lines.jsonl", encoding="utf-8") if l.strip()]

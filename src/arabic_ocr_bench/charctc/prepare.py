"""Write a resumable on-disk line cache (png + labels.jsonl) from training sources."""

from __future__ import annotations

import json
import random
from pathlib import Path

from arabic_ocr_bench.charctc.charset import canonicalize_label
from arabic_ocr_bench.charctc.config import CharCTCConfig
from arabic_ocr_bench.training.data import iter_source_samples

MANIFEST = "manifest.json"
LABELS = "labels.jsonl"


def prepare_cache(cfg: CharCTCConfig, out: Path, *, log=print) -> dict:
    """Materialise train/val line images. Skips pngs that already exist."""
    out = Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_skipped = 0
    for src in cfg.data.sources:
        n_src = 0
        for sample in iter_source_samples(src):
            if not sample.ground_truth or not sample.ground_truth.strip():
                n_skipped += 1
                continue
            label = canonicalize_label(sample.ground_truth)
            if not label:
                n_skipped += 1
                continue
            safe_id = sample.id.replace("/", "_").replace("\\", "_")
            rel = f"images/{safe_id}.png"
            dest = out / rel
            if not dest.exists():
                sample.image.convert("RGB").save(dest)
            rows.append(
                {
                    "id": sample.id,
                    "path": rel,
                    "text": label,
                    "source": src.name,
                }
            )
            n_src += 1
            if n_src % 200 == 0:
                log(f"prepare: {src.name} {n_src} lines written")
        log(f"prepare: {src.name} done ({n_src} lines)")

    rng = random.Random(cfg.train.seed)
    rng.shuffle(rows)
    n_val = round(len(rows) * cfg.data.val_fraction)
    for i, row in enumerate(rows):
        row["split"] = "val" if i < n_val else "train"

    labels_path = out / LABELS
    with labels_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val_actual = len(rows) - n_train
    manifest = {
        "n": len(rows),
        "n_train": n_train,
        "n_val": n_val_actual,
        "n_skipped_empty": n_skipped,
        "seed": cfg.train.seed,
        "val_fraction": cfg.data.val_fraction,
        "sources": [s.name for s in cfg.data.sources],
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"prepare: {n_train} train / {n_val_actual} val  ({n_skipped} empty dropped)")
    return manifest


def write_charset(cache: Path, dest: Path, *, log=print):
    from arabic_ocr_bench.charctc.charset import build_charset

    rows = load_label_rows(cache)
    charset = build_charset([r["text"] for r in rows])
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    charset.save(dest)
    oov = {"train": 0, "val": 0}
    for row in rows:
        split = row.get("split", "train")
        if charset.oov_chars(row["text"]):
            oov[split] = oov.get(split, 0) + 1
    log(
        f"charset: {charset.n_classes} classes "
        f"(blank+unk+{charset.n_classes - 2} glyphs) -> {dest}"
    )
    log(f"charset: sample tokens {list(charset.tokens[:24])}")
    log(f"charset: OOV lines vs this charset {oov}")
    return charset


def load_label_rows(cache: Path, split: str | None = None) -> list[dict]:
    rows = []
    with (cache / LABELS).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split is None or row.get("split") == split:
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    from arabic_ocr_bench.charctc.config import load_charctc_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="cache directory (images + labels.jsonl)")
    ap.add_argument("--charset-out", default=None, help="also write charset.json here")
    args = ap.parse_args(argv)
    cfg = load_charctc_config(args.config)
    cache = Path(args.out)
    prepare_cache(cfg, cache)
    if args.charset_out:
        write_charset(cache, Path(args.charset_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


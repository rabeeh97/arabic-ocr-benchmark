"""Convert one of our training configs' `data.sources` into PaddleOCR /
PaddleX recognition-fine-tuning format: a flat directory of cropped line
images plus tab-separated label files.

    python scripts/prep_paddleocr_labels.py \\
        --config configs/training/suite_v1/qwen3vl4b.yaml \\
        --out data/paddleocr_labels/suite_v1

Writes, for each requested split:

  * ``{split}_list.txt`` — PaddleOCR ``tools/train.py`` convention
    (``image_path<TAB>label``)
  * ``train.txt`` / ``val.txt`` — PaddleX ``MSTextRecDataset`` names

Deliberately reuses the SAME `data.sources` a Qwen fine-tune trains on (same
adapters — MuharafLines, AraMSHTR, KitabBench — same leak guard via
`check_source_not_scored`/`resolve_source_hf_id`) rather than re-implementing
dataset loading a second time for a second framework. This only prepares
DATA — it does not install or invoke PaddleOCR's own training tooling
(PaddlePaddle, not PyTorch; needs a SEPARATE venv from the one this repo's
[local,train] extra builds — see docs/experiments/paddleocr-runbook.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_PADDLEX_NAME = {"train": "train.txt", "val": "val.txt"}


def write_split(rows: list[dict], out: Path, split: str) -> int:
    """Save ``rows`` from ``assemble_dataset`` as images + label files.

    Returns the number of non-empty labels written.
    """
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    list_path = out / f"{split}_list.txt"
    paddlex_path = out / _PADDLEX_NAME[split]
    n_written = 0
    with list_path.open("w", encoding="utf-8") as list_fh, paddlex_path.open(
        "w", encoding="utf-8"
    ) as paddlex_fh:
        for row in rows:
            # assemble_dataset's rows are chat-shaped for the VLM trainer
            # (messages/images/source/id); recover the one image + its label
            # (the assistant turn's text) rather than re-deriving from the
            # adapter directly, so this is byte-for-byte the same data a
            # Qwen fine-tune on this config would train on.
            image = row["images"][0]
            label = row["messages"][-1]["content"][0]["text"]
            if not label.strip():
                continue
            safe_id = row["id"].replace("/", "_").replace("\\", "_")
            rel_path = f"images/{safe_id}.png"
            image.save(out / rel_path)
            line = f"{rel_path}\t{label}\n"
            list_fh.write(line)
            paddlex_fh.write(line)
            n_written += 1
    return n_written


def _cap_sources(cfg, per_source_limit: int | None):
    if per_source_limit is None:
        return
    for src in cfg.data.sources:
        if src.limit is None or src.limit > per_source_limit:
            src.limit = per_source_limit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        required=True,
        help="a configs/training/*.yaml — reuses its data.sources",
    )
    ap.add_argument("--out", required=True, help="destination directory for images/ + labels")
    ap.add_argument(
        "--split",
        default="both",
        choices=["train", "val", "both"],
        help="which half of the config's own val_fraction split to write (default: both)",
    )
    ap.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="cap every data.sources[].limit (smoke fine-tune / quick label check)",
    )
    args = ap.parse_args(argv)

    from arabic_ocr_bench.training.config import load_training_config
    from arabic_ocr_bench.training.data import assemble_dataset

    cfg = load_training_config(args.config)
    _cap_sources(cfg, args.per_source_limit)
    assembled = assemble_dataset(cfg.data)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        n = write_split(assembled[split], out, split)
        print(f"wrote {n} {split} lines to {out / f'{split}_list.txt'}")
    print(f"images under {out / 'images'}")
    print(
        "Next: PaddleOCR recognition fine-tuning — see "
        "docs/experiments/paddleocr-runbook.md and scripts/finetune_paddleocr.py. "
        "Needs a separate venv from this repo's torch/transformers one, and "
        "must not share the GPU with an in-flight VLM training run."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

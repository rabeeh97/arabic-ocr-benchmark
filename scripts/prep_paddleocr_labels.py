"""Convert one of our training configs' `data.sources` into PaddleOCR's
recognition-fine-tuning format: a flat directory of cropped line images plus
a tab-separated label file, `<relative_image_path>\\t<label>` per line.

    python scripts/prep_paddleocr_labels.py \\
        --config configs/training/suite_v1/qwen3vl4b.yaml \\
        --out data/paddleocr_labels/suite_v1

Deliberately reuses the SAME `data.sources` a Qwen fine-tune trains on (same
adapters — MuharafLines, AraMSHTR, KitabBench — same leak guard via
`check_source_not_scored`/`resolve_source_hf_id`) rather than re-implementing
dataset loading a second time for a second framework. This only prepares
DATA — it does not install or invoke PaddleOCR's own training tooling
(PaddlePaddle, not PyTorch; needs a SEPARATE venv from the one this repo's
[local,train] extra builds — see docs/experiments/paddleocr-runbook.md for
why, and for the exact `tools/train.py` invocation once this data exists).

Not run in this session, per the same "write scripts, don't fetch/run heavy
things automatically" convention as fetch_arams.py / fetch_training_corpus.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a configs/training/*.yaml — reuses its data.sources")
    ap.add_argument("--out", required=True, help="destination directory for images/ + label.txt")
    ap.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="which half of the config's own val_fraction split to write (default: train)",
    )
    args = ap.parse_args()

    from arabic_ocr_bench.training.config import load_training_config
    from arabic_ocr_bench.training.data import assemble_dataset

    cfg = load_training_config(args.config)
    rows = assemble_dataset(cfg.data)[args.split]

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    label_path = out / f"{args.split}_list.txt"
    n_written = 0
    with label_path.open("w", encoding="utf-8") as fh:
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
            safe_id = row["id"].replace("/", "_")
            rel_path = f"images/{safe_id}.png"
            image.save(out / rel_path)
            fh.write(f"{rel_path}\t{label}\n")
            n_written += 1

    print(f"wrote {n_written} lines to {label_path}")
    print(f"images under {img_dir}")
    print(
        "Next: PaddleOCR's own recognition fine-tuning "
        "(tools/train.py -c <rec_config>.yml, pretrained_model pointing at "
        "arabic_PP-OCRv5_mobile_rec's checkpoint) — see "
        "docs/experiments/paddleocr-runbook.md for the exact invocation and "
        "why it needs a separate venv from this repo's torch/transformers one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

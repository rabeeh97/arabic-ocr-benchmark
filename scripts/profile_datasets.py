"""Text-layer profile of every dataset we have ground truth for (characterisation P1).

Ground truth is recovered from the prediction files we already have (every
row carries `ground_truth`), deduplicated by `sample_id`, so no dataset needs
to be re-downloaded. Muharaf `train` -- our fine-tuning data -- is read from the
local Hugging Face cache (text column only) with --muharaf-train.

    .venv/bin/python scripts/profile_datasets.py --muharaf-train

Writes results/characterize/<UTC stamp>/{summary.json,lines.jsonl} and prints
the output directory. scripts/build_atlas.py turns summary.json into the Atlas
page.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from arabic_ocr_bench.characterize.text import (
    ALPHABET,
    POSITIONS,
    RASM_CLASSES,
    analyze_letters,
    entropy,
    gini,
    line_features,
)
from arabic_ocr_bench.normalize import normalize

PRED_GLOBS = (
    "results/**/predictions*.jsonl",
    "exp3090-results/**/predictions*.jsonl",
    "suite-v1-artifacts/**/predictions*.jsonl",
)
MIN_N = 50  # below this a dataset is shown but flagged as too small to read


def collect_ground_truth() -> tuple[dict[str, dict], int]:
    """sample_id -> {dataset, split, track, gt}. Returns (samples, n_files)."""
    samples: dict[str, dict] = {}
    files = sorted({f for g in PRED_GLOBS for f in glob.glob(g, recursive=True)})
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid, gt = r.get("sample_id"), r.get("ground_truth")
                if not sid or not gt or sid in samples:
                    continue
                parts = sid.split("/")
                samples[sid] = {
                    "dataset": r.get("dataset") or parts[0],
                    "split": parts[1] if len(parts) > 2 else "?",
                    "track": r.get("track") or "line",
                    "gt": gt,
                }
    return samples, len(files)


def muharaf_train() -> dict[str, dict]:
    from datasets import load_dataset

    ds = load_dataset("aamijar/muharaf-public", split="train").select_columns(["text"])
    return {
        f"muharaf_train/train/{i:06d}": {
            "dataset": "muharaf_train", "split": "train", "track": "line", "gt": t,
        }
        for i, t in enumerate(ds["text"]) if t
    }


def _words(gt: str) -> list[str]:
    return normalize(gt, "diac_insensitive").split()


def _example(sid: str, gt: str) -> dict:
    letters = analyze_letters(gt)
    from arabic_ocr_bench.characterize.text import prepare

    return {
        "sample_id": sid,
        "text": prepare(gt),
        "letters": [
            {"i": li.index, "ch": li.letter, "pos": li.position, "rasm": li.rasm,
             "amb": li.ambiguous, "w": li.word, "paw": li.paw, "marks": li.marks}
            for li in letters
        ],
    }


def pick_examples(rows: list[tuple[str, str, dict]], k: int = 4) -> list[dict]:
    """Short readable lines, preferring a mix with and without diacritics."""
    cands = [r for r in rows if 18 <= r[2]["chars"] <= 55 and r[2]["letters"] >= 12]
    cands.sort(key=lambda r: r[0])
    with_d = [r for r in cands if r[2]["diac_marks"] > 0]
    without = [r for r in cands if r[2]["diac_marks"] == 0]
    chosen = (with_d[: k // 2] + without)[:k] if with_d else cands[:k]
    if len(chosen) < k:
        chosen += [r for r in rows if r not in chosen][: k - len(chosen)]
    return [_example(sid, gt) for sid, gt, _ in chosen]


def summarize(
    name: str,
    rows: list[tuple[str, str, dict]],
    train_vocab: set[str] | None,
    train_lines: set[str] | None,
) -> dict:
    n = len(rows)
    feats = [f for _, _, f in rows]
    shapes: Counter = Counter()
    letters: Counter = Counter()
    rasm: Counter = Counter()
    dot_pairs: Counter = Counter()  # rasm class -> letter counts (dot-ambiguous only)
    for _, gt, _ in rows:
        for li in analyze_letters(gt):
            shapes[f"{li.letter}|{li.position}"] += 1
            letters[li.letter] += 1
            if li.rasm:
                rasm[f"{li.rasm}|{li.position}"] += 1
            if li.ambiguous:
                dot_pairs[f"{li.rasm}|{li.letter}"] += 1
    tot_letters = sum(letters.values()) or 1
    pos = Counter()
    for f in feats:
        pos.update(f["positions"])
    ppw = Counter()
    for f in feats:
        ppw.update(min(x, 6) for x in f["paws_per_word"])
    diac_types = Counter()
    for f in feats:
        diac_types.update(f["diac_types"])
    hyg_tot = Counter()
    hyg_lines = Counter()
    for f in feats:
        for k, v in f["hygiene"].items():
            hyg_tot[k] += v
            hyg_lines[k] += v > 0
    tot_chars = sum(f["chars"] for f in feats) or 1
    tot_words = sum(f["words"] for f in feats) or 1
    word_tokens = [w for _, gt, _ in rows for w in _words(gt)]
    vocab = Counter(word_tokens)
    shape_freq = sorted(shapes.values(), reverse=True)
    cum, k95 = 0, 0
    for c in shape_freq:
        cum += c
        k95 += 1
        if cum >= 0.95 * tot_letters:
            break
    h_shape = entropy(shapes)
    h_rasm = entropy(rasm)
    out = {
        "name": name,
        "n": n,
        "small": n < MIN_N,
        # the prediction rows all say "line"; flag by length instead
        "page_scale": bool(feats) and statistics.median(f["chars"] for f in feats) > 200,
        "chars_median": statistics.median(f["chars"] for f in feats) if feats else 0,
        "words_median": statistics.median(f["words"] for f in feats) if feats else 0,
        "letters_total": tot_letters,
        "positions": {p: pos[p] / tot_letters for p in POSITIONS},
        "shapes": dict(shapes),
        "letters": dict(letters),
        "rasm": dict(rasm),
        "dot_pairs": dict(dot_pairs),
        "shape_rank_freq": [c / tot_letters for c in shape_freq],
        "distinct_shapes": len(shapes),
        "shapes_for_95pct": k95,
        "shape_gini": gini(list(shapes.values())),
        "rare_shapes_lt10": sum(c < 10 for c in shapes.values()),
        "dotted_share": sum(f["dotted"] for f in feats) / tot_letters,
        "ambiguous_share": sum(f["ambiguous"] for f in feats) / tot_letters,
        "hamza_share": sum(f["hamza"] for f in feats) / tot_letters,
        "h_letter": entropy(letters),
        "h_shape": h_shape,
        "h_rasm": h_rasm,
        "dot_bits": h_shape - h_rasm,
        "paws_per_word": {str(k): v for k, v in sorted(ppw.items())},
        "paws_per_word_mean": (sum(k * v for k, v in ppw.items()) / sum(ppw.values())) if ppw else 0,
        "lam_alef_per_100_words": 100 * sum(f["lam_alef"] for f in feats) / tot_words,
        "diac_levels": {lv: sum(f["diac_level"] == lv for f in feats) / max(n, 1)
                        for lv in ("none", "sparse", "partial", "full")},
        "diac_coverage_mean": statistics.fmean(f["diac_coverage"] for f in feats) if feats else 0,
        "diac_types": dict(diac_types),
        "diac_invalid_per_1k_letters": 1000 * sum(f["diac_invalid"] for f in feats) / tot_letters,
        "hygiene_per_1k_chars": {k: 1000 * v / tot_chars for k, v in hyg_tot.items()},
        "hygiene_line_share": {k: v / max(n, 1) for k, v in hyg_lines.items()},
        "word_tokens": len(word_tokens),
        "word_types": len(vocab),
        "type_token_ratio": len(vocab) / max(len(word_tokens), 1),
        "hapax_share": sum(c == 1 for c in vocab.values()) / max(len(vocab), 1),
        "oov_vs_muharaf_train": (
            sum(w not in train_vocab for w in word_tokens) / max(len(word_tokens), 1)
            if train_vocab is not None else None
        ),
        "overlap_muharaf_train": _overlap(rows, train_lines),
        "examples": pick_examples(rows),
    }
    return out


def _overlap(rows, train_lines: set[str] | None) -> dict | None:
    """Share of lines whose normalised text appears verbatim in Muharaf train.
    `long` restricts to lines >= 20 chars, where a match is unlikely by chance."""
    if train_lines is None:
        return None
    norm = [normalize(gt, "diac_insensitive") for _, gt, _ in rows]
    long_ = [t for t in norm if len(t) >= 20]
    return {
        "all": sum(t in train_lines for t in norm) / max(len(norm), 1),
        "long": sum(t in train_lines for t in long_) / max(len(long_), 1),
        "long_n": len(long_),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--muharaf-train", action="store_true", help="also profile Muharaf train from the HF cache")
    ap.add_argument("--out", default="results/characterize")
    args = ap.parse_args()

    samples, n_files = collect_ground_truth()
    if args.muharaf_train:
        samples.update(muharaf_train())

    by_ds: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    out_dir = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lines.jsonl", "w", encoding="utf-8") as fh:
        for sid in sorted(samples):
            s = samples[sid]
            f = line_features(s["gt"])
            f["_track"] = s["track"]
            by_ds[s["dataset"]].append((sid, s["gt"], f))
            fh.write(json.dumps({"sample_id": sid, "dataset": s["dataset"], "split": s["split"],
                                 "track": s["track"], **f}, ensure_ascii=False) + "\n")

    train_vocab = train_lines = None
    if "muharaf_train" in by_ds:
        train_vocab = {w for _, gt, _ in by_ds["muharaf_train"] for w in _words(gt)}
        train_lines = {normalize(gt, "diac_insensitive") for _, gt, _ in by_ds["muharaf_train"]}

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"prediction_files": n_files, "samples": len(samples)},
        "alphabet": ALPHABET,
        "positions": list(POSITIONS),
        "rasm_classes": RASM_CLASSES,
        "datasets": {name: summarize(name, rows, train_vocab, train_lines if name != "muharaf_train" else None) for name, rows in sorted(by_ds.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()

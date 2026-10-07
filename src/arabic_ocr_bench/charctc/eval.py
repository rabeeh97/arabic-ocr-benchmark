"""Score a CharCTC checkpoint on registered line datasets (same metrics as bench)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arabic_ocr_bench.charctc.charset import canonicalize_label
from arabic_ocr_bench.charctc.infer import CharCTCRecognizer
from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs
from arabic_ocr_bench.metrics import aggregate, score_pair
from arabic_ocr_bench.normalize import PROFILES

EVAL_DATASETS = (
    "muharaf_lines",
    "kitab_khatt",
    "nakba_lines",
    "rasam1",
    "rasam2",
)


def _already_scored(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("id")
        if sid:
            seen.add(sid)
    return seen


def eval_dataset(
    rec: CharCTCRecognizer,
    dataset: str,
    *,
    out_dir: Path,
    limit: int,
    log=print,
) -> dict:
    cfgs = load_dataset_configs()
    if dataset not in cfgs:
        raise SystemExit(f"unknown dataset {dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    score_path = out_dir / "scores.jsonl"
    seen = _already_scored(pred_path)
    adapter = build_dataset(
        dataset, cfgs[dataset], limit=limit, offset=len(seen)
    )
    pair_scores = []
    n_new = 0
    n_oov = 0
    # re-read already-scored pair scores so accumulate + resume stay honest
    if score_path.exists():
        for line in score_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pair_scores.append(json.loads(line)["metrics"])

    pred_fh = pred_path.open("a", encoding="utf-8")
    score_fh = score_path.open("a", encoding="utf-8")
    try:
        for sample in adapter:
            if sample.id in seen:
                continue
            t0 = time.monotonic()
            try:
                hyp = rec.transcribe(sample.image)
                err = None
            except Exception as exc:  # noqa: BLE001
                hyp = ""
                err = f"{type(exc).__name__}: {exc}"
            latency = (time.monotonic() - t0) * 1000
            oov = rec.charset.oov_chars(sample.ground_truth)
            if oov:
                n_oov += 1
            pred_row = {
                "id": sample.id,
                "dataset": dataset,
                "model": "charctc-cpu",
                "raw_text": hyp,
                "cleaned_text": hyp,
                "ground_truth": sample.ground_truth,
                "latency_ms": latency,
                "error": err,
                "oov_chars": oov,
            }
            pred_fh.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            pred_fh.flush()
            metrics = score_pair(sample.ground_truth, hyp)
            score_fh.write(
                json.dumps(
                    {
                        "id": sample.id,
                        "dataset": dataset,
                        "metrics": metrics,
                        "gt_canonical": canonicalize_label(sample.ground_truth),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            score_fh.flush()
            pair_scores.append(metrics)
            n_new += 1
            if n_new % 20 == 0:
                log(f"eval {dataset}: {n_new} new / {len(seen) + n_new} total")
    finally:
        pred_fh.close()
        score_fh.close()

    agg = aggregate(pair_scores, profiles=PROFILES) if pair_scores else {"n": 0}
    summary = {
        "dataset": dataset,
        "n": agg.get("n", 0),
        "n_new": n_new,
        "n_oov_lines": n_oov,
        "cer_micro_light": (agg.get("micro_cer") or {}).get("light"),
        "cer_micro_raw": (agg.get("micro_cer") or {}).get("raw"),
        "cer_micro_diac_insensitive": (agg.get("micro_cer") or {}).get("diac_insensitive"),
        "wer_micro_light": (agg.get("micro_wer") or {}).get("light"),
        "diac_f1": (agg.get("diacritic") or {}).get("f1"),
        "checkpoint": str(rec.ckpt),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        f"# charctc-cpu / {dataset}\n\n"
        f"- n = {summary['n']} ({n_new} new this pass)\n"
        f"- CER-light = {summary['cer_micro_light']}\n"
        f"- CER-raw = {summary['cer_micro_raw']}\n"
        f"- CER-diac-insensitive = {summary['cer_micro_diac_insensitive']}\n"
        f"- WER-light = {summary['wer_micro_light']}\n"
        f"- diac-F1 = {summary['diac_f1']}\n"
        f"- lines with GT OOV vs train charset = {n_oov}\n"
        f"- checkpoint = `{rec.ckpt}`\n",
        encoding="utf-8",
    )
    log(
        f"eval {dataset}: n={summary['n']} CER-light={summary['cer_micro_light']} "
        f"OOV-lines={n_oov}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    rec = CharCTCRecognizer(args.ckpt, device=args.device)
    eval_dataset(rec, args.dataset, out_dir=Path(args.out), limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

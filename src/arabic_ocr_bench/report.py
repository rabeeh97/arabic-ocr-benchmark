"""predictions.jsonl -> per-sample scores.jsonl + summary.{csv,md}."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import aggregate, score_pair
from .normalize import PROFILES


def _load_predictions(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_run(run_dir: str | Path, profiles=PROFILES, *, progress: bool = True) -> Path:
    run_dir = Path(run_dir)
    preds = _load_predictions(run_dir)

    from tqdm.auto import tqdm

    scored: list[dict] = []
    for r in tqdm(preds, desc="scoring", unit="pred", disable=not progress):
        hyp = "" if r.get("error") else (r.get("cleaned_text") or "")
        ps = score_pair(r["ground_truth"], hyp, profiles=profiles)
        scored.append({
            "sample_id": r["sample_id"],
            "model": r["model"],
            "dataset": r["dataset"],
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms", 0.0),
            "cost_usd": r.get("cost_usd", 0.0),
            **{f"cer_{p}": ps["cer"][p] for p in profiles},
            **{f"wer_{p}": ps["wer"][p] for p in profiles},
            **{f"exact_{p}": ps["exact_match"][p] for p in profiles},
            "diac_precision": ps["diacritic"]["precision"],
            "diac_recall": ps["diacritic"]["recall"],
            "diac_f1": ps["diacritic"]["f1"],
            "_pair": ps,  # kept for aggregation, dropped before writing
        })

    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps({k: v for k, v in s.items() if k != "_pair"},
                                ensure_ascii=False) + "\n")

    summary_rows = []
    for (model, dataset), group in _group_by(scored, ("model", "dataset")):
        agg = aggregate([g["_pair"] for g in group], profiles=profiles)
        n = len(group)
        n_err = sum(1 for g in group if g["error"])
        row = {
            "model": model,
            "dataset": dataset,
            "n": n,
            "errors": n_err,
            "refusal_rate": round(n_err / n, 4) if n else 0.0,
            "latency_ms_mean": round(sum(g["latency_ms"] for g in group) / n, 1) if n else 0.0,
            "cost_usd_total": round(sum(g["cost_usd"] for g in group), 6),
        }
        for p in profiles:
            row[f"cer_micro_{p}"] = round(agg["micro_cer"][p], 4)
            row[f"cer_macro_{p}"] = round(agg["cer_macro"][p], 4)
            row[f"wer_micro_{p}"] = round(agg["micro_wer"][p], 4)
            row[f"exact_{p}"] = round(agg["exact_match_rate"][p], 4)
        row["diac_f1"] = round(agg["diacritic"]["f1"], 4)
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows).sort_values(["dataset", "cer_micro_raw"])
    df.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.md").write_text(_render_md(df, run_dir.name, profiles))
    return run_dir / "summary.md"


def _group_by(rows: list[dict], keys: tuple[str, ...]):
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in keys), []).append(r)
    return sorted(buckets.items())


def _render_md(df: pd.DataFrame, run_id: str, profiles) -> str:
    lines = [
        f"# Benchmark summary — `{run_id}`",
        "",
        "CER/WER: lower is better. `micro` = corpus-weighted, `macro` = per-sample mean.",
        "`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).",
        "",
    ]
    headline = ["model", "dataset", "n", "errors"]
    headline += [f"cer_micro_{p}" for p in profiles]
    headline += ["wer_micro_raw", "exact_raw", "diac_f1", "latency_ms_mean", "cost_usd_total"]
    view = df[[c for c in headline if c in df.columns]]
    lines.append(_df_to_md(view))
    lines.append("")
    lines.append("Full per-profile numbers in `summary.csv`; per-sample in `scores.jsonl`.")
    return "\n".join(lines) + "\n"


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

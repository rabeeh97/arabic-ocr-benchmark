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

    # Rows that came back with a transport / API error (r["error"] set) are
    # QUARANTINED: they are not scored, not counted in n, and not aggregated —
    # a 429 or a 500 is missing data, not a CER-1.0 transcription. They stay in
    # predictions.jsonl (and are echoed to scores.jsonl with error set, no CER)
    # so a later run can retry exactly those sample ids. A genuine empty answer
    # (error is None, cleaned_text "") is still a real prediction and is scored.
    scored: list[dict] = []
    errored: list[dict] = []
    for r in tqdm(preds, desc="scoring", unit="pred", disable=not progress):
        if r.get("error"):
            errored.append({
                "sample_id": r["sample_id"], "model": r["model"],
                "dataset": r["dataset"], "error": r["error"],
                "latency_ms": r.get("latency_ms", 0.0), "cost_usd": r.get("cost_usd", 0.0),
            })
            continue
        ps = score_pair(r["ground_truth"], r.get("cleaned_text") or "", profiles=profiles)
        scored.append({
            "sample_id": r["sample_id"],
            "model": r["model"],
            "dataset": r["dataset"],
            "error": None,
            "latency_ms": r.get("latency_ms", 0.0),
            "cost_usd": r.get("cost_usd", 0.0),
            **{f"cer_{p}": ps["cer"][p] for p in profiles},
            **{f"car_{p}": ps["car"][p] for p in profiles},
            **{f"wer_{p}": ps["wer"][p] for p in profiles},
            **{f"mer_{p}": ps["mer"][p] for p in profiles},
            **{f"wil_{p}": ps["wil"][p] for p in profiles},
            **{f"wip_{p}": ps["wip"][p] for p in profiles},
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
        for e in errored:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    err_by_key: dict[tuple[str, str], int] = {}
    for e in errored:
        k = (e["model"], e["dataset"])
        err_by_key[k] = err_by_key.get(k, 0) + 1

    summary_rows = []
    keys = sorted(
        {(s["model"], s["dataset"]) for s in scored} | set(err_by_key)
    )
    grouped = {k: g for k, g in _group_by(scored, ("model", "dataset"))}
    err_cost = {k: 0.0 for k in err_by_key}
    for e in errored:
        err_cost[(e["model"], e["dataset"])] += e["cost_usd"]
    for model, dataset in keys:
        group = grouped.get((model, dataset), [])
        n = len(group)
        n_err = err_by_key.get((model, dataset), 0)
        row = {
            "model": model,
            "dataset": dataset,
            "n": n,
            "errors": n_err,
            "refusal_rate": round(n_err / (n + n_err), 4) if (n + n_err) else 0.0,
            "latency_ms_mean": round(sum(g["latency_ms"] for g in group) / n, 1) if n else 0.0,
            "cost_usd_total": round(
                sum(g["cost_usd"] for g in group) + err_cost.get((model, dataset), 0.0), 6
            ),
        }
        if n:
            agg = aggregate([g["_pair"] for g in group], profiles=profiles)
            for p in profiles:
                row[f"cer_micro_{p}"] = round(agg["micro_cer"][p], 4)
                row[f"cer_macro_{p}"] = round(agg["cer_macro"][p], 4)
                row[f"car_micro_{p}"] = round(agg["micro_car"][p], 4)
                row[f"car_macro_{p}"] = round(agg["car_macro"][p], 4)
                row[f"wer_micro_{p}"] = round(agg["micro_wer"][p], 4)
                row[f"mer_micro_{p}"] = round(agg["micro_mer"][p], 4)
                row[f"mer_macro_{p}"] = round(agg["mer_macro"][p], 4)
                row[f"wil_micro_{p}"] = round(agg["micro_wil"][p], 4)
                row[f"wip_micro_{p}"] = round(agg["micro_wip"][p], 4)
                row[f"exact_{p}"] = round(agg["exact_match_rate"][p], 4)
            row["diac_f1"] = round(agg["diacritic"]["f1"], 4)
        else:
            # nothing scored (every attempt errored) — leave the metric cells
            # empty so the board shows "—", not a spurious 0.00 CER.
            for p in profiles:
                for m in ("cer_micro", "cer_macro", "car_micro", "car_macro",
                          "wer_micro", "mer_micro", "mer_macro",
                          "wil_micro", "wip_micro", "exact"):
                    row[f"{m}_{p}"] = ""
            row["diac_f1"] = ""
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "cer_micro_raw"],
        key=lambda c: pd.to_numeric(c, errors="coerce") if c.name == "cer_micro_raw" else c,
    )
    df.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.md").write_text(_render_md(df, run_dir.name, profiles))
    return run_dir / "summary.md"


def _group_by(rows: list[dict], keys: tuple[str, ...]):
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in keys), []).append(r)
    return sorted(buckets.items())


def _render_md(df: pd.DataFrame, run_id: str, profiles) -> str:
    note1 = ("CER/WER/MER/WIL: lower is better. CAR/WIP: higher is better. "
             "`micro` = corpus-weighted, `macro` = per-sample mean.")
    note3 = ("Full per-metric, per-profile numbers (CAR / MER / WIL / WIP too) "
             "are in `summary.csv`; see `docs/metrics.md` for definitions.")
    lines = [
        f"# Benchmark summary — `{run_id}`",
        "",
        note1,
        "`exact` = fraction of samples transcribed perfectly. `diac_f1` = harakat F1 (raw text).",
        note3,
        "",
    ]
    headline = ["model", "dataset", "n", "errors"]
    headline += [f"cer_micro_{p}" for p in profiles]
    headline += ["car_micro_light", "wer_micro_raw", "mer_micro_raw",
                 "exact_raw", "diac_f1", "latency_ms_mean", "cost_usd_total"]
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

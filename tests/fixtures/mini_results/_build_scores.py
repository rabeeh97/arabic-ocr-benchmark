# tests/fixtures/mini_results/_build_scores.py  (committed; documents provenance)
import json, pathlib
from arabic_ocr_bench.metrics import score_pair

for run in ("run_a", "run_b"):
    d = pathlib.Path(__file__).parent / run
    preds = [json.loads(l) for l in (d / "predictions.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for p in preds:
        if p["error"]:
            continue
        s = score_pair(p["ground_truth"], p["cleaned_text"])
        row = {"sample_id": p["sample_id"], "model": p["model"], "dataset": p["dataset"],
               "error": None, "latency_ms": p["latency_ms"], "cost_usd": p["cost_usd"]}
        for prof in ("raw", "light", "diac_insensitive"):
            row[f"cer_{prof}"] = s["cer"][prof]
            row[f"car_{prof}"] = s["car"][prof]
            row[f"wer_{prof}"] = s["wer"][prof]
            row[f"mer_{prof}"] = s["mer"][prof]
            row[f"wil_{prof}"] = s["wil"][prof]
            row[f"wip_{prof}"] = s["wip"][prof]
            row[f"exact_{prof}"] = s["exact_match"][prof]
        row["diac_precision"] = s["diacritic"]["precision"]
        row["diac_recall"] = s["diacritic"]["recall"]
        row["diac_f1"] = s["diacritic"]["f1"]
        out.append(row)
    (d / "scores.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")

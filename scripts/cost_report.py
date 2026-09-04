"""Cost report for a hosted-API benchmark run.

Reads the per-line token counts a run actually recorded, the pricing block in
configs/models.yaml, and the dataset sizes, then reports for each model:

  * mean input / output tokens per line (pooled, and per dataset)
  * cost per line
  * the run's own measured cost (sum of predictions' cost_usd)
  * projected cost to transcribe every line of each full dataset, and the total

Projection = mean cost/line (from this run's lines) x dataset row count. It is a
linear estimate: real cost varies with image size and how much a model rambles.

    uv run python scripts/cost_report.py results/<run_id> [--md report.md]

With no run dir, uses the newest results/<stamp>/ that contains *-paid models.
Dataset sizes are read from configs (via the adapters) unless --sizes-json is
given; a small on-disk cache avoids re-downloading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

import arabic_ocr_bench  # noqa: F401  -- loads .env

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "configs" / "models.yaml"
SIZES_CACHE = ROOT / "results" / "_dataset_sizes.json"

# The five real datasets a full sweep would cover (synthetic is a QA stub).
ALL_DATASETS = [
    "muharaf_lines",
    "kitab_khatt",
    "kitab_muharaf",
    "kitab_historicalbooks",
    "kitab_hindawi",
]


def _pricing() -> dict[str, dict]:
    cfgs = yaml.safe_load(MODELS_YAML.read_text())
    return {name: (c.get("pricing") or {}) for name, c in cfgs.items()}


def _dataset_sizes(explicit: Path | None) -> dict[str, int]:
    if explicit:
        return json.loads(explicit.read_text())
    if SIZES_CACHE.exists():
        cached = json.loads(SIZES_CACHE.read_text())
        if all(d in cached for d in ALL_DATASETS):
            return cached
    from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs

    dcfg = load_dataset_configs()
    sizes: dict[str, int] = {}
    for name in ALL_DATASETS:
        ds = build_dataset(name, dcfg[name], limit=None, offset=0)
        sizes[name] = len(ds)
    SIZES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SIZES_CACHE.write_text(json.dumps(sizes, indent=2))
    return sizes


def _latest_paid_run() -> Path:
    for d in sorted((ROOT / "results").glob("2026*/"), reverse=True):
        pred = d / "predictions.jsonl"
        if not pred.exists():
            continue
        if '-paid"' in pred.read_text(encoding="utf-8"):
            return d
    sys.exit("no results/<stamp>/ run dir with *-paid models found")


def _cost(pricing: dict, in_tok: float, out_tok: float) -> float:
    return (
        in_tok / 1e6 * pricing.get("input_usd_per_1m", 0.0)
        + out_tok / 1e6 * pricing.get("output_usd_per_1m", 0.0)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--md", type=Path, help="also write the tables to this markdown file")
    ap.add_argument("--sizes-json", type=Path, help="{dataset: rows} JSON, skips adapter load")
    args = ap.parse_args()

    run_dir = args.run_dir or _latest_paid_run()
    rows = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines() if l.strip()]
    pricing = _pricing()
    sizes = _dataset_sizes(args.sizes_json)

    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    # per (model): scored (non-error) rows only for token means; cost sums include all
    lines: list[str] = []
    lines.append(f"# Cost report — `{run_dir.name}`\n")
    lines.append(
        f"Run covers **{', '.join(datasets)}** at n per (model,dataset). "
        f"Token means are over non-errored lines. Projection = mean cost/line x full "
        f"dataset size. Pricing from `configs/models.yaml` (USD per 1M tokens).\n"
    )
    lines.append("Dataset sizes: " + ", ".join(f"`{d}` {sizes.get(d, '?')}" for d in ALL_DATASETS) + "\n")

    # --- Table: per-model token + cost, measured, and all-datasets projection ---
    hdr = (
        "| model | in $/1M | out $/1M | n | err | mean in tok | mean out tok | "
        "$/line | run cost (measured) | "
        + " | ".join(f"proj {d}" for d in ALL_DATASETS)
        + " | **proj all 5** |"
    )
    sep = "|" + "---|" * (9 + len(ALL_DATASETS) + 1)
    lines += ["## Per-model cost\n", hdr, sep]

    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        ok = [r for r in mrows if not r.get("error")]
        n, err = len(mrows), sum(1 for r in mrows if r.get("error"))
        pr = pricing.get(m, {})
        mean_in = sum(r["input_tokens"] for r in ok) / len(ok) if ok else 0.0
        mean_out = sum(r["output_tokens"] for r in ok) / len(ok) if ok else 0.0
        cost_line = _cost(pr, mean_in, mean_out)
        measured = sum(r.get("cost_usd", 0.0) for r in mrows)
        projs = [cost_line * sizes.get(d, 0) for d in ALL_DATASETS]
        row = (
            f"| {m} | {pr.get('input_usd_per_1m', 0):.2f} | {pr.get('output_usd_per_1m', 0):.2f} "
            f"| {n} | {err} | {mean_in:.0f} | {mean_out:.0f} | ${cost_line:.5f} | ${measured:.4f} | "
            + " | ".join(f"${p:.2f}" for p in projs)
            + f" | **${sum(projs):.2f}** |"
        )
        lines.append(row)

    # --- Table: per (model,dataset) measured detail ---
    lines += ["\n## Per (model, dataset) — measured\n",
              "| model | dataset | n | err | mean in tok | mean out tok | $/line | measured $ |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for d in datasets:
            g = [r for r in rows if r["model"] == m and r["dataset"] == d]
            if not g:
                continue
            ok = [r for r in g if not r.get("error")]
            mi = sum(r["input_tokens"] for r in ok) / len(ok) if ok else 0.0
            mo = sum(r["output_tokens"] for r in ok) / len(ok) if ok else 0.0
            cl = _cost(pricing.get(m, {}), mi, mo)
            lines.append(
                f"| {m} | {d} | {len(g)} | {sum(1 for r in g if r.get('error'))} | "
                f"{mi:.0f} | {mo:.0f} | ${cl:.5f} | ${sum(r.get('cost_usd', 0.0) for r in g):.4f} |"
            )

    grand_measured = sum(r.get("cost_usd", 0.0) for r in rows)
    lines.append(f"\n**Total measured cost of this run: ${grand_measured:.4f}** "
                 f"({len(rows)} lines, {sum(1 for r in rows if r.get('error'))} errored)\n")

    out = "\n".join(lines)
    print(out)
    if args.md:
        args.md.write_text(out)
        print(f"\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

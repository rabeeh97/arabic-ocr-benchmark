"""X6: which models' stored readings stop at their output-token cap.

    uv run python scripts/token_cap_report.py [--root .] [--json results/characterize/token_cap.json]

CPU only, reads predictions files, makes no API calls. Prints one row per model
(share of readings at the cap, split into truncated = text lost vs runaway =
loop/preamble hit the cap) and the models over the X6 threshold (> 5 %).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_ocr_bench.characterize import token_cap
from arabic_ocr_bench.models.registry import load_model_configs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--json", type=Path, default=None, help="also write per-pair rows here")
    ap.add_argument("--share", type=float, default=token_cap.DONE_SHARE)
    a = ap.parse_args()

    pairs = token_cap.scan(a.root, load_model_configs(a.models))
    models = token_cap.by_model(pairs)
    print(f"{'model':42s} {'caps':>12s} {'n':>6s} {'at cap':>7s} {'trunc':>6s} {'runaway':>7s} {'unknown':>7s}")
    for m, v in sorted(models.items(), key=lambda kv: -(kv[1]["share_at_cap"] or 0)):
        share = "-" if v["share_at_cap"] is None else f"{100 * v['share_at_cap']:.1f}%"
        caps = ",".join(map(str, v["caps"])) or "-"
        print(f"{m:42s} {caps:>12s} {v['n']:6d} {share:>7s} {v['truncated']:6d} {v['runaway']:7d} {v['unknown']:7d}")
    bad = token_cap.flagged(models, a.share)
    print(f"\nover {100 * a.share:.0f} % at the cap ({len(bad)}): {', '.join(bad) or 'none'}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"model": m, "dataset": d, **v} for (m, d), v in sorted(pairs.items())]
        a.json.write_text(json.dumps({"pairs": rows, "flagged": bad}, indent=1, ensure_ascii=False),
                          encoding="utf-8")


if __name__ == "__main__":
    main()

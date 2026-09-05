"""Audit a distributed sweep: coverage, per-runner tallies, duplicate work.

    uv run python scripts/sweep_audit.py results [--claims DIR] [--json]

Multiple Colab runners (see ``arabic_ocr_bench.coordination``) each write their
own ``results/<run_id>/predictions.jsonl``; nothing is overwritten, so the full
who-did-what history is on disk. This script reads every one of them and reports:

* coverage per ``(model, dataset)`` — distinct samples with a real result;
* per-runner line tallies (how much each Colab account contributed);
* duplicates — any sample transcribed for real (non-cached, non-errored) by more
  than one runner/run, with the runners and run ids involved;
* stale leases still sitting in the ``claims/`` directory (with ``--claims``).

Exit status is 1 if any duplicate is found, else 0.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _iter_predictions(root: Path):
    for pred in sorted(root.glob("*/predictions.jsonl")):
        run_id = pred.parent.name
        try:
            lines = pred.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_run"] = run_id
            yield row


def audit(root: Path) -> dict:
    # (model, dataset, sample_id) -> list of contributions
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in _iter_predictions(root):
        key = (r.get("model", "?"), r.get("dataset", "?"), r.get("sample_id", "?"))
        by_key[key].append({
            "run": r["_run"],
            "runner": r.get("runner", r.get("served_to", "")) or "(unstamped)",
            "session": r.get("session", ""),
            "cached": bool(r.get("cached")),
            "error": r.get("error"),
        })

    coverage: dict[tuple, dict] = defaultdict(lambda: {"samples": set(), "scored": set()})
    runner_lines: dict[str, int] = defaultdict(int)
    duplicates: list[dict] = []

    for (model, dataset, sid), contribs in sorted(by_key.items()):
        cov = coverage[(model, dataset)]
        cov["samples"].add(sid)
        real = [c for c in contribs if not c["cached"] and not c["error"]]
        if real:
            cov["scored"].add(sid)
        for c in real:
            runner_lines[c["runner"]] += 1
        # a sample really computed by >1 runner (or >1 run of one runner) is
        # duplicated work
        real_runs = {(c["runner"], c["run"]) for c in real}
        if len(real_runs) > 1:
            duplicates.append({
                "model": model, "dataset": dataset, "sample_id": sid,
                "by": sorted(f"{rn} @ {ru}" for rn, ru in real_runs),
            })

    cov_out = {
        f"{m} / {d}": {"scored": len(v["scored"]), "seen": len(v["samples"])}
        for (m, d), v in sorted(coverage.items())
    }
    return {
        "coverage": cov_out,
        "runner_lines": dict(sorted(runner_lines.items(), key=lambda kv: -kv[1])),
        "duplicates": duplicates,
    }


def _leases(claims_dir: Path) -> list[dict]:
    import time

    out = []
    for p in sorted(claims_dir.glob("*.json")):
        try:
            b = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        age_min = (time.time() - float(b.get("claimed_at", 0))) / 60
        out.append({
            "file": p.name, "status": b.get("status"), "runner": b.get("runner_id"),
            "model": b.get("model"), "dataset": b.get("dataset"),
            "range": f"{b.get('i0')}-{b.get('i1')}", "age_min": round(age_min, 1),
            "run_id": b.get("run_id", ""),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results", help="results/ directory")
    ap.add_argument("--claims", default=None, help="claims/ dir to list leases from")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    report = audit(Path(args.results))
    if args.claims:
        report["leases"] = _leases(Path(args.claims))

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["duplicates"] else 0

    print("== coverage (scored / seen) ==")
    for k, v in report["coverage"].items():
        print(f"  {k:38s} {v['scored']:>4d} / {v['seen']:<4d}")
    print("\n== lines computed per runner ==")
    for runner, n in report["runner_lines"].items():
        print(f"  {runner:40s} {n}")
    dups = report["duplicates"]
    print(f"\n== duplicates: {len(dups)} ==")
    for d in dups[:200]:
        print(f"  {d['model']} / {d['dataset']} / {d['sample_id']}")
        for b in d["by"]:
            print(f"      {b}")
    if len(dups) > 200:
        print(f"  ... and {len(dups) - 200} more")
    if "leases" in report:
        print("\n== leases in claims dir ==")
        for lease in report["leases"]:
            flag = " STALE?" if lease["status"] == "reserved" and lease["age_min"] > 30 else ""
            print(f"  [{lease['status']:8s}] {lease['runner']} "
                  f"{lease['model']}/{lease['dataset']} {lease['range']} "
                  f"age {lease['age_min']}m{flag}")
    return 1 if dups else 0


if __name__ == "__main__":
    raise SystemExit(main())

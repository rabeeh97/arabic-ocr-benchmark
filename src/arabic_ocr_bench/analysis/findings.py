"""Assemble every analysis section into one JSON-serialisable findings document.

`build()` is the keystone: it loads prediction rows once, aligns them once, and
fans the shared `(rows, records, rows_by_key)` triple out to all eight reducers.
The result is a plain dict of plain dicts/lists/str/int/float/bool -- safe for
`json.dumps` with the stdlib encoder (no numpy scalars, no NaN/Infinity).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .align import build_record
from .charstats import error_anatomy, script_letters
from .corpus import corpus_health
from .diacritics import diacritics
from .difficulty import difficulty
from .loader import iter_pred_rows, run_dirs
from .modelcmp import model_landscape
from .operational import operational

# Documentation of the row caps the reducers already hardcode. Surfaced in
# `meta` so downstream consumers know a list was truncated; NOT threaded back
# into the reducers.
ROW_CAPS = {
    "confusion_pairs": 40,
    "troublesome_letters": 30,
    "rewrite_patterns": 25,
    "difficulty_samples": 30,
    "length_bins": 12,
}


def build(results_dir, *, seed: int = 1234, contamination_path=None) -> dict:
    """Build the full findings document for one results directory."""
    rows = iter_pred_rows(results_dir)
    records = [build_record(r) for r in rows]
    rbk = {(r.model, r.sample_id): r for r in rows}
    return {
        "meta": {
            "seed": seed,
            "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "run_ids": sorted(p.name for p in run_dirs(results_dir)),
            "models": sorted({r.model for r in rows}),
            "datasets": sorted({r.dataset for r in rows}),
            "n_pred_rows": len(rows),
            "row_caps": ROW_CAPS,
        },
        "error_anatomy": error_anatomy(records, rbk),
        "script_letters": script_letters(records, rbk),
        "diacritics": diacritics(records, rbk),
        "difficulty": difficulty(rows, records),
        "model_landscape": model_landscape(rows, records, seed=seed),
        "operational": operational(rows, records),
        "corpus_health": corpus_health(rows, records, contamination_path=contamination_path),
    }


def _default_out(results_dir: str) -> str:
    p = Path(results_dir)
    if p.name == "results":
        return str(p / "findings.json")
    return "results/findings.json"


def _dump_records(results_dir, path: str) -> None:
    """Thin debug hatch: alignment records as a parquet table.

    `char_ops` / `word_ops` / `rasm_ops` are nested (list[CharOp] / dict), which
    pyarrow chokes on, so JSON-stringify those three columns first.
    """
    import dataclasses

    import pandas as pd

    recs = [build_record(r) for r in iter_pred_rows(results_dir)]
    frame = []
    for r in recs:
        d = dataclasses.asdict(r)
        for col in ("char_ops", "word_ops", "rasm_ops"):
            d[col] = json.dumps(d[col], ensure_ascii=False)
        frame.append(d)
    pd.DataFrame(frame).to_parquet(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench analyze")
    ap.add_argument("results_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--dump-records", metavar="PATH")
    a = ap.parse_args(argv)

    doc = build(a.results_dir)
    out = a.out or _default_out(a.results_dir)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"findings -> {out}")

    if a.dump_records:
        _dump_records(a.results_dir, a.dump_records)
        print(f"records  -> {a.dump_records}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

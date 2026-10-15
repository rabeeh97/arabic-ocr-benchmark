"""Feasibility of an allograph library (shape programme, 2026-10-15): are CTC letter spans usable as cut lines, and do
handwritten PAWs separate into connected components?

    .geomvenv/bin/python scripts/analyze_shape_feasibility.py

Runs scripts/_feasibility_spans.py (contact sheet of base-CRNN span crops with the span drawn, and span-width statistics)
and scripts/_feasibility_paws.py (large connected components vs label PAWs on Muharaf train, KHATT, NAKBA; lines per RASAM
manuscript). Writes results/analysis/S/span_sheet.png and results/analysis/S/shape_feasibility.json (the printed log).
"""
import contextlib
import io
import json
import runpy
from pathlib import Path

W = Path(__file__).resolve().parent.parent
out = W / "results/analysis/S"
out.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(W / "scripts/_feasibility_spans.py"), run_name="__main__")
    runpy.run_path(str(W / "scripts/_feasibility_paws.py"), run_name="__main__")
lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
print("\n".join(lines))
(out / "shape_feasibility.json").write_text(json.dumps({"log": lines}, ensure_ascii=False, indent=1), encoding="utf-8")

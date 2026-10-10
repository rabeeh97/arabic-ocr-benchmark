"""Build the attribution explorer page from xai_attribute_qwen3vl.py output.

    .venv/bin/python scripts/build_xai_explorer.py OUT_DIR --panel letters=PATH --panel errors=PATH

Writes OUT_DIR/explorer.html (data inlined) and OUT_DIR/sprites/<panel>_<line>.jpg: one JPEG per line
holding the original and, per chosen token, the overlay of each method (one column of equal cells).
The page crops cells with CSS background-position, so a line's maps cost one request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = ["attn", "attn_rel", "gxi"]
CELL_W = 560
TEMPLATE = Path(__file__).with_name("xai_explorer_template.html")


def sprite(line_dir: Path, toks: list[dict], dest: Path) -> dict:
    cells = [Image.open(line_dir / "orig.png").convert("RGB")]
    for t in toks:
        cells += [Image.open(line_dir / f"ov_{m}_{t['t']}.png").convert("RGB") for m in METHODS]
    h = round(cells[0].height * CELL_W / cells[0].width)
    sheet = Image.new("RGB", (CELL_W, h * len(cells)))
    for i, c in enumerate(cells):
        sheet.paste(c.resize((CELL_W, h), Image.LANCZOS), (0, i * h))
    sheet.save(dest, quality=62, optimize=True)
    return {"w": CELL_W, "h": h, "n": len(cells)}


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--panel", action="append", required=True, help="key=dir (a run dir of xai_attribute_qwen3vl.py)")
    a = ap.parse_args()
    (a.out / "sprites").mkdir(parents=True, exist_ok=True)
    panels, all_tokens = [], []
    for spec in a.panel:
        key, path = spec.split("=", 1)
        lines = []
        for d in sorted(Path(path).glob("[0-9][0-9][0-9]")):
            j = json.loads((d / "tokens.json").read_text(encoding="utf-8"))
            toks = sorted(j["tokens"], key=lambda t: t["t"])
            fn = f"{key}_{d.name}.jpg"
            geo = sprite(d, toks, a.out / "sprites" / fn)
            tk = []
            for k, t in enumerate(toks):
                row = {"t": t["t"], "text": t["text"], "err": t["err"], "gt_span": t["gt_span"], "lp": t["lp"],
                       "targets": t["targets"], "cell": {m: 1 + 3 * k + i for i, m in enumerate(METHODS)},
                       "conc": {m: t["conc"][m] for m in METHODS}, "ink": {m: t["ink"][m] for m in METHODS},
                       "del": {m: t["deletion"].get(m, {}).get("0.1") for m in METHODS + ["random"]}}
                tk.append(row)
                all_tokens.append((key, row))
            lines.append({"id": j["id"], "dataset": j["dataset"], "gt": j["gt"], "hyp": j["hyp"], "stored_hyp": j.get("stored_hyp"),
                          "cer": j.get("cer"), "group": j.get("group"), "rtl": j["rtl"], "sprite": f"sprites/{fn}", **geo, "tokens": tk})
        panels.append({"key": key, "lines": lines})

    def agg(rows):
        return {
            "n": len(rows), "lp": mean([r["lp"] for r in rows]),
            "conc": {m: mean([r["conc"][m] for r in rows]) for m in METHODS},
            "ink": {m: mean([r["ink"][m] for r in rows]) for m in METHODS},
            "del": {m: mean([r["del"][m] for r in rows]) for m in METHODS + ["random"]},
        }

    rows = [r for _, r in all_tokens]
    families = {"alef": "ا", "lam": "ل", "jim": "ج", "ha": "ح", "kha": "خ"}
    # The error panel is built from the worst lines and oversamples wrong tokens, so per-letter
    # error rates come from the letter panel only (lines picked for their letters, not their errors).
    lrows = [r for key, r in all_tokens if key == "letters"]
    summary = {
        "all": agg(rows),
        "err": agg([r for r in rows if r["err"]]),
        "ok": agg([r for r in rows if not r["err"]]),
        "letters": {k: {**agg([r for r in lrows if k in r["targets"]]),
                        "err_rate": mean([1.0 if r["err"] else 0.0 for r in lrows if k in r["targets"]])} for k in families},
        "rtl": {m: mean([ln["rtl"].get(m) for p in panels for ln in p["lines"]]) for m in METHODS},
        "n_lines": sum(len(p["lines"]) for p in panels),
    }
    data = {"panels": panels, "summary": summary, "methods": METHODS}
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    (a.out / "explorer.html").write_text(html, encoding="utf-8")
    size = sum(f.stat().st_size for f in (a.out / "sprites").iterdir()) / 1e6
    print(f"{summary['n_lines']} lines, {len(rows)} tokens, sprites {size:.1f} MB -> {a.out / 'explorer.html'}")


if __name__ == "__main__":
    main()

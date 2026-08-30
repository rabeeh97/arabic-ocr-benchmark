"""Turn one or more results/<run_id>/ directories into a single self-contained
HTML leaderboard.

    uv run python scripts/make_leaderboard.py results -o results/leaderboard.html
    uv run python scripts/make_leaderboard.py results/20260101T000000Z -o board.html

The output is a style + content fragment (no <html>/<head>/<body> wrapper): it
opens fine on double-click AND can be published directly as an Artifact.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path

_PROFILES = ("raw", "light", "diac_insensitive")


def _run_dirs(target: Path) -> list[Path]:
    if (target / "summary.csv").exists():
        return [target]
    return sorted(p.parent for p in target.glob("*/summary.csv"))


def _run_meta(run_dir: Path) -> dict:
    meta = {"run": run_dir.name, "when": "", "limit": "", "prompt_version": ""}
    try:
        parsed = dt.datetime.strptime(run_dir.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
        meta["when"] = parsed.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        pass
    snap = run_dir / "run_config.snapshot.yaml"
    if snap.exists():
        for line in snap.read_text().splitlines():
            if line.startswith("limit:"):
                meta["limit"] = line.split(":", 1)[1].strip()
            if line.startswith("prompt_version:"):
                meta["prompt_version"] = line.split(":", 1)[1].strip()
    return meta


def collect(target: Path) -> list[dict]:
    rows: list[dict] = []
    for rd in _run_dirs(target):
        meta = _run_meta(rd)
        with (rd / "summary.csv").open() as fh:
            for r in csv.DictReader(fh):
                rows.append({**meta, **r})
    return _dedupe_keep_largest(rows)


def _dedupe_keep_largest(rows: list[dict]) -> list[dict]:
    """One row per (model, dataset): keep the run with the most samples, so a
    later full run supersedes an earlier partial one on the board."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r.get("model", ""), r.get("dataset", ""))
        if k not in best or int(r.get("n", 0) or 0) > int(best[k].get("n", 0) or 0):
            best[k] = r
    return list(best.values())


def _fmt_pct(v: str | float) -> str:
    try:
        return f"{float(v) * 100:.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: str | float, nd: int = 1) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _is_preview(rows: list[dict]) -> bool:
    """True while the only data is the offline self-test (mock models / synthetic set)."""
    return all(
        r.get("model", "").startswith("mock") or r.get("dataset") == "synthetic"
        for r in rows
    )


def render(rows: list[dict], title: str) -> str:
    rows = sorted(rows, key=lambda r: (r.get("dataset", ""),
                                       float(r.get("cer_micro_light", 1) or 1)))
    preview = _is_preview(rows)
    best_run_key = None
    real = [r for r in rows if not (r["model"].startswith("mock") or r["dataset"] == "synthetic")]
    if real:
        best = min(real, key=lambda r: float(r.get("cer_micro_light", 1) or 1))
        best_run_key = (best["run"], best["model"], best["dataset"])

    gen = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_runs = len({r["run"] for r in rows})

    out = [_HEAD.replace("{{TITLE}}", html.escape(title))]
    out.append("<main>")
    out.append("<header class='top'>")
    out.append("<p class='eyebrow'>Handwritten Arabic · line-level recognition</p>")
    out.append(f"<h1>{html.escape(title)}</h1>")
    out.append(
        "<p class='lede'>Character- and word-error rates for OCR / vision-language "
        "models transcribing cursive handwritten Arabic, scored on the same held-out "
        "lines with the same metrics.</p>"
    )
    if preview:
        out.append(
            "<p class='status status--preview'><span class='dot'></span>"
            "Pipeline verified — awaiting the first model run. The row below is the "
            "offline self-test: an echo baseline scores 0.00 CER, an empty baseline "
            "scores 100.00.</p>"
        )
    else:
        out.append(
            f"<p class='status status--live'><span class='dot'></span>"
            f"{len({r['model'] for r in real})} model(s) · {n_runs} run(s) · "
            f"dataset: Muharaf-public test split</p>"
        )
    out.append("</header>")

    out.append("<section class='legend'><dl>")
    for term, desc in _LEGEND:
        out.append(f"<div><dt>{term}</dt><dd>{desc}</dd></div>")
    out.append("</dl></section>")

    cols = [
        ("model", "model", "t"), ("dataset", "dataset", "t"), ("n", "n", "n"),
        ("cer_micro_raw", "CER · raw", "n"),
        ("cer_micro_light", "CER · light", "n hero"),
        ("cer_micro_diac_insensitive", "CER · no-diac", "n"),
        ("cer_macro_light", "CER · macro", "n"),
        ("wer_micro_light", "WER · light", "n"),
        ("exact_light", "exact %", "n"),
        ("diac_f1", "diac F1", "n"),
        ("latency_ms_mean", "ms / line", "n"),
        ("cost_usd_total", "cost $", "n"),
        ("errors", "err", "n"),
    ]
    thead = "".join(
        f"<th class='{cls}'>{html.escape(label)}</th>" for _, label, cls in cols
    )
    trs = []
    for r in rows:
        is_best = best_run_key == (r["run"], r["model"], r["dataset"])
        is_mock = r["model"].startswith("mock") or r["dataset"] == "synthetic"
        cls = " best" if is_best else (" mock" if is_mock else "")
        tds = []
        for key, _, ccls in cols:
            if key in ("model", "dataset"):
                val = html.escape(r.get(key, ""))
            elif key in ("n", "errors"):
                val = html.escape(str(r.get(key, "")))
            elif key == "diac_f1":
                val = _fmt_num(r.get(key), 3)
            elif key == "latency_ms_mean":
                val = _fmt_num(r.get(key), 0)
            elif key == "cost_usd_total":
                val = _fmt_num(r.get(key), 4)
            else:
                val = _fmt_pct(r.get(key))
            tds.append(f"<td class='{ccls}'>{val}</td>")
        trs.append(f"<tr class='{cls.strip()}'>{''.join(tds)}</tr>")
    out.append(
        "<section class='board'><div class='scroll'><table>"
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
        "<p class='board-note'>All error figures are percentages; lower is better. "
        "The <b>CER · light</b> column is the headline metric.</p></section>"
    )

    runs = sorted({(r["run"], r.get("when", ""), r.get("limit", ""),
                    r.get("prompt_version", "")) for r in rows}, reverse=True)
    out.append("<section class='runs'><h2>Runs</h2><ul>")
    for run, when, limit, pv in runs:
        out.append(
            f"<li><span class='rid'>{html.escape(run)}</span>"
            f"<span class='rmeta'>{html.escape(when or 'time unknown')} · "
            f"limit {html.escape(str(limit) or 'all')} · prompt {html.escape(pv or '?')}</span></li>"
        )
    out.append("</ul></section>")

    out.append(
        "<footer><p>Dataset: <b>Muharaf-public</b> — Manuscripts of Handwritten "
        "Arabic (NeurIPS 2024), CC-BY-NC-SA-4.0, <code>test</code> split. "
        "Harness: <code>arabic-ocr-bench</code>. "
        f"Generated {gen} by <code>scripts/make_leaderboard.py</code>.</p></footer>"
    )
    out.append("</main>")
    return "\n".join(out)


_LEGEND = [
    ("CER / WER", "edit distance to the reference, over characters / words. 0 = perfect."),
    ("raw", "exact strings — every orthographic and diacritic difference counts."),
    ("light", "alef forms, digits, tatweel and spacing normalized first."),
    ("no-diac", "light, and harakat stripped from both sides."),
    ("micro / macro", "corpus-weighted total vs. mean of per-line rates."),
    ("diac F1", "precision/recall of harakat marks, measured on raw text."),
]

_HEAD = """<title>{{TITLE}}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f6f6f3; --surface:#ffffff; --ink:#1b1d21; --muted:#5d6470;
  --line:#e4e3de; --accent:#2f4b99; --accent-soft:#eaeef7;
  --mock:#8a8f99; --best-bar:#2f4b99;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#14161a; --surface:#1b1e23; --ink:#e8e8e4; --muted:#989fab;
    --line:#2b2e35; --accent:#93a9e8; --accent-soft:#1f2740;
    --mock:#727783; --best-bar:#93a9e8;
  }
}
:root[data-theme="dark"]{
  --ground:#14161a; --surface:#1b1e23; --ink:#e8e8e4; --muted:#989fab;
  --line:#2b2e35; --accent:#93a9e8; --accent-soft:#1f2740;
  --mock:#727783; --best-bar:#93a9e8;
}

*{box-sizing:border-box;}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  -webkit-font-smoothing:antialiased;}
main{max-width:1060px;margin:0 auto;padding:clamp(24px,5vw,64px) clamp(16px,4vw,32px) 72px;}

.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin:0 0 14px;}
h1{font-family:Spectral,Georgia,serif;font-weight:600;font-size:clamp(28px,4.4vw,40px);
  line-height:1.12;letter-spacing:-.01em;margin:0 0 14px;text-wrap:balance;}
.lede{max-width:60ch;color:var(--muted);font-size:15.5px;line-height:1.6;margin:0 0 22px;}

.status{display:flex;align-items:center;gap:9px;font-size:13.5px;line-height:1.5;
  padding:11px 15px;border:1px solid var(--line);border-radius:10px;
  background:var(--surface);margin:0;max-width:64ch;}
.status .dot{flex:none;width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 4px var(--accent-soft);}
.status--preview{color:var(--muted);}
.status--live .dot{background:var(--accent);}

.legend{margin:32px 0 8px;}
.legend dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:2px 28px;margin:0;}
.legend div{padding:9px 0;border-top:1px solid var(--line);}
.legend dt{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500;
  color:var(--ink);margin-bottom:3px;}
.legend dd{margin:0;font-size:12.5px;color:var(--muted);line-height:1.5;}

.board{margin:24px 0 0;}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--surface);}
table{border-collapse:collapse;width:100%;font-size:13px;}
th,td{padding:11px 14px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line);}
thead th{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:11px;
  letter-spacing:.03em;text-transform:uppercase;color:var(--muted);
  background:var(--ground);border-bottom:1px solid var(--line);}
th.n,td.n{text-align:right;font-variant-numeric:tabular-nums;}
td.n{font-family:"IBM Plex Mono",monospace;}
td.t{font-weight:500;}
th.hero,td.hero{background:var(--accent-soft);}
td.hero{font-weight:600;color:var(--ink);}
tbody tr:last-child td{border-bottom:none;}
tbody tr.best td{background:var(--accent-soft);}
tbody tr.best td:first-child{box-shadow:inset 3px 0 0 var(--best-bar);}
tbody tr.mock td{color:var(--mock);}
tbody tr.mock td.t{font-weight:400;font-style:italic;}
.board-note{font-size:12px;color:var(--muted);margin:10px 2px 0;}

.runs{margin:38px 0 0;}
h2{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin:0 0 12px;}
.runs ul{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:1px;}
.runs li{display:flex;flex-wrap:wrap;gap:4px 14px;padding:9px 0;
  border-top:1px solid var(--line);font-size:12.5px;}
.rid{font-family:"IBM Plex Mono",monospace;color:var(--ink);}
.rmeta{color:var(--muted);}

footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);}
footer p{font-size:12px;color:var(--muted);line-height:1.65;max-width:70ch;margin:0;}
code{font-family:"IBM Plex Mono",monospace;font-size:.92em;
  background:var(--accent-soft);padding:1px 5px;border-radius:4px;}
b{font-weight:600;color:var(--ink);}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="results/ dir, or a single results/<run_id>/ dir")
    ap.add_argument("-o", "--out", default="results/leaderboard.html")
    ap.add_argument("--title", default="Arabic Handwriting OCR Leaderboard")
    args = ap.parse_args()

    rows = collect(Path(args.target))
    if not rows:
        print(f"no summary.csv found under {args.target}")
        return 1
    Path(args.out).write_text(render(rows, args.title), encoding="utf-8")
    print(f"leaderboard -> {args.out}  ({len(rows)} rows from "
          f"{len({r['run'] for r in rows})} run(s))")
    # also dump a machine-readable bundle next to it
    bundle = Path(args.out).with_suffix(".json")
    bundle.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"bundle      -> {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

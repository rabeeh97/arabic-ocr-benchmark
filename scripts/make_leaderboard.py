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
    return rows


def _fmt_pct(v: str | float) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: str | float, nd: int = 1) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render(rows: list[dict], title: str) -> str:
    rows = sorted(rows, key=lambda r: (r.get("dataset", ""), float(r.get("cer_micro_light", 1) or 1)))
    body = [_STYLE, f"<main><h1>{html.escape(title)}</h1>"]
    body.append(
        "<p class='sub'>Arabic cursive script recognition — character/word error rate "
        "(lower is better). <b>micro</b> = corpus-weighted, <b>macro</b> = per-line mean. "
        "<b>light</b> normalizes alef/digits/tatweel/spacing; <b>diac</b> also strips harakat. "
        "<b>diac F1</b> = harakat F1 on raw text.</p>"
    )

    header = (
        "<tr><th>model</th><th>dataset</th><th>n</th>"
        "<th>CER<br>raw</th><th>CER<br>light</th><th>CER<br>diac</th>"
        "<th>CER<br>macro·light</th><th>WER<br>light</th>"
        "<th>exact<br>light</th><th>diac<br>F1</th>"
        "<th>latency<br>ms</th><th>cost<br>USD</th><th>errors</th></tr>"
    )
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td class='m'>{html.escape(r.get('model',''))}</td>"
            f"<td>{html.escape(r.get('dataset',''))}</td>"
            f"<td class='n'>{html.escape(str(r.get('n','')))}</td>"
            f"<td class='n'>{_fmt_pct(r.get('cer_micro_raw'))}</td>"
            f"<td class='n hi'>{_fmt_pct(r.get('cer_micro_light'))}</td>"
            f"<td class='n'>{_fmt_pct(r.get('cer_micro_diac_insensitive'))}</td>"
            f"<td class='n'>{_fmt_pct(r.get('cer_macro_light'))}</td>"
            f"<td class='n'>{_fmt_pct(r.get('wer_micro_light'))}</td>"
            f"<td class='n'>{_fmt_pct(r.get('exact_light'))}</td>"
            f"<td class='n'>{_fmt_num(r.get('diac_f1'), 3)}</td>"
            f"<td class='n'>{_fmt_num(r.get('latency_ms_mean'))}</td>"
            f"<td class='n'>{_fmt_num(r.get('cost_usd_total'), 4)}</td>"
            f"<td class='n'>{html.escape(str(r.get('errors','')))}</td>"
            "</tr>"
        )
    body.append("<div class='wrap'><table><thead>" + header + "</thead><tbody>"
                + "".join(trs) + "</tbody></table></div>")

    runs = sorted({(r["run"], r.get("when", ""), r.get("limit", ""),
                    r.get("prompt_version", "")) for r in rows})
    body.append("<h2>Runs</h2><ul class='runs'>")
    for run, when, limit, pv in runs:
        body.append(
            f"<li><code>{html.escape(run)}</code> — {html.escape(when or '?')} · "
            f"limit={html.escape(str(limit) or 'all')} · prompt={html.escape(pv or '?')}</li>"
        )
    body.append("</ul>")
    body.append(
        f"<p class='foot'>Generated {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M UTC')} "
        "by <code>scripts/make_leaderboard.py</code> · "
        "dataset: Muharaf-public (CC-BY-NC-SA-4.0) · harness: arabic-ocr-bench</p>")
    body.append("</main>")
    return "\n".join(body)


_STYLE = """<style>
:root{
  --bg:#fbfbfa; --fg:#1a1a1a; --muted:#666; --line:#e3e3e0;
  --head:#f0f0ee; --hi:#eef4ff; --accent:#2a5db0;
}
:root:not([data-theme="light"]){ }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#16171a; --fg:#e9e9e7; --muted:#9a9a97; --line:#2c2e33;
    --head:#22242a; --hi:#1d2735; --accent:#7aa6f0;
  }
}
:root[data-theme="dark"]{
  --bg:#16171a; --fg:#e9e9e7; --muted:#9a9a97; --line:#2c2e33;
  --head:#22242a; --hi:#1d2735; --accent:#7aa6f0;
}
body{background:var(--bg);color:var(--fg);}
main{max-width:1040px;margin:0 auto;padding:28px 20px 56px;
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
h1{font-size:22px;margin:0 0 4px;}
h2{font-size:15px;margin:28px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
.sub{color:var(--muted);margin:0 0 20px;}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;}
thead th{background:var(--head);font-weight:600;vertical-align:bottom;position:sticky;top:0;}
tbody tr:last-child td{border-bottom:none;}
td.n{text-align:right;font-variant-numeric:tabular-nums;}
td.m{font-weight:600;}
td.hi,th:nth-child(5){background:var(--hi);}
.runs{color:var(--muted);padding-left:18px;}
.runs code,.foot code{background:var(--head);padding:1px 5px;border-radius:4px;}
.foot{color:var(--muted);font-size:12px;margin-top:24px;}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;}
</style>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="results/ dir, or a single results/<run_id>/ dir")
    ap.add_argument("-o", "--out", default="results/leaderboard.html")
    ap.add_argument("--title", default="Arabic Cursive OCR — Benchmark v1")
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

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

import yaml

_PROFILES = ("raw", "light", "diac_insensitive")
_REPO = Path(__file__).resolve().parent.parent
_SAMPLE_CATS = ("anchor", "best", "worst", "typical", "perfect", "longest")


def load_cards() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {"models": {}, "datasets": {}}
    for kind, fname in (("models", "models.yaml"), ("datasets", "datasets.yaml")):
        try:
            cfg = yaml.safe_load((_REPO / "configs" / fname).read_text()) or {}
        except OSError:
            continue
        for name, entry in cfg.items():
            if isinstance(entry, dict) and entry.get("card"):
                out[kind][name] = " ".join(str(entry["card"]).split())
    return out


def collect_samples(target: Path) -> dict:
    """Merge every results/*/samples.json into {sample_id -> image + GT + per-model preds}."""
    files = (
        [target / "samples.json"]
        if (target / "samples.json").exists()
        else sorted(target.glob("*/samples.json"))
    )
    by_id: dict[str, dict] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for s in data.get("samples", []):
            if s.get("dataset") == "synthetic":
                continue
            sid = s["sample_id"]
            slot = by_id.setdefault(sid, {
                "row": s.get("row"), "dataset": s.get("dataset", ""),
                "ground_truth": s.get("ground_truth", ""), "image": "", "preds": {},
            })
            if not slot["image"] and s.get("image"):
                slot["image"] = s["image"]
            if not slot["ground_truth"] and s.get("ground_truth"):
                slot["ground_truth"] = s["ground_truth"]
            prev = slot["preds"].get(s["model"], {})
            cats = sorted(set(prev.get("categories", [])) | set(s.get("categories", [])),
                          key=lambda c: _SAMPLE_CATS.index(c) if c in _SAMPLE_CATS else 9)
            slot["preds"][s["model"]] = {
                "prediction": s.get("prediction", ""),
                "error": s.get("error"),
                "cer_light": s.get("cer_light"),
                "cer_raw": s.get("cer_raw"),
                "categories": cats,
            }
    return by_id


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
    run_dirs: dict[str, Path] = {}
    for rd in _run_dirs(target):
        run_dirs[rd.name] = rd
        meta = _run_meta(rd)
        with (rd / "summary.csv").open() as fh:
            for r in csv.DictReader(fh):
                # `synthetic` is an offline sanity-check dataset (near-0 CER for
                # every real model) — informative for pipeline QA, not for the
                # published board, so it never makes it into `rows`.
                if r.get("dataset") == "synthetic":
                    continue
                rows.append({**meta, **r})
    return _merge_duplicate_runs(rows, run_dirs)


def _merge_duplicate_runs(rows: list[dict], run_dirs: dict[str, Path]) -> list[dict]:
    """One row per (model, dataset). A single run is kept as-is; when the same
    pair was scored across more than one run, the rows are MERGED into one
    combined row recomputed over the union of their (distinct, deduped-by-id)
    samples, rather than discarding all but the largest run — so re-running a
    model on new lines grows its reported n instead of throwing work away."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("model", ""), r.get("dataset", "")), []).append(r)
    out = []
    for (model, dataset), group in groups.items():
        out.append(group[0] if len(group) == 1 else _merge_runs(model, dataset, group, run_dirs))
    return out


def _merge_runs(model: str, dataset: str, group: list[dict], run_dirs: dict[str, Path]) -> dict:
    from arabic_ocr_bench.metrics import aggregate, score_pair

    seen: dict[str, dict] = {}
    for r in group:
        rd = run_dirs.get(r.get("run", ""))
        pf = rd / "predictions.jsonl" if rd else None
        if not pf or not pf.exists():
            continue
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                if p.get("model") == model and p.get("dataset") == dataset:
                    seen.setdefault(p["sample_id"], p)  # first-seen wins if ids ever collide

    if not seen:
        # predictions.jsonl missing for every contributing run (older run dir) —
        # fall back to the previous behaviour rather than reporting nothing.
        return max(group, key=lambda r: int(r.get("n", 0) or 0))

    preds = list(seen.values())
    scored = [
        (p, score_pair(p["ground_truth"], "" if p.get("error") else (p.get("cleaned_text") or "")))
        for p in preds
    ]
    n = len(scored)
    n_err = sum(1 for p, _ in scored if p.get("error"))
    agg = aggregate([ps for _, ps in scored])

    row = {
        "run": "+".join(sorted({r["run"] for r in group})),
        "when": max((r.get("when", "") for r in group), default=""),
        "limit": str(n),
        "prompt_version": "/".join(sorted({r.get("prompt_version", "") for r in group} - {""})),
        "model": model,
        "dataset": dataset,
        "n": str(n),
        "errors": str(n_err),
        "refusal_rate": f"{n_err / n:.4f}" if n else "0.0",
        "latency_ms_mean": f"{sum(p.get('latency_ms', 0.0) for p, _ in scored) / n:.1f}" if n else "0.0",
        "cost_usd_total": f"{sum(p.get('cost_usd', 0.0) for p, _ in scored):.6f}",
    }
    for p in _PROFILES:
        row[f"cer_micro_{p}"] = f"{agg['micro_cer'][p]:.4f}"
        row[f"cer_macro_{p}"] = f"{agg['cer_macro'][p]:.4f}"
        row[f"wer_micro_{p}"] = f"{agg['micro_wer'][p]:.4f}"
        row[f"exact_{p}"] = f"{agg['exact_match_rate'][p]:.4f}"
    row["diac_f1"] = f"{agg['diacritic']['f1']:.4f}"
    return row


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


def _cell(r: dict, key: str, ccls: str) -> str:
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
    return f"<td class='{ccls}' data-v='{html.escape(str(r.get(key, '')))}'>{val}</td>"


def _table(rows: list[dict], cols: list[tuple], best_run_key) -> str:
    thead = "".join(
        "<th class='{cls}' data-key='{key}' data-type='{typ}'>{lbl}</th>".format(
            cls=cls, key=key, lbl=html.escape(label),
            typ="text" if key in ("model", "dataset") else "num",
        )
        for key, label, cls in cols
    )
    trs = []
    for r in rows:
        is_best = best_run_key == (r["run"], r["model"], r["dataset"])
        is_mock = r["model"].startswith("mock") or r["dataset"] == "synthetic"
        cls = "best" if is_best else ("mock" if is_mock else "")
        tds = "".join(_cell(r, key, ccls) for key, _, ccls in cols)
        trs.append(f"<tr class='{cls}'>{tds}</tr>")
    return (
        "<div class='scroll'><table class='sortable'>"
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
    )


def render(rows: list[dict], title: str, target: Path | None = None) -> str:
    _RENDER_CTX["target"] = target or Path("results")
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
    out.append(
        "<section class='board'>" + _table(rows, cols, best_run_key)
        + "<p class='board-note'>All error figures are percentages; lower is better. "
        "The <b>CER · light</b> column is the headline metric. "
        "Click a column header to sort; click again to reverse.</p></section>"
    )

    datasets = sorted({r.get("dataset", "") for r in rows})
    if len(datasets) > 1:
        sub_cols = [c for c in cols if c[0] != "dataset"]
        parts = [
            "<section class='board by-dataset'><h2>By dataset</h2>",
            (
                "<p class='board-note'>The same results, split out one table per "
                "dataset &mdash; so each model is only lined up against others "
                "scored on the <b>same</b> lines. Each table sorts on its own.</p>"
            ),
        ]
        for ds in datasets:
            drows = sorted(
                (r for r in rows if r.get("dataset", "") == ds),
                key=lambda r: float(r.get("cer_micro_light", 1) or 1),
            )
            parts.append(
                f"<h3>{html.escape(ds)} "
                f"<span class='dscount'>{len(drows)} model(s)</span></h3>"
            )
            parts.append(_table(drows, sub_cols, best_run_key))
        parts.append("</section>")
        out.append("".join(parts))

    out.append(_SORT_JS)

    out.append(_render_cards(rows))
    out.append(_render_samples(rows))

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


_SORT_JS = """<script>
(function(){
  Array.prototype.forEach.call(document.querySelectorAll('table.sortable'),function(t){
    if(!t.tHead||!t.tBodies[0]) return;
    var tb=t.tBodies[0];
    var ths=Array.prototype.slice.call(t.tHead.rows[0].cells);
    ths.forEach(function(th,i){
      th.addEventListener('click',function(){
        var dir=th.getAttribute('aria-sort')==='ascending'?'descending':'ascending';
        ths.forEach(function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',dir);
        var num=th.dataset.type==='num', s=dir==='ascending'?1:-1;
        var rows=Array.prototype.slice.call(tb.rows);
        rows.sort(function(a,b){
          var x=a.cells[i].getAttribute('data-v')||'', y=b.cells[i].getAttribute('data-v')||'';
          if(num){
            var nx=parseFloat(x), ny=parseFloat(y);
            var ax=isNaN(nx), ay=isNaN(ny);
            if(ax&&ay) return 0;
            if(ax) return 1;          /* blanks / dashes always sort last */
            if(ay) return -1;
            return (nx-ny)*s;
          }
          return x.localeCompare(y)*s;
        });
        rows.forEach(function(r){tb.appendChild(r);});
      });
    });
  });
})();
</script>"""

_LEGEND = [
    ("CER / WER", "edit distance to the reference, over characters / words. 0 = perfect."),
    ("raw", "exact strings — every orthographic and diacritic difference counts."),
    ("light", "alef forms, digits, tatweel and spacing normalized first."),
    ("no-diac", "light, and harakat stripped from both sides."),
    ("micro / macro", "corpus-weighted total vs. mean of per-line rates."),
    ("diac F1", "precision/recall of harakat marks, measured on raw text."),
]

_RENDER_CTX: dict = {"target": Path("results")}
_CAT_LABEL = {
    "anchor": "shared", "best": "best", "worst": "worst",
    "typical": "typical", "perfect": "perfect", "longest": "longest line",
}


def _cer_pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _render_cards(rows: list[dict]) -> str:
    cards = load_cards()
    models = [m for m in dict.fromkeys(r["model"] for r in rows) if m in cards["models"]]
    datasets = [d for d in dict.fromkeys(r["dataset"] for r in rows) if d in cards["datasets"]]
    if not models and not datasets:
        return ""
    out = ["<section class='cards'>"]
    if models:
        out.append("<h2>Models</h2><dl class='cardlist'>")
        for m in models:
            out.append(f"<div><dt>{html.escape(m)}</dt>"
                       f"<dd>{html.escape(cards['models'][m])}</dd></div>")
        out.append("</dl>")
    if datasets:
        out.append("<h2>Datasets</h2><dl class='cardlist'>")
        for d in datasets:
            out.append(f"<div><dt>{html.escape(d)}</dt>"
                       f"<dd>{html.escape(cards['datasets'][d])}</dd></div>")
        out.append("</dl>")
    out.append("</section>")
    return "\n".join(out)


def _chips(cats: list[str], dataset: str = "") -> str:
    out = ""
    if dataset:
        out += f"<span class='chip chip--ds'>{html.escape(dataset)}</span>"
    out += "".join(
        f"<span class='chip chip--{c}'>{html.escape(_CAT_LABEL.get(c, c))}</span>"
        for c in cats
    )
    return out


def _pred_line(text: str, error: str | None) -> str:
    if error:
        return f"<span class='pred err'>— {html.escape(error)}</span>"
    return f"<span class='pred' dir='auto'>{html.escape(text) or '<em>(empty)</em>'}</span>"


def _render_samples(rows: list[dict]) -> str:
    by_id = collect_samples(Path(_RENDER_CTX["target"]))
    if not by_id:
        return ""
    all_models = list(dict.fromkeys(r["model"] for r in rows))
    real_models = [m for m in all_models if not m.startswith("mock")] or all_models

    datasets = sorted({slot["dataset"] for slot in by_id.values() if slot.get("dataset")})

    out = [
        "<section class='samples'><h2>Sample transcriptions</h2>",
        (
            "<p class='samp-intro'>Each image is one line from a test set; "
            "<span dir='auto'>Arabic</span> reads right-to-left. Percentages are CER "
            "(light) — lower is better. Samples are drawn from every dataset a model "
            "was run on (Muharaf handwriting and the KITAB-Bench subsets), each "
            "tagged with its dataset.</p>"
        ),
    ]

    # 1. shared lines, every model side by side — grouped by dataset
    for ds in datasets:
        shared = [
            (sid, slot) for sid, slot in sorted(by_id.items())
            if slot.get("dataset") == ds and len(slot["preds"]) >= 2
        ]
        # prefer explicitly anchored lines, then fall back to any multi-model line
        shared.sort(key=lambda t: 0 if "anchor" in {
            c for p in t[1]["preds"].values() for c in p["categories"]
        } else 1)
        shared = shared[:4]
        if not shared:
            continue
        out.append(f"<h3>Same lines, every model &mdash; <code>{html.escape(ds)}</code></h3>")
        for _sid, slot in shared:
            out.append("<div class='cmp'>")
            out.append(_img_block(slot))
            out.append(f"<p class='gt' dir='auto'><b>GT</b> {html.escape(slot['ground_truth'])}</p>")
            out.append("<table class='cmp-t'><tbody>")
            for m in real_models:
                p = slot["preds"].get(m)
                if not p:
                    continue
                out.append(
                    f"<tr><td class='cm'>{html.escape(m)}</td>"
                    f"<td class='cc'>{_cer_pct(p['cer_light'])}</td>"
                    f"<td>{_pred_line(p['prediction'], p['error'])}</td></tr>"
                )
            out.append("</tbody></table></div>")

    # 2. best / worst per model — up to 2 tiles per dataset the model ran on
    rank = {"best": 0, "typical": 1, "longest": 2, "worst": 3}
    for m in real_models:
        picks = [
            (slot["dataset"], slot, slot["preds"][m])
            for slot in by_id.values()
            if m in slot["preds"]
            and {c for c in slot["preds"][m]["categories"]} & set(rank)
        ]
        if not picks:
            continue
        picks.sort(key=lambda t: (
            t[0],
            min((rank.get(c, 5) for c in t[2]["categories"]), default=5),
            float(t[2].get("cer_light") or 0),
        ))
        seen: dict[str, int] = {}
        kept = []
        for ds, slot, p in picks:
            if seen.get(ds, 0) >= 2:
                continue
            seen[ds] = seen.get(ds, 0) + 1
            kept.append((ds, slot, p))
        out.append(f"<h3>{html.escape(m)} — best &amp; worst</h3><div class='grid'>")
        for ds, slot, p in kept:
            out.append("<div class='samp'>")
            out.append(_chips(p["categories"], ds))
            out.append(_img_block(slot))
            out.append(f"<p class='gt' dir='auto'><b>GT</b> {html.escape(slot['ground_truth'])}</p>")
            out.append(f"<p class='pw'>{_pred_line(p['prediction'], p['error'])}"
                       f"<span class='cc'>{_cer_pct(p['cer_light'])}</span></p>")
            out.append("</div>")
        out.append("</div>")

    out.append("</section>")
    return "\n".join(out)


def _img_block(slot: dict) -> str:
    if not slot.get("image"):
        return "<div class='samp-img samp-img--missing'>image unavailable</div>"
    return (f"<div class='samp-img'><img loading='lazy' alt='handwritten Arabic line' "
            f"src='{slot['image']}'></div>")

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
table.sortable thead th{cursor:pointer;}
table.sortable thead th:hover{color:var(--ink);}
thead th[aria-sort]{color:var(--ink);}
thead th[aria-sort="ascending"]::after{content:" \2191";}
thead th[aria-sort="descending"]::after{content:" \2193";}
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
.by-dataset{margin:44px 0 0;}
.by-dataset h3{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500;
  letter-spacing:.04em;text-transform:uppercase;color:var(--ink);
  margin:26px 2px 10px;display:flex;align-items:baseline;gap:10px;}
.by-dataset h3:first-of-type{margin-top:16px;}
.by-dataset .dscount{font-size:11px;font-weight:400;color:var(--muted);
  text-transform:none;letter-spacing:0;}
.by-dataset .scroll+h3{margin-top:26px;}

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

/* --- model / dataset cards --- */
.cards{margin:40px 0 0;}
.cardlist{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:2px 28px;margin:0 0 20px;}
.cardlist div{padding:11px 0;border-top:1px solid var(--line);}
.cardlist dt{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:500;
  color:var(--ink);margin-bottom:4px;}
.cardlist dd{margin:0;font-size:13px;color:var(--muted);line-height:1.55;max-width:52ch;}

/* --- sample transcriptions --- */
.samples{margin:44px 0 0;}
.samp-intro{color:var(--muted);font-size:13px;margin:0 0 18px;max-width:64ch;}
.samples h3{font-family:Spectral,Georgia,serif;font-weight:600;font-size:17px;
  color:var(--ink);margin:30px 0 12px;letter-spacing:-.01em;}
.samp-img{border:1px solid var(--line);border-radius:8px;overflow:hidden;
  background:#fff;line-height:0;}
.samp-img img{width:100%;height:auto;display:block;}
.samp-img--missing{padding:16px;font-size:12px;color:var(--muted);
  background:var(--surface);line-height:1.4;}
.gt{font-size:14px;line-height:1.7;margin:8px 0 4px;color:var(--ink);}
.gt b,.pw .cc,.cmp-t .cm{font-family:"IBM Plex Mono",monospace;font-size:11px;
  font-weight:500;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
  margin-inline-end:6px;}
.pred{font-size:14px;line-height:1.7;color:var(--ink);}
.pred.err{color:var(--muted);font-style:italic;}

.cmp{border-top:1px solid var(--line);padding:16px 0;}
.cmp .samp-img{max-width:640px;}
.cmp-t{width:100%;font-size:13px;margin-top:8px;}
.cmp-t td{padding:6px 10px 6px 0;border:none;white-space:normal;vertical-align:top;}
.cmp-t .cm{white-space:nowrap;}
.cmp-t .cc{font-family:"IBM Plex Mono",monospace;text-align:right;
  font-variant-numeric:tabular-nums;color:var(--muted);white-space:nowrap;}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px;}
.samp{border:1px solid var(--line);border-radius:10px;padding:12px;background:var(--surface);}
.samp .samp-img{margin:6px 0;}
.pw{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  font-size:14px;line-height:1.6;margin:4px 0 0;}
.pw .cc{flex:none;}
.chip{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:10px;
  letter-spacing:.03em;text-transform:uppercase;padding:2px 7px;border-radius:999px;
  margin:0 5px 0 0;background:var(--accent-soft);color:var(--accent);}
.chip--worst{background:color-mix(in srgb,#a5443a 16%,transparent);color:#a5443a;}
.chip--best{background:color-mix(in srgb,#3a7d5c 16%,transparent);color:#3a7d5c;}
.chip--ds{background:var(--ink);color:var(--ground);}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .samp-img,
  :root:not([data-theme="light"]) .samp-img img{background:#f2f2ef;}
  :root:not([data-theme="light"]) .chip--worst{color:#e0857b;}
  :root:not([data-theme="light"]) .chip--best{color:#79c6a0;}
}
:root[data-theme="dark"] .samp-img,:root[data-theme="dark"] .samp-img img{background:#f2f2ef;}
</style>"""


def _render_empty(title: str) -> str:
    gen = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return _HEAD.replace("{{TITLE}}", html.escape(title)) + f"""
<main>
<header class='top'>
<p class='eyebrow'>Handwritten Arabic · line-level recognition</p>
<h1>{html.escape(title)}</h1>
<p class='lede'>Character- and word-error rates for OCR / vision-language models
transcribing cursive handwritten Arabic, scored on the same held-out lines with
the same metrics.</p>
<p class='status status--preview'><span class='dot'></span>
No benchmark runs yet — the leaderboard, model cards and sample transcriptions
appear here after the first sweep.</p>
</header>
<footer><p>Dataset: <b>Muharaf-public</b> — Manuscripts of Handwritten Arabic
(NeurIPS 2024), CC-BY-NC-SA-4.0, <code>test</code> split. Harness:
<code>arabic-ocr-bench</code>. Generated {gen} by
<code>scripts/make_leaderboard.py</code>.</p></footer>
</main>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="results/ dir, or a single results/<run_id>/ dir")
    ap.add_argument("-o", "--out", default="results/leaderboard.html")
    ap.add_argument("--title", default="Arabic Handwriting OCR Leaderboard")
    args = ap.parse_args()

    target = Path(args.target)
    rows = collect(target)
    if not rows:
        Path(args.out).write_text(_render_empty(args.title), encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text("[]", encoding="utf-8")
        print(f"leaderboard -> {args.out}  (no runs yet — placeholder page)")
        return 0
    Path(args.out).write_text(render(rows, args.title, target), encoding="utf-8")
    print(f"leaderboard -> {args.out}  ({len(rows)} rows from "
          f"{len({r['run'] for r in rows})} run(s))")
    # also dump a machine-readable bundle next to it
    bundle = Path(args.out).with_suffix(".json")
    bundle.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"bundle      -> {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

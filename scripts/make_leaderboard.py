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
import re
import sys
from pathlib import Path

import _leaderboard_findings as _lf  # scripts/ sibling; resolvable on sys.path[0]
import _leaderboard_prompt_study as _lps  # scripts/ sibling; the Research tab
import yaml

_PROFILES = ("raw", "light", "diac_insensitive")
_REPO = Path(__file__).resolve().parent.parent
_SAMPLE_CATS = ("anchor", "best", "worst", "typical", "perfect", "longest")

# Full row counts per dataset, for the "cost to run the whole dataset" projection.
# Overridden by results/_dataset_sizes.json when present. historicalbooks / hindawi
# are approximate until a size probe fills them in.
_DATASET_ROWS = {
    "muharaf_lines": 1334,
    "kitab_khatt": 200,
    "kitab_muharaf": 200,
    "kitab_historicalbooks": 200,
    "kitab_hindawi": 200,
}


def _dataset_rows() -> dict[str, int]:
    out = dict(_DATASET_ROWS)
    f = _REPO / "results" / "_dataset_sizes.json"
    try:
        out.update({k: int(v) for k, v in json.loads(f.read_text()).items()})
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return out


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


def collect_review(target: Path) -> dict[str, list[dict]]:
    """Every scored line, per model: {model -> [ {sample_id, dataset, cer_light,
    cer_raw, wer_light, exact, error, gt, pred, image}, ... ]}.

    Joins each run's scores.jsonl (metrics) with its predictions.jsonl (text),
    then dedupes by (model, sample_id) across all runs — a later successful
    reading supersedes an earlier errored one, otherwise the first is kept.
    Errored lines are included with cer_light = None so they can head the
    'worst' list. Images are attached from samples.json where one exists.
    """
    # image source: a full run-level images.json (every line the run touched)
    # wins; otherwise the curated samples.json slot image.
    images = {sid: slot.get("image", "") for sid, slot in collect_samples(target).items()}
    img_files = (
        [target / "images.json"]
        if (target / "images.json").exists()
        else sorted(target.glob("*/images.json"))
    )
    for f in img_files:
        try:
            for sid, uri in json.loads(f.read_text(encoding="utf-8")).items():
                if uri:
                    images[sid] = uri
        except (OSError, json.JSONDecodeError):
            continue
    merged: dict[tuple[str, str], dict] = {}
    for rd in _run_dirs(target):
        preds: dict[tuple[str, str], dict] = {}
        pf = rd / "predictions.jsonl"
        if pf.exists():
            for line in pf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preds[(p.get("model", ""), p.get("sample_id", ""))] = p
        sf = rd / "scores.jsonl"
        if not sf.exists():
            continue
        for line in sf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            model, sid = s.get("model", ""), s.get("sample_id", "")
            ds = s.get("dataset", "")
            if not model or model.startswith("mock") or ds == "synthetic":
                continue
            p = preds.get((model, sid), {})
            rec = {
                "sample_id": sid,
                "dataset": ds,
                "cer_light": s.get("cer_light"),
                "cer_raw": s.get("cer_raw"),
                "wer_light": s.get("wer_light"),
                "exact": bool(s.get("exact_light")),
                "error": s.get("error") or p.get("error"),
                "gt": p.get("ground_truth", ""),
                "pred": p.get("cleaned_text", ""),
                "image": images.get(sid, ""),
            }
            prev = merged.get((model, sid))
            if prev is None or (prev.get("error") and not rec.get("error")):
                merged[(model, sid)] = rec
    out: dict[str, list[dict]] = {}
    for (model, _sid), rec in merged.items():
        out.setdefault(model, []).append(rec)
    return out


def collect_costs(target: Path) -> dict[str, dict]:
    """Per model, from every run's predictions.jsonl (deduped by sample_id, last
    successful wins): number of billable calls, mean & median $/call, mean in/out
    tokens, and mean $/line per dataset. Only models with a pricing block in
    configs/models.yaml are returned."""
    import statistics

    priced = {
        name for name, c in (yaml.safe_load((_REPO / "configs" / "models.yaml").read_text()) or {}).items()
        if isinstance(c, dict) and (c.get("pricing") or {}).get("input_usd_per_1m") is not None
    }
    pricing = {
        name: (c.get("pricing") or {})
        for name, c in (yaml.safe_load((_REPO / "configs" / "models.yaml").read_text()) or {}).items()
        if isinstance(c, dict)
    }
    seen: dict[tuple[str, str], dict] = {}
    for rd in _run_dirs(target):
        pf = rd / "predictions.jsonl"
        if not pf.exists():
            continue
        for line in pf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = p.get("model", "")
            if m not in priced or p.get("dataset") == "synthetic":
                continue
            key = (m, p.get("sample_id", ""))
            prev = seen.get(key)
            if prev is None or (prev.get("error") and not p.get("error")):
                seen[key] = p

    out: dict[str, dict] = {}
    for (m, _sid), p in seen.items():
        if p.get("error"):
            continue
        d = out.setdefault(m, {
            "pricing": pricing.get(m, {}), "costs": [], "in": [], "out": [],
            "per_ds": {},  # dataset -> list[cost_usd]
        })
        c = float(p.get("cost_usd", 0.0) or 0.0)
        d["costs"].append(c)
        d["in"].append(int(p.get("input_tokens", 0) or 0))
        d["out"].append(int(p.get("output_tokens", 0) or 0))
        d["per_ds"].setdefault(p.get("dataset", ""), []).append(c)

    for m, d in out.items():
        d["calls"] = len(d["costs"])
        d["mean_call"] = statistics.fmean(d["costs"]) if d["costs"] else 0.0
        d["median_call"] = statistics.median(d["costs"]) if d["costs"] else 0.0
        d["mean_in"] = statistics.fmean(d["in"]) if d["in"] else 0.0
        d["mean_out"] = statistics.fmean(d["out"]) if d["out"] else 0.0
        d["ds_line"] = {
            ds: (statistics.fmean(cs) if cs else 0.0) for ds, cs in d["per_ds"].items()
        }
    return out


def _run_dirs(target: Path) -> list[Path]:
    if (target / "summary.csv").exists():
        return [target]
    return sorted(p.parent for p in target.glob("*/summary.csv"))


def _ensure_findings(target: Path) -> dict:
    """Return the parsed findings doc for ``target``, (re)building
    ``results/findings.json`` when it is missing or older than the newest run
    file under ``target``, else reading the cached file. Never raises on an
    analysis failure — logs to stderr and returns ``{}``."""
    fj = (target / "findings.json") if target.is_dir() else target.parent / "findings.json"
    newest = 0.0
    for rd in _run_dirs(target):
        for name in ("predictions.jsonl", "scores.jsonl"):
            p = rd / name
            if p.exists():
                newest = max(newest, p.stat().st_mtime)
    if fj.exists() and fj.stat().st_mtime >= newest:
        try:
            doc = json.loads(fj.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
            # valid JSON but not an object — treat like a decode error, rebuild.
        except (OSError, json.JSONDecodeError):
            pass
    try:
        from arabic_ocr_bench.analysis.findings import build

        doc = build(target)
        fj.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc
    except Exception as e:  # noqa: BLE001
        print(f"findings build failed: {e}", file=sys.stderr)
        return {}


def _run_meta(run_dir: Path) -> dict:
    meta = {"run": run_dir.name, "when": "", "limit": "", "prompt_version": ""}
    try:
        stamp = run_dir.name.split("-")[0]  # tolerate a "-2" same-second suffix
        parsed = dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
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


_MIN_N = 10  # a handful of lines (a smoke batch, a rate-limit-truncated pass)
             # isn't a meaningful score -- keep it off the board entirely
             # rather than show a misleadingly precise CER from n<10.


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
    merged = _merge_duplicate_runs(rows, run_dirs)
    _compute_coverage(merged, run_dirs)
    return [r for r in merged if int(r.get("n") or 0) >= _MIN_N]


def _sample_index(sample_id: str) -> int | None:
    """`<dataset>/<split>/000123` -> 123, the row's position in that split."""
    tail = sample_id.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _compute_coverage(rows: list[dict], run_dirs: dict[str, Path]) -> None:
    """Attach `_full` (bool) and `_pfx` (int) to every row for the "fair
    comparison" leaderboard views: `_full` is whether the model was scored on
    every line of the dataset (per `_dataset_rows()`); `_pfx` is the length of
    the largest *clean, zero-based prefix* of sample indices it covers (0 if
    the covered indices aren't a contiguous 0..k-1 run -- e.g. accumulate mode
    filling in scattered gaps across passes). A model with `_pfx >= 20` was
    scored on exactly the same first-20 lines as any other model with
    `_pfx >= 20` on that dataset, so comparing them there is apples-to-apples
    even if neither has finished the whole dataset.

    Reads each contributing run's predictions.jsonl once (cached per run_dir,
    not per row) rather than trusting `n` alone, since `n` doesn't say *which*
    lines were scored -- only a contiguous-from-0 index set does.
    """
    sizes = _dataset_rows()
    cache: dict[str, dict[tuple[str, str], set[int]]] = {}

    def idx_for(run_name: str, model: str, dataset: str) -> set[int]:
        if run_name not in cache:
            m: dict[tuple[str, str], set[int]] = {}
            rd = run_dirs.get(run_name)
            pf = rd / "predictions.jsonl" if rd else None
            if pf and pf.exists():
                with pf.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            p = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if p.get("error"):
                            continue
                        i = _sample_index(p.get("sample_id", ""))
                        if i is not None:
                            m.setdefault((p.get("model"), p.get("dataset")), set()).add(i)
            cache[run_name] = m
        return cache[run_name].get((model, dataset), set())

    for r in rows:
        idxs: set[int] = set()
        for run_name in (r.get("run") or "").split("+"):
            idxs |= idx_for(run_name, r["model"], r["dataset"])
        total = sizes.get(r["dataset"])
        r["_full"] = bool(total and idxs == set(range(total)))
        r["_pfx"] = len(idxs) if idxs and idxs == set(range(len(idxs))) else 0


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
                if p.get("model") != model or p.get("dataset") != dataset:
                    continue
                prev = seen.get(p["sample_id"])
                # a later, successful retry supersedes an earlier errored attempt;
                # otherwise keep the first successful reading we saw.
                if prev is None or (prev.get("error") and not p.get("error")):
                    seen[p["sample_id"]] = p

    if not seen:
        # predictions.jsonl missing for every contributing run (older run dir) —
        # fall back to the previous behaviour rather than reporting nothing.
        return max(group, key=lambda r: int(r.get("n", 0) or 0))

    ok = [p for p in seen.values() if not p.get("error")]
    n_err = len(seen) - len(ok)
    n = len(ok)
    scored = [
        (p, score_pair(p["ground_truth"], p.get("cleaned_text") or "")) for p in ok
    ]
    agg = aggregate([ps for _, ps in scored]) if scored else None

    row = {
        "run": "+".join(sorted({r["run"] for r in group})),
        "when": max((r.get("when", "") for r in group), default=""),
        "limit": str(n),
        "prompt_version": "/".join(sorted({r.get("prompt_version", "") for r in group} - {""})),
        "model": model,
        "dataset": dataset,
        "n": str(n),
        "errors": str(n_err),
        "refusal_rate": f"{n_err / (n + n_err):.4f}" if (n + n_err) else "0.0",
        "latency_ms_mean": (
            f"{sum(p.get('latency_ms', 0.0) for p, _ in scored) / n:.1f}" if n else "0.0"
        ),
        "cost_usd_total": (
            f"{sum(p.get('cost_usd', 0.0) for p in seen.values()):.6f}"
        ),
    }
    for p in _PROFILES:
        row[f"cer_micro_{p}"] = f"{agg['micro_cer'][p]:.4f}" if agg else ""
        row[f"cer_macro_{p}"] = f"{agg['cer_macro'][p]:.4f}" if agg else ""
        row[f"car_micro_{p}"] = f"{agg['micro_car'][p]:.4f}" if agg else ""
        row[f"car_macro_{p}"] = f"{agg['car_macro'][p]:.4f}" if agg else ""
        row[f"wer_micro_{p}"] = f"{agg['micro_wer'][p]:.4f}" if agg else ""
        row[f"mer_micro_{p}"] = f"{agg['micro_mer'][p]:.4f}" if agg else ""
        row[f"mer_macro_{p}"] = f"{agg['mer_macro'][p]:.4f}" if agg else ""
        row[f"wil_micro_{p}"] = f"{agg['micro_wil'][p]:.4f}" if agg else ""
        row[f"wip_micro_{p}"] = f"{agg['micro_wip'][p]:.4f}" if agg else ""
        row[f"exact_{p}"] = f"{agg['exact_match_rate'][p]:.4f}" if agg else ""
    row["diac_f1"] = f"{agg['diacritic']['f1']:.4f}" if agg else ""
    return row


_EXPLORER_COLS = [
    # (key in payload, label, decimals, *100, best = "lo" | "hi" | "", hover def)
    ("n", "n", 0, False, "", "lines scored for this model on this dataset"),
    ("err", "err", 0, False, "lo", "lines that errored or were refused (excluded from the rates)"),
    ("cl", "CER·light", 1, True, "lo",
     "character error rate after light normalization (alef forms, digits, tatweel, "
     "spacing) — corpus-weighted. 0 = perfect. The headline metric."),
    ("cr", "CER·raw", 1, True, "lo",
     "character error rate on exact strings — every orthographic and diacritic "
     "difference counts. Corpus-weighted."),
    ("cnd", "CER·no-diac", 1, True, "lo",
     "character error rate with light normalization and harakat stripped from both "
     "sides — how well the skeleton letters were read."),
    ("cmac", "CER·macro", 1, True, "lo",
     "mean of the per-line light CER (every line weighted equally) rather than "
     "corpus-weighted — more sensitive to short lines."),
    ("car", "CAR·light", 1, True, "hi",
     "character accuracy rate after light normalization = 100 − CER·light "
     "(clamped at 0). Higher is better."),
    ("wl", "WER·light", 1, True, "lo",
     "word error rate after light normalization, corpus-weighted."),
    ("mer", "MER·light", 1, True, "lo",
     "match error rate — like WER but hits are in the denominator, so it never "
     "exceeds 100. Light normalization, corpus-weighted."),
    ("wil", "WIL·light", 1, True, "lo",
     "word information lost = 100·(1 − (H/N_ref)·(H/N_hyp)); penalises both "
     "missed and invented words. Light normalization."),
    ("ex", "exact %", 1, True, "hi",
     "share of lines transcribed exactly right after light normalization."),
    ("df", "diac F1", 3, False, "hi",
     "F1 of harakat (diacritic) marks against the reference, measured on raw text."),
    ("ms", "ms/line", 0, False, "lo", "mean wall-clock latency per line, milliseconds"),
    ("cost", "cost $", 4, False, "lo",
     "total USD billed for this model on this dataset in the stored runs"),
]

_EXPLORER_JS = r"""
(function(){
  var el=function(id){return document.getElementById(id);};
  var D={}; try{D=JSON.parse(el('ex-data').textContent);}catch(e){return;}
  var COLS=D.cols, DS=D.ds;
  var sel=el('ex-ds'), card=el('ex-card'), out=el('ex-out');
  var sortKey='cl', sortDir=1;
  function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function fmt(v,dp,pct){
    if(v==null||v==='') return '—';
    var n=Number(v); if(isNaN(n)) return '—';
    return (pct?n*100:n).toFixed(dp);
  }
  function render(){
    var d=DS[sel.value]; if(!d) return;
    card.textContent=d.card||'';
    var rows=d.rows.slice().sort(function(a,b){
      var x=a[sortKey], y=b[sortKey];
      x=(x==null||x==='')?Infinity:Number(x); y=(y==null||y==='')?Infinity:Number(y);
      return (x-y)*sortDir;
    });
    var head="<th class='exm' data-k='m'>model</th>"+COLS.map(function(c){
      var tip=c[5]?" title='"+esc(c[5])+"'":"";
      var lab=c[5]?"<abbr class='gloss'"+tip+">"+esc(c[1])+"</abbr>":esc(c[1]);
      return "<th data-k='"+c[0]+"'"+(c[0]===sortKey?" aria-sort='"+(sortDir>0?'ascending':'descending')+"'":"")+">"+lab+"</th>";
    }).join('');
    // best value per column (for the bold-best highlight, like a benchmark table)
    var best={};
    COLS.forEach(function(c){
      if(!c[4]) return;
      var vals=rows.map(function(r){return r[c[0]];})
                   .filter(function(v){return v!=null && v!=='' && !isNaN(Number(v));})
                   .map(Number);
      if(vals.length) best[c[0]] = c[4]==='hi' ? Math.max.apply(null,vals) : Math.min.apply(null,vals);
    });
    var body=rows.map(function(r){
      return "<tr><td class='exm'>"+esc(r.m)+"</td>"+COLS.map(function(c){
        var v=r[c[0]];
        var isBest = (c[0] in best) && v!=null && v!=='' && Number(v)===best[c[0]];
        return "<td class='exn"+(isBest?" exbest":"")+"'>"+fmt(v,c[2],c[3])+"</td>";
      }).join('')+"</tr>";
    }).join('');
    out.innerHTML="<div class='scroll'><table class='ex-tbl'><thead><tr>"+head+
      "</tr></thead><tbody>"+body+"</tbody></table></div>";
    Array.prototype.forEach.call(out.querySelectorAll('th[data-k]'),function(th){
      th.addEventListener('click',function(){
        var k=th.getAttribute('data-k'); if(k==='m') return;
        if(k===sortKey){ sortDir=-sortDir; } else { sortKey=k; sortDir=1; }
        render();
      });
    });
  }
  sel.addEventListener('change',render);
  render();
})();
"""

_MODELREF_JS = r"""
(function(){
  var el=function(id){return document.getElementById(id);};
  var M={}; try{M=JSON.parse(el('mr-data').textContent);}catch(e){return;}
  var sel=el('mr-sel'), out=el('mr-out');
  function render(){ out.textContent=M[sel.value]||'(no description)'; }
  sel.addEventListener('change',render); render();
})();
"""


def _render_explorer(rows: list[dict], target: Path) -> str:
    """Pick a dataset -> its description + a sortable table of every model on it."""
    cards = load_cards()
    datasets = sorted({r["dataset"] for r in rows})
    if not datasets:
        return ""
    payload = {"cols": [list(c) for c in _EXPLORER_COLS], "ds": {}}
    for ds in datasets:
        drows = sorted(
            (r for r in rows if r["dataset"] == ds),
            key=lambda r: float(r.get("cer_micro_light", 1) or 1),
        )
        payload["ds"][ds] = {
            "card": cards["datasets"].get(ds, ""),
            "rows": [
                {
                    "m": r["model"], "n": r.get("n", ""), "err": r.get("errors", ""),
                    "cl": r.get("cer_micro_light", ""), "cr": r.get("cer_micro_raw", ""),
                    "cnd": r.get("cer_micro_diac_insensitive", ""),
                    "cmac": r.get("cer_macro_light", ""),
                    "car": r.get("car_micro_light", ""),
                    "wl": r.get("wer_micro_light", ""),
                    "mer": r.get("mer_micro_light", ""),
                    "wil": r.get("wil_micro_light", ""),
                    "ex": r.get("exact_light", ""), "df": r.get("diac_f1", ""),
                    "ms": r.get("latency_ms_mean", ""), "cost": r.get("cost_usd_total", ""),
                }
                for r in drows
            ],
        }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    opts = "".join(f"<option value='{html.escape(d)}'>{html.escape(d)}</option>" for d in datasets)
    return f"""<section class='explorer'><h2>By dataset</h2>
<p class='board-note'>Pick a dataset to see its description and every model scored on it —
same lines, same metrics. Click a column header to sort.</p>
<div class='ex-controls'><label>dataset <select id='ex-ds'>{opts}</select></label></div>
<p class='ex-card' id='ex-card'></p>
<div id='ex-out'></div>
<script id='ex-data' type='application/json'>{blob}</script>
<script>{_EXPLORER_JS}</script>
</section>"""


def _render_model_reference(rows: list[dict]) -> str:
    """Pick a model -> its one-line description."""
    cards = load_cards()
    models = [m for m in _model_order(rows) if m in cards["models"]]
    if not models:
        return ""
    payload = {m: cards["models"][m] for m in models}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    opts = "".join(f"<option value='{html.escape(m)}'>{html.escape(m)}</option>" for m in models)
    return f"""<section class='modelref'><h2>Model reference</h2>
<div class='mr-controls'><label>model <select id='mr-sel'>{opts}</select></label></div>
<p class='mr-out' id='mr-out'></p>
<script id='mr-data' type='application/json'>{blob}</script>
<script>{_MODELREF_JS}</script>
</section>"""


def _is_preview(rows: list[dict]) -> bool:
    """True while the only data is the offline self-test (mock models / synthetic set)."""
    return all(
        r.get("model", "").startswith("mock") or r.get("dataset") == "synthetic"
        for r in rows
    )


def _num(v, factor=100.0):
    try:
        return float(v) * factor
    except (TypeError, ValueError):
        return None


def _pct1(v) -> str:
    n = _num(v)
    return "—" if n is None else f"{n:.1f}"


def _model_order(rows: list[dict]) -> list[str]:
    """Models, best mean CER-light first (the order matrix columns use)."""
    by_model: dict[str, list[float]] = {}
    for r in rows:
        n = _num(r.get("cer_micro_light"))
        if n is not None:
            by_model.setdefault(r["model"], []).append(n)
    return sorted(
        by_model, key=lambda m: sum(by_model[m]) / len(by_model[m])
    ) + [m for m in dict.fromkeys(r["model"] for r in rows) if m not in by_model]


_MATRIX_JS = r"""
(function(){
  var el=function(id){return document.getElementById(id);};
  var P={}; try{P=JSON.parse(el('mx-data').textContent);}catch(e){return;}
  var DS=P.ds, M=P.m, C=P.c, FULL=P.full, PFX=P.pfx;  // C[di][mi]=[cerLight,werLight]|null
  var out=el('mx-out'), note=el('mx-sortnote'), modeSel=el('mx-mode');
  var meanOrder=P.order;                     // default model column order (indices)
  var sortDi=-1;                             // >=0 => columns sorted by that dataset
  function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function pc(v){return (v==null||isNaN(v))?'—':(v*100).toFixed(1);}
  function tcls(cer){ if(cer==null||isNaN(cer)) return ''; return 'mx-t'+Math.min(7,Math.max(0,Math.floor(cer/0.15))); }
  function mode(){ return modeSel ? modeSel.value : 'all'; }
  // Is this cell shown under the current comparison mode? 'full' = model
  // scored the entire dataset; 'pfxN' = model's scored lines are a clean,
  // zero-based prefix covering at least the first N -- i.e. the SAME N lines
  // every other qualifying model was scored on, for a fair head-to-head.
  function pass(di,mi){
    if(!C[di][mi]) return false;
    var md=mode();
    if(md==='all') return true;
    if(md==='full') return !!(FULL && FULL[di] && FULL[di][mi]);
    var need = md==='pfx10'?10 : md==='pfx20'?20 : md==='pfx40'?40 : 0;
    return !!(PFX && PFX[di] && PFX[di][mi] >= need);
  }
  function colOrder(){
    if(sortDi<0) return meanOrder.slice();
    var row=C[sortDi];
    return M.map(function(_,i){return i;}).sort(function(a,b){
      var x=(pass(sortDi,a) && row[a])?row[a][0]:Infinity;
      var y=(pass(sortDi,b) && row[b])?row[b][0]:Infinity;
      return x-y;
    });
  }
  function render(){
    var ord=colOrder();
    var head="<th class='mx-corner'"+(sortDi>=0?" data-reset='1' title='reset column order'":"")+">dataset \\ model</th>"
      + ord.map(function(i){return "<th title='"+esc(M[i])+"'>"+esc(M[i])+"</th>";}).join('');
    var body=DS.map(function(ds,di){
      var row=C[di];
      var best=Infinity;
      row.forEach(function(c,mi){ if(c && pass(di,mi) && c[0]<best) best=c[0]; });
      var cells="<th class='mx-ds"+(di===sortDi?" mx-ds-on":"")+"' data-di='"+di+"' role='button' tabindex='0'>"+esc(ds)+"</th>";
      cells+=ord.map(function(i){
        var c=row[i];
        if(!c || !pass(di,i)) return "<td class='mx-empty'>·</td>";
        var isBest=c[0]===best;
        return "<td class='"+tcls(c[0])+(isBest?" mx-best":"")+"'>"+
          "<span class='mx-cer'>"+pc(c[0])+"</span><span class='mx-wer'>"+pc(c[1])+"</span></td>";
      }).join('');
      return "<tr>"+cells+"</tr>";
    }).join('');
    out.innerHTML="<div class='scroll'><table class='mx-tbl'><thead><tr>"+head+
      "</tr></thead><tbody>"+body+"</tbody></table></div>";
    var modeNote={full:' — only models scored on every line of the dataset are shown.',
      pfx10:' — only models scored on the same first 10 lines are shown.',
      pfx20:' — only models scored on the same first 20 lines are shown.',
      pfx40:' — only models scored on the same first 40 lines are shown.'}[mode()] || '';
    note.textContent = (sortDi>=0
      ? ("columns sorted best→worst on "+DS[sortDi]+" — click it again (or the corner) to reset")
      : "click a dataset name to sort the model columns best→worst on it") + modeNote;
  }
  out.addEventListener('click',function(e){
    var t=e.target.closest('[data-di],[data-reset]'); if(!t) return;
    if(t.hasAttribute('data-reset')){ sortDi=-1; }
    else { var di=+t.getAttribute('data-di'); sortDi = (di===sortDi ? -1 : di); }
    render();
  });
  out.addEventListener('keydown',function(e){
    if(e.key!=='Enter' && e.key!==' ') return;
    var t=e.target.closest('[data-di]'); if(!t) return;
    e.preventDefault(); var di=+t.getAttribute('data-di'); sortDi=(di===sortDi?-1:di); render();
  });
  if(modeSel){
    try{ var saved=localStorage.getItem('lb_mx_mode'); if(saved) modeSel.value=saved; }catch(_){}
    modeSel.addEventListener('change',function(){
      try{ localStorage.setItem('lb_mx_mode', modeSel.value); }catch(_){}
      render();
    });
  }
  render();
})();
"""


_FT_CSS = """<style>
.ft-view{max-width:900px}
.ft-view>h2{margin:.2rem 0 .3rem}
.ft-exp{border:1px solid var(--line);border-radius:10px;padding:.9rem 1.05rem;margin:0 0 1.1rem}
.ft-exp h3{margin:.1rem 0 .55rem;font-size:1.02rem}
.ft-meta{font-size:.84rem;color:var(--muted);margin:.18rem 0;line-height:1.55}
.ft-meta code{font-size:.86em}
.ft-head{font-size:.92rem;margin:.55rem 0 .7rem}
.ft-head b{font-variant-numeric:tabular-nums}
.ft-tbl{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}
.ft-tbl th,.ft-tbl td{padding:.32rem .55rem;border-bottom:1px solid var(--line);text-align:right}
.ft-tbl th:first-child,.ft-tbl td:first-child{text-align:left}
.ft-tbl thead th{color:var(--muted);font-weight:600}
.ft-tbl tr.ft-mean td{border-top:2px solid var(--line);font-weight:600}
.ft-gain{color:#0a7d3c}.ft-loss{color:#c0392b}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]) .ft-gain{color:#4ade80}
  :root:not([data-theme=light]) .ft-loss{color:#f87171}}
:root[data-theme=dark] .ft-gain{color:#4ade80}
:root[data-theme=dark] .ft-loss{color:#f87171}
.ft-train{margin:.4rem 0 .7rem;max-width:340px}
.ft-train .ft-curve{display:block;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
.ft-hyp{font-size:.84rem;font-style:italic;color:var(--muted);margin:.6rem 0 .1rem}
.ft-note{font-size:.85rem;line-height:1.55;margin:.5rem 0 0;padding-left:1.15rem}
.ft-note li{margin:.22rem 0}
.ft-empty{color:var(--muted);font-size:.9rem}
</style>"""


def _loss_curve_svg(train: list[tuple[float, float]], evals: list[tuple[float, float]]) -> str:
    """Tiny inline SVG: training loss line + eval-loss dots vs optimizer step."""
    if len(train) < 2:
        return ""
    W, H, ml, mr, mt, mb = 320, 120, 34, 8, 8, 20
    xs = [s for s, _ in train] + [s for s, _ in evals]
    ys = [v for _, v in train] + [v for _, v in evals]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    yr = (y1 - y0) or 1.0
    xr = (x1 - x0) or 1.0

    def fx(s):
        return ml + (W - mr - ml) * (s - x0) / xr

    def fy(v):
        return (H - mb) - (H - mb - mt) * (v - y0) / yr

    pts = " ".join(f"{fx(s):.1f},{fy(v):.1f}" for s, v in train)
    dots = "".join(
        f'<circle cx="{fx(s):.1f}" cy="{fy(v):.1f}" r="2.6" fill="var(--accent)"></circle>'
        for s, v in evals
    )
    return (
        f'<svg class="ft-curve" viewBox="0 0 {W} {H}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="var(--line)"></line>'
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}" stroke="var(--line)"></line>'
        f'<text x="{ml-4}" y="{fy(y1):.1f}" text-anchor="end" font-size="8" '
        f'fill="var(--muted)">{y1:.2f}</text>'
        f'<text x="{ml-4}" y="{fy(y0):.1f}" text-anchor="end" font-size="8" '
        f'fill="var(--muted)">{y0:.2f}</text>'
        f'<text x="{(ml+W-mr)/2:.0f}" y="{H-4}" text-anchor="middle" font-size="8" '
        f'fill="var(--muted)">optimizer step</text>'
        f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"></polyline>'
        f'{dots}</svg>'
    )


def _ft_training_block(e: dict) -> str:
    """Per-experiment training snapshot: loss curve + a couple of numbers, from
    the fetched ``<report_dir>/trainer_state.json`` if it has been committed."""
    rd = e.get("report_dir")
    if not rd:
        return ""
    tsp = _REPO / rd / "trainer_state.json"
    if not tsp.exists():
        return (f"<p class='ft-meta'>Training curve: drop the run's "
                f"<code>trainer_state.json</code> at <code>{html.escape(rd)}/</code> "
                f"(from the pipeline's <code>reports/&lt;exp&gt;/</code> snapshot).</p>")
    try:
        ts = json.loads(tsp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    lh = ts.get("log_history") or []
    train = [(float(x["step"]), float(x["loss"])) for x in lh if "loss" in x and "step" in x]
    evals = [(float(x["step"]), float(x["eval_loss"])) for x in lh if "eval_loss" in x]
    bits = []
    if ts.get("global_step"):
        bits.append(f"{int(ts['global_step'])} steps")
    if ts.get("epoch"):
        bits.append(f"{float(ts['epoch']):.2f} epoch")
    if train:
        bits.append(f"loss {train[0][1]:.2f} &rarr; {train[-1][1]:.2f}")
    if evals:
        bits.append(f"eval-loss {evals[-1][1]:.2f}")
    svg = _loss_curve_svg(train, evals)
    head = " &middot; ".join(bits)
    return (f"<div class='ft-train'>{svg}"
            + (f"<p class='ft-meta'>{head}</p>" if head else "")
            + "</div>")


def _render_finetuning(rows: list[dict], target: Path) -> tuple[str, list[tuple[str, str]]]:
    """The Fine-tuning tab: per training experiment (from
    ``configs/training/experiments.yaml``), base vs fine-tune on held-out
    datasets. Returns ``(html, toc_items)``; ``toc_items`` empty => no tab."""
    try:
        manifest = yaml.safe_load(
            (_REPO / "configs" / "training" / "experiments.yaml").read_text(encoding="utf-8")
        ) or {}
    except OSError:
        manifest = {}

    idx: dict[tuple[str, str], dict] = {
        (r["model"], r.get("dataset", "")): r for r in rows
    }

    def _cer(model: str, ds: str):
        r = idx.get((model, ds))
        try:
            return float(r["cer_micro_light"]) if r and r.get("cer_micro_light") else None
        except (TypeError, ValueError):
            return None

    parts = [
        _FT_CSS, "<div class='ft-view'>",
        "<h2>Fine-tuning experiments</h2>",
        "<p class='board-note'>Each training run, its recipe, and what it gained "
        "over the untuned base on datasets it never trained on. CER is "
        "%-of-characters (light-normalised), lower is better; Δ is fine-tune "
        "minus base.</p>",
    ]
    toc: list[tuple[str, str]] = []
    if not manifest:
        parts.append("<p class='ft-empty'>No experiments defined "
                     "(<code>configs/training/experiments.yaml</code>).</p></div>")
        return "".join(parts), toc

    for exp_id, e in manifest.items():
        if not isinstance(e, dict):
            continue
        sid = f"ft-{exp_id}"
        toc.append((sid, e.get("title", exp_id)))
        base = e.get("base_model", "")
        fts = e.get("ft_models") or []
        ft = fts[0] if fts else ""
        dsets = e.get("holdout_datasets") or []

        trip = []  # (dataset, n, base_cer, ft_cer)
        for ds in dsets:
            bc, fc = _cer(base, ds), _cer(ft, ds)
            if bc is not None and fc is not None:
                n = (idx.get((ft, ds)) or {}).get("n", "?")
                trip.append((ds, n, bc, fc))

        body = ""
        if trip:
            rows_html = ""
            for ds, n, bc, fc in trip:
                d, dp = fc - bc, ((fc - bc) / bc * 100 if bc else 0.0)
                cls = "ft-gain" if d < 0 else ("ft-loss" if d > 0 else "")
                rows_html += (
                    f"<tr><td>{html.escape(ds)}</td><td>{html.escape(str(n))}</td>"
                    f"<td>{bc * 100:.1f}</td><td>{fc * 100:.1f}</td>"
                    f"<td class='{cls}'>{d * 100:+.1f}</td>"
                    f"<td class='{cls}'>{dp:+.0f}%</td></tr>"
                )
            mb = sum(t[2] for t in trip) / len(trip)
            mf = sum(t[3] for t in trip) / len(trip)
            md = mf - mb
            mcls = "ft-gain" if md < 0 else ("ft-loss" if md > 0 else "")
            rows_html += (
                f"<tr class='ft-mean'><td>mean · {len(trip)} set(s)</td><td></td>"
                f"<td>{mb * 100:.1f}</td><td>{mf * 100:.1f}</td>"
                f"<td class='{mcls}'>{md * 100:+.1f}</td>"
                f"<td class='{mcls}'>{(md / mb * 100 if mb else 0):+.0f}%</td></tr>"
            )
            headline = (
                f"held-out CER <b>{mb * 100:.1f}% &rarr; {mf * 100:.1f}%</b> "
                f"<span class='{mcls}'>({md * 100:+.1f}, {(md / mb * 100 if mb else 0):+.0f}%)</span> "
                f"across {len(trip)} dataset(s)"
            )
            body = (
                "<div class='scroll'><table class='ft-tbl'><thead><tr>"
                "<th>dataset</th><th>n</th><th>base CER</th><th>ft CER</th>"
                "<th>&Delta;</th><th>&Delta;%</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>"
            )
        else:
            headline = "no held-out results scored yet"

        io = e.get("in_domain") or {}
        indom = ""
        if io.get("dataset"):
            bc, fc = _cer(base, io["dataset"]), _cer(ft, io["dataset"])
            indom = (
                f"<p class='ft-meta'>In-domain (<code>{html.escape(io['dataset'])}</code>"
                + (f" &mdash; {html.escape(io.get('note', ''))}" if io.get("note") else "")
                + f"): base {bc * 100:.1f}% &middot; fine-tune {fc * 100:.1f}%</p>"
                if bc is not None and fc is not None else ""
            )

        cfg = e.get("config", "")
        block = [
            f"<section id='{sid}' class='ft-exp'>",
            f"<h3>{html.escape(e.get('title', exp_id))}</h3>",
            f"<p class='ft-meta'><b>Base</b> <code>{html.escape(base)}</code> "
            f"&nbsp;&nbsp; <b>Fine-tune</b> <code>{html.escape(ft)}</code></p>",
            f"<p class='ft-meta'>{html.escape(e.get('training', ''))}"
            + (f" &nbsp;&middot;&nbsp; recipe <code>{html.escape(cfg)}</code>" if cfg else "")
            + "</p>",
            f"<p class='ft-head'>{headline}</p>",
            _ft_training_block(e),
            indom, body,
        ]
        if e.get("hypothesis"):
            block.append(f"<p class='ft-hyp'>Hypothesis &mdash; {html.escape(e['hypothesis'].strip())}</p>")
        notes = e.get("notes") or []
        if notes:
            block.append("<ul class='ft-note'>" + "".join(
                f"<li>{html.escape(str(x))}</li>" for x in notes) + "</ul>")
        block.append("</section>")
        parts.append("".join(block))

    parts.append("</div>")
    return "".join(parts), toc


def _render_matrix(rows: list[dict]) -> str:
    """Datasets down the side, models across the top, each cell CER-light (big)
    over WER-light (small). Best model per dataset is bold; a faint green→red CER
    wash tints every cell. Click a dataset name to re-sort the model columns
    best→worst on that dataset."""
    datasets = sorted({r["dataset"] for r in rows})
    models = _model_order(rows)
    if not datasets or not models:
        return ""
    m_ix = {m: i for i, m in enumerate(models)}
    idx = {(r["model"], r["dataset"]): r for r in rows}

    cell, full, pfx = [], [], []
    for ds in datasets:
        row, frow, prow = [], [], []
        for m in models:
            r = idx.get((m, ds))
            cl = _num(r.get("cer_micro_light"), 1.0) if r else None
            wl = _num(r.get("wer_micro_light"), 1.0) if r else None
            row.append([cl, wl] if cl is not None else None)
            frow.append(bool(r and r.get("_full")))
            prow.append(int(r["_pfx"]) if r else 0)
        cell.append(row)
        full.append(frow)
        pfx.append(prow)

    payload = {"ds": datasets, "m": models, "c": cell, "full": full, "pfx": pfx,
               "order": list(range(len(models)))}  # already best-mean-CER first
    _ = m_ix  # (kept for clarity; order is identity since `models` is pre-sorted)
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    _cl = ("<abbr class='gloss' title='character error rate after light normalization "
           "(alef forms, digits, tatweel, spacing), corpus-weighted. 0 = perfect.'>"
           "CER&middot;light</abbr>")
    _wl = ("<abbr class='gloss' title='word error rate after light normalization, "
           "corpus-weighted.'>WER&middot;light</abbr>")
    return (
        f"<section class='matrix'><h2>All results — {_cl} / {_wl}</h2>"
        f"<p class='board-note'>Every model &times; dataset in one grid (models scored on "
        f"fewer than {_MIN_N} lines are left off the board entirely). Top number is "
        f"<b>{_cl}&nbsp;%</b>, smaller number below is <b>{_wl}&nbsp;%</b> "
        "(both lower = better). The <b>best model on each dataset is bold</b>; cells carry a "
        "faint green&rarr;red wash by CER. <b>Click a dataset name</b> to sort the model "
        "columns best&rarr;worst on it. <code>&middot;</code> = not run on that dataset "
        "(or not run under the comparison mode below).</p>"
        "<div class='mx-controls'><label>comparison <select id='mx-mode'>"
        "<option value='all'>All scored (n&ge;10, uneven coverage)</option>"
        "<option value='full'>Full dataset only (model scored every line)</option>"
        "<option value='pfx10'>First 10 lines (same lines across models)</option>"
        "<option value='pfx20'>First 20 lines (same lines across models)</option>"
        "<option value='pfx40'>First 40 lines (same lines across models)</option>"
        "</select></label></div>"
        "<p class='board-note mx-sortnote' id='mx-sortnote'></p>"
        f"<div id='mx-out'></div>"
        f"<script id='mx-data' type='application/json'>{blob}</script>"
        f"<script>{_MATRIX_JS}</script></section>"
    )


_TOC_JS = r"""
(function(){
  var toc=document.getElementById('toc'); if(!toc) return;
  var links=[].slice.call(toc.querySelectorAll('a'));
  var byId={};
  function openAncestor(t){ var d=t&&t.closest('details'); if(d && !d.open) d.open=true; }
  links.forEach(function(a){
    var id=a.getAttribute('href').slice(1); byId[id]=a;
    a.addEventListener('click',function(e){
      var t=document.getElementById(id); if(!t) return;
      e.preventDefault(); openAncestor(t);
      requestAnimationFrame(function(){
        t.scrollIntoView({block:'start'});
        try{ history.replaceState(null,'','#'+id); }catch(_){}
      });
    });
  });
  try{
    var obs=new IntersectionObserver(function(ents){
      ents.forEach(function(en){
        if(!en.isIntersecting) return;
        var a=byId[en.target.id]; if(!a) return;
        links.forEach(function(x){x.classList.remove('active');});
        a.classList.add('active');
      });
    },{rootMargin:'-42% 0px -52% 0px',threshold:0});
    Object.keys(byId).forEach(function(id){
      var s=document.getElementById(id); if(s) obs.observe(s);
    });
  }catch(_){}
  // persist each panel's open/closed state
  [].forEach.call(document.querySelectorAll('details.panel-d[data-key]'),function(d){
    var k='lb_panel_'+d.getAttribute('data-key');
    try{ var v=localStorage.getItem(k); if(v==='1') d.open=true; else if(v==='0') d.open=false; }catch(_){}
    d.addEventListener('toggle',function(){
      try{ localStorage.setItem(k, d.open?'1':'0'); }catch(_){}
    });
  });
  if(location.hash){
    var t=document.getElementById(location.hash.slice(1));
    if(t){ openAncestor(t); requestAnimationFrame(function(){ t.scrollIntoView(); }); }
  }
})();
"""


_TABS_JS = r"""
(function(){
  var bar=document.querySelector('.tabbar');
  var btns=bar?[].slice.call(bar.querySelectorAll('.tab')):[];
  var views={};
  [].forEach.call(document.querySelectorAll('.view[data-view]'),function(v){
    views[v.getAttribute('data-view')]=v;
  });
  var TABS=btns.map(function(b){return b.getAttribute('data-tab');});
  var DEF=TABS[0]||'leaderboard';
  var navs={};
  TABS.forEach(function(n){ navs[n]=document.getElementById('toc-'+n); });

  function want(){
    var h=location.hash||'';
    if(h.indexOf('#f-')===0||h.indexOf('#findings')===0) return 'findings';
    if(h.indexOf('#ft-')===0||h.indexOf('#finetuning')===0) return 'finetuning';
    if(h.indexOf('#rs-')===0||h.indexOf('#research')===0) return 'research';
    try{ var s=localStorage.getItem('lb_tab');
         if(TABS.indexOf(s)>=0) return s; }catch(_){}
    return DEF;
  }
  function activate(name,push){
    if(TABS.indexOf(name)<0) name=DEF;
    btns.forEach(function(b){
      b.classList.toggle('tab--on', b.getAttribute('data-tab')===name);
    });
    Object.keys(views).forEach(function(k){ if(views[k]) views[k].hidden=(k!==name); });
    TABS.forEach(function(k){ if(navs[k]) navs[k].hidden=(k!==name); });
    try{ localStorage.setItem('lb_tab',name); }catch(_){}
    if(push){ try{ history.replaceState(null,'',name===DEF?'#':'#'+name); }catch(_){} }
  }

  var init=want();
  activate(init,false);
  if((init==='findings' && location.hash.indexOf('#f-')===0)
     || (init==='research' && location.hash.indexOf('#rs-')===0)){
    var t=document.getElementById(location.hash.slice(1));
    if(t) requestAnimationFrame(function(){ t.scrollIntoView({block:'start'}); });
  }
  btns.forEach(function(b){
    b.addEventListener('click',function(){ activate(b.getAttribute('data-tab'),true); });
  });

  // one shared cursor tooltip for any [data-tip] element (charts live here)
  var tip=null;
  function host(el){
    while(el && el.getAttribute){
      if(el.hasAttribute('data-tip')) return el;
      el=el.parentNode;
    }
    return null;
  }
  document.addEventListener('mousemove',function(e){
    var h=host(e.target);
    if(!h){ if(tip) tip.classList.remove('on'); return; }
    if(!tip){ tip=document.createElement('div'); tip.className='chart-tip';
              document.body.appendChild(tip); }
    tip.textContent=h.getAttribute('data-tip')||'';
    tip.classList.add('on');
    var pad=14, w=tip.offsetWidth, ht=tip.offsetHeight;
    var x=e.clientX+pad, y=e.clientY+pad;
    if(x+w>window.innerWidth-4) x=e.clientX-pad-w;
    if(y+ht>window.innerHeight-4) y=e.clientY-pad-ht;
    tip.style.left=Math.max(4,x)+'px';
    tip.style.top=Math.max(4,y)+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(tip && !host(e.target)) tip.classList.remove('on');
  });
})();
"""


def _toc(items: list[tuple[str, str]],
         findings_items: list[tuple[str, str]] | None = None,
         finetuning_items: list[tuple[str, str]] | None = None,
         research_items: list[tuple[str, str]] | None = None) -> str:
    """A vertical section index, fixed to the left of the page (scroll-spy in
    ``_TOC_JS`` highlights the section currently in view). One list per tab
    shares the nav; ``_TABS_JS`` shows only the one matching the active tab."""
    def _ol(pairs: list[tuple[str, str]], oid: str, hidden: bool) -> str:
        lis = "".join(
            f"<li><a href='#{html.escape(sid)}'>{html.escape(label)}</a></li>"
            for sid, label in pairs
        )
        return f"<ol id='{oid}'{' hidden' if hidden else ''}>{lis}</ol>"

    return (
        "<nav class='sidenav' id='toc' aria-label='sections'>"
        "<p class='sidenav-h'>Contents</p>"
        f"{_ol(items, 'toc-leaderboard', False)}"
        f"{_ol(finetuning_items or [], 'toc-finetuning', True)}"
        f"{_ol(findings_items or [], 'toc-findings', True)}"
        f"{_ol(research_items or [], 'toc-research', True)}"
        "</nav>"
    )


_PANEL_RE = re.compile(r"^<section class='([^']+)'>\s*<h2>(.*?)</h2>\s*(.*)$", re.S)


def _panel(section_html: str, *, pid: str, collapsible: bool,
           open: bool = False, hint: str = "") -> str:
    """Add a stable id to a rendered `<section>`; when `collapsible`, fold its
    body into a `<details>` whose `<summary>` carries the section title."""
    if not section_html:
        return ""
    m = _PANEL_RE.match(section_html)
    if not m:
        return section_html.replace("<section class='", f"<section id='{pid}' class='", 1)
    cls, h2, rest = m.groups()
    rest = rest.rstrip()
    if rest.endswith("</section>"):
        rest = rest[: -len("</section>")].rstrip()
    if not collapsible:
        return f"<section id='{pid}' class='{cls}'><h2>{h2}</h2>\n{rest}</section>"
    hint_html = f"<span class='panel-hint'>{html.escape(hint)}</span>" if hint else ""
    op = " open" if open else ""
    return (
        f"<section id='{pid}' class='{cls} panel'>"
        f"<details class='panel-d' data-key='{pid}'{op}>"
        f"<summary class='panel-sum'><span class='panel-ttl'>{h2}</span>"
        f"{hint_html}<span class='panel-chev' aria-hidden='true'></span></summary>"
        f"<div class='panel-body'>{rest}</div></details></section>"
    )


def render(rows: list[dict], title: str, target: Path | None = None,
           findings: dict | None = None) -> str:
    _RENDER_CTX["target"] = target or Path("results")
    rows = sorted(rows, key=lambda r: (r.get("dataset", ""),
                                       float(r.get("cer_micro_light", 1) or 1)))
    preview = _is_preview(rows)
    real = [r for r in rows if not (r["model"].startswith("mock") or r["dataset"] == "synthetic")]

    gen = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_runs = len({r["run"] for r in rows})

    out = [_HEAD.replace("{{TITLE}}", html.escape(title))]
    out.append("<div class='layout'>")
    _SIDENAV_AT = len(out)
    out.append("")                       # sidenav spliced in once `present` is known
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
        n_ds = len({r.get("dataset", "") for r in real if r.get("dataset")})
        out.append(
            f"<p class='status status--live'><span class='dot'></span>"
            f"{len({r['model'] for r in real})} model(s) · {n_runs} run(s) · "
            f"{n_ds} dataset(s) — Muharaf handwriting + KITAB-Bench subsets</p>"
        )
    out.append("</header>")

    ft_html, ft_toc = _render_finetuning(rows, Path(_RENDER_CTX["target"]))
    rs_html, rs_toc = _lps.render_research_view(Path(_RENDER_CTX["target"]))
    out.append(
        "<div class=\"tabbar\" role=\"tablist\">"
        "<button type=\"button\" class=\"tab tab--on\" data-tab=\"leaderboard\" "
        "role=\"tab\">Leaderboard</button>"
        + ("<button type=\"button\" class=\"tab\" data-tab=\"finetuning\" "
           "role=\"tab\">Fine-tuning</button>" if ft_toc else "")
        + "<button type=\"button\" class=\"tab\" data-tab=\"findings\" "
        "role=\"tab\">Findings &amp; analysis</button>"
        "<button type=\"button\" class=\"tab\" data-tab=\"research\" "
        "role=\"tab\">Research</button>"
        "</div>"
    )
    out.append('<div class="view" data-view="leaderboard">')

    real_rows = [
        r for r in rows
        if not (r["model"].startswith("mock") or r["dataset"] == "synthetic")
    ] or rows

    tgt = Path(_RENDER_CTX["target"])
    n_models_real = len({r["model"] for r in real_rows})
    n_ds_real = len({r["dataset"] for r in real_rows if r.get("dataset")})
    try:
        n_cost_models = len(collect_costs(tgt))
    except Exception:  # noqa: BLE001
        n_cost_models = 0

    runs = sorted({(r["run"], r.get("when", ""), r.get("limit", ""),
                    r.get("prompt_version", "")) for r in rows}, reverse=True)
    runs_html = ["<section class='runs'><h2>Runs</h2><ul>"]
    for run, when, limit_, pv in runs:
        runs_html.append(
            f"<li><span class='rid'>{html.escape(run)}</span>"
            f"<span class='rmeta'>{html.escape(when or 'time unknown')} · "
            f"limit {html.escape(str(limit_) or 'all')} · prompt {html.escape(pv or '?')}</span></li>"
        )
    runs_html.append("</ul></section>")

    # (id, label for TOC, rendered html, collapsible?, open-by-default?, summary hint)
    sections = [
        ("matrix", "Matrix", _render_matrix(real_rows), False, True, ""),
        ("by-dataset", "By dataset", _render_explorer(real_rows, tgt), True, True,
         f"{n_ds_real} datasets"),
        ("models", "Models", _render_model_reference(real_rows), True, False,
         f"{n_models_real} models"),
        ("cost", "Cost", _render_cost(rows, tgt), True, False,
         f"{n_cost_models} priced models" if n_cost_models else ""),
        ("review", "Review", _render_review(rows, tgt), True, False,
         "best & worst lines, per model"),
        ("samples", "Samples", _render_samples(rows), True, False, "with images"),
        ("runs", "Runs", "".join(runs_html), True, False, f"{n_runs} runs"),
    ]
    present = [(sid, label) for sid, label, h, *_ in sections if h]
    for sid, _label, section_html, collapsible, open_default, hint in sections:
        if not section_html:
            continue
        out.append(_panel(section_html, pid=sid, collapsible=collapsible,
                          open=open_default, hint=hint))

    # metric definitions live at the foot of the page; each term is also a
    # hover tooltip on the table headers above.
    out.append("<section id='legend' class='legend'><h2>Metric definitions</h2><dl>")
    for term, desc in _LEGEND:
        out.append(f"<div><dt>{html.escape(term)}</dt><dd>{desc}</dd></div>")
    out.append("</dl></section>")
    present.append(("legend", "Definitions"))
    out[_SIDENAV_AT] = _toc(
        present,
        [(f"f-{slug}", label) for slug, label in _lf.FINDINGS_SECTIONS],
        ft_toc,
        rs_toc,
    )

    out.append("</div>")  # close data-view="leaderboard"
    if ft_toc:
        out.append('<div class="view" data-view="finetuning" hidden>')
        out.append(ft_html)
        out.append("</div>")
    out.append('<div class="view" data-view="findings" hidden>')
    # A structurally-wrong (but JSON-valid) findings doc can make a "trust the
    # golden shape" renderer raise; degrade the tab to its empty shell rather
    # than crash the whole leaderboard build.
    try:
        _fv = _lf.render_findings_view(findings or {})
    except Exception as e:  # noqa: BLE001
        print(f"findings view render failed: {e}", file=sys.stderr)
        _fv = _lf.render_findings_view({})
    out.append(_fv)
    out.append("</div>")

    out.append('<div class="view" data-view="research" hidden>')
    out.append(rs_html)
    out.append("</div>")

    out.append(
        "<footer><p>Datasets: <b>Muharaf-public</b> (Manuscripts of Handwritten "
        "Arabic, NeurIPS 2024, CC-BY-NC-SA-4.0) and the <b>KITAB-Bench</b> "
        "text-recognition subsets (KHATT, Muharaf, HistoricalBooks, Hindawi, "
        "HistoryAr, OnlineKHATT). Line track, research use. "
        "Harness: <code>arabic-ocr-bench</code>. "
        f"Generated {gen} by <code>scripts/make_leaderboard.py</code>.</p></footer>"
    )
    out.append(f"<script>{_TABS_JS}</script>")
    out.append(f"<script>{_TOC_JS}</script>")
    out.append("</main>")
    out.append("</div>")
    return "\n".join(out)


_REVIEW_JS = r"""
(function(){
  var el=function(id){return document.getElementById(id);};
  var P={}; try{P=JSON.parse(el('rv-data').textContent);}catch(e){return;}
  var IMG=P.img||[], DATA=P.models||{};
  var mSel=el('rv-model'), dSel=el('rv-ds'), nSel=el('rv-n'),
      out=el('rv-out'), meta=el('rv-meta');
  var lb=el('rv-lb'), lbImg=el('rv-lb-img'), lbWrap=el('rv-lb-imgwrap'),
      lbMeta=el('rv-lb-meta'), lbGt=el('rv-lb-gt'), lbPd=el('rv-lb-pd'),
      lbPdLab=el('rv-lb-pdlab');
  var SHOWN=[];          // rows currently on screen, indexed by data-si
  var lastFocus=null;
  function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function pct(v){return (v==null||isNaN(v))?'—':(v*100).toFixed(1)+'%';}
  function imgOf(r){return (r.im>=0 && IMG[r.im])?IMG[r.im]:"";}

  // --- ground-truth vs prediction diff -------------------------------------
  var diffMode='word', curGT='', curPD='', curErr=null;
  function lcsDiff(a,b){
    var n=a.length, m=b.length;
    var dp=[]; for(var i=0;i<=n;i++) dp.push(new Uint16Array(m+1));
    for(var i=n-1;i>=0;i--) for(var j=m-1;j>=0;j--)
      dp[i][j]= a[i]===b[j] ? dp[i+1][j+1]+1
        : (dp[i+1][j]>=dp[i][j+1] ? dp[i+1][j] : dp[i][j+1]);
    var ops=[], i=0, j=0;
    while(i<n && j<m){
      if(a[i]===b[j]){ ops.push({t:'e',s:a[i]}); i++; j++; }
      else if(dp[i+1][j]>=dp[i][j+1]){ ops.push({t:'d',s:a[i]}); i++; }
      else { ops.push({t:'i',s:b[j]}); j++; }
    }
    while(i<n) ops.push({t:'d',s:a[i++]});
    while(j<m) ops.push({t:'i',s:b[j++]});
    for(var k=0;k<ops.length;){                 // a deletion run followed by an
      if(ops[k].t==='d'){                        // insertion run == a substitution
        var p=k; while(p<ops.length && ops[p].t==='d') p++;
        if(p<ops.length && ops[p].t==='i'){
          var q=p; while(q<ops.length && ops[q].t==='i') q++;
          for(var x=k;x<q;x++) ops[x].sub=true;
          k=q; continue;
        }
      }
      k++;
    }
    return ops;
  }
  function renderDiff(){
    var gt=el('rv-lb-gt'), pd=el('rv-lb-pd'), stats=el('rv-lb-stats');
    if(curErr){ gt.textContent=curGT; pd.innerHTML="<span class='rv-err'>"+esc(curErr)+"</span>"; stats.textContent=''; return; }
    if(diffMode==='plain'){ gt.textContent=curGT; pd.textContent=curPD||'(empty)'; stats.textContent=''; return; }
    var isChar=diffMode==='char', sep=isChar?'':' ';
    var A=isChar?Array.from(curGT):curGT.split(/\s+/).filter(Boolean);
    var B=isChar?Array.from(curPD):curPD.split(/\s+/).filter(Boolean);
    var ops=lcsDiff(A,B);
    function tok(s){ return isChar ? (s===' '?'␣':esc(s)) : esc(s); }
    var gh='', ph='', keep=0, miss=0, extra=0, chg=0;
    ops.forEach(function(o){
      var cls=o.sub?'d-sub':(o.t==='d'?'d-del':'d-ins');
      if(o.t==='e'){ gh+=tok(o.s)+sep; ph+=tok(o.s)+sep; keep++; }
      else if(o.t==='d'){ gh+="<span class='"+cls+"'>"+tok(o.s)+"</span>"+sep; if(o.sub) chg++; else miss++; }
      else { ph+="<span class='"+cls+"'>"+tok(o.s)+"</span>"+sep; if(!o.sub) extra++; }
    });
    gt.innerHTML=gh.trim(); pd.innerHTML=ph.trim()||'<em>(empty)</em>';
    stats.textContent=A.length+' '+(isChar?'chars':'words')+' · '+keep+' kept · '+chg+' changed · '+miss+' missed · '+extra+' extra';
  }

  function fillDatasets(){
    var rows=DATA[mSel.value]||[];
    var seen={},ds=[];
    rows.forEach(function(r){if(!seen[r.d]){seen[r.d]=1;ds.push(r.d);}});
    ds.sort();
    dSel.innerHTML="<option value=''>all datasets</option>"+
      ds.map(function(d){return "<option value='"+esc(d)+"'>"+esc(d)+"</option>";}).join('');
  }
  function tbl(rows,title,cls){
    var h="<h4 class='rv-h "+cls+"'>"+esc(title)+" <span>("+rows.length+")</span></h4>";
    if(!rows.length) return h+"<p class='rv-empty'>none</p>";
    var head="<tr><th>#</th><th>dataset</th><th>CER·light</th><th>CER·raw</th>"+
      "<th>image</th><th>ground truth</th><th>model reading</th></tr>";
    var body=rows.map(function(r,i){
      var si=SHOWN.push(r)-1;
      var uri=imgOf(r);
      var cer=r.e?"<span class='rv-err'>"+esc(r.e)+"</span>":pct(r.cl);
      var thumb=uri?"<img loading='lazy' alt='' src='"+uri+"'>":"<span class='rv-noimg'>—</span>";
      var pred=r.e?"<em>—</em>":(esc(r.p)||"<em>(empty)</em>");
      return "<tr class='rv-row"+(uri?" rv-has-img":"")+"' data-si='"+si+"' tabindex='0'>"+
        "<td class='rv-i'>"+(i+1)+"</td><td class='rv-ds'>"+esc(r.d)+
        "</td><td class='rv-n'>"+cer+"</td><td class='rv-n'>"+pct(r.cr)+"</td>"+
        "<td class='rv-img'>"+thumb+"</td>"+
        "<td class='rv-x' dir='auto'>"+esc(r.gt)+"</td>"+
        "<td class='rv-x' dir='auto'>"+pred+"</td></tr>";
    }).join('');
    return h+"<div class='scroll'><table class='rv-tbl'><thead>"+head+"</thead><tbody>"+body+"</tbody></table></div>";
  }
  function render(){
    var rows=(DATA[mSel.value]||[]).slice();
    if(dSel.value) rows=rows.filter(function(r){return r.d===dSel.value;});
    var N=nSel.value==='all'?1e9:parseInt(nSel.value,10);
    var ok=rows.filter(function(r){return !r.e && r.cl!=null;});
    var bad=rows.filter(function(r){return r.e;});
    var best=ok.slice().sort(function(a,b){return a.cl-b.cl;}).slice(0,N);
    var worst=bad.concat(ok.slice().sort(function(a,b){return b.cl-a.cl;})).slice(0,N);
    var mean=ok.length?(ok.reduce(function(s,r){return s+r.cl;},0)/ok.length):null;
    meta.textContent=mSel.value+" — "+rows.length+" scored line(s)"+
      (bad.length?(" · "+bad.length+" errored/refused"):"")+
      (mean!=null?(" · mean CER·light "+(mean*100).toFixed(1)+"%"):"");
    SHOWN=[];
    out.innerHTML=tbl(best,'Best','rv-best')+tbl(worst,'Worst','rv-worst');
  }
  function openLB(r){
    if(!r) return;
    lastFocus=document.activeElement;
    var uri=imgOf(r);
    if(uri){ lbImg.src=uri; lbWrap.hidden=false; }
    else { lbImg.removeAttribute('src'); lbWrap.hidden=true; }
    lbImg.classList.remove('rv-zoom');
    lbMeta.textContent=[mSel.value, r.d, r.s,
      (r.e?("error: "+r.e):("CER·light "+pct(r.cl)+"  ·  CER·raw "+pct(r.cr)))
    ].join('  ·  ');
    curGT=r.gt||''; curPD=r.p||''; curErr=r.e||null;
    el('rv-lb-diffbar').hidden=!!curErr;
    renderDiff();
    lb.hidden=false;
    document.body.style.overflow='hidden';
    el('rv-lb').querySelector('.rv-lb-x').focus();
  }
  function closeLB(){
    lb.hidden=true;
    document.body.style.overflow='';
    if(lastFocus && lastFocus.focus) lastFocus.focus();
  }
  out.addEventListener('click',function(e){
    var tr=e.target.closest('tr[data-si]'); if(!tr) return;
    openLB(SHOWN[+tr.getAttribute('data-si')]);
  });
  out.addEventListener('keydown',function(e){
    if(e.key!=='Enter' && e.key!==' ') return;
    var tr=e.target.closest('tr[data-si]'); if(!tr) return;
    e.preventDefault(); openLB(SHOWN[+tr.getAttribute('data-si')]);
  });
  lb.addEventListener('click',function(e){
    if(e.target.dataset.close || e.target.classList.contains('rv-lb-x')) closeLB();
  });
  lbImg.addEventListener('click',function(){ lbImg.classList.toggle('rv-zoom'); });
  el('rv-lb-diffbar').addEventListener('click',function(e){
    var b=e.target.closest('.rv-lb-mode'); if(!b) return;
    diffMode=b.getAttribute('data-mode');
    Array.prototype.forEach.call(this.querySelectorAll('.rv-lb-mode'),function(x){
      x.classList.toggle('on', x===b);
    });
    renderDiff();
  });
  document.addEventListener('keydown',function(e){ if(!lb.hidden && e.key==='Escape') closeLB(); });
  mSel.addEventListener('change',function(){fillDatasets();render();});
  dSel.addEventListener('change',render);
  nSel.addEventListener('change',render);
  fillDatasets(); render();
})();
"""

_LEGEND = [
    ("CER / WER",
     "edit distance to the reference over characters / words: (S+D+I) ÷ N_ref. "
     "0 = perfect; can exceed 100% when the output is much longer than the reference."),
    ("CAR",
     "character accuracy rate = max(0, 100 − CER). The same information as CER, "
     "framed as “how much was right”. Higher is better."),
    ("MER",
     "match error rate = (S+D+I) ÷ (S+D+I+H) — like WER but hits sit in the "
     "denominator too, so it is always between 0 and 100%."),
    ("WIL / WIP",
     "word information lost / preserved. WIP = (H/N_ref)·(H/N_hyp) rewards "
     "correct words and punishes invented ones; WIL = 100% − WIP. Lower WIL is better."),
    ("micro / macro",
     "micro = one corpus-wide rate (sum the S/D/I/H counts, then divide) — "
     "long lines weigh more. macro = the plain mean of the per-line rates."),
    ("diac F1", "precision/recall of harakat marks, measured on raw (un-normalized) text."),
    ("raw",
     ("encoding canonicalisation only — strip bidi / zero-width controls, fold "
      "Arabic Presentation Forms to nominal letters, then NFC. Every ORTHOGRAPHIC "
      "difference (alef form, tatweel, digit, harakat, spacing) still counts. "
      "Strictest.")),
    ("light",
     ("raw, plus strip tatweel (ـ), unify alef forms (أإآٱ→ا), fold Persian/Urdu "
      "look-alikes (ک→ك, ی→ي, ہ→ه), map Arabic-Indic digits to ASCII, collapse "
      "whitespace. Keeps harakat, taa-marbuta, alef-maqsura. The headline "
      "profile — matches CAMeL-Tools / Lucene practice.")),
    ("no-diac",
     "light, plus alef-maqsura→yaa, taa-marbuta→haa, hamza carriers folded (ؤ→و, "
     "ئ→ي) and all harakat stripped from both sides — scores the consonant skeleton."),
    ("(details)",
     "full formulae and the normalization rationale: docs/metrics.md in the repo."),
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

    # (per-model best/worst now lives in the interactive "Review by model"
    #  section — see _render_review — which covers every scored line, not just a
    #  curated handful.)
    out.append("</section>")
    return "\n".join(out)


def _shrink_uri(uri: str, max_dim: int = 440, quality: int = 42) -> str:
    """Re-encode a base64 image data URI small. The page only needs the line
    legible, not full-res; this keeps the artifact light enough to load fast.
    Caps the longest edge (some datasets are page-sized or very wide)."""
    if not uri.startswith("data:image/") or ";base64," not in uri:
        return uri
    try:
        import base64
        import io

        from PIL import Image

        raw = base64.b64decode(uri.split(";base64,", 1)[1])
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        longest = max(im.size)
        if longest > max_dim:
            s = max_dim / longest
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        out = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return out if len(out) < len(uri) else uri
    except Exception:  # noqa: BLE001 - keep the original on any decode/encode issue
        return uri


def _render_review(rows: list[dict], target: Path) -> str:
    """Interactive best/worst browser: pick a model (and optionally a dataset),
    get its most- and least-accurate lines as a table, from every scored run."""
    review = collect_review(target)
    if not review:
        return ""
    # keep the leaderboard's model ordering, then append any extras
    ordered = [m for m in dict.fromkeys(r["model"] for r in rows) if m in review]
    ordered += [m for m in sorted(review) if m not in ordered]

    # Deduplicate images: a data URI is stored once in `img` and rows reference
    # it by index. Without this the same 20 line images are repeated per model
    # (12x+) and the page balloons past what the Artifact renderer loads quickly.
    img_idx: dict[str, int] = {}
    imgs: list[str] = []

    def _im(uri: str) -> int:
        if not uri:
            return -1
        i = img_idx.get(uri)
        if i is None:
            i = img_idx[uri] = len(imgs)
            imgs.append(uri)
        return i

    # The review browser only ever shows a model's most- and least-accurate
    # lines, so embed just those: the WORST_PER_MODEL hardest (errored / refused
    # first, then highest CER) plus the BEST_PER_MODEL cleanest. Carrying every
    # scored line's ground-truth + prediction text for every model is what makes
    # the page multi-MB (8k+ rows -> ~7 MB) and too big for the Artifact viewer
    # to share. Thumbnails go on the worst IMAGES_PER_MODEL of the kept set.
    WORST_PER_MODEL = 30
    BEST_PER_MODEL = 12
    IMAGES_PER_MODEL = 6

    def _worst_first(r: dict) -> tuple:
        return (0 if r.get("error") else 1, -(r["cer_light"] if r["cer_light"] is not None else 1e9))

    def _best_first(r: dict) -> tuple:
        return (1 if r.get("error") else 0, r["cer_light"] if r["cer_light"] is not None else 1e9)

    models = {}
    for m in ordered:
        recs = review[m]
        worst = sorted(recs, key=_worst_first)[:WORST_PER_MODEL]
        best = [r for r in sorted(recs, key=_best_first) if not r.get("error")][:BEST_PER_MODEL]
        kept, seen = [], set()
        for r in worst + best:
            if id(r) not in seen:
                seen.add(id(r))
                kept.append(r)
        keep_img = {id(r) for r in worst[:IMAGES_PER_MODEL]}
        models[m] = [
            {
                "s": r["sample_id"], "d": r["dataset"],
                "cl": r["cer_light"], "cr": r["cer_raw"],
                "e": r["error"], "gt": r["gt"], "p": r["pred"],
                "im": _im(r.get("image", "")) if id(r) in keep_img else -1,
            }
            for r in kept
        ]
    payload = {"img": [_shrink_uri(u) for u in imgs], "models": models}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

    opts = "".join(f"<option value='{html.escape(m)}'>{html.escape(m)}</option>" for m in ordered)
    return f"""<section class='review'><h2>Review — best &amp; worst by model</h2>
<p class='samp-intro'>Pick a model to see its most- and least-accurate lines across
every dataset it was scored on, from the stored per-sample results. CER = character
error rate, light-normalised; lower is better. Errored / refused lines head the
worst list. <b>Click any row</b> to enlarge the line image with its ground truth
and the model's reading.</p>
<div class='rv-controls'>
  <label>model <select id='rv-model'>{opts}</select></label>
  <label>dataset <select id='rv-ds'></select></label>
  <label>rows <select id='rv-n'>
    <option>10</option><option selected>25</option><option>50</option><option value='all'>all</option>
  </select></label>
</div>
<p class='rv-meta' id='rv-meta'></p>
<div id='rv-out'></div>
<div id='rv-lb' class='rv-lb' hidden role='dialog' aria-modal='true' aria-label='sample detail'>
  <div class='rv-lb-scrim' data-close='1'></div>
  <div class='rv-lb-card'>
    <button type='button' class='rv-lb-x' aria-label='Close'>&times;</button>
    <p class='rv-lb-meta' id='rv-lb-meta'></p>
    <div class='rv-lb-imgwrap' id='rv-lb-imgwrap'>
      <img id='rv-lb-img' alt='handwritten Arabic line' />
    </div>
    <div class='rv-lb-diffbar' id='rv-lb-diffbar'>
      <span class='rv-lb-modes'>
        <button type='button' data-mode='word' class='rv-lb-mode on'>words</button
        ><button type='button' data-mode='char' class='rv-lb-mode'>chars</button
        ><button type='button' data-mode='plain' class='rv-lb-mode'>plain</button>
      </span>
      <span class='rv-lb-legend'>
        <span class='d-ins'>added / wrong</span>
        <span class='d-del'>missed</span>
        <span class='d-sub'>changed</span>
      </span>
      <span class='rv-lb-stats' id='rv-lb-stats'></span>
    </div>
    <div class='rv-lb-pair'>
      <div class='rv-lb-block rv-lb-gt'>
        <span class='rv-lb-lab'>Ground truth</span>
        <p id='rv-lb-gt' dir='auto'></p>
      </div>
      <div class='rv-lb-block rv-lb-pd'>
        <span class='rv-lb-lab' id='rv-lb-pdlab'>Model reading</span>
        <p id='rv-lb-pd' dir='auto'></p>
      </div>
    </div>
  </div>
</div>
<script id='rv-data' type='application/json'>{blob}</script>
<script>{_REVIEW_JS}</script>
</section>"""


def _usd(v: float, nd: int = 4) -> str:
    return f"${v:,.{nd}f}"


def _render_cost(rows: list[dict], target: Path) -> str:
    """Per-model cost: average / median $ for one image, plus the projected cost
    to run the whole of each dataset (mean $/line on that dataset x row count)."""
    costs = collect_costs(target)
    if not costs:
        return ""
    ds_rows = _dataset_rows()
    # datasets that actually appear on the board, in the board's order
    board_ds = [d for d in dict.fromkeys(r["dataset"] for r in rows) if d in ds_rows]
    board_ds = [d for d in board_ds if any(d in c["ds_line"] for c in costs.values())]

    def total(m: str) -> float:
        c = costs[m]
        return sum(c["ds_line"][d] * ds_rows[d] for d in board_ds if d in c["ds_line"])

    models = sorted(costs, key=total, reverse=True)

    def _g(label: str, tip: str) -> str:
        return f"<abbr class='gloss' title='{html.escape(tip)}'>{label}</abbr>"

    head = (
        "<tr><th>model</th>"
        f"<th>{_g('in $/1M', 'list price per 1M input tokens (image tokens included)')}</th>"
        f"<th>{_g('out $/1M', 'list price per 1M output tokens')}</th>"
        f"<th>{_g('calls', 'billable API calls for this model across the stored runs')}</th>"
        f"<th>{_g('avg $/image', 'mean USD per line: provider-reported tokens x the price above')}</th>"
        f"<th>{_g('median $/image', 'median USD per line over this model billable calls')}</th>"
        + "".join(f"<th>{html.escape(d)}<br><span>({ds_rows[d]:,} lines)</span></th>" for d in board_ds)
        + f"<th>{_g('full sweep', 'sum of the per-dataset projections for datasets this model has run')}</th></tr>"
    )
    trs = []
    for m in models:
        c = costs[m]
        p = c["pricing"]
        cells = [
            f"<td>{html.escape(m)}</td>",
            f"<td>{p.get('input_usd_per_1m', 0):.2f}</td>",
            f"<td>{p.get('output_usd_per_1m', 0):.2f}</td>",
            f"<td>{c['calls']}</td>",
            f"<td>{_usd(c['mean_call'], 5)}</td>",
            f"<td>{_usd(c['median_call'], 5)}</td>",
        ]
        for d in board_ds:
            if d in c["ds_line"]:
                cells.append(f"<td>{_usd(c['ds_line'][d] * ds_rows[d], 2)}</td>")
            else:
                cells.append("<td>—</td>")
        cells.append(f"<td class='cost-hi'>{_usd(total(m), 2)}</td>")
        trs.append("<tr>" + "".join(cells) + "</tr>")

    miss = [d for d in ds_rows if d not in board_ds]
    note = (
        "Average and median are over this model's billable calls in the stored runs "
        "(provider-reported tokens × the price above; image tokens are already inside "
        "the input count). A dataset's projection is that model's mean $/line <em>on "
        "that dataset</em> × its full row count; “full sweep” sums the datasets the "
        "model has actually run. "
        + (f"Not shown (no runs yet): {', '.join(sorted(miss))}. " if miss else "")
        + "Row counts: "
        + ", ".join(f"<code>{html.escape(d)}</code> {ds_rows[d]:,}" for d in board_ds)
        + ". Reasoning models (GPT-5 / GPT-5.6, and Claude with adaptive thinking) spend "
        "hidden output tokens that dominate cost."
    )
    return (
        "<section class='cost'><h2>Cost — per image and per full dataset</h2>"
        "<div class='scroll'><table class='cost-tbl'><thead>" + head
        + "</thead><tbody>" + "".join(trs) + "</tbody></table></div>"
        + f"<p class='cost-note'>{note}</p></section>"
    )


def _img_block(slot: dict) -> str:
    if not slot.get("image"):
        return "<div class='samp-img samp-img--missing'>image unavailable</div>"
    return (f"<div class='samp-img'><img loading='lazy' alt='handwritten Arabic line' "
            f"src='{_shrink_uri(slot['image'], max_dim=620)}'></div>")

_HEAD = """<meta charset="utf-8">
<title>{{TITLE}}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f6f6f3; --surface:#ffffff; --ink:#1b1d21; --muted:#5d6470;
  --line:#e4e3de; --accent:#2f4b99; --accent-soft:#eaeef7;
  --mock:#8a8f99; --best-bar:#2f4b99;
  --toc-h:0px;   /* sticky table headers stick to the very top now */
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

.legend{margin:44px 0 8px;padding-top:26px;border-top:2px solid var(--line);}
.legend h2{font-family:Spectral,Georgia,serif;font-weight:600;font-size:19px;
  margin:0 0 14px;}
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
.board-note{font-size:12px;color:var(--muted);margin:10px 2px 0;max-width:74ch;line-height:1.55;}

/* --- results matrix (datasets x models) --- */
.matrix{margin:28px 0 0;}
.mx-tbl{font-size:12px;border-collapse:separate;border-spacing:0;}
.mx-tbl th,.mx-tbl td{white-space:nowrap;text-align:right;padding:8px 12px;
  border-bottom:1px solid var(--line);}
.mx-tbl thead th{position:sticky;top:var(--toc-h);background:var(--ground);z-index:2;
  font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:500;
  text-transform:none;letter-spacing:0;color:var(--muted);max-width:130px;
  overflow:hidden;text-overflow:ellipsis;}
.mx-tbl th.mx-corner{position:sticky;left:0;z-index:3;text-align:left;}
.mx-tbl th.mx-corner{cursor:pointer;}
.mx-tbl th.mx-ds{position:sticky;left:0;z-index:1;background:var(--ground);
  text-align:left;font-family:"IBM Plex Mono",monospace;font-size:11px;
  font-weight:500;color:var(--ink);text-transform:none;letter-spacing:0;
  cursor:pointer;}
.mx-tbl th.mx-ds:hover{color:var(--accent);}
.mx-tbl th.mx-ds.mx-ds-on{color:var(--accent);box-shadow:inset 2px 0 0 var(--accent);}
.mx-sortnote{font-style:italic;}
.mx-tbl td{font-variant-numeric:tabular-nums;}
.mx-tbl .mx-cer{display:block;font-family:"IBM Plex Mono",monospace;color:var(--ink);}
.mx-tbl .mx-wer{display:block;font-size:10px;color:var(--muted);margin-top:2px;}
.mx-tbl td.mx-empty{color:var(--line);}
.mx-tbl td.mx-best .mx-cer{font-weight:700;}
.mx-tbl td.mx-best{box-shadow:inset 0 0 0 1px var(--accent);}
/* CER wash — green (low) -> amber -> red (high), .15 alpha; theme-agnostic */
.mx-tbl td.mx-t0{background:rgba(58,125,92,.16);}
.mx-tbl td.mx-t1{background:rgba(94,132,79,.16);}
.mx-tbl td.mx-t2{background:rgba(130,140,67,.16);}
.mx-tbl td.mx-t3{background:rgba(166,147,55,.16);}
.mx-tbl td.mx-t4{background:rgba(201,154,58,.16);}
.mx-tbl td.mx-t5{background:rgba(190,125,58,.16);}
.mx-tbl td.mx-t6{background:rgba(178,96,58,.16);}
.mx-tbl td.mx-t7{background:rgba(165,68,58,.18);}
.mx-tbl tbody tr:hover td{background:var(--accent-soft)!important;}
.mx-tbl tbody tr:hover th.mx-ds{background:var(--accent-soft);}

/* --- dataset explorer + model reference --- */
.explorer,.modelref{margin:40px 0 0;}
.ex-controls,.mr-controls,.mx-controls{display:flex;gap:10px;align-items:center;margin:10px 0 8px;}
.ex-controls label,.mr-controls label,.mx-controls label{display:flex;align-items:center;gap:7px;
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);}
.ex-controls select,.mr-controls select,.mx-controls select{font:inherit;font-size:12.5px;padding:5px 9px;
  border:1px solid var(--line);border-radius:7px;background:var(--surface);
  color:var(--ink);text-transform:none;letter-spacing:0;max-width:70vw;}
.ex-card,.mr-out{font-size:13.5px;line-height:1.6;color:var(--ink);max-width:70ch;
  margin:4px 2px 12px;}
.ex-tbl{font-size:13px;}
.ex-tbl thead th{cursor:pointer;position:sticky;top:var(--toc-h);background:var(--ground);}
.ex-tbl thead th:hover{color:var(--ink);}
.ex-tbl thead th[aria-sort]{color:var(--ink);}
.ex-tbl thead th[aria-sort="ascending"]::after{content:" \2191";}
.ex-tbl thead th[aria-sort="descending"]::after{content:" \2193";}
.ex-tbl td,.ex-tbl th{white-space:nowrap;}
.ex-tbl td.exn,.ex-tbl th:not(.exm){text-align:right;font-variant-numeric:tabular-nums;}
.ex-tbl td.exn{font-family:"IBM Plex Mono",monospace;}
.ex-tbl td.exn.exbest{font-weight:700;color:var(--ink);}
.ex-tbl td.exm{font-weight:500;}
.ex-tbl tbody tr:hover td{background:var(--accent-soft);}

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

/* --- interactive review (best/worst by model) --- */
.review{margin:44px 0 0;}
.rv-controls{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0 8px;}
.rv-controls label{display:flex;align-items:center;gap:7px;
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);}
.rv-controls select{font:inherit;font-size:12.5px;padding:5px 9px;border:1px solid var(--line);
  border-radius:7px;background:var(--surface);color:var(--ink);
  text-transform:none;letter-spacing:0;max-width:60vw;}
.rv-meta{font-size:12.5px;color:var(--muted);margin:6px 2px 4px;font-variant-numeric:tabular-nums;}
.rv-h{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
  letter-spacing:.05em;text-transform:uppercase;margin:20px 2px 8px;}
.rv-h span{font-weight:400;color:var(--muted);}
.rv-h.rv-best{color:#3a7d5c;}
.rv-h.rv-worst{color:#a5443a;}
.rv-empty{font-size:12px;color:var(--muted);margin:0 2px 6px;}
.rv-tbl{width:100%;font-size:13px;}
.rv-tbl thead th{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:11px;
  letter-spacing:.03em;text-transform:uppercase;color:var(--muted);background:var(--ground);
  position:sticky;top:var(--toc-h);}
.rv-tbl td{border-bottom:1px solid var(--line);vertical-align:top;white-space:normal;line-height:1.6;}
.rv-tbl tbody tr:last-child td{border-bottom:none;}
.rv-tbl tbody tr.rv-row{cursor:zoom-in;transition:background .12s;}
.rv-tbl tbody tr.rv-row:hover,.rv-tbl tbody tr.rv-row:focus-visible{background:var(--accent-soft);outline:none;}
.rv-tbl tbody tr.rv-row:focus-visible{box-shadow:inset 3px 0 0 var(--accent);}
.rv-tbl td.rv-i,.rv-tbl td.rv-n{font-family:"IBM Plex Mono",monospace;text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink);}
.rv-tbl td.rv-ds{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);white-space:nowrap;}
.rv-tbl td.rv-x{max-width:40ch;color:var(--ink);}
.rv-tbl td.rv-img{width:150px;}
.rv-tbl td.rv-img img{width:150px;height:auto;display:block;border:1px solid var(--line);
  border-radius:5px;background:#fff;}
.rv-noimg{color:var(--muted);font-family:"IBM Plex Mono",monospace;}
.rv-err{color:#a5443a;font-style:italic;}

/* lightbox */
.rv-lb{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;padding:4vh 3vw;}
.rv-lb[hidden]{display:none!important;}   /* class sets display:flex, so hide explicitly */
.rv-lb-scrim{position:absolute;inset:0;background:rgba(10,11,14,.82);backdrop-filter:blur(2px);}
.rv-lb-card{position:relative;z-index:1;width:min(1080px,96vw);max-height:92vh;overflow:auto;
  background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:14px;
  padding:22px clamp(16px,3vw,30px) 26px;box-shadow:0 24px 60px rgba(0,0,0,.4);}
.rv-lb-x{position:absolute;top:10px;right:12px;width:32px;height:32px;border:1px solid var(--line);
  border-radius:8px;background:var(--ground);color:var(--ink);font-size:20px;line-height:1;cursor:pointer;}
.rv-lb-x:hover{background:var(--accent-soft);}
.rv-lb-meta{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);
  margin:0 40px 14px 0;letter-spacing:.02em;word-break:break-word;}
.rv-lb-imgwrap{border:1px solid var(--line);border-radius:10px;overflow:auto;background:#f2f2ef;
  display:flex;justify-content:center;margin:0 0 18px;}
#rv-lb-img{max-width:100%;max-height:62vh;height:auto;object-fit:contain;display:block;cursor:zoom-in;}
#rv-lb-img.rv-zoom{max-height:none;max-width:none;cursor:zoom-out;}
.rv-lb-pair{display:grid;gap:12px;}
.rv-lb-block{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--ground);}
.rv-lb-gt{border-inline-start:3px solid #3a7d5c;}
.rv-lb-pd{border-inline-start:3px solid var(--accent);}
.rv-lb-lab{display:block;font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:5px;}
.rv-lb-block p{margin:0;font-size:16px;line-height:1.9;color:var(--ink);}
@media (min-width:720px){.rv-lb-pair{grid-template-columns:1fr 1fr;}}

/* GT-vs-prediction diff */
.rv-lb-diffbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;
  margin:0 0 12px;font-family:"IBM Plex Mono",monospace;font-size:11px;}
.rv-lb-modes{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden;}
.rv-lb-mode{font:inherit;font-size:11px;padding:4px 10px;border:0;cursor:pointer;
  background:var(--surface);color:var(--muted);border-inline-start:1px solid var(--line);}
.rv-lb-mode:first-child{border-inline-start:0;}
.rv-lb-mode.on{background:var(--accent-soft);color:var(--accent);font-weight:600;}
.rv-lb-legend{display:inline-flex;gap:8px;color:var(--muted);}
.rv-lb-legend span{padding:1px 6px;border-radius:4px;}
.rv-lb-stats{color:var(--muted);margin-inline-start:auto;font-variant-numeric:tabular-nums;}
.d-del{background:rgba(165,68,58,.16);color:#a5443a;text-decoration:line-through;
  border-radius:3px;padding:0 2px;}
.d-ins{background:rgba(58,125,92,.16);color:#3a7d5c;border-radius:3px;padding:0 2px;}
.d-sub{background:rgba(201,154,58,.20);color:#9a7420;border-radius:3px;padding:0 2px;}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .d-del{color:#e0857b;}
  :root:not([data-theme="light"]) .d-ins{color:#79c6a0;}
  :root:not([data-theme="light"]) .d-sub{color:#d9b25f;}
}
:root[data-theme="dark"] .d-del{color:#e0857b;}
:root[data-theme="dark"] .d-ins{color:#79c6a0;}
:root[data-theme="dark"] .d-sub{color:#d9b25f;}

@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .rv-h.rv-best{color:#79c6a0;}
  :root:not([data-theme="light"]) .rv-h.rv-worst{color:#e0857b;}
  :root:not([data-theme="light"]) .rv-err{color:#e0857b;}
  :root:not([data-theme="light"]) .rv-tbl td.rv-img img{background:#f2f2ef;}
  :root:not([data-theme="light"]) .rv-lb-gt{border-inline-start-color:#79c6a0;}
}
:root[data-theme="dark"] .rv-h.rv-best{color:#79c6a0;}
:root[data-theme="dark"] .rv-h.rv-worst{color:#e0857b;}
:root[data-theme="dark"] .rv-tbl td.rv-img img{background:#f2f2ef;}
:root[data-theme="dark"] .rv-lb-gt{border-inline-start-color:#79c6a0;}

/* --- cost projection --- */
.cost{margin:44px 0 0;}
.cost .scroll{margin-top:6px;}
.cost-tbl{width:100%;font-size:13px;}
.cost-tbl th{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:11px;
  letter-spacing:.03em;text-transform:uppercase;color:var(--muted);background:var(--ground);
  text-align:right;white-space:nowrap;padding:10px 12px;}
.cost-tbl th:first-child{text-align:left;}
.cost-tbl td{border-bottom:1px solid var(--line);padding:9px 12px;text-align:right;
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink);}
.cost-tbl td:first-child{text-align:left;font-family:"IBM Plex Sans",system-ui,sans-serif;font-weight:500;}
.cost-tbl tbody tr:last-child td{border-bottom:none;}
.cost-tbl td.cost-hi{background:var(--accent-soft);font-weight:600;}
.cost-note{font-size:12px;color:var(--muted);margin:10px 2px 0;max-width:72ch;line-height:1.55;}

/* --- table of contents: fixed left sidebar --- */
html{scroll-behavior:smooth;}
main section{scroll-margin-top:20px;}
.layout{max-width:1320px;margin:0 auto;display:grid;
  grid-template-columns:214px minmax(0,1fr);
  padding:0 clamp(16px,4vw,32px);}
.layout>main{max-width:1060px;margin:0;
  padding:clamp(24px,5vw,64px) 0 72px clamp(20px,3.2vw,48px);}
.sidenav{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;
  padding:clamp(24px,5vw,64px) 22px 28px 0;
  border-right:1px solid var(--line);scrollbar-width:thin;}
.sidenav-h{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin:0 0 12px 10px;font-weight:500;}
.sidenav ol[hidden]{display:none;}   /* per-tab list; .sidenav ol sets display:flex, so hide explicitly */
.sidenav ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;
  gap:1px;counter-reset:sn;}
.sidenav a{display:flex;gap:9px;align-items:baseline;padding:7px 10px;border-radius:8px;
  font-size:13px;line-height:1.3;color:var(--muted);text-decoration:none;
  border-left:2px solid transparent;transition:color .12s,background .12s;}
.sidenav li{counter-increment:sn;}
.sidenav a::before{content:counter(sn,decimal-leading-zero);
  font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--line);flex:none;}
.sidenav a:hover{color:var(--ink);background:var(--accent-soft);}
.sidenav a:hover::before{color:var(--muted);}
.sidenav a.active{color:var(--accent);background:var(--accent-soft);font-weight:600;
  border-left-color:var(--accent);}
.sidenav a.active::before{color:var(--accent);}
@media (max-width:1120px){
  .layout{grid-template-columns:1fr;}
  .layout>main{padding-left:0;}
  .sidenav{position:static;max-height:none;overflow:visible;border-right:0;
    border-bottom:1px solid var(--line);padding:14px 0 12px;margin-bottom:6px;}
  .sidenav ol{flex-direction:row;flex-wrap:wrap;gap:3px;}
  .sidenav a::before{display:none;}
}

/* --- hoverable metric glossary --- */
abbr.gloss{text-decoration:none;border-bottom:1px dotted currentColor;cursor:help;}

/* --- collapsible panels --- */
.panel{margin:30px 0 0;}
.panel>.panel-d{border:1px solid var(--line);border-radius:12px;
  background:var(--surface);overflow:hidden;}
.panel>.panel-d[open]{background:transparent;}
.panel-sum{display:flex;align-items:center;gap:12px;cursor:pointer;list-style:none;
  padding:14px 16px;font-family:"IBM Plex Mono",monospace;font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--muted);user-select:none;}
.panel-sum::-webkit-details-marker{display:none;}
.panel-sum:hover{color:var(--ink);}
.panel-d[open]>.panel-sum{color:var(--ink);border-bottom:1px solid var(--line);
  background:var(--surface);}
.panel-ttl{flex:1;}
.panel-hint{font-size:10px;letter-spacing:.03em;color:var(--muted);
  text-transform:none;font-weight:400;white-space:nowrap;}
.panel-chev{flex:none;width:7px;height:7px;border-right:1.5px solid currentColor;
  border-bottom:1.5px solid currentColor;transform:rotate(-45deg);
  transition:transform .15s;margin-inline-start:2px;}
.panel-d[open]>.panel-sum .panel-chev{transform:rotate(45deg);}
.panel-body{padding:18px 16px 8px;}
.panel-body>h2{display:none;}
.panel-body>*:first-child{margin-top:0;}

/* --- sticky first column: cost + explorer tables --- */
.ex-tbl,.cost-tbl{border-collapse:separate;border-spacing:0;}
.cost-tbl thead th{position:sticky;top:var(--toc-h);z-index:2;}
.ex-tbl thead th:first-child,.cost-tbl thead th:first-child{left:0;z-index:4;
  box-shadow:1px 0 0 var(--line);}
.ex-tbl tbody td.exm,.cost-tbl tbody td:first-child{position:sticky;left:0;z-index:1;
  background:var(--surface);box-shadow:1px 0 0 var(--line);}
.ex-tbl tbody tr:hover td.exm{background:var(--accent-soft);}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto;}}

/* --- tab bar + view split (Findings & analysis) --- */
.tabbar{display:flex;gap:2px;margin:26px 0 2px;border-bottom:1px solid var(--line);}
.tab{appearance:none;-webkit-appearance:none;background:none;border:0;
  border-bottom:2px solid transparent;margin-bottom:-1px;padding:9px 15px;
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);cursor:pointer;
  transition:color .12s,border-color .12s;}
.tab:hover{color:var(--ink);}
.tab--on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600;}
.view[hidden]{display:none;}
/* the shared cursor tooltip is appended to <body>, so it needs unscoped rules
   (FINDINGS_CSS only styles .findings-view .chart-tip). */
.chart-tip{position:fixed;z-index:60;left:0;top:0;pointer-events:none;opacity:0;
  max-width:280px;padding:6px 9px;border:1px solid var(--line);border-radius:8px;
  background:var(--surface);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:11.5px;line-height:1.45;
  box-shadow:0 4px 16px rgba(0,0,0,.18);transition:opacity .1s;}
.chart-tip.on{opacity:1;}
</style>"""

# Fold the Findings view's own stylesheet (scoped under ``.findings-view``) into
# the single ``<style>`` block so the published fragment stays self-contained.
_HEAD = _HEAD.replace("\n</style>", "\n" + _lf.FINDINGS_CSS.strip() + "\n</style>")


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
    findings = _ensure_findings(target)
    if not rows:
        Path(args.out).write_text(_render_empty(args.title), encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text("[]", encoding="utf-8")
        print(f"leaderboard -> {args.out}  (no runs yet — placeholder page)")
        return 0
    # Model outputs occasionally contain U+FFFD (an undecodable byte). It is
    # valid text but the Artifact deploy validator rejects a raw U+FFFD, so drop
    # it from the published page and the bundle — nothing meaningful is lost.
    page = render(rows, args.title, target, findings=findings).replace("�", "")
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"leaderboard -> {args.out}  ({len(rows)} rows from "
          f"{len({r['run'] for r in rows})} run(s))")
    # also dump a machine-readable bundle next to it
    bundle = Path(args.out).with_suffix(".json")
    bundle.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False).replace("�", ""),
        encoding="utf-8",
    )
    print(f"bundle      -> {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

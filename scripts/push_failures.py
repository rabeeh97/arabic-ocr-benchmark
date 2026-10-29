"""Lines where both models still fail after the push (2026-10-29): patterns + a local HTML viewer.

    PYTHONPATH=src python scripts/push_failures.py [--thr 20] [--out results/rasam/push/both_fail.html]

Readings per model and set: results/rasam/lm/NB_PX_<cand>_<set>__<set>__test.json (beam 1-best with the char LM mix); when
NB_PX_<cand>_<set>__<set>__rescored.json exists (AraGPT2, rasam_nbest_rescore.py), its per-line readings replace the 1-best.
"Both fail" = line CER >= --thr % for E60 AND for PT (raw profile). Patterns are rates among those lines against all test lines:
manuscript, line length, pixels per character, ink contrast, region, digits / punctuation in the reference, where the errors sit in the
line, edit types, character confusions, and whether the two models agree on the same wrong reading (a sign of a label / crop problem).
Writes the HTML (line image, reference, E60 / PT / ENS readings with differences marked) and results/rasam/push/both_fail_patterns.json.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import html
import io
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from arabic_ocr_bench.metrics import char_edit_counts
from arabic_ocr_bench.normalize import normalize

PACK = "/Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_tm_h96.pkl"
LM = Path("results/rasam/lm")
SETS = {"r1": "RASAM-1", "r2": "RASAM-2", "tm": "TariMa"}
CANDS = ("E60", "PT", "ENS")


def line_cer(ref: str, hyp: str) -> float:
    c = char_edit_counts(normalize(ref, "raw").strip(), normalize(hyp, "raw").strip())
    return 100 * (c["sub"] + c["del"] + c["ins"]) / max(1, c["ref_len"])


def readings(cand: str, t: str) -> tuple[dict, str]:
    j = json.loads((LM / f"NB_PX_{cand}_{t}__{t}__test.json").read_text())
    hyp = {l["id"]: l["hyp"] for l in j["lines"]}
    src = "char-LM 1-best"
    rf = LM / f"NB_PX_{cand}_{t}__{t}__rescored.json"
    if rf.exists() and "lines" in json.loads(rf.read_text()):
        hyp.update({l["id"]: l["hyp"] for l in json.loads(rf.read_text())["lines"]})
        src = "AraGPT2-rescored"
    return hyp, src


def diff_html(ref: str, hyp: str) -> str:
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(hyp[j1:j2]))
        elif op in ("replace", "insert"):
            out.append(f'<span class="bad">{html.escape(hyp[j1:j2])}</span>')
        if op in ("replace", "delete"):
            out.append(f'<span class="miss" title="missing: {html.escape(ref[i1:i2])}">{html.escape(ref[i1:i2])}</span>')
    return "".join(out)


def error_profile(ref: str, hyp: str):
    """Edit types, position (tenths of the reference) and substitution pairs, from a character alignment."""
    types, pos, subs = Counter(), Counter(), Counter()
    n = max(1, len(ref))
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        types[op] += max(i2 - i1, j2 - j1)
        pos[min(9, int(10 * i1 / n))] += max(1, i2 - i1)
        if op == "replace" and i2 - i1 == j2 - j1:
            subs.update(zip(ref[i1:i2], hyp[j1:j2]))
    return types, pos, subs


def contrast(img: np.ndarray) -> float:
    g = img.astype(np.float32)
    return float(np.percentile(g, 90) - np.percentile(g, 2))


def png_b64(img: np.ndarray, h: int = 64) -> str:
    im = Image.fromarray(img)
    im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=Path("results/rasam/push/both_fail.html"))
    a = ap.parse_args()
    rows = {r["id"]: r for r in pickle.load(open(PACK, "rb"))["rows"] if r["role"] == "test"}
    src = {}
    lines = []
    for t in SETS:
        hy = {}
        for c in CANDS:
            hy[c], src[c, t] = readings(c, t)
        for i in hy["E60"]:
            r = rows[i]
            ref = r["label"]
            d = {"id": i, "set": t, "ms": r["manuscript"], "region": r["region"], "ref": ref, **{c: hy[c][i] for c in CANDS}}
            for c in CANDS:
                d[f"cer_{c}"] = line_cer(ref, hy[c][i])
            img = r["img"]
            d["chars"] = len(ref)
            d["px_per_char"] = img.shape[1] / max(1, len(ref))
            d["contrast"] = contrast(img)
            d["digits_punct"] = any(ch.isdigit() or ch in ".,:;/-#()،؟" for ch in ref)
            d["cer_E60_vs_PT"] = line_cer(hy["E60"][i], hy["PT"][i])  # the two models' readings against each other
            d["models_agree"] = d["cer_E60_vs_PT"] <= 10.0
            d["both_fail"] = d["cer_E60"] >= a.thr and d["cer_PT"] >= a.thr
            d["one_fails"] = (d["cer_E60"] >= a.thr) != (d["cer_PT"] >= a.thr)
            d["cat"] = ("MS 1936 (degenerate crops)" if d["ms"] == "BULAC_MS_ARA_1936" else
                        "models agree, reference differs (label?)" if d["models_agree"] else
                        "catchword / short line" if (d["region"] or "").endswith("catchword") or d["chars"] < 15 else
                        "margin text" if "arg" in (d["region"] or "") else "other hard line")
            lines.append(d)

    bf = [d for d in lines if d["both_fail"]]
    pat = {"threshold_cer": a.thr, "sources": {f"{c}_{t}": s for (c, t), s in src.items()}, "n_lines": len(lines), "n_both_fail": len(bf),
           "n_only_one_fails": sum(d["one_fails"] for d in lines)}
    # share of all character errors (E60) that the both-fail lines carry
    err = lambda ds, c: sum(d[f"cer_{c}"] * d["chars"] for d in ds)  # noqa: E731
    pat["share_of_errors"] = {c: round(100 * err(bf, c) / max(1, err(lines, c)), 1) for c in CANDS}
    pat["per_set"] = {t: {"lines": sum(d["set"] == t for d in lines), "both_fail": sum(d["set"] == t for d in bf)} for t in SETS}
    ms_all, ms_bf = Counter(d["ms"] for d in lines), Counter(d["ms"] for d in bf)
    pat["manuscripts"] = sorted(([m, ms_bf[m], ms_all[m], round(100 * ms_bf[m] / ms_all[m], 1)] for m in ms_bf), key=lambda x: -x[1])[:15]
    pat["base_rate_pct"] = round(100 * len(bf) / len(lines), 1)
    cats = Counter(d["cat"] for d in bf)
    pat["categories"] = {k: {"lines": v, "share_of_E60_errors_pct": round(100 * err([d for d in bf if d["cat"] == k], "E60") / max(1, err(lines, "E60")), 1)}
                         for k, v in cats.most_common()}

    def cmp(key, f=statistics.median):
        return {"both_fail": round(f([d[key] for d in bf]), 2) if bf else None, "all": round(f([d[key] for d in lines]), 2)}
    pat["median"] = {k: cmp(k) for k in ("chars", "px_per_char", "contrast")}
    pat["rate"] = {k: {"both_fail": round(100 * sum(d[k] for d in bf) / max(1, len(bf)), 1), "all": round(100 * sum(d[k] for d in lines) / len(lines), 1)}
                   for k in ("digits_punct", "models_agree")}
    short = [d for d in lines if d["chars"] < 15]
    pat["short_lines_lt15"] = {"lines": len(short), "both_fail_pct": round(100 * sum(d["both_fail"] for d in short) / max(1, len(short)), 1)}
    lowc = sorted(lines, key=lambda d: d["contrast"])[: len(lines) // 5]
    pat["lowest_contrast_fifth"] = {"both_fail_pct": round(100 * sum(d["both_fail"] for d in lowc) / len(lowc), 1)}
    types, pos, subs = Counter(), Counter(), Counter()
    for d in bf:
        ty, po, su = error_profile(d["ref"], d["E60"])
        types += ty; pos += po; subs += su
    tot = max(1, sum(types.values()))
    pat["edit_types_E60_pct"] = {k: round(100 * v / tot, 1) for k, v in types.items()}
    pt = max(1, sum(pos.values()))
    pat["error_position_tenths_pct"] = [round(100 * pos[k] / pt, 1) for k in range(10)]
    pat["top_substitutions_E60"] = [[f"{x}->{y}", n] for (x, y), n in subs.most_common(20)]
    regs_all, regs_bf = Counter(d["region"] for d in lines), Counter(d["region"] for d in bf)
    pat["regions"] = {str(k): [regs_bf[k], regs_all[k]] for k in regs_all}
    Path("results/rasam/push").mkdir(parents=True, exist_ok=True)
    Path("results/rasam/push/both_fail_patterns.json").write_text(json.dumps(pat, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in pat.items() if k not in ("manuscripts", "top_substitutions_E60", "sources")}, ensure_ascii=False))
    print("manuscripts:", pat["manuscripts"][:8])
    print("top substitutions:", pat["top_substitutions_E60"][:12])

    # ---- HTML ----
    bf.sort(key=lambda d: -(d["cer_E60"] + d["cer_PT"]))
    cards = []
    for d in bf:
        img = rows[d["id"]]["img"]
        tags = [d["cat"], d["set"], d["ms"] or "", d["region"] or "", f"{d['chars']} chars", f"E60 vs PT {d['cer_E60_vs_PT']:.0f}%", f"{d['px_per_char']:.0f} px/char", f"contrast {d['contrast']:.0f}"]
        if d["digits_punct"]:
            tags.append("digits/punct")
        rd = "".join(f'<div class="rd"><b>{c}</b><span class="cer">{d[f"cer_{c}"]:.0f}%</span><div class="txt">{diff_html(d["ref"], d[c])}</div></div>'
                     for c in CANDS)
        cards.append(f'<div class="card" data-set="{d["set"]}" data-agree="{int(d["models_agree"])}" data-cat="{html.escape(d["cat"])}"><div class="meta">{html.escape(d["id"])} · '
                     + " · ".join(f"<span class=tag>{html.escape(x)}</span>" for x in tags)
                     + f'</div><img src="data:image/png;base64,{png_b64(img)}"><div class="rd"><b>GT</b><span class="cer"></span>'
                     f'<div class="txt ref">{html.escape(d["ref"])}</div></div>{rd}</div>')
    cat_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['lines']}</td><td>{v['share_of_E60_errors_pct']}%</td></tr>" for k, v in pat["categories"].items())
    summ_rows = "".join(f"<tr><td>{html.escape(str(m))}</td><td>{n}</td><td>{tot_}</td><td>{p}%</td></tr>" for m, n, tot_, p in pat["manuscripts"])
    from rasam_data_issues import section as data_issues  # noqa: E402
    issues = data_issues(rows, {d["id"]: {c: d[c] for c in CANDS} for d in lines}, diff_html, png_b64, CANDS)
    subs_html = " ".join(f"<span class=tag>{html.escape(s)} ×{n}</span>" for s, n in pat["top_substitutions_E60"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Both-fail lines</title><style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--bd:#e3e2dc;--bad:#c62828;--badbg:#fde7e7;--miss:#1565c0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#a0a09a;--card:#1f1f1d;--bd:#33332f;--bad:#ff7b7b;--badbg:#4a1f1f;--miss:#7fb2ff}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}}
h1{{font-size:22px}} .card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;margin:12px 0}}
.card img{{max-width:100%;display:block;margin:8px 0;background:#fff}} .meta{{color:var(--mut);font-size:12px}}
.tag{{display:inline-block;border:1px solid var(--bd);border-radius:10px;padding:0 6px;margin:1px;font-size:12px}}
.rd{{display:grid;grid-template-columns:48px 44px 1fr;gap:6px;align-items:baseline;border-top:1px dashed var(--bd);padding:3px 0}}
.txt{{direction:rtl;text-align:right;font-family:'Amiri','Noto Naskh Arabic','Geeza Pro',serif;font-size:20px}}
.ref{{font-weight:600}} .bad{{color:var(--bad);background:var(--badbg);border-radius:3px}}
.miss{{color:var(--miss);text-decoration:line-through;opacity:.8;font-size:15px}} .cer{{color:var(--mut);font-size:12px}}
.tabs button{{font:inherit;padding:6px 14px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:8px 8px 0 0;cursor:pointer}}
.tabs button.on{{background:var(--fg);color:var(--bg)}} .warn{{border-left:4px solid var(--bad);padding:6px 10px;background:var(--badbg)}}
.note{{margin:4px 0}} .lab{{color:var(--mut);font-size:12px;margin-top:6px}} .card img{{max-height:340px;object-fit:contain}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--bd);padding:2px 8px}} .ctl{{position:sticky;top:0;background:var(--bg);padding:8px 0}}
</style></head><body>
<div class="tabs"><button id="b1" class="on">Lines where both models fail</button><button id="b2">Data issues (examples)</button></div>
<div id="t1"><h1>Lines where both models fail (line CER ≥ {a.thr:.0f}% for E60 and PT)</h1>
<p>{len(bf)} of {len(lines)} official test lines ({pat['base_rate_pct']}%); they carry {pat['share_of_errors']['E60']}% of E60's and
{pat['share_of_errors']['PT']}% of PT's character errors. Readings: {', '.join(sorted(set(src.values())))}.
<b>Red</b> = wrong or extra characters in the reading; <span style="color:var(--miss);text-decoration:line-through">blue struck</span> = reference characters the reading missed.</p>
<h2>Categories</h2><table><tr><th>category</th><th>lines</th><th>share of all E60 errors</th></tr>{cat_rows}</table>
<details><summary>Patterns</summary><table><tr><th>manuscript</th><th>both fail</th><th>test lines</th><th>rate</th></tr>{summ_rows}</table>
<p>Median chars: {pat['median']['chars']}; px per char: {pat['median']['px_per_char']}; ink contrast: {pat['median']['contrast']} (both-fail vs all).
The two models agree with each other (≤10% apart) while both miss the reference: {pat['rate']['models_agree']['both_fail']}% (all lines: {pat['rate']['models_agree']['all']}%).
Error position by tenth of the line: {pat['error_position_tenths_pct']}. Edit types: {pat['edit_types_E60_pct']}.</p><p>Top substitutions (E60, ref→read): {subs_html}</p></details>
<div class="ctl">Set: <select id="s"><option value="">all</option>{''.join(f'<option value="{t}">{n}</option>' for t, n in SETS.items())}</select>
Category: <select id="ct"><option value="">all</option>{''.join(f'<option>{html.escape(k)}</option>' for k in pat["categories"])}</select>
<label><input type="checkbox" id="ag"> only lines where the two models agree (≤10% apart)</label> <span id="n"></span></div>
<div id="list">{''.join(cards)}</div></div>
<div id="t2" hidden><h1>Data issues in the RASAM / TariMa test sets</h1>{issues}</div>
<script>
const ct=document.getElementById('ct'),s=document.getElementById('s'),ag=document.getElementById('ag'),n=document.getElementById('n');
function f(){{let k=0;document.querySelectorAll('.card').forEach(c=>{{const ok=(!s.value||c.dataset.set===s.value)&&(!ag.checked||c.dataset.agree==='1')&&(!ct.value||c.dataset.cat===ct.value);c.style.display=ok?'':'none';k+=ok}});n.textContent=k+' lines'}}
s.onchange=f;ag.onchange=f;ct.onchange=f;f();
const b1=document.getElementById('b1'),b2=document.getElementById('b2'),t1=document.getElementById('t1'),t2=document.getElementById('t2');
function tab(k){{t1.hidden=k!==1;t2.hidden=k!==2;b1.classList.toggle('on',k===1);b2.classList.toggle('on',k===2);try{{localStorage.setItem('tab',k)}}catch(e){{}}}}
b1.onclick=()=>tab(1);b2.onclick=()=>tab(2);try{{if(localStorage.getItem('tab')==='2')tab(2)}}catch(e){{}}
</script></body></html>"""
    a.out.write_text(page, encoding="utf-8")
    print("wrote", a.out, f"{a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

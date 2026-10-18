"""One-page report for the PAW-inventory coverage curve (checklist 2.6): curves, table, decision, and a per-line gallery.

    .geomvenv/bin/python scripts/shape_paw_report.py --json results/characterize/shape/paw_coverage.json \
        --html results/characterize/shape/paw_coverage.html

Gallery: per test set, the lines with the largest share of PAW tokens outside the arwiki top-8,000 (worst), fully covered
lines (best) and random lines, with uncovered PAWs marked, plus the line image.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from shape_paw_coverage import PANEL, TEST_SETS, WORD_RE, _letters_view, paws  # noqa: E402

COL = {"kitab_khatt_full": "#d1495b", "muharaf_lines": "#edae49", "nakba_lines": "#2e86ab", "nakba_blind": "#00798c",
       "rasam1": "#6a4c93", "rasam2": "#8a9b0f"}


def svg_curves(res, view, ref="arwiki", W=560, H=300):
    pad_l, pad_b, pad_t, pad_r = 46, 34, 10, 10
    xs = lambda n: pad_l + (np.log10(n) - 1) / 5 * (W - pad_l - pad_r)  # 10 .. 1e6
    ys = lambda c: pad_t + (1 - (c - 0.5) / 0.5) * (H - pad_t - pad_b)  # 0.5 .. 1
    out = [f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:{W}px' role='img' aria-label='coverage curves {view}'>"]
    for c in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
        out.append(f"<line x1='{pad_l}' x2='{W - pad_r}' y1='{ys(c):.1f}' y2='{ys(c):.1f}' stroke='var(--line)' stroke-width='{1.5 if c in (0.95, 0.99) else 0.6}'/>"
                   f"<text x='{pad_l - 4}' y='{ys(c) + 4:.1f}' text-anchor='end' font-size='10' fill='var(--mut)'>{c:.0%}</text>")
    for n in (10, 100, 1000, 8000, 10**4, 10**5, 10**6):
        dash = " stroke-dasharray='4 3'" if n == 8000 else ""
        out.append(f"<line x1='{xs(n):.1f}' x2='{xs(n):.1f}' y1='{pad_t}' y2='{H - pad_b}' stroke='var(--line)'{dash}/>"
                   f"<text x='{xs(n):.1f}' y='{H - pad_b + 14}' text-anchor='middle' font-size='10' fill='var(--mut)'>{n:,}</text>")
    out.append(f"<text x='{(W + pad_l) / 2}' y='{H - 4}' text-anchor='middle' font-size='11' fill='var(--mut)'>PAW types in the inventory (top-N by {ref} frequency, log)</text>")
    for t in TEST_SETS:
        r = res["heldout"][t][view].get(ref)
        if not r:
            continue
        pts = sorted((int(k), v) for k, v in r["coverage"].items())
        ci = {int(k): v for k, v in r["coverage_ci"].items()}
        band = [(xs(n), ys(ci[n][1])) for n, _ in pts] + [(xs(n), ys(ci[n][0])) for n, _ in pts[::-1]]
        out.append(f"<polygon points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in band)}' fill='{COL[t]}' opacity='0.18'/>")
        out.append(f"<polyline fill='none' stroke='{COL[t]}' stroke-width='2' points='{' '.join(f'{xs(n):.1f},{ys(max(c, 0.5)):.1f}' for n, c in pts)}'/>")
    out.append("</svg>")
    return "".join(out)


def b64_line(img, maxw=1200):
    im = Image.fromarray(img)
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--html", type=Path, required=True)
    a = ap.parse_args()
    res = json.loads(a.json.read_text(encoding="utf-8"))
    cap = res["vocab_cap"]
    inv = set(res["arwiki_inventory_at_cap"]["dotted"])
    H = ["""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAW Coverage Curve</title><style>
:root{--bg:#fbfaf7;--fg:#1d2329;--mut:#5f6b76;--card:#fff;--line:#e3e0d8;--miss:#d1495b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181b;--fg:#e6e8ea;--mut:#9aa5ae;--card:#1d2125;--line:#30363c}}
:root[data-theme="dark"]{--bg:#15181b;--fg:#e6e8ea;--mut:#9aa5ae;--card:#1d2125;--line:#30363c}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1300px}
h1{font-size:22px}h2{font-size:18px;margin-top:26px;border-bottom:1px solid var(--line)}.mut{color:var(--mut)}
table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0}.ar{font-family:'Noto Naskh Arabic','Geeza Pro',serif;font-size:18px;direction:rtl}
.miss{color:var(--miss);text-decoration:underline}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
img{max-width:100%;background:#fff}.leg span{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px;border-radius:2px}
</style></head><body><h1>PAW-inventory coverage curve</h1>
<p class='mut'>Checklist 2.6 (map P-L3). Share of each panel test set's PAW tokens covered by the N most frequent PAW types of a reference text.
PAW = letters up to and including a right-joining letter or ء; dotless = position-aware rasm. Bands: 95 % cluster bootstrap (pages for NAKBA/RASAM, identical texts for KHATT, lines for Muharaf).</p>"""]
    H.append(f"<h2>Decision</h2><div class='card'><pre style='white-space:pre-wrap;margin:0'>{html.escape(res['decision_rule'])}</pre>"
             f"<p><b>Outcome: {res['decision']['outcome']}</b> (lower 95 % bound of dotless coverage at {cap:,} arwiki types: "
             + ", ".join(f"{t} {v:.3f}" for t, v in res["decision"]["lower_bound_coverage_at_cap"].items()) + ")</p></div>")
    H.append("<p class='leg'>" + "".join(f"<span style='background:{COL[t]}'></span>{t}" for t in TEST_SETS) + "</p><div class='grid'>")
    for v in ("dotted", "dotless"):
        H.append(f"<div class='card'><b>{v} PAWs, arwiki-ranked</b>{svg_curves(res, v)}</div>")
    H.append("</div><h2>Held-out N95 / N99</h2><table><tr><th>test set</th><th>view</th><th>reference</th><th>PAW tokens</th><th>N95 [95 % CI]</th><th>N99 [95 % CI]</th><th>coverage at 8k [CI]</th><th>max coverage</th></tr>")
    fmt = lambda x: "—" if x is None else f"{x:,}"
    for t in TEST_SETS:
        for v in ("dotted", "dotless"):
            for rn, r in res["heldout"][t][v].items():
                c = r[f"coverage_at_{cap}"]
                H.append(f"<tr><td>{t}</td><td>{v}</td><td>{rn}</td><td>{r['tokens']:,}</td><td>{fmt(r['N95'])} [{fmt(r['N95_ci'][0])}, {fmt(r['N95_ci'][1])}]</td>"
                         f"<td>{fmt(r['N99'])} [{fmt(r['N99_ci'][0])}, {fmt(r['N99_ci'][1])}]</td><td>{c[0]:.2%} [{c[1]:.2%}, {c[2]:.2%}]</td><td>{r['max_coverage']:.2%}</td></tr>")
    H.append("</table><p class='mut'>— = not reached: the whole reference inventory covers less (see max coverage). A test set's own training text is the reference only where it has one.</p>")
    H.append("<h2>Does the inventory saturate? In-sample on the Wikipedia sample at growing size</h2><table><tr><th>PAW tokens</th><th>dotted distinct</th><th>dotted N95</th><th>dotted N99</th><th>dotless distinct</th><th>dotless N95</th><th>dotless N99</th></tr>")
    for x, y in zip(res["arwiki_saturation"]["dotted"], res["arwiki_saturation"]["dotless"]):
        H.append(f"<tr><td>{x['tokens']:,}</td><td>{x['distinct']:,}</td><td>{x['N95']:,}</td><td>{x['N99']:,}</td><td>{y['distinct']:,}</td><td>{y['N95']:,}</td><td>{y['N99']:,}</td></tr>")
    H.append("</table><h2>In-sample inventories</h2><table><tr><th>corpus</th><th>PAW tokens</th><th>dotted distinct / N95 / N99</th><th>dotless distinct / N95 / N99</th></tr>")
    for k, r in res["in_sample"].items():
        H.append(f"<tr><td>{k}</td><td>{r['dotted']['tokens']:,}</td><td>{r['dotted']['distinct']:,} / {r['dotted']['N95']:,} / {r['dotted']['N99']:,}</td>"
                 f"<td>{r['dotless']['distinct']:,} / {r['dotless']['N95']:,} / {r['dotless']['N99']:,}</td></tr>")
    H.append("</table><h2>Most frequent PAWs outside the arwiki top-8,000 (dotted)</h2>")
    for t, ex in res["uncovered_examples"].items():
        H.append(f"<div class='card'><b>{t}</b> <span class='ar'>" + "، ".join(f"{html.escape(p)} ({c})" for p, c in ex) + "</span></div>")
    # gallery
    rows = [r for r in pickle.load(open(PANEL / "panel.pkl", "rb"))["rows"] if r["dataset"] in TEST_SETS]
    rows += [r for r in pickle.load(open(PANEL / "khatt_full_panel.pkl", "rb"))["rows"] if r["dataset"] == "kitab_khatt_full"]
    rng = np.random.default_rng(0)
    H.append(f"<h2>Per-line gallery</h2><p class='mut'>Per test set: the 3 lines with the largest share of dotted PAWs outside the arwiki top-{cap:,} (worst), 2 fully covered lines (best), 2 random. Uncovered PAWs underlined.</p>")
    for t in TEST_SETS:
        rr = [r for r in rows if r["dataset"] == t]
        scored = []
        for r in rr:
            words = WORD_RE.findall(_letters_view(r.get("label_raw") or r["label"]))
            ps = [p for w in words for p in paws(w)]
            if len(ps) < 4:
                continue
            miss = sum(p not in inv for p in ps) / len(ps)
            scored.append((miss, r, words))
        scored.sort(key=lambda x: -x[0])
        full = [s for s in scored if s[0] == 0]
        picks = [("worst", s) for s in scored[:3]] + [("best", full[i]) for i in rng.permutation(len(full))[:2]] + \
                [("random", scored[i]) for i in rng.permutation(len(scored))[:2]]
        H.append(f"<div class='card'><b>{t}</b>")
        for kind, (miss, r, words) in picks:
            txt = " ".join("".join(f"<span class='miss'>{html.escape(p)}</span>" if p not in inv else html.escape(p) for p in paws(w)) for w in words)
            H.append(f"<div style='margin:8px 0'><span class='mut'>{kind} · {html.escape(r['id'])} · uncovered {miss:.0%}</span><br>"
                     f"<img src='data:image/png;base64,{b64_line(r['img'])}' alt='line image'><div class='ar'>{txt}</div></div>")
        H.append("</div>")
    H.append("</body></html>")
    a.html.write_text("\n".join(H), encoding="utf-8")
    print("wrote", a.html, f"{a.html.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

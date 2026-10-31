import html, sys
sys.path.insert(0, "scripts")
import build_rasam_leaderboard as B

T = "/Users/rabeehjouni/.claude/jobs/abb5a0e7/tmp/"
SCALE = 45.0
PILL = {"ours": ("p-ours", "ours"), "rescored": ("p-rescored", "re-run by us"), "reported-same-lines": ("p-reported", "reported")}


def cell(v, best):
    if not isinstance(v, (int, float)):
        return "<td class='num'>–</td>"
    w = min(100, 100 * v / SCALE)
    val = f"<b class='best'>{v:.2f}</b>" if v == best else f"{v:.2f}"
    return f"<td class='num'>{val}<span class='bar'><i style='width:{w:.1f}%'></i></span></td>"


best = {c: min(r[c] for r in B.STRICT if isinstance(r[c], (int, float))) for c in ("r1", "r2", "tm")}
rows = []
for r in sorted(B.STRICT, key=lambda r: r["r2"] if isinstance(r["r2"], (int, float)) else 999):
    cls, lab = PILL[r["kind"]]
    src = f" <a href='{B.LINKS[r['src']]}'>source</a>" if r["src"] in B.LINKS else ""
    rows.append(f"<tr class='{'ours' if r['kind'] == 'ours' else ''}'><td>{html.escape(r['sys'])} <span class='pill {cls}'>{lab}</span></td>"
                + "".join(cell(r[c], best[c]) for c in ("r1", "r2", "tm"))
                + f"<td class='note'>{html.escape(r['note'])}{src}</td></tr>")
strict = ("<div class='tbl'><table><thead><tr><th>System</th><th class='num'>RASAM-1 (734)</th><th class='num'>RASAM-2 (372)</th>"
          "<th class='num'>TariMa (229)</th><th>Notes</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
ideas = ("<div class='tbl'><table><thead><tr><th>Idea</th><th>Effect</th><th>Decision</th></tr></thead><tbody>"
         + "".join(f"<tr><td>{html.escape(i)}</td><td class='note'>{html.escape(e)}</td><td><span class='pill p-{v}'>{v}</span></td></tr>"
                   for i, e, v in B.IDEAS)
         + "<tr><td>Train on AraMS-style unseen hands (AraMS Maghrebi books)</td><td class='note'>not yet run; proposed next step</td>"
           "<td><span class='pill p-optional'>optional</span></td></tr></tbody></table></div>")
import json
bench = json.load(open("results/rasam/clean/bench.json"))  # scripts/clean_bench.py


def ci(x, best=None):
    if not x:
        return "<td class='num'>–</td>"
    v = f"<b class='best'>{x['cer']:.2f}</b>" if best is not None and x["cer"] == best else f"{x['cer']:.2f}"
    return f"<td class='num'>{v}<span class='ci'>{x['ci95'][0]:.1f}–{x['ci95'][1]:.1f}</span></td>"


VIEWS = ("official", "clean", "clean-indep")
SETS = ("r1", "r2", "tm")
bbest = {(v, t): min(r[f"{v}/{t}"]["cer"] for r in bench["systems"].values() if f"{v}/{t}" in r) for v in VIEWS for t in SETS}
crow = []
for name, r in sorted(bench["systems"].items(), key=lambda kv: kv[1].get("clean/r2", kv[1].get("official/r2", {"cer": 999}))["cer"]):
    cls, lab = PILL[r["kind"]]
    crow.append(f"<tr class='{'ours' if r['kind'] == 'ours' else ''}'><td>{html.escape(name)} <span class='pill {cls}'>{lab}</span>"
                f"<span class='sub'>{html.escape(r.get('note', ''))}</span></td>"
                + "".join(ci(r.get(f"{v}/{t}"), bbest[v, t]) for v in VIEWS for t in SETS) + "</tr>")
nv = bench["views"]
chead = "".join(f"<th colspan='3' class='grp'>{lab}<span class='sub'>{nv[v]['r1']} / {nv[v]['r2']} / {nv[v]['tm']} lines</span></th>"
                for v, lab in zip(VIEWS, ("Official", "Clean", "Clean, people-found drops only")))
cleanbench = ("<div class='tbl'><table class='cb'><thead><tr><th rowspan='2'>System</th>" + chead + "</tr><tr>"
              + "<th class='num'>R-1</th><th class='num'>R-2</th><th class='num'>TM</th>" * len(VIEWS) + "</tr></thead><tbody>"
              + "".join(crow) + "</tbody></table></div>")
pn = bench["panel"]
prow = []
for name, r in sorted(pn.items(), key=lambda kv: kv[1].get("official/r1", {"cer": 999})["cer"]):
    if not ("ft" in name or name.startswith(("Ours", "AMIDDA", "HAFITH"))):
        continue
    prow.append(f"<tr class='{'ours' if name.startswith(('Ours', 'VLM')) else ''}'><td>{html.escape(name.replace('VLM fine-tune: ', 'VLM fine-tune (ours): '))}</td>"
                + "".join(ci(r.get(f"{v}/{t}")) for v in ("official", "drop-only") for t in ("r1", "r2")) + "</tr>")
panel = ("<div class='tbl'><table class='cb'><thead><tr><th rowspan='2'>System</th><th colspan='2' class='grp'>Official crops, lines 0–199</th>"
         "<th colspan='2' class='grp'>Bad references dropped</th></tr><tr><th class='num'>R-1</th><th class='num'>R-2</th><th class='num'>R-1</th>"
         "<th class='num'>R-2</th></tr></thead><tbody>" + "".join(prow) + "</tbody></table></div>")
pairs = []
for k, v in bench["paired"].items():
    a, rest = k.split(" − ")
    b, vt = rest.split(" | ")
    if vt.split("/")[0] != "clean":
        continue
    pairs.append(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{vt.split('/')[1].replace('r1', 'RASAM-1').replace('r2', 'RASAM-2').replace('tm', 'TariMa')}</td>"
                 f"<td class='num'>{v['diff']:+.2f}</td><td class='num'>{v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f}</td></tr>")
paired = ("<div class='tbl'><table><thead><tr><th>System A</th><th>minus system B</th><th>Set</th><th class='num'>Difference</th>"
          "<th class='num'>95 % interval</th></tr></thead><tbody>" + "".join(pairs) + "</tbody></table></div>")
page = open("scripts/record/record_template.html").read()
page = page.replace("{{CLEANBENCH}}", cleanbench).replace("{{PANEL}}", panel).replace("{{PAIRED}}", paired)
page = page.replace("{{STRICT}}", strict).replace("{{IDEAS}}", ideas)
page = page.replace("{{GALLERY}}", open("results/rasam/clean/record_gallery.html").read())  # scripts/record/data_gallery.py
page = page.replace("{{PROGRESS}}", open(T + "chart_progress.svg").read()).replace("{{LOMO}}", open(T + "chart_lomo.svg").read())
assert "{{" not in page
open("docs/research/maghrebi_htr_record.html", "w").write(page)
print(len(page), "bytes;", len(rows), "board rows;", best)

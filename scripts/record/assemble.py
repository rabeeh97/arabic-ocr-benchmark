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
page = open("scripts/record/record_template.html").read()
page = page.replace("{{STRICT}}", strict).replace("{{IDEAS}}", ideas)
page = page.replace("{{GALLERY}}", open("results/rasam/clean/record_gallery.html").read())  # scripts/record/data_gallery.py
page = page.replace("{{PROGRESS}}", open(T + "chart_progress.svg").read()).replace("{{LOMO}}", open(T + "chart_lomo.svg").read())
assert "{{" not in page
open("docs/research/maghrebi_htr_record.html", "w").write(page)
print(len(page), "bytes;", len(rows), "board rows;", best)

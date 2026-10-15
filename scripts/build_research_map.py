"""Render docs/research/research_map.json into the ledger's top "Research map" section and docs/research/RESEARCH-MAP.md.

    .geomvenv/bin/python scripts/build_research_map.py

The map groups every finding, experiment, idea and fix under the research question it serves (an item can sit under
several). Run scripts/research_map_coverage.py first: it proves every ID from every source is placed.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

W = Path(__file__).resolve().parent.parent
LEDGER = W / "docs/research/paper-ledger.html"
MAP = W / "docs/research/research_map.json"
MD = W / "docs/research/RESEARCH-MAP.md"

GRADE = {"A": ("A · strong, replicated", "ok"), "B": ("B · solid, one gap", "acc"), "C": ("C · suggestive", "part"), "D": ("D · weak", "none"), "N": ("settled negative", "risk")}
STATUS = [("next", "Proposed next round"), ("running", "Running"), ("planned", "Planned"), ("gated", "Gated on something else"),
          ("parked", "Parked"), ("untouched", "Not started (borrowed or ours)"), ("dropped", "Dropped"), ("done", "Done")]
KIND = {"finding": "finding", "experiment": "experiment", "idea": "idea", "fix": "fix", "check": "check", "design": "design", "literature": "literature"}
LABEL = {"key-findings": "key findings", "htr-dossier": "deep dive", "run-next": "test queue", "d-evidence": "evidence", "d-aug": "augmentations",
         "d-pool": "idea pool", "d-ideas": "ideas log", "d-geo": "geometry lit.", "d-geom": "geometry", "d-xfer": "transfer", "d-day2": "10-12 (1)",
         "d-day3": "10-12 (2)", "d-day4": "10-13", "d-xai": "attribution", "d-ctc": "CTC aligner", "d-r26": "R2.6", "d-labels": "labels",
         "d-cross": "cross-lingual", "d-styles": "styles", "d-gap": "gap paper", "d-exp": "experiments"}
KEY_GRADES = {"Writing size inside the line image": "A", "Hamza spelling convention": "A", "Letter-shape novelty": "B", "Slant": "B",
              "Text distance": "B", "Stroke width and pen contrast": "C", "Dots at low resolution": "C", "Where models look": "B",
              "Slant and stroke augmentation in training": "N", "LLM decoder on CRNN features (E8)": "N", "Run-to-run noise (yardstick)": "A",
              "Measured clearly, little effect": "N"}
STYLE = """<style>
.rm{display:flex;flex-direction:column;gap:14px}
.rm-glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.rm-glance .stat b{font-size:22px}
.rm-list{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:5px}
.rm-list li{max-width:none}
.rm-theme{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;scroll-margin-top:16px}
.rm-theme>summary{cursor:pointer;display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;list-style:none}
.rm-theme>summary::-webkit-details-marker{display:none}
.rm-theme>summary::before{content:"▸";color:var(--muted);font-size:12px}
.rm-theme[open]>summary::before{content:"▾"}
.rm-theme>summary .t{font-weight:600;font-size:15.5px}
.rm-theme>summary .c{font-size:12.5px;color:var(--muted)}
.rm-body{display:flex;flex-direction:column;gap:10px;margin-top:10px}
.rm-q{font-style:italic;color:var(--ink);margin:0;max-width:80ch}
.rm-sum{margin:0;max-width:80ch;color:var(--ink)}
.rm-grp h5{margin:6px 0 4px;font:500 11.5px var(--f-mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.rm-it{display:grid;grid-template-columns:minmax(0,1fr);gap:2px;padding:6px 0;border-top:1px solid var(--line)}
.rm-it:first-of-type{border-top:0}
.rm-it .h{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:baseline}
.rm-it .id{font:500 11.5px var(--f-mono);color:var(--accent)}
.rm-it .ti{font-weight:500}
.rm-it .r{font-size:13.5px;color:var(--ink)}
.rm-it .m{font-size:12px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 10px}
.rm-it .m a{color:var(--accent);text-decoration:none}
.g{font:500 11px var(--f-mono);padding:1px 7px;border-radius:99px;white-space:nowrap}
.g-ok{color:var(--ok);background:var(--ok-soft)}.g-acc{color:var(--accent);background:var(--accent-soft)}.g-part{color:var(--part);background:var(--part-soft)}
.g-none{color:var(--none);background:var(--none-soft)}.g-risk{color:var(--risk);background:var(--risk-soft)}
.k{font:500 10.5px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 5px}
.rm-ev{font-size:12.5px}.rm-ev a,.rm-list a{color:var(--accent);text-decoration:none}
.also a{font-size:11.5px;color:var(--muted);text-decoration:none;border:1px solid var(--line);border-radius:99px;padding:0 7px}
</style>"""


def esc(s):
    return html.escape(s or "")


def links(ev):
    return " ".join(f'<a href="#{e}">{esc(LABEL.get(e, e.replace("ev-", "evidence §")))}</a>' for e in ev)


def item_html(it, theme, tmap):
    g = it.get("grade")
    gchip = f'<span class="g g-{GRADE[g][1]}">{esc(GRADE[g][0])}</span>' if g in GRADE else ""
    also = [t for t in it["themes"] if t != theme]
    also_html = ('<span class="also">also in ' + " ".join(f'<a href="#rm-{t}">{t} {esc(tmap[t]["title"].split(":")[0].split(" (")[0])}</a>' for t in also) + "</span>") if also else ""
    meta = []
    if it.get("src"):
        meta.append(f"source: {esc(it['src'])}")
    if it.get("ev"):
        meta.append("evidence: " + links(it["ev"]))
    if also_html:
        meta.append(also_html)
    return (f'<div class="rm-it"><div class="h"><span class="id">{esc(it["id"])}</span><span class="k">{esc(KIND.get(it["kind"], it["kind"]))}</span>'
            f'{gchip}<span class="ti">{esc(it["title"])}</span></div>'
            + (f'<div class="r">{esc(it["result"])}</div>' if it.get("result") else "")
            + (f'<div class="m">{" · ".join(meta)}</div>' if meta else "") + "</div>")


def main() -> None:
    R = json.loads(MAP.read_text(encoding="utf-8"))
    items, themes = R["items"], R["themes"]
    tmap = {t["id"]: t for t in themes}
    st = Counter(it["status"] for it in items)
    gr = Counter(it.get("grade") for it in items if it["kind"] == "finding")
    tiles = [(gr["A"], "findings graded A"), (gr["B"], "graded B"), (gr["N"], "settled negatives"), (st["next"], "proposed next"),
             (st["planned"] + st["gated"], "planned or gated"), (st["untouched"], "not started"), (st["dropped"], "dropped"), (len(items), "items in the map")]
    glance = '<div class="rm-glance">' + "".join(f'<div class="stat"><b>{v}</b><span>{esc(l)}</span></div>' for v, l in tiles) + "</div>"
    strong = [it for it in items if it.get("grade") == "A"]
    strong_html = "<ul class=\"rm-list\">" + "".join(f'<li><b>{esc(it["title"])}</b>: {esc(it.get("result", ""))}'
                                                  + (f' <span class="rm-ev">({links(it["ev"])})</span>' if it.get("ev") else "") + "</li>" for it in strong) + "</ul>"
    nxt = [it for it in items if it["status"] == "next"]
    next_html = "<ul class=\"rm-list\">" + "".join(f'<li><span class="id" style="font:500 11.5px var(--f-mono);color:var(--accent)">{esc(it["id"])}</span> {esc(it["title"])}'
                                                f'{(" — " + esc(it["result"])) if it.get("result") else ""} <span class="also">' + " ".join(f'<a href="#rm-{t}">{t}</a>' for t in it["themes"]) + "</span></li>" for it in nxt) + "</ul>"
    index = '<nav class="dnav" aria-label="Research map themes" style="display:flex">' + "".join(f'<a href="#rm-{t["id"]}">{t["id"]} · {esc(t["title"].split(":")[0])}</a>' for t in themes) + "</nav>"
    cards = []
    for t in themes:
        its = [it for it in items if t["id"] in it["themes"]]
        c = Counter(it["status"] for it in its)
        counts = ", ".join(f"{c[s]} {lab.lower()}" for s, lab in STATUS if c[s])
        finds = [it for it in its if it["kind"] == "finding"]
        finds.sort(key=lambda it: "ABCDN".find(it.get("grade", "D")) if it.get("grade") else 9)
        body = [f'<p class="rm-q">{esc(t["question"])}</p>', f'<p class="rm-sum">{esc(t["summary"])}</p>']
        if finds:
            body.append('<div class="rm-grp"><h5>What we found</h5>' + "".join(item_html(it, t["id"], tmap) for it in finds) + "</div>")
        for s, lab in STATUS:
            grp = [it for it in its if it["status"] == s and it["kind"] != "finding"]
            if grp:
                body.append(f'<div class="rm-grp"><h5>{esc(lab)}</h5>' + "".join(item_html(it, t["id"], tmap) for it in grp) + "</div>")
        cards.append(f'<details class="rm-theme" id="rm-{t["id"]}"><summary><span class="t">{t["id"]} · {esc(t["title"])}</span><span class="c">{counts}</span></summary>'
                     f'<div class="rm-body">{"".join(body)}</div></details>')
    legend = ("<p style=\"font-size:13px;color:var(--muted);max-width:none\">Grades for findings: <span class=\"g g-ok\">A</span> strong and replicated · <span class=\"g g-acc\">B</span> solid, one gap "
              "· <span class=\"g g-part\">C</span> suggestive · <span class=\"g g-none\">D</span> weak · <span class=\"g g-risk\">settled negative</span>. Every item sits under each question it "
              "serves (“also in”). The map is built from <span class=\"mono\">docs/research/research_map.json</span>; <span class=\"mono\">scripts/research_map_coverage.py</span> checks that "
              "every ID from the experiment backlog, the test queue, the ideas log, the idea pool, the augmentation catalogue, the paper records, NEXT_STEPS, the plan's detectors and the chat is placed.</p>")
    SECTION = f"""
<!-- research map -->
<section id="research-map">
  {STYLE}
  <div class="section-head">
    <div class="eyebrow">Overview · updated {esc(R["updated"])}</div>
    <h2>Research map: what we asked, what we found, and what is next</h2>
    <p>Everything in this ledger, grouped by the question it serves: findings with a strength grade and links to their evidence, then every experiment and idea by status, including those not started yet. Open a theme to see its items.</p>
  </div>
  <div class="rm">
    {glance}
    <h3 style="font-size:16px">The strongest results so far</h3>
    {strong_html}
    <h3 style="font-size:16px">Proposed for the next round</h3>
    {next_html}
    {index}
    {"".join(cards)}
    {legend}
  </div>
</section>
"""
    h = LEDGER.read_text(encoding="utf-8")
    h = re.sub(r"\n<!-- research map -->\n<section id=\"research-map\">.*?</section>\n", "\n", h, flags=re.S)
    # grade chips in the key-findings table, same scale as the map
    h = re.sub(r'<span class="g g-[a-z]+ kfg">[^<]*</span> ', "", h)
    i = h.index('<section id="key-findings">')
    j = h.index("</section>", i)
    kf = h[i:j]
    for title, g in KEY_GRADES.items():
        kf = kf.replace(f"<tr><td><b>{title}</b></td>", f'<tr><td><span class="g g-{GRADE[g][1]} kfg">{g if g != "N" else "neg."}</span> <b>{title}</b></td>', 1)
        kf = kf.replace(f"<tr><td>{title}</td>", f'<tr><td><span class="g g-{GRADE[g][1]} kfg">{g if g != "N" else "neg."}</span> {title}</td>', 1)
    h = h[:i] + kf + h[j:]
    h = h.replace('<div class="summary" id="summary"></div>\n', '<div class="summary" id="summary"></div>\n' + SECTION, 1)
    if 'href="#research-map"' not in h:
        h = h.replace('<li><a href="#key-findings">Key findings</a></li>', '<li><a href="#research-map">Research map</a></li>\n    <li><a href="#key-findings">Key findings</a></li>', 1)
    LEDGER.write_text(h, encoding="utf-8")

    md = [f"# Research map ({R['updated']})", "", "Built from `docs/research/research_map.json` by `scripts/build_research_map.py`; the ledger's top section shows the same content. "
          "Every item sits under each question it serves. Grades: A strong and replicated, B solid with one gap, C suggestive, D weak, N settled negative.", "",
          "## At a glance", ""] + [f"- {v} {l}" for v, l in tiles] + ["", "## Strongest results", ""] + [f"- **{it['title']}**: {it.get('result', '')}" for it in strong] + \
         ["", "## Proposed next round", ""] + [f"- `{it['id']}` {it['title']}" + (f" — {it['result']}" if it.get("result") else "") + f" ({', '.join(it['themes'])})" for it in nxt]
    for t in themes:
        its = [it for it in items if t["id"] in it["themes"]]
        md += ["", f"## {t['id']} · {t['title']}", "", f"*{t['question']}*", "", t["summary"], ""]
        finds = sorted([it for it in its if it["kind"] == "finding"], key=lambda it: "ABCDN".find(it.get("grade", "D")) if it.get("grade") else 9)
        if finds:
            md += ["**What we found**", ""] + [f"- [{it.get('grade', '–')}] `{it['id']}` {it['title']}" + (f": {it['result']}" if it.get("result") else "") for it in finds] + [""]
        for s, lab in STATUS:
            grp = [it for it in its if it["status"] == s and it["kind"] != "finding"]
            if grp:
                md += [f"**{lab}**", ""] + [f"- `{it['id']}` ({it['kind']}) {it['title']}" + (f": {it['result']}" if it.get("result") else "") + (f" — source: {it['src']}" if it.get("src") else "")
                                            + (f" — also in {', '.join(x for x in it['themes'] if x != t['id'])}" if len(it["themes"]) > 1 else "") for it in grp] + [""]
    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("ledger", len(h), "| map items", len(items), "| md lines", len(md))


if __name__ == "__main__":
    main()

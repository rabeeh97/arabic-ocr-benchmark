"""Render docs/research/MASTER-CHECKLIST.md into the ledger as the "Master checklist" section (after the research map).

    .geomvenv/bin/python scripts/build_checklist_section.py

Headings become sub-headings, checkboxes become ☐ / ☑, inline code and bold are kept, and links to design documents become
their repository paths (the published page cannot open repository files).
"""
from __future__ import annotations

import html
import re
from pathlib import Path

W = Path(__file__).resolve().parent.parent
LEDGER = W / "docs/research/paper-ledger.html"
MD = W / "docs/research/MASTER-CHECKLIST.md"


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<span class="mono" title="docs/research/{m.group(2)}">{m.group(1)}</span>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r'<code class="mono">\1</code>', s)
    return s


def main() -> None:
    raw = MD.read_text(encoding="utf-8").splitlines()
    lines = []  # join hard-wrapped continuation lines onto the line they continue
    for ln in raw:
        cont = ln.strip() and not ln.lstrip().startswith(("- ", "#", "---")) and lines and lines[-1].strip() and not lines[-1].startswith("#")
        if cont:
            lines[-1] = lines[-1].rstrip() + " " + ln.strip()
        else:
            lines.append(ln)
    out, in_list, depth = [], False, 0
    total = done = 0
    for ln in lines[1:]:
        if not ln.strip() or ln.strip() == "---":
            if in_list:
                out.append("</ul>" * depth)
                in_list, depth = False, 0
            continue
        m = re.match(r"^(\s*)- (\[( |x)\] )?(.*)$", ln)
        if ln.startswith("## "):
            if in_list:
                out.append("</ul>" * depth); in_list, depth = False, 0
            out.append(f'<h4 class="ck-h">{inline(ln[3:])}</h4>')
        elif m:
            lvl = len(m.group(1)) // 2 + 1
            if not in_list:
                in_list = True
            while depth < lvl:
                out.append('<ul class="ck">'); depth += 1
            while depth > lvl:
                out.append("</ul>"); depth -= 1
            box = ""
            if m.group(2):
                total += 1
                done += m.group(3) == "x"
                box = "☑ " if m.group(3) == "x" else "☐ "
            out.append(f"<li>{box}{inline(m.group(4))}</li>")
        else:
            if in_list:
                out.append("</ul>" * depth); in_list, depth = False, 0
            out.append(f"<p>{inline(ln)}</p>")
    if in_list:
        out.append("</ul>" * depth)
    style = ("<style>.ck{margin:4px 0 8px;padding-left:18px;display:flex;flex-direction:column;gap:5px}.ck li{max-width:none;font-size:14px}"
             ".ck .ck{margin-top:4px}.ck-h{margin-top:14px;font-size:15px}#checklist code.mono{font-size:.85em;color:var(--accent)}</style>")
    section = f"""
<!-- master checklist -->
<section id="checklist">
  {style}
  <div class="section-head">
    <div class="eyebrow">Plan · the single list we work from</div>
    <h2>Master checklist: everything that remains, in order</h2>
    <p>{done} of {total} items done. Every unfinished research-map item is here (checked by <span class="mono">scripts/check_master_checklist.py</span>); explanations are in <span class="mono">docs/research/designs/</span>. The source is <span class="mono">docs/research/MASTER-CHECKLIST.md</span>.</p>
  </div>
  <div class="dpart">{"".join(out)}</div>
</section>
"""
    h = LEDGER.read_text(encoding="utf-8")
    h = re.sub(r"\n<!-- master checklist -->\n<section id=\"checklist\">.*?</section>\n", "\n", h, flags=re.S)
    i = h.index('<section id="research-map">')
    j = h.index("</section>", i) + len("</section>")
    h = h[:j] + "\n" + section + h[j:]
    if 'href="#checklist"' not in h:
        h = h.replace('<li><a href="#research-map">Research map</a></li>', '<li><a href="#research-map">Research map</a></li>\n    <li><a href="#checklist">Master checklist</a></li>', 1)
    LEDGER.write_text(h, encoding="utf-8")
    print("ledger", len(h), "| checklist items", total, "done", done)


if __name__ == "__main__":
    main()

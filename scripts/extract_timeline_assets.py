"""Collect the evidence images for the project timeline into docs/research/timeline_assets.json.

Sources: the ledger (docs/research/paper-ledger.html, images already WebP) and the augmentation preview file of the
main checkout (aug-gallery-data.js, PNG, re-encoded here). Run once; scripts/build_timeline.py reads the result.
Usage: .geomvenv/bin/python scripts/extract_timeline_assets.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from slim_ledger import to_webp  # noqa: E402
from PIL import Image  # noqa: E402

LEDGER = Path("docs/research/paper-ledger.html")
AUG_JS = Path("/Volumes/PortableSSD/repos/arabic_ocr/docs/research/aug-gallery-data.js")
OUT = Path("docs/research/timeline_assets.json")
h = LEDGER.read_text(encoding="utf8")
IMGRE = re.compile(r'<img\b[^>]*?alt="([^"]*)"[^>]*?src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"|<img\b[^>]*?src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"[^>]*?alt="([^"]*)"')


def nth(anchor: str, n: int = 1) -> int:
    i = -1
    for _ in range(n):
        i = h.find(anchor, i + 1)
        if i < 0:
            raise SystemExit(f"anchor not found: {anchor!r} (#{n})")
    return i


def text(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


def imgs_in(block: str):
    out = []
    for m in IMGRE.finditer(block):
        alt, src = (m.group(1), m.group(2)) if m.group(2) else (m.group(4), m.group(3))
        out.append((html.unescape(alt or ""), src))
    return out


def gcards(anchor: str, which: str, k: int, n: int = 1):
    i = nth(anchor, n)
    seg = h[i:i + 70000]
    j = seg.find(f"<h4>{which}</h4>")
    if j < 0:
        return []
    blk = seg[j + len(which) + 9:]
    end = blk.find("<h4>")
    blk = blk[: end if end > 0 else 40000]
    cards = []
    for c in re.split(r'<div class="gcard [a-z]+">', blk)[1:k + 1]:
        ims = imgs_in(c)
        errs = re.findall(r'<span class="k">([^<]+)</span><span class="e">(\d+) err</span>', c)
        truth = re.search(r'<span class="k">truth</span><bdi[^>]*>(.*?)</bdi>', c, re.S)
        cards.append({"images": [{"alt": a, "src": s} for a, s in ims], "errs": [[html.unescape(a), int(b)] for a, b in errs],
                      "truth": text(truth.group(1)) if truth else ""})
    return cards


def xrows(anchor: str, k: int):
    i = nth(anchor)
    seg = h[i:i + 60000]
    stop = seg.find("<h4", 10)
    seg = seg[: stop if stop > 0 else 60000]
    rows = []
    for chunk in seg.split('<div class="xrow">')[1:]:
        ims = imgs_in(chunk)
        if len(ims) >= 2:
            rows.append([{"alt": a, "src": s} for a, s in ims[:2]])
        if len(rows) >= k:
            break
    cap = text(seg[seg.find("<p>"): seg.find("</p>")]) if "<p>" in seg else ""
    return {"rows": rows, "caption": cap[:420]}


out: dict = {}
# --- galleries (ledger)
G = {
    "crop": ("KHATT, cropped to the ink band before resizing", "Helped most", 2, 1),
    "aug_khatt": ("KHATT: original CRNN against the augmented CRNN", "Helped most", 1, 1),
    "aug_khatt_hurt": ("KHATT: original CRNN against the augmented CRNN", "Hurt most", 1, 1),
    "e8_khatt": ("KHATT: CRNN against E8", "Helped most", 1, 1),
    "e8_khatt_hurt": ("KHATT: CRNN against E8", "Hurt most", 1, 1),
    "hamza": ("NAKBA: the CRNN", "Helped most", 1, 1),
    "targeted": ("KHATT half B (held out from the selection)", "Helped most", 1, 1),
    "stack": ("KHATT (ink-cropped)", "Helped most", 1, 1),
    "tta": ("KHATT (ink-cropped)", "Helped most", 1, 2),
}
out["galleries"] = {}
for key, (anc, which, k, n) in G.items():
    try:
        out["galleries"][key] = gcards(anc, which, k, n)
    except SystemExit as e:
        out["galleries"][key] = []
        print("skip", key, e)
# --- geometry transfer demos
out["transfer"] = {}
for key, anc in {"muharaf_to_khatt": ">Muharaf → KHATT</h4>", "khatt_to_muharaf": ">KHATT → Muharaf</h4>", "nakba_to_muharaf": ">NAKBA → Muharaf</h4>"}.items():
    try:
        out["transfer"][key] = xrows(anc, 2)
    except SystemExit as e:
        print("skip", key, e)
# --- detector overlays
i = nth(">Do the detectors work?</h4>")
seg = h[i:i + 120000]
figs = re.findall(r'<figure class="fig">(.*?)</figure>', seg[: seg.find("<h4") if "<h4" in seg[10:] else 120000], re.S)
out["detectors"] = []
for f in figs[:3]:
    ims = imgs_in(f)
    cap = re.search(r"<figcaption>(.*?)</figcaption>", f, re.S)
    if ims:
        out["detectors"].append({"src": ims[0][1], "caption": text(cap.group(1)) if cap else ims[0][0]})
# --- allograph bank
i = nth(">Allographs in the training bank</h4>")
seg = h[i:i + 90000]
out["allographs"] = []
for m in re.finditer(r'<div class="al"><div class="alh"><bdi class="arb">(.*?)</bdi> ([a-z]+) · (\d+) examples</div><div class="alr">(.*?)</div></div>', seg, re.S):
    tiles = re.findall(r'<img[^>]*src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"[^>]*><figcaption>(\d+) %', m.group(4))
    if len(tiles) >= 5:
        out["allographs"].append({"char": html.unescape(m.group(1)), "form": m.group(2), "n": int(m.group(3)), "tiles": [{"src": s, "pct": int(p)} for s, p in tiles[:6]]})
seen_chars, pick = set(), []
for a in sorted(out["allographs"], key=lambda a: -a["n"]):
    if a["char"] not in seen_chars and len(pick) < 4:
        seen_chars.add(a["char"]); pick.append(a)
out["allographs"] = pick
# --- letter contact sheet (feasibility)
i = nth(">18 · Can we build a library of real letter shapes?")
seg = h[i:i + 140000]
m = re.search(r'<figure class="fig"><img src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"[^>]*>', seg)
out["contact"] = {"src": m.group(1) if m else ""}
# --- hard-core lines nobody reads
i = nth(">3 · Which lines nobody reads")
seg = h[i:i + 40000]
out["hardcore"] = []
for c in re.split(r'<div class="gcard worse">', seg)[1:4]:
    ims = imgs_in(c)
    tr = re.search(r'<span class="k">truth</span><bdi[^>]*>(.*?)</bdi>', c, re.S)
    ce = re.search(r'<span class="k">median CER</span><span class="e">([0-9.]+) %', c)
    if ims:
        out["hardcore"].append({"src": ims[0][1], "truth": text(tr.group(1)) if tr else "", "median_cer": ce.group(1) if ce else ""})
# --- attribution spike figures (separate JPEG files next to the ledger; resized and embedded here)
out["attribution"] = []
for fn, alt in [("xai-sheet.jpg", "A RASAM-1 line and three overlays for one generated token: attention, relative attention, gradient times input minus paper"),
                ("xai-locany-pad.jpg", "LocateAnything-3B zero-shot probe on handwritten lines: word boxes in grey, dot and letter queries in teal and purple")]:
    im = Image.open(Path("docs/research") / fn).convert("RGB")
    if im.size[0] > 900:
        im = im.resize((900, round(im.size[1] * 900 / im.size[0])))
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=58, method=6)
    out["attribution"].append({"alt": alt, "src": "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(), "size": list(im.size)})
# --- augmentation previews (main checkout js, PNG -> WebP)
s = AUG_JS.read_text(encoding="utf8")
d = json.loads(s[s.index("=") + 1:].strip().rstrip(";"))
NAMES = {"1": "Rotation, shear and horizontal scale", "2": "Elastic grid warp", "3": "Stroke thickening (dilation), with a reject guard",
         "4": "Gamma, contrast and noise", "12": "Resolution round trip (down, then back up)", "14": "Gaps between word parts widened or closed",
         "15": "Dots removed or moved", "13": "Joins stretched"}
out["aug"] = []
for key, name in NAMES.items():
    panels = d["items"][key]["panels"]
    grp, seen = [], 0
    for p in panels:
        if p.get("orig"):
            seen += 1
            if seen > 1:
                break
        if p.get("kind") == "img" and len(grp) < 4:
            m = re.match(r"data:image/png;base64,(.*)", p["src"])
            raw = base64.b64decode(m.group(1)) if m else b""
            w = to_webp(raw, 62) if raw else None
            src = "data:image/webp;base64," + base64.b64encode(w or raw).decode() if raw else p["src"]
            grp.append({"src": src, "caption": p.get("caption", ""), "orig": bool(p.get("orig"))})
    out["aug"].append({"id": key, "name": name, "panels": grp})
OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
print("wrote", OUT, round(OUT.stat().st_size / 1024), "KB")
for k, v in out["galleries"].items():
    print(" gallery", k, [len(c["images"]) for c in v], [c["errs"][:3] for c in v][:1])
for k, v in out["transfer"].items():
    print(" transfer", k, len(v["rows"]))
print(" detectors", len(out["detectors"]), "allographs", [(a["char"], a["form"]) for a in out["allographs"]], "contact", bool(out["contact"]["src"]),
      "hardcore", len(out["hardcore"]), "attribution", [(a["alt"][:30], len(a["src"])) for a in out["attribution"]])
print(" aug", [(a["id"], len(a["panels"])) for a in out["aug"]])

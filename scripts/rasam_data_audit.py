"""Screen RASAM-1/2 and TariMa for data defects: broken crops and wrong references.

    PYTHONPATH=src .venv/bin/python scripts/rasam_data_audit.py

Every line of every split is measured from the published Hugging Face parquet (original crop size, ink share) and its reference
is scored with the OpenITI character LM (bits per character; the LM never saw RASAM text, so it measures "how plausible is this
as Arabic", not "how close is it to RASAM"). Test lines are also compared with our readings (`results/rasam/lm/NB_PX_*`).

Screens (candidates, not verdicts — each needs a look at the image):
  image   degenerate (< 8 px tall), vertical (> 1.8 x taller than wide), tall (> 1.9 x the manuscript's median line height:
          neighbouring lines inside), blank (test: almost no ink while the reference has text and the reading misses it)
  ref     garbled   (test: reference >= 15 chars and >= 1.0 bit/char less plausible than our reading, >= 25 % CER apart;
                     not MS 1936, whose Berber text is legitimately implausible for an Arabic LM)
          truncated (test: both models agree and read >= 15 % + 5 more characters than the reference)
          repeated  (a >= 10-character span of the reference recurs right after itself; our reading does not repeat it)
          shifted   (test: the reference is much closer to our reading of another line on the same page than to this line's)
Writes results/rasam/push/data_audit.json (counts, per-manuscript reference plausibility, flagged ids).
"""
from __future__ import annotations

import glob
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import rasam_prepare_crnn_data as P  # noqa: E402
from big_char_lm import BigCharLM  # noqa: E402
from arabic_ocr_bench.metrics import char_edit_counts  # noqa: E402

BIG = "/Volumes/PortableSSD/models/rasam-replication/lm/openiti_o8.npz"
SETS = {"RASAM-1": "r1", "RASAM-2": "r2", "tarima": "tm"}
OUT = Path("results/rasam/push/data_audit.json")


def cer(ref: str, hyp: str) -> float:
    k = char_edit_counts(ref, hyp)
    return 100 * (k["sub"] + k["del"] + k["ins"]) / max(1, k["ref_len"])


def bpc(lm: BigCharLM, text: str) -> float:
    v = text[::-1]  # the LM is in visual order, like the decoder
    if not v:
        return float("nan")
    return -sum(lm.logp(c, v[:k]) for k, c in enumerate(v)) / len(v) / math.log(2)


def ink_share(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    paper = g[g > 0]  # HF crops black out everything outside the line polygon
    if paper.size < 50:
        return 0.0
    return float((paper < 0.6 * np.median(paper)).mean())


def repeated_span(t: str, n: int = 10, gap: int = 12) -> str:
    """A >= n-character span that occurs again right after itself (within `gap` characters): a doubled phrase, not a formula."""
    for i in range(len(t) - n + 1):
        s = t[i:i + n]
        j = t.find(s, i + n)
        if s.strip() and 0 <= j <= i + n + gap:
            return s
    return ""


def main() -> None:
    lm = BigCharLM(Path(BIG))
    reads = {}
    for t in SETS.values():
        for c in ("E60", "PT", "ENS"):
            for L in json.load(open(f"results/rasam/lm/NB_PX_{c}_{t}__{t}__test.json"))["lines"]:
                reads.setdefault(L["id"], {})[c] = L["hyp"]
    lines = []
    for name, tag in SETS.items():
        for split in ("train", "validation", "test"):
            k = -1
            for f in sorted(glob.glob(f"{P.ROOT}{name}/snapshots/*/data/{split}-*.parquet")):
                for rec in pq.read_table(f).to_pylist():
                    k += 1
                    im = Image.open(io.BytesIO(rec["image"]["bytes"]))
                    lab = P.clean(rec["transcription"], "raw")
                    lines.append({"id": f"{name}/{split}/{k:06d}", "set": tag, "split": split, "ms": rec["manuscript_name"],
                                  "page": rec["image_name"], "region": rec["region_type"], "w": im.size[0], "h": im.size[1],
                                  "ink": round(ink_share(im), 4), "ref": lab, "bpc_ref": round(bpc(lm, lab), 3)})
            print(name, split, k + 1, flush=True)
    mh = defaultdict(list)
    for d in lines:
        if d["h"] >= 8:
            mh[d["ms"]].append(d["h"])
    mh = {m: median(v) for m, v in mh.items()}
    by_page = defaultdict(list)
    for d in lines:
        if d["id"] in reads:
            by_page[(d["ms"], d["page"])].append(d)
    for d in lines:
        f = []
        if d["h"] < 8:
            f.append("degenerate crop")
        elif d["h"] > 1.8 * d["w"]:
            f.append("vertical line")
        elif d["h"] > 1.9 * mh[d["ms"]]:
            f.append("tall crop (neighbour lines)")
        rp = repeated_span(d["ref"])
        r = reads.get(d["id"])
        if r:
            hyp = r["ENS"]
            d["read"] = r
            d["bpc_read"] = round(bpc(lm, hyp), 3)
            d["cer_ref_read"] = round(cer(d["ref"], hyp), 1)
            agree = cer(r["E60"], r["PT"]) <= 10
            if d["h"] >= 8 and d["ink"] < 0.01 and d["cer_ref_read"] >= 50:
                f.append("blank crop (no text where the reference has some)")
            if d["ms"] != "BULAC_MS_ARA_1936" and len(d["ref"]) >= 15 and d["bpc_ref"] - d["bpc_read"] >= 1.0 and d["cer_ref_read"] >= 25:
                f.append("garbled reference?")
            if agree and len(hyp) >= 1.15 * len(d["ref"]) + 5 and d["h"] >= 8:
                f.append("truncated reference?")
            if rp and rp not in hyp:
                f.append("repeated text in reference?")
            own = d["cer_ref_read"]
            others = [(cer(d["ref"], o["read"]["ENS"] if "read" in o else reads[o["id"]]["ENS"]), o["id"])
                      for o in by_page[(d["ms"], d["page"])] if o["id"] != d["id"]]
            if others and own >= 30:
                best = min(others)
                if best[0] < 0.5 * own:
                    f.append("shifted reference?")
                    d["matches_reading_of"] = best[1]
        elif rp:
            f.append("repeated text in reference?")
        d["flags"] = f
    # reference plausibility per manuscript (all splits), with the share of references above the corpus 95th percentile
    allb = sorted(d["bpc_ref"] for d in lines if d["ref"] and not math.isnan(d["bpc_ref"]))
    p95 = allb[int(0.95 * len(allb))]
    per_ms = defaultdict(list)
    for d in lines:
        if d["ref"] and not math.isnan(d["bpc_ref"]):
            per_ms[(d["set"], d["ms"])].append(d["bpc_ref"])
    ms_tab = sorted(({"set": s, "ms": m, "lines": len(v), "median_bpc": round(median(v), 2),
                      "share_above_p95": round(100 * sum(x > p95 for x in v) / len(v), 1)} for (s, m), v in per_ms.items()),
                    key=lambda x: -x["share_above_p95"])
    test_reads = [d for d in lines if "bpc_read" in d]
    per_pg = defaultdict(list)
    for d in lines:
        if d["ref"] and not math.isnan(d["bpc_ref"]):
            per_pg[(d["set"], d["ms"], d["page"])].append(d["bpc_ref"])
    pg_tab = sorted(({"set": s, "ms": m, "page": g, "lines": len(v), "median_bpc": round(median(v), 2),
                      "share_above_p95": round(100 * sum(x > p95 for x in v) / len(v), 1)}
                     for (s, m, g), v in per_pg.items() if len(v) >= 8), key=lambda x: -x["median_bpc"])
    counts = defaultdict(lambda: defaultdict(int))
    for d in lines:
        for x in d["flags"]:
            counts[f"{d['set']}/{d['split']}"][x] += 1
    out = {"note": "Screens give candidates, not verdicts. bits/char = OpenITI char-LM cross-entropy of the text (lower = more plausible Arabic). "
                   "MS 1936 contains Berber in Arabic script, so its references score high without being wrong.",
           "p95_bpc_ref": round(p95, 3),
           "median_bpc_test_reference": round(median(d["bpc_ref"] for d in test_reads), 3),
           "median_bpc_test_reading": round(median(d["bpc_read"] for d in test_reads), 3),
           "counts": {k: dict(v) for k, v in sorted(counts.items())},
           "per_manuscript": ms_tab,
           "per_page_top40": pg_tab[:40],
           "flagged": [{k: v for k, v in d.items() if k != "read"} | ({"read_ENS": d["read"]["ENS"]} if "read" in d else {})
                       for d in lines if d["flags"]]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("p95 bpc", round(p95, 2), "| test median ref", out["median_bpc_test_reference"], "read", out["median_bpc_test_reading"])
    for k, v in out["counts"].items():
        print(k, dict(v))
    for r in ms_tab[:8]:
        print(r)
    print("pages with the least plausible references:")
    for r in pg_tab[:25]:
        print(r)
    print("->", OUT, len(out["flagged"]), "flagged")


if __name__ == "__main__":
    main()

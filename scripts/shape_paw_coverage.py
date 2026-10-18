"""PAW-inventory coverage curve (checklist 2.6, map P-L3; shape-programme "related checks"). Text only, CPU.

    .geomvenv/bin/python scripts/shape_paw_coverage.py --out results/characterize/shape/paw_coverage.json

Extends the 2026-10-12 slice (geom_paw_coverage.py), whose in-sample "types for 95 / 99 %" depend on corpus size (on a 26k-token
test set N99 is little more than "distinct types minus a few hapaxes"). Here:

1. Units. Text view = canonical, harakat and tatweel stripped (samples._letters_view). Word = maximal run of Arabic letters.
   A PAW ends after every right-joining letter (ا أ إ آ ٱ د ذ ر ز و ؤ ة) and after ء. Dotless PAW = position-aware rasm of each
   letter (characterize.text.rasm_of: ن ي ئ ق take their tooth / ف form inside the PAW and ں ى ٯ at its end), so hamza seats
   fold too (أ→ا, ؤ→و, ئ→ب/ى); ء stays.
2. Held-out curve (the answer to "how big must a PAW vocabulary be"): rank PAW types by frequency in a reference corpus
   (Arabic Wikipedia sample, 25.7M words; and each dataset's own training text where it has one), take the top N, and
   measure the share of a panel test set's PAW tokens covered, for N on a log grid. N95 / N99 = smallest N reaching 95 / 99 %
   (None when even the full reference inventory falls short). 95 % intervals: cluster bootstrap, 1,000 reps; clusters are
   pages for NAKBA and RASAM, identical texts for KHATT (its test set is mostly a few fixed paragraphs written by many
   writers), lines for Muharaf.
3. Saturation: in-sample N95 / N99 and distinct types of the Wikipedia sample at growing sizes (does the inventory level off?).

Decision rule (fixed before the run; vocabulary option (3) "shape BPE up to PAW-sized units" of shape-programme step 5):
  KEEP PAW-level units     if an arwiki-ranked dotless-PAW inventory of <= 8,000 types covers >= 99 % of PAW tokens on every
                           panel test set (lower 95 % bound);
  PAW + sub-PAW fallback   if <= 8,000 types reach 95 % (lower bound) everywhere but not 99 %;
  DROP PAW-level units     otherwise.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")  # read-only import, as analyze_letters.py does
from arabic_ocr_bench.characterize.samples import _letters_view  # noqa: E402
from arabic_ocr_bench.characterize.text import rasm_of  # noqa: E402

ARWIKI = Path("/Volumes/PortableSSD/data/text/arwiki/arwiki_20231101_shard2.txt.gz")
BOX = Path("/Volumes/PortableSSD/aocr-box-backup/xai-2026-10-13/xai/aligner")
PANEL = W / "results/panel"
BREAK = set("اأإآٱدذرزوؤةء")
WORD_RE = re.compile("[ء-غف-يٱ-ۓ]+")
VOCAB_CAP = 8000
GRID = [10, 30, 100, 300, 1000, 2000, 3000, 5000, 8000, 10000, 20000, 30000, 50000, 100000, 200000, 500000, 1000000]
TEST_SETS = ["kitab_khatt_full", "muharaf_lines", "nakba_lines", "nakba_blind", "rasam1", "rasam2"]


def paws(word: str) -> list[str]:
    out, cur = [], ""
    for ch in word:
        cur += ch
        if ch in BREAK:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def dotless(paw: str) -> str:
    n = len(paw)
    out = []
    for i, ch in enumerate(paw):
        pos = "medial" if 0 < i < n - 1 else ("initial" if i == 0 and n > 1 else ("final" if n > 1 else "isolated"))
        r = rasm_of(ch, pos)
        out.append(r if r is not None else ch)
    return "".join(out)


def line_paws(text: str) -> list[str]:
    return [p for w in WORD_RE.findall(_letters_view(text)) for p in paws(w)]


def page_of(r: dict) -> str:
    m, t = r.get("meta") or {}, r["dataset"]
    if t == "nakba_lines" and "filename" in m:
        return m["filename"].rsplit("-", 1)[0]
    if t == "nakba_blind" and "filename" in m:
        return m["filename"].rsplit("_l", 1)[0]
    if t == "kitab_khatt_full" and m.get("text_norm"):
        return "text:" + m["text_norm"]  # KHATT repeats a few fixed paragraphs across writers: cluster lines by text
    if t.startswith("rasam") and "manuscript_name" in m:
        return f'{m["manuscript_name"]}/{m.get("image_name", "")}'
    return r["id"]


def load_tests():
    rows = [r for r in pickle.load(open(PANEL / "panel.pkl", "rb"))["rows"] if r["dataset"] in TEST_SETS]
    rows += [r for r in pickle.load(open(PANEL / "khatt_full_panel.pkl", "rb"))["rows"] if r["dataset"] == "kitab_khatt_full"]
    out = defaultdict(list)
    for r in rows:
        out[r["dataset"]].append((page_of(r), line_paws(r.get("label_raw") or r["label"])))
    return out


def load_train_texts():
    tr = {}
    rows = pickle.load(open(BOX / "aligner_data.pkl", "rb"))["rows"]
    tr["muharaf_lines"] = [line_paws(r["label"]) for r in rows if r["dataset"] == "muharaf_lines" and r["role"] == "train"]
    del rows
    rows = pickle.load(open(PANEL / "aligner_data_nakba.pkl", "rb"))["rows"]
    tr["nakba"] = [line_paws(r["label"]) for r in rows if r["role"] == "train"]
    del rows
    rows = pickle.load(open(PANEL / "khatt_full_train.pkl", "rb"))["rows"]
    tr["kitab_khatt_full"] = [line_paws(r["label"]) for r in rows if r["role"] == "train"]
    return tr


def rank_of(counter: Counter) -> dict:
    return {p: k for k, (p, _) in enumerate(counter.most_common(), 1)}


def coverage_curve(test_lines, rank, reps=1000, seed=0):
    """test_lines: [(page, [paw...])]. Returns per-N coverage with page-cluster bootstrap CI, and N95/N99 with CIs."""
    pages = sorted({p for p, _ in test_lines})
    pidx = {p: i for i, p in enumerate(pages)}
    big = len(rank) + 1
    grid = [g for g in GRID if g < len(rank)] + [len(rank)]
    # per page: token count and count covered at each grid N; plus all ranks for exact N95/N99
    tot = np.zeros(len(pages))
    cov = np.zeros((len(pages), len(grid)))
    page_ranks = defaultdict(list)
    for p, toks in test_lines:
        i = pidx[p]
        rk = np.array([rank.get(t, big) for t in toks], dtype=np.int64)
        tot[i] += len(rk)
        cov[i] += (rk[:, None] <= np.array(grid)[None]).sum(0)
        page_ranks[i].append(rk)
    page_ranks = [np.concatenate(page_ranks[i]) if page_ranks[i] else np.zeros(0, np.int64) for i in range(len(pages))]

    def n_for(ranks_all, share):
        rk = np.sort(ranks_all)
        k = int(np.ceil(share * len(rk))) - 1
        v = rk[k]
        return None if v >= big else int(v)

    allr = np.concatenate(page_ranks)
    point = {"coverage": dict(zip(grid, (cov.sum(0) / tot.sum()).round(5).tolist())),
             "N95": n_for(allr, 0.95), "N99": n_for(allr, 0.99), "max_coverage": float((allr < big).mean())}
    rng = np.random.default_rng(seed)
    bc, b95, b99 = [], [], []
    for _ in range(reps):
        s = rng.integers(0, len(pages), len(pages))
        bc.append(cov[s].sum(0) / tot[s].sum())
        rr = np.concatenate([page_ranks[i] for i in s])
        a, b = n_for(rr, 0.95), n_for(rr, 0.99)
        b95.append(np.inf if a is None else a)
        b99.append(np.inf if b is None else b)
    bc = np.array(bc)
    lo, hi = np.percentile(bc, [2.5, 97.5], axis=0)
    point["coverage_ci"] = {g: [round(float(l), 5), round(float(h), 5)] for g, l, h in zip(grid, lo, hi)}
    for nm, b in (("N95_ci", b95), ("N99_ci", b99)):
        q = np.percentile(np.array(b, float), [2.5, 97.5])
        point[nm] = [None if not np.isfinite(v) else int(v) for v in q]
    at = max(g for g in grid if g <= VOCAB_CAP)
    point[f"coverage_at_{VOCAB_CAP}"] = [float(cov[:, grid.index(at)].sum() / tot.sum()), *point["coverage_ci"][at]]
    point["tokens"], point["lines"], point["clusters"] = int(tot.sum()), len(test_lines), len(pages)
    point["cluster_unit"] = "cluster" if len(pages) < len(test_lines) else "line"
    return point


def in_sample(counter: Counter) -> dict:
    v = np.array(sorted(counter.values(), reverse=True), float)
    c = np.cumsum(v) / v.sum()
    return {"tokens": int(v.sum()), "distinct": len(v), "N95": int(np.searchsorted(c, 0.95) + 1), "N99": int(np.searchsorted(c, 0.99) + 1),
            "hapax_share_of_types": float((v == 1).mean())}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    tests = load_tests()
    print({k: len(v) for k, v in tests.items()}, flush=True)
    train = load_train_texts()
    print({k: len(v) for k, v in train.items()}, flush=True)
    # Wikipedia sample, streamed; saturation checkpoints by token count
    wk = {"dotted": Counter(), "dotless": Counter()}
    checkpoints = [10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7, 3 * 10**7]
    sat = {"dotted": [], "dotless": []}
    ntok, ci = 0, 0
    with gzip.open(ARWIKI, "rt", encoding="utf-8") as fh:
        for par in fh:
            ps = line_paws(par)
            wk["dotted"].update(ps)
            wk["dotless"].update(dotless(p) for p in ps)
            ntok += len(ps)
            while ci < len(checkpoints) and ntok >= checkpoints[ci]:
                for v in ("dotted", "dotless"):
                    sat[v].append(in_sample(wk[v]))
                ci += 1
    for v in ("dotted", "dotless"):
        sat[v].append(in_sample(wk[v]))
    print("arwiki PAW tokens", ntok, flush=True)
    res = {"decision_rule": __doc__.split("Decision rule")[1].strip(), "vocab_cap": VOCAB_CAP, "arwiki_saturation": sat,
           "in_sample": {}, "heldout": {}}
    refs = {"arwiki": wk}
    for name, lines in train.items():
        refs[f"train:{name}"] = {"dotted": Counter(p for l in lines for p in l), "dotless": Counter(dotless(p) for l in lines for p in l)}
    refs["train:all"] = {v: sum((refs[f"train:{n}"][v] for n in train), Counter()) for v in ("dotted", "dotless")}
    for rn, rc in refs.items():
        res["in_sample"][rn] = {v: in_sample(rc[v]) for v in ("dotted", "dotless")}
    for t, lines in tests.items():
        cnt = {"dotted": Counter(p for _, l in lines for p in l), "dotless": Counter(dotless(p) for _, l in lines for p in l)}
        res["in_sample"][f"test:{t}"] = {v: in_sample(cnt[v]) for v in ("dotted", "dotless")}
    own = {"kitab_khatt_full": "train:kitab_khatt_full", "muharaf_lines": "train:muharaf_lines", "nakba_lines": "train:nakba", "nakba_blind": "train:nakba"}
    for v in ("dotted", "dotless"):
        ranks = {rn: rank_of(rc[v]) for rn, rc in refs.items()}
        for t, lines in tests.items():
            tl = [(p, l if v == "dotted" else [dotless(x) for x in l]) for p, l in lines]
            for rn in ["arwiki", "train:all"] + ([own[t]] if t in own else []):
                res["heldout"].setdefault(t, {}).setdefault(v, {})[rn] = coverage_curve(tl, ranks[rn], a.reps)
                r = res["heldout"][t][v][rn]
                print(t, v, rn, "N95", r["N95"], r["N95_ci"], "N99", r["N99"], r["N99_ci"], f"cov@{VOCAB_CAP}", [round(x, 4) for x in r[f"coverage_at_{VOCAB_CAP}"]], flush=True)
    # decision (dotless PAWs, arwiki ranking, lower bounds)
    lows = {t: res["heldout"][t]["dotless"]["arwiki"][f"coverage_at_{VOCAB_CAP}"][1] for t in tests}
    res["decision"] = {"lower_bound_coverage_at_cap": lows,
                       "outcome": "KEEP PAW-level units" if min(lows.values()) >= 0.99 else
                       ("PAW + sub-PAW fallback" if min(lows.values()) >= 0.95 else "DROP PAW-level units")}
    print(res["decision"], flush=True)
    # examples for the page: most frequent uncovered test PAWs
    top = rank_of(wk["dotted"])
    res["uncovered_examples"] = {t: Counter(p for _, l in lines for p in l if top.get(p, 10**9) > VOCAB_CAP).most_common(25) for t, lines in tests.items()}
    res["top_arwiki_paws"] = {v: wk[v].most_common(60) for v in ("dotted", "dotless")}
    res["arwiki_inventory_at_cap"] = {v: [p for p, _ in wk[v].most_common(VOCAB_CAP)] for v in ("dotted", "dotless")}  # for the report page
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

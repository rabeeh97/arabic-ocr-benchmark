"""Checklist 2.7: calibration-sentence candidates (design: docs/research/designs/writer-adaptation.md, part C).

Set cover over letter x positional form (isolated / initial / medial / final), plus lam-alef ligatures and
hamza-carrier forms, drawn from real text: the Arabic Wikipedia sample (CC BY-SA 4.0) and the KHATT train
line labels; the lexicon (results/lexicon/LICENCES.md) supplies word frequency and name tags for a
naturalness score and the "patch words" for any form a candidate misses.

CPU only. Outputs under results/calibration/:
  candidates.json   ranked sentence sets with per-target coverage
  review.html       page for the native-speaker check (checklist 0.5)

Usage: .geomvenv/bin/python scripts/calib_sentence_cover.py [--max-lines 400000]
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration"
ARWIKI = Path("/Volumes/PortableSSD/data/text/arwiki/arwiki_20231101_shard2.txt.gz")
VOCAB = Path("/Volumes/PortableSSD/data/lexicon/build/arwiki_vocab.tsv.gz")
LEXICON = Path("/Volumes/PortableSSD/data/lexicon/build/lexicon.tsv.gz")
KHATT = ROOT / "results" / "panel" / "khatt_full_train.pkl.gz"

# ---------------------------------------------------------------- script model
DUAL = set("بتثجحخسشصضطظعغفقكلمنهيئ")          # join on both sides
RIGHT = set("اأإآدذرزوؤةى")                     # join to the previous letter only
NONJ = set("ء")                                 # never joins
LETTERS = DUAL | RIGHT | NONJ
ALEFS = set("اأإآ")
HARAKAT = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # marks + tatweel
FORMS = ("iso", "ini", "med", "fin")
FORM_AR = {"iso": "منفصل", "ini": "أول", "med": "وسط", "fin": "آخر"}

BASE28 = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
CORE_EXTRA = "ةى"
HAMZA = "أإآؤئء"


def letter_targets(ch: str) -> list[str]:
    if ch in DUAL:
        return list(FORMS)
    if ch in RIGHT:
        return ["iso", "fin"]
    return ["iso"]


TIER1 = [(c, f) for c in BASE28 + CORE_EXTRA for f in letter_targets(c)]          # required
TIER2 = [(c, f) for c in HAMZA for f in letter_targets(c)]                          # desired
TIER2 += [("ل" + a, f) for a in ALEFS for f in ("iso", "fin")]                     # lam-alef ligatures
ALL = TIER1 + TIER2
IDX = {t: i for i, t in enumerate(ALL)}
N1 = len(TIER1)
MASK1 = (1 << N1) - 1
MASK2 = ((1 << len(ALL)) - 1) ^ MASK1

# rough sensitivity weights: forms that are rare in running text count more in the cover
WORD_RE = re.compile(r"^[ء-ي]+$")


def word_targets(w: str) -> list[tuple[str, str]]:
    """Glyph targets of one undiacritised word, with lam-alef taken as a ligature."""
    out = []
    i, n = 0, len(w)
    prev_joins = False  # does the previous glyph connect to this one
    while i < n:
        c = w[i]
        if c == "ل" and i + 1 < n and w[i + 1] in ALEFS:
            out.append(("ل" + w[i + 1], "fin" if prev_joins else "iso"))
            prev_joins = False
            i += 2
            continue
        nxt = i + 1 < n
        if c in DUAL:
            f = ("med" if nxt else "fin") if prev_joins else ("ini" if nxt else "iso")
            prev_joins = True
        elif c in RIGHT:
            f = "fin" if prev_joins else "iso"
            prev_joins = False
        else:
            f = "iso"
            prev_joins = False
        out.append((c, f))
        i += 1
    return out


def mask_of(words) -> int:
    m = 0
    for w in words:
        for t in word_targets(w):
            j = IDX.get(t)
            if j is not None:
                m |= 1 << j
    return m


def popcount(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------- candidates
SENT_SPLIT = re.compile(r"(?<=[.!؟?؛])\s+|\n")
ALLOWED = re.compile(r"^[ء-يً-ْـ\s،,.؟?!؛:]+$")


def clean_words(s: str) -> list[str]:
    s = HARAKAT.sub("", s)
    return [w for w in re.split(r"[\s،,.؟?!؛:]+", s) if w]


# topics a writer should not be asked to copy out (sexual/anatomical, violence, sectarian); substring match
BLOCK = ("القضيب", "الانتصاب", "الحيض", "الثدي", "النطاف", "الجنس", "جنسي", "المهبل", "الشرج", "الخصي", "الاغتصاب",
         "اغتصاب", "جرائم", "جريمة", "داعش", "صاروخ", "قنبلة", "السرطان", "المهزوز", "الجثث", "قتل", "القتل", "الذبح", "الانتحار", "الإرهاب", "إرهاب", "الكفار", "الشيعة",
         "السنة والشيعة", "اليهود", "الخمر", "الدعارة", "المخدرات", "البراز", "البول")


def arwiki_sentences(max_lines: int):
    with gzip.open(ARWIKI, "rt", encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if k >= max_lines:
                break
            for s in SENT_SPLIT.split(line.strip()):
                s = s.strip()
                # full sentences only (no headings or captions): terminal punctuation and >= 6 words
                if 25 <= len(s) <= 220 and s[-1] in ".؟!" and len(s.split()) >= 6 and ALLOWED.match(s) \
                        and not any(b in HARAKAT.sub("", s) for b in BLOCK) and ".." not in s:
                    yield s


def khatt_lines():
    if not KHATT.exists():
        return
    d = pickle.load(gzip.open(KHATT))
    seen = set()
    for r in d["rows"]:
        s = " ".join(r["label"].split())
        if s not in seen and 20 <= len(s) and ALLOWED.match(s):
            seen.add(s)
            yield s


def load_vocab():
    rank, freq = {}, {}
    with gzip.open(VOCAB, "rt", encoding="utf-8") as fh:
        for r, line in enumerate(fh):
            w, c = line.rstrip("\n").split("\t")[:2]
            rank[w] = r
            freq[w] = int(c)
    return rank, freq


PROCLITICS = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")
SPELL = {"aspell", "ayaspell"}
NAME_SRC = {"wikidata", "geonames", "anetac"}


def stems_of(w: str) -> list[str]:
    out = [w]
    for p in PROCLITICS:
        if w.startswith(p) and len(w) - len(p) >= 2:
            out.append(w[len(p):])
    return out


def load_lexicon_classes(words: set[str]) -> dict[str, tuple[set[str], set[str]]]:
    """word -> (sources, classes) for the words and their proclitic-stripped stems."""
    want = {v for w in words for v in stems_of(w)}
    out = {}
    with gzip.open(LEXICON, "rt", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[0] in want:
                out[p[0]] = (set(p[2].split(",")), set(p[3].split(",")))
    return out


def is_name(w: str, lexcls) -> bool:
    """Name-like: listed by a name source (Wikidata, GeoNames, ANETAC) while neither the word nor its
    proclitic-stripped stem is in a curated spelling dictionary (aspell, Ayaspell). Heuristic."""
    src = lexcls.get(w, (set(), set()))[0]
    if not src & NAME_SRC:
        return False
    return not any(lexcls.get(v, (set(), set()))[0] & SPELL for v in stems_of(w))


def naturalness(words, rank, lexcls) -> float:
    """0..1: share of words that are common in Wikipedia and not name-like (single letters count 0)."""
    good = 0.0
    for w in words:
        if len(w) == 1 and w != "و":
            continue
        r = rank.get(w, 10**7)
        s = 1.0 if r < 30000 else 0.6 if r < 150000 else 0.2
        if is_name(w, lexcls):
            s *= 0.2
        good += s
    return good / max(1, len(words))


# ---------------------------------------------------------------- search
def iter_bits(x):
    while x:
        lb = x & -x
        yield lb.bit_length() - 1
        x ^= lb


def set_value(ch, alpha, beta):
    m, cost = 0, 0.0
    for i in ch:
        m |= cand_mask[i]
        cost += cand_len[i] * (1.6 - cand_nat[i])
    return alpha * popcount(m & MASK1) + beta * popcount(m & MASK2) - cost


def solve_cover(pool, k_max, alpha, beta, rng, restarts=30, banned=()):
    """Max-coverage with a length cost, at most k_max sentences.

    value(S) = alpha * |tier-1 forms covered| + beta * |tier-2 forms covered| - sum_i chars_i * (1.6 - naturalness_i).
    Randomised greedy (gain per cost, top-5 choice) followed by best-improvement 1-swap / drop local search.
    """
    pool = [i for i in pool if i not in banned]
    cost = {i: cand_len[i] * (1.6 - cand_nat[i]) for i in pool}
    best, best_v = None, -1e18
    for _ in range(restarts):
        ch, m = [], 0
        for _ in range(k_max):
            sc = []
            for i in pool:
                g = cand_mask[i] & ~m
                gv = alpha * popcount(g & MASK1) + beta * popcount(g & MASK2) - cost[i]
                if gv > 0:
                    sc.append((gv / cost[i], i))
            if not sc:
                break
            sc.sort(reverse=True)
            i = rng.choice(sc[:5])[1]
            ch.append(i)
            m |= cand_mask[i]
        v = set_value(ch, alpha, beta)
        improved = True
        while improved:
            improved = False
            for pos in range(len(ch)):
                rest = ch[:pos] + ch[pos + 1:]
                rm = 0
                for k in rest:
                    rm |= cand_mask[k]
                base = alpha * popcount(rm & MASK1) + beta * popcount(rm & MASK2) - sum(cost[k] for k in rest)
                if base > v + 1e-9:  # dropping helps
                    ch, v, improved = rest, base, True
                    break
                bi, bv = None, v
                for i in pool:
                    if i in ch:
                        continue
                    nm = rm | cand_mask[i]
                    nv = base + alpha * (popcount(nm & MASK1) - popcount(rm & MASK1)) \
                        + beta * (popcount(nm & MASK2) - popcount(rm & MASK2)) - cost[i]
                    if nv > bv + 1e-9:
                        bi, bv = i, nv
                if bi is not None:
                    ch = rest[:pos] + [bi] + rest[pos:]
                    v, improved = bv, True
                    break
        if v > best_v:
            best, best_v = tuple(sorted(ch)), v
    return best


# ---------------------------------------------------------------- page
def coverage_table(mask, mask_patch) -> str:
    """Letter x form grid: covered by the sentences / added by the patch words / still missing / impossible."""
    def cell(t):
        if t not in IDX:
            return '<td class="na">·</td>'
        j = IDX[t]
        if mask >> j & 1:
            return '<td class="ok">✓</td>'
        if mask_patch >> j & 1:
            return '<td class="pat">+</td>'
        return '<td class="miss">✗</td>'
    rows = []
    for c in BASE28 + CORE_EXTRA + HAMZA:
        tier = "t2" if c in HAMZA else ""
        rows.append(f'<tr class="{tier}"><th>{c}</th>{"".join(cell((c, f)) for f in FORMS)}</tr>')
    for a in ALEFS:
        lig = "ل" + a
        rows.append(f'<tr class="t2"><th>{lig}</th>{"".join(cell((lig, f)) for f in FORMS)}</tr>')
    head = "".join(f"<th>{FORM_AR[f]}<br><small>{f}</small></th>" for f in FORMS)
    return (f'<table class="cov"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def target_label(t) -> str:
    return f"{t[0]} ({t[1]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=400000)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--depth", type=int, default=12, help="max sentences per full-cover set")
    ap.add_argument("--short", type=int, default=3, help="max sentences per short set")
    ap.add_argument("--alpha-full", type=float, default=150.0)
    ap.add_argument("--patch-rank", type=int, default=40000, help="patch words come from the top-N arwiki vocab")
    ap.add_argument("--per-k", type=int, default=2, help="disjoint solutions per set size")
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=60.0, help="chars a letter form is worth")
    ap.add_argument("--min-nat", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=10.0, help="chars a hamza/lam-alef form is worth")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    global cand_mask, cand_len, cand_nat
    texts, sources = [], []
    seen = set()
    for s in arwiki_sentences(args.max_lines):
        if s not in seen:
            seen.add(s); texts.append(s); sources.append("arwiki")
    n_wiki = len(texts)
    for s in khatt_lines():
        if s not in seen:
            seen.add(s); texts.append(s); sources.append("khatt-train")
    print(f"candidates: {n_wiki} arwiki sentences, {len(texts) - n_wiki} KHATT train lines")

    words_per = [clean_words(s) for s in texts]
    rank, freq = load_vocab()
    allw = set(w for ws in words_per for w in ws) | {w for w, r in rank.items() if r < args.patch_rank}
    lexcls = load_lexicon_classes(allw)
    cand_mask = [mask_of(ws) for ws in words_per]
    cand_len = [len(HARAKAT.sub("", s)) for s in texts]
    cand_nat = [naturalness(ws, rank, lexcls) for ws in words_per]

    # pool: natural sentences, reduced to the efficient ones plus the shortest carriers of every target
    ok = [i for i in range(len(texts)) if cand_nat[i] >= args.min_nat]
    tf = Counter(j for i in ok for j in iter_bits(cand_mask[i]))
    rare_w = [1.0 + 2.0 / math.log(2 + tf[j]) for j in range(len(ALL))]

    def eff(i):
        return sum(rare_w[j] for j in iter_bits(cand_mask[i] & MASK1)) / cand_len[i] * (0.4 + cand_nat[i])
    def val(i):
        return args.alpha * popcount(cand_mask[i] & MASK1) - cand_len[i] * (1.6 - cand_nat[i])
    pool = sorted(ok, key=eff, reverse=True)[: args.pool] + sorted(ok, key=val, reverse=True)[: args.pool]
    by_t = defaultdict(list)
    for i in ok:
        for j in iter_bits(cand_mask[i]):
            by_t[j].append(i)
    extra = set()
    for j, lst in by_t.items():
        lst.sort(key=lambda i: -eff(i))
        extra.update(lst[:60])
    pool = list(dict.fromkeys(pool + sorted(extra)))
    print(f"natural sentences: {len(ok)}; search pool: {len(pool)}")

    def mask_set(ch):
        m = 0
        for i in ch:
            m |= cand_mask[i]
        return m

    # (a) short sets: 1..--short sentences, length-penalised max coverage, best + one disjoint alternative
    # (b) full sets: alpha_full makes every reachable letter form worth buying; grow k until 104/104
    picked = []
    for k in range(1, args.short + 1):
        used = set()
        for r in range(args.per_k):
            ch = solve_cover(pool, k, args.alpha, args.beta, random.Random(100 * k + r),
                             restarts=args.restarts, banned=used)
            if not ch:
                break
            m = mask_set(ch)
            print(f"  short k={k}: {sum(cand_len[i] for i in ch)} chars, {popcount(m & MASK1)}/{N1} + "
                  f"{popcount(m & MASK2)}/{len(ALL) - N1}")
            picked.append(("short", ch, m))
            used |= set(ch)
    n_full = 0
    for k in range(args.short + 1, args.depth + 1):
        ch = solve_cover(pool, k, args.alpha_full, args.beta, random.Random(7000 + k), restarts=args.restarts)
        m = mask_set(ch)
        print(f"  full k={k}: {len(ch)} sentences, {sum(cand_len[i] for i in ch)} chars, "
              f"{popcount(m & MASK1)}/{N1} + {popcount(m & MASK2)}/{len(ALL) - N1}")
        if popcount(m & MASK1) == N1:
            picked.append(("full", ch, m))
            n_full += 1
            if n_full >= args.per_k:
                break

    # patch words from the lexicon side: common, non-name words; smallest greedy set covering what is missing
    patch_pool = []
    for w, r in rank.items():
        if r < args.patch_rank and 2 <= len(w) <= 8 and WORD_RE.match(w):
            if is_name(w, lexcls) or "name" in lexcls.get(w, (set(), set()))[1] and r > 3000:
                continue
            patch_pool.append((w, r, mask_of([w])))

    def patch(m):
        need1 = MASK1 & ~m
        need2 = MASK2 & ~m
        words = []
        while need1:
            w, r, wm = max(patch_pool, key=lambda x: (popcount(x[2] & need1) * 4 + popcount(x[2] & need2)
                                                      - 0.15 * len(x[0]) - x[1] / args.patch_rank))
            if not wm & need1:
                break
            words.append(w)
            need1 &= ~wm
            need2 &= ~wm
        return words, need1

    carriers = defaultdict(list)
    for w, r, wm in sorted(patch_pool, key=lambda x: x[1]):
        for j in iter_bits(wm):
            if len(carriers[j]) < 4:
                carriers[j].append(w)

    results = []
    for kind, ch, m in picked:
        miss1 = [ALL[j] for j in range(N1) if not m >> j & 1]
        miss2 = [ALL[j] for j in range(N1, len(ALL)) if not m >> j & 1]
        L = sum(cand_len[i] for i in ch)
        nat = sum(cand_nat[i] * cand_len[i] for i in ch) / L
        pw, left = patch(m)
        pm = m | mask_of(pw)
        results.append({
            "kind": kind,
            "sentences": [{"text": texts[i], "source": sources[i], "chars": cand_len[i],
                           "words": len(words_per[i]), "naturalness": round(cand_nat[i], 3)} for i in ch],
            "chars": L, "words": sum(len(words_per[i]) for i in ch), "naturalness": round(nat, 3),
            "tier1_covered": popcount(m & MASK1), "tier1_total": N1,
            "tier2_covered": popcount(m & MASK2), "tier2_total": len(ALL) - N1,
            "missing_tier1": [target_label(t) for t in miss1],
            "missing_tier2": [target_label(t) for t in miss2],
            "patch_set": pw, "patch_chars": sum(len(w) + 1 for w in pw),
            "with_patch_tier1": popcount(pm & MASK1), "with_patch_tier2": popcount(pm & MASK2),
            "carriers": {target_label(t): carriers.get(IDX[t], []) for t in miss1 + miss2},
            "mask": m, "mask_with_patch": pm,
        })
    # ranking: 1-3 sentence sets plus patch words first (shortest total, then naturalness), then full sets
    results.sort(key=lambda r: (r["kind"] != "short", r["chars"] + r["patch_chars"] - 60 * r["naturalness"]))
    for k, r in enumerate(results, 1):
        r["rank"] = k
    meta = {
        "design": "docs/research/designs/writer-adaptation.md part C; checklist 2.7",
        "sources": {"arwiki": str(ARWIKI), "arwiki_lines_scanned": args.max_lines, "arwiki_sentences": n_wiki,
                    "khatt_train_lines": len(texts) - n_wiki, "vocab": str(VOCAB), "lexicon": str(LEXICON)},
        "targets": {"tier1": [target_label(t) for t in TIER1], "tier2": [target_label(t) for t in TIER2]},
        "search": {"method": "randomised greedy + 1-swap local search", "alpha": args.alpha, "alpha_full": args.alpha_full, "short": args.short, "patch_rank": args.patch_rank, "restarts": args.restarts, "depth": args.depth, "per_k": args.per_k, "pool": args.pool, "min_nat": args.min_nat, "beta": args.beta},
    }
    (OUT / "candidates.json").write_text(json.dumps({"meta": meta, "candidates": results}, ensure_ascii=False, indent=1))
    write_page(results, meta)
    for r in results:
        print(r["rank"], r["kind"], len(r["sentences"]), r["chars"], r["naturalness"], f'{r["tier1_covered"]}/{N1}',
              f'{r["tier2_covered"]}/{len(ALL) - N1}', "patch:", " ".join(r["patch_set"]), r["patch_chars"],
              f'-> {r["with_patch_tier1"]}/{N1}')


def write_page(results, meta):
    n1, n2 = len(TIER1), len(TIER2)
    esc = html.escape
    cards = []
    for r in results:
        sents = "".join(
            f'<li><p class="ar big" lang="ar" dir="rtl">{esc(s["text"])}</p>'
            f'<p class="meta">{s["source"]} · {s["words"]} words · {s["chars"]} chars · naturalness {s["naturalness"]:.2f}</p></li>'
            for s in r["sentences"])
        miss1 = "، ".join(r["missing_tier1"]) or "none"
        miss2 = "، ".join(r["missing_tier2"]) or "none"
        if r["patch_set"]:
            patch = (f'<p class="ar big" lang="ar" dir="rtl">{esc(" ".join(r["patch_set"]))}</p>'
                     f'<p class="meta">{len(r["patch_set"])} words, {r["patch_chars"]} chars; with them: letter forms '
                     f'{r["with_patch_tier1"]}/{n1}, hamza and lam-alef {r["with_patch_tier2"]}/{n2}. '
                     f'A native speaker could fold these into one extra sentence.</p>')
        else:
            patch = '<p class="meta">None needed: the sentences cover every letter form.</p>'
        carriers = "".join(
            f'<li><span class="ar">{esc(t)}</span>: <span class="ar">{"، ".join(ws) or "–"}</span></li>'
            for t, ws in r["carriers"].items())
        label = "Short set" if r["kind"] == "short" else "Full real-text cover"
        cards.append(f"""
<section class="card" id="c{r["rank"]}">
  <h2>{r["rank"]}. {label} <span class="sub">{len(r["sentences"])} sentence(s) · {r["words"]} words · {r["chars"]} chars ·
  letter forms {r["tier1_covered"]}/{n1} · hamza and lam-alef {r["tier2_covered"]}/{n2} · naturalness {r["naturalness"]:.2f}</span></h2>
  <ol class="sents">{sents}</ol>
  <h3>Patch words (lexicon, common words) for the missing letter forms</h3>{patch}
  <div class="grid">
    <div>{coverage_table(r["mask"], r["mask_with_patch"])}
      <p class="meta">✓ in the sentences · + only via patch words · ✗ missing · · form does not exist</p></div>
    <div class="notes">
      <h3>Letter forms the sentences miss</h3><p class="ar">{miss1}</p>
      <h3>Hamza and lam-alef forms they miss</h3><p class="ar">{miss2}</p>
      <details><summary>Other common words carrying each missing form</summary><ul class="patch">{carriers or "<li>none</li>"}</ul></details>
      <h3>Your check (0.5)</h3>
      <ul class="check"><li>Natural modern Arabic? (yes / awkward / no)</li>
      <li>Any word or name a writer might stumble on?</li><li>Edits you would make:</li></ul>
    </div>
  </div>
</section>""")
    short = [r for r in results if r["kind"] == "short"]
    full = [r for r in results if r["kind"] == "full"]
    summary = "".join(
        f'<tr><td><a href="#c{r["rank"]}">{r["rank"]}</a></td><td>{r["kind"]}</td><td>{len(r["sentences"])}</td>'
        f'<td>{r["chars"]}</td><td>{r["tier1_covered"]}/{n1}</td><td>{r["tier2_covered"]}/{n2}</td>'
        f'<td>{len(r["patch_set"])} ({r["patch_chars"]} ch)</td><td>{r["with_patch_tier1"]}/{n1}</td>'
        f'<td>{r["naturalness"]:.2f}</td></tr>' for r in results)
    src = meta["sources"]
    n_dual = sum(c in DUAL for c in BASE28 + CORE_EXTRA)
    b1 = max(short, key=lambda r: r["tier1_covered"]) if short else None
    best_short = (f'{b1["tier1_covered"]}/{n1} letter forms with {len(b1["sentences"])} sentences ({b1["words"]} words)'
                  if b1 else "n/a")
    best_full = (f'{len(full[0]["sentences"])} sentences, {full[0]["words"]} words, {full[0]["chars"]} chars'
                 if full else "none found")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Calibration Sentence Candidates</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e3e0d8;--ok:#2f7d4f;--okbg:#e5f3ea;--miss:#b3261e;--missbg:#fbe6e4;--pat:#8a5a00;--patbg:#fbf0d9}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#171715;--fg:#ecebe6;--mut:#a3a19a;--card:#21211e;--line:#3a3934;--ok:#7fd1a0;--okbg:#1e3327;--miss:#f19a92;--missbg:#3b2220;--pat:#f0c674;--patbg:#3a2f18}}}}
:root[data-theme="dark"]{{--bg:#171715;--fg:#ecebe6;--mut:#a3a19a;--card:#21211e;--line:#3a3934;--ok:#7fd1a0;--okbg:#1e3327;--miss:#f19a92;--missbg:#3b2220;--pat:#f0c674;--patbg:#3a2f18}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:980px;margin:0 auto}} h1{{font-size:1.5rem;margin:0 0 .3rem}} .lead{{color:var(--mut)}} a{{color:inherit}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0}}
h2{{font-size:1.1rem;margin:0 0 .5rem}} .sub{{display:block;font-weight:400;color:var(--mut);font-size:.85rem}}
h3{{font-size:.9rem;margin:.8rem 0 .2rem}} .ar{{font-family:"Noto Naskh Arabic","Geeza Pro","Amiri",serif;font-size:1.2rem;direction:rtl;unicode-bidi:isolate}}
p.ar.big{{font-size:1.5rem;line-height:2;margin:.2rem 0}} .meta{{color:var(--mut);font-size:.8rem;margin:0}}
.grid{{display:grid;grid-template-columns:auto 1fr;gap:20px;margin-top:.6rem}} @media (max-width:640px){{.grid{{grid-template-columns:1fr}}}}
.cov{{border-collapse:collapse;font-size:.85rem}} .cov th,.cov td{{border:1px solid var(--line);padding:2px 6px;text-align:center}}
.cov tbody th{{font-family:"Noto Naskh Arabic","Geeza Pro",serif;font-size:1.1rem}} .cov small{{color:var(--mut)}}
.ok{{color:var(--ok);background:var(--okbg)}} .miss{{color:var(--miss);background:var(--missbg);font-weight:700}}
.pat{{color:var(--pat);background:var(--patbg);font-weight:700}} .na{{color:var(--mut)}}
tr.t2 th{{color:var(--mut)}} .patch,.check{{padding-left:1.1rem;margin:.2rem 0}} code{{font-size:.85em}}
.wrap{{overflow-x:auto}} table.sum{{border-collapse:collapse;font-size:.85rem;width:100%}}
table.sum th,table.sum td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}}
</style></head><body><main>
<h1>Calibration sentence candidates</h1>
<p class="lead">Checklist 2.7 (design: writer-adaptation, part C), for the native-speaker check (0.5). Mark each
candidate natural / awkward / unusable and note edits. Cards 1–{len(short)} are short sets (1–3 real sentences) plus a
few common patch words for what they miss; the last {len(full)} are full covers built from real sentences only.</p>
<p class="lead"><b>Why it cannot be one or two lines:</b> a dual-joining letter takes its final or isolated form only at the
end of a word, so every word supplies at most one of those {2 * n_dual} targets ({n_dual} letters × final/isolated). Full
coverage therefore needs at least {2 * n_dual} words, i.e. several handwritten lines. Best found: {best_short} for 1–3 real
sentences; {best_full} for a real-text-only full cover. The practical calibration text is a short set plus a patch sentence the
native speaker composes from the listed words.</p>
<div class="wrap"><table class="sum"><thead><tr><th>#</th><th>kind</th><th>sentences</th><th>chars</th><th>letter forms</th>
<th>hamza / lam-alef</th><th>patch words</th><th>with patch</th><th>naturalness</th></tr></thead><tbody>{summary}</tbody></table></div>
<p class="lead">Targets: <b>{n1} letter forms</b> (28 letters + ة ى; dual-joining letters × isolated/initial/medial/final,
right-joining ا د ذ ر ز و ة ى × isolated/final) and <b>{n2} extras</b> (hamza forms أ إ آ ؤ ئ ء and the lam-alef
ligatures لا لأ لإ لآ, isolated and final). Lam before alef counts as the ligature, not as a lam form. Dotted pairs
(ب ت ث ن ي, ج ح خ, د ذ, ر ز, س ش, ص ض, ط ظ, ع غ, ف ق) are covered whenever their letter forms are. Diacritics ignored.</p>
<p class="lead">Sources: {src["arwiki_sentences"]:,} sentences from Arabic Wikipedia (2023-11-01, shard 2, first
{src["arwiki_lines_scanned"]:,} lines; CC BY-SA 4.0, attribution: Wikipedia contributors) and {src["khatt_train_lines"]:,}
KHATT train line labels. Patch words: top-{meta["search"]["patch_rank"]:,} Wikipedia words, excluding words the lexicon
tags only as names. Naturalness (0–1) is the share of words that are common in Wikipedia and not lexicon-only names.
Data: <code>results/calibration/candidates.json</code>; script: <code>scripts/calib_sentence_cover.py</code>.</p>
{"".join(cards)}
</main></body></html>"""
    (OUT / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Lexicon coverage of the panel's ground-truth tokens (checklist 1.7, lexicon.md Assembly 5).

    python -I scripts/lexicon/coverage.py [--panel results/panel/panel.pkl] [--out results/lexicon]

For each test set (one row per line; the __inkcrop copies are skipped) and each source, the share of running
Arabic-letter tokens that are
  known   - the key-normalised token is in the source (affix lexicons: accepted by the Hunspell-style checker);
  variant - only a hamza / alif / yā' / tā' marbūṭa respelling is (diac_insensitive key, or a checker-accepted respelling);
  unseen  - neither.
"Common" = Ayaspell ∪ aspell-ar ∪ Pecina ∪ Arabic-Wikipedia sample. For tokens unseen in Common we report the share
found in a name source (Wikidata, GeoNames, ANETAC, Ayaspell's name section), also after stripping proclitics.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import OUT, load_checker, tokens, vkey  # noqa: E402

W = HERE.parents[1]
COMMON = ("ayaspell", "aspell", "pecina", "arwiki")
NAMES = ("wikidata", "geonames", "anetac", "ayaspell_names")
PROCLITICS = sorted({"و", "ف", "ب", "ل", "ك", "ال", "وال", "فال", "بال", "كال", "لل", "ولل", "فلل", "وب", "فب", "ول", "فل",
                     "وك", "فك"}, key=len, reverse=True)
ALEFS = "اأإآ"
# letters that share a dotless skeleton (rasm); swapping one for another is a dot error
DOT_GROUPS = ["بتثنيى", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة"]


def controls(types, n=3000, seed=0):
    """Pseudo-words for the false-accept check: letters shuffled, and one dot-group letter swapped (a dot error)."""
    import random
    rng = random.Random(seed)
    pool = sorted(t for t in types if len(t) >= 3)
    rng.shuffle(pool)
    shuf, dots = set(), set()
    for t in pool:
        if len(shuf) < n:
            c = list(t)
            rng.shuffle(c)
            s = "".join(c)
            if s not in types:
                shuf.add(s)
        if len(dots) < n:
            idx = [i for i, ch in enumerate(t) if any(ch in g for g in DOT_GROUPS)]
            if idx:
                i = rng.choice(idx)
                g = next(g for g in DOT_GROUPS if t[i] in g)
                alt = [x for x in g if x != t[i]]
                s = t[:i] + rng.choice(alt) + t[i + 1:]
                if s not in types:
                    dots.add(s)
    return shuf, dots


def respellings(t: str, cap: int = 64):
    """Orthographic variants: any alif form at each alif position, final ي/ى and ه/ة, hamza carriers to bare."""
    opts = []
    for i, ch in enumerate(t):
        last = i == len(t) - 1
        if ch in ALEFS:
            opts.append(ALEFS)
        elif last and ch in "يى":
            opts.append("يى")
        elif last and ch in "هة":
            opts.append("هة")
        elif ch == "ؤ":
            opts.append("ؤو")
        elif ch == "ئ":
            opts.append("ئي")
        else:
            opts.append(ch)
    for c in itertools.islice(itertools.product(*opts), cap):
        s = "".join(c)
        if s != t:
            yield s


def strip_clitics(t: str):
    for p in PROCLITICS:
        if t.startswith(p) and len(t) - len(p) >= 2:
            yield t[len(p):]


def stream_forms(name: str):
    if name == "arwiki":
        with gzip.open(OUT / "arwiki_vocab.tsv.gz", "rt", encoding="utf-8") as fh:
            for ln in fh:
                yield ln.split("\t", 1)[0]
        return
    if name == "ayaspell_names":
        with gzip.open(OUT / "ayaspell_stems.tsv.gz", "rt", encoding="utf-8") as fh:
            for ln in fh:
                w, _, c = ln.rstrip("\n").split("\t")
                if c in ("name", "person", "place", "organisation"):
                    yield w
        return
    with gzip.open(OUT / f"{name}.tsv.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            yield ln.split("\t", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=W / "results/panel/panel.pkl")
    ap.add_argument("--out", type=Path, default=W / "results/lexicon")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = pickle.load(open(a.panel, "rb"))["rows"]
    sets: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["dataset"].endswith("__inkcrop"):
            continue
        sets[r["dataset"]].update(tokens(r.get("label_raw") or r["label"]))
    types = set().union(*sets.values())
    vk = {t: vkey(t) for t in types}
    vk_inv = defaultdict(set)
    for t, v in vk.items():
        vk_inv[v].add(t)
    probe = set(types)
    for t in types:
        probe.update(strip_clitics(t))
    shuf, dots = controls(types)
    probe |= shuf | dots
    print(f"{len(rows)} rows, {len(types)} token types, {len(probe)} probe types", flush=True)

    exact: dict[str, set] = {}
    variant: dict[str, set] = {}
    false_accept: dict[str, dict] = {}
    for src in (*COMMON, *NAMES):
        t0 = time.time()
        ex, va = set(), set()
        f = {"arwiki": "arwiki_vocab", "ayaspell": "ayaspell_stems", "aspell": "aspell_stems",
             "ayaspell_names": "ayaspell_stems"}.get(src, src)
        if not (OUT / f"{f}.tsv.gz").exists():
            print(f"  {src}: MISSING", flush=True)
            exact[src], variant[src], false_accept[src] = set(), set(), {"missing": True}
            continue
        elif src in ("ayaspell", "aspell"):
            chk = load_checker(src)
            for t in probe:
                if chk.known(t):
                    ex.add(t)
                elif t in types and any(chk.known(s) for s in respellings(t)):
                    va.add(t)
        else:
            for f in stream_forms(src):
                if f in probe:
                    ex.add(f)
                v = vkey(f)
                if v in vk_inv:
                    va.update(vk_inv[v])
            va -= ex
        exact[src], variant[src] = ex, va
        false_accept[src] = {"shuffled": round(len(ex & shuf) / len(shuf), 4), "dot_swapped": round(len(ex & dots) / len(dots), 4)}
        print(f"  {src}: false-accept {false_accept[src]}; exact {len(ex & types)} variant {len(va)} types  ({time.time() - t0:.0f}s)", flush=True)

    def tok_share(c: Counter, pred) -> float:
        n = sum(c.values())
        return round(sum(v for t, v in c.items() if pred(t)) / n, 4) if n else 0.0

    com_ex = set().union(*(exact[s] for s in COMMON))
    com_va = set().union(*(variant[s] for s in COMMON)) - com_ex
    name_ex = set().union(*(exact[s] for s in NAMES))
    name_va = set().union(*(variant[s] for s in NAMES))

    def is_name(t):
        return t in name_ex or t in name_va or any(s in name_ex for s in strip_clitics(t))

    report = {"normalisation": "key = NFC + bidi/ZW + presentation forms + lookalikes, tatweel and harakat stripped; "
                               "variant = repo diac_insensitive profile (alif, yā', tā' marbūṭa, hamza folded)",
              "false_accept_controls": {"n_shuffled": len(shuf), "n_dot_swapped": len(dots), **false_accept},
              "sets": {}}
    for ds in sorted(sets):
        c = sets[ds]
        unseen = {t for t in c if t not in com_ex and t not in com_va}
        d = {"tokens": sum(c.values()), "types": len(c),
             "by_source": {s: {"known": tok_share(c, lambda t, s=s: t in exact[s]),
                               "variant": tok_share(c, lambda t, s=s: t in variant[s])} for s in (*COMMON, *NAMES)},
             "common": {"known": tok_share(c, lambda t: t in com_ex), "variant": tok_share(c, lambda t: t in com_va),
                        "unseen": tok_share(c, lambda t: t in unseen)},
             "any_source_incl_names": tok_share(c, lambda t: t in com_ex or t in com_va or is_name(t)),
             "unseen_tokens": sum(c[t] for t in unseen),
             "unseen_name_share": round(sum(c[t] for t in unseen if is_name(t)) / max(1, sum(c[t] for t in unseen)), 4),
             "top_unseen": [(t, c[t], is_name(t)) for t in sorted(unseen, key=lambda t: -c[t])[:25]]}
        report["sets"][ds] = d
    (a.out / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    L = ["# Lexicon coverage of the panel ground truth", "",
         "Generated by `scripts/lexicon/coverage.py` from `results/panel/panel.pkl` (one row per line, ink-crop copies "
         "skipped) and the lexicon build on the SSD (`/Volumes/PortableSSD/data/lexicon/build`). Token = maximal run of "
         "Arabic letters; shares are of running tokens.", "",
         f"Normalisation: {report['normalisation']}.", "",
         "## Common-word lexicons (Ayaspell ∪ aspell-ar ∪ Pecina ∪ Wikipedia sample)", "",
         "| Set | Tokens | Types | Known % | Variant-only % | Unseen % | Unseen that are names % | Any source incl. names % |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, d in report["sets"].items():
        cm = d["common"]
        L.append(f"| {ds} | {d['tokens']} | {d['types']} | {pct(cm['known'])} | {pct(cm['variant'])} | {pct(cm['unseen'])} | "
                 f"{pct(d['unseen_name_share'])} | {pct(d['any_source_incl_names'])} |")
    L += ["", "## Per source: known % (variant-only %)", "",
          "| Set | " + " | ".join((*COMMON, *NAMES)) + " |", "|---|" + "---:|" * len((*COMMON, *NAMES))]
    for ds, d in report["sets"].items():
        L.append(f"| {ds} | " + " | ".join(f"{pct(d['by_source'][s]['known'])} ({pct(d['by_source'][s]['variant'])})"
                                           for s in (*COMMON, *NAMES)) + " |")
    L += ["", "## False-accept controls (permissiveness)", "",
          f"Share of pseudo-words a source accepts: {len(shuf)} panel types with letters shuffled, and {len(dots)} with one "
          "letter swapped inside its dot group (ب/ت/ث/ن/ي, ج/ح/خ, د/ذ, ر/ز, س/ش, ص/ض, ط/ظ, ع/غ, ف/ق, ه/ة) -- i.e. a dot error "
          "that a lexicon-constrained decoder could not catch. Both exclude strings that are themselves panel types.", "",
          "| Source | Shuffled % | Dot-swapped % |", "|---|---:|---:|"]
    for s_ in (*COMMON, *NAMES):
        fa = false_accept.get(s_, {})
        L.append(f"| {s_} | {pct(fa.get('shuffled', 0))} | {pct(fa.get('dot_swapped', 0))} |")
    L += ["", "## Most frequent unseen tokens (✓ = found in a name source)", ""]
    for ds, d in report["sets"].items():
        L.append(f"- **{ds}:** " + "، ".join(f"{t} ({n}{' ✓' if nm else ''})" for t, n, nm in d["top_unseen"][:15]))
    L += ["", "## Reading notes", "",
          "- Coverage is near-saturated (96-99 % known), so for decoding the binding number is the false-accept table: "
          "the common union accepts at least ~23 % of single dot errors (the max over its members), so a lexicon "
          "constraint alone cannot restore dots; frequency (Wikipedia counts) or an LM has to rank candidates.",
          "- Ayaspell and Pecina enforce hamza spelling; Muharaf and RASAM texts often omit it, which shows up as their "
          "large variant-only share (12-16 %) against aspell-ar's ~1.5 %.",
          "- Name sources contain many common words (organisation and place labels), so their own coverage is not a name "
          "rate; the name share of OOV (column 7 of the first table) is the relevant figure.",
          "- Affix lexicons (Ayaspell, aspell-ar) are tested with `scripts/lexicon/affix.py` (`Checker`), which accepts "
          "exactly what the expander generates (checked on 107k forms of 300 random Ayaspell stems: 0 misses)."]
    (a.out / "COVERAGE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

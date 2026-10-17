"""Checklist 2.10: spelling inconsistency inside a label set (R4.3) and the per-dataset conventions table (R4.4).

    .geomvenv/bin/python scripts/audit_spelling.py [--out results/audit]

R4.3  Tokens are taken at the no-harakat view (canonical, tatweel and digits folded, harakat stripped, punctuation removed).
      Two surface forms are VARIANTS when they share the orthographic-fold key (alif/hamza seats, yāʾ/alif maqṣūra, tāʾ
      marbūṭa/hāʾ, lookalikes; errors.LADDER "orthographic"). Per set: the share of recurring word keys (>= 2 tokens) written
      more than one way, the share of tokens in a minority variant (line-bootstrap 95 % CI), split by the letter class that
      differs, and how many minority tokens are NON-STANDARD (absent from the lexicon, or < 1 % of the majority variant's
      Arabic-Wikipedia frequency) while the majority variant is attested. Examples per set.
R4.4  One row per set: the measured fingerprint (results/analysis/B/conventions.json: hamza on alif, tāʾ marbūṭa share,
      alif maqṣūra share, digit system, double spaces, space before punctuation; harakat share from audit_diacritics) plus
      R4.3's inconsistency rate, the source description, and what the dataset documentation says ("not yet extracted" where
      we have not read the paper's annotation section; nothing is filled in from memory).

Decision rule (written before the run):
  If a set's minority-variant token rate has a 95 % CI lower bound >= 1 % of tokens, its labels are internally inconsistent
  in hamza / yāʾ / tāʾ spelling, and its headline CER must be quoted at the orthographic rung (folds applied) with the
  canonical rung beside it; below that, the canonical rung may stand alone. Sets whose test splits differ in a convention by
  more than 5 points (e.g. NAKBA test vs blind hamza) are reported per split, never pooled.
Outputs: OUT/spelling.json, OUT/conventions_table.json, OUT/spelling_gallery.html.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import HTML_HEAD, OUT, W, load_samples, panel_images, png_b64  # noqa: E402
from audit_leakage import TEXT_COL, TRAIN_TEXT, files  # noqa: E402
from arabic_ocr_bench.characterize import errors  # noqa: E402
from arabic_ocr_bench.normalize import canonicalize_encoding, normalize_digits, strip_harakat, strip_tatweel  # noqa: E402

LEX = Path("/Volumes/PortableSSD/data/lexicon/build/lexicon.tsv.gz")
CLASS = {**{c: "alif/hamza" for c in "اأإآٱءؤئ"}, **{c: "yaa/maqsura" for c in "يىی"}, **{c: "taa/haa" for c in "هةہھە"}}
SOURCE = {
    "kitab_khatt": "KHATT handwritten Arabic (KITAB-Bench 200-line sample); many writers copying fixed paragraphs plus free text",
    "kitab_khatt_full": "KHATT v1.0 test (Teklia copy), 1,038 lines",
    "nakba_lines": "NAKBA contest test split: Omar al-Saleh manuscripts (U4RASD)",
    "nakba_blind": "NAKBA contest blind test split: Omar al-Saleh manuscripts (U4RASD)",
    "muharaf_lines": "Muharaf public test split (aamijar/muharaf-public): Levantine historical manuscripts",
    "rasam1": "RASAM-1 test (calfa-ai): Maghrebi manuscripts",
    "rasam2": "RASAM-2 test (calfa-ai): Maghrebi manuscripts",
}


def tokens(raw: str) -> list[str]:
    """The surface spelling: ladder rungs canonical + tatweel/digits, harakat stripped, no orthographic folds."""
    surf = strip_harakat(normalize_digits(strip_tatweel(canonicalize_encoding(raw or ""))))
    return [t for t in "".join(c if (c.isalpha() or c.isspace()) else " " for c in surf).split() if any("؀" <= c <= "ۿ" for c in t)]


def fold(tok: str) -> str:
    return errors._orthographic(tok)


def diff_class(a: str, b: str) -> str:
    if len(a) != len(b):
        return "alif/hamza"  # آ vs ءا style length changes only arise from hamza seats
    cls = {CLASS.get(x, "other") for x, y in zip(a, b) if x != y} | {CLASS.get(y, "other") for x, y in zip(a, b) if x != y}
    cls.discard("other")
    return "+".join(sorted(cls)) or "other"


def analyse(lines: list[tuple[str, str]], lexf: dict[str, int], lex: set[str], reps: int, seed: int = 0) -> dict:
    toks = [(lid, tokens(t)) for lid, t in lines]
    var = defaultdict(Counter)
    for _, ts in toks:
        for t in ts:
            var[fold(t)][t] += 1
    major = {k: c.most_common(1)[0][0] for k, c in var.items()}
    rec = {k: c for k, c in var.items() if sum(c.values()) >= 2}
    inc = {k: c for k, c in rec.items() if len(c) > 1}
    per_line_min, per_line_tok, per_line_ns = [], [], []
    by_class, ns_by_class, examples = Counter(), Counter(), []
    for lid, ts in toks:
        m = ns = 0
        for t in ts:
            k = fold(t)
            if len(var[k]) > 1 and t != major[k]:
                m += 1
                c = diff_class(t, major[k])
                by_class[c] += 1
                mj = major[k]
                nonstd = (mj in lex) and (t not in lex or lexf.get(t, 0) < 0.01 * max(1, lexf.get(mj, 0)))
                if nonstd:
                    ns += 1
                    ns_by_class[c] += 1
        per_line_min.append(m)
        per_line_tok.append(len(ts))
        per_line_ns.append(ns)
    mn, tk, nsa = np.array(per_line_min, float), np.array(per_line_tok, float), np.array(per_line_ns, float)
    rng = np.random.default_rng(seed)
    bs, bns = [], []
    for _ in range(reps):
        i = rng.integers(0, len(mn), len(mn))
        bs.append(mn[i].sum() / max(1, tk[i].sum()))
        bns.append(nsa[i].sum() / max(1, tk[i].sum()))
    for k, c in sorted(inc.items(), key=lambda kv: -sum(kv[1].values()))[:25]:
        examples.append({"key": k, "variants": c.most_common(), "class": diff_class(c.most_common(2)[1][0], c.most_common(1)[0][0]),
                         "lexicon": {v: (v in lex, lexf.get(v, 0)) for v in c}})
    T = max(1, tk.sum())
    return {"lines": len(lines), "tokens": int(tk.sum()), "word_keys_recurring": len(rec), "keys_inconsistent_share": len(inc) / max(1, len(rec)),
            "minority_token_rate": float(mn.sum() / T), "minority_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "nonstandard_token_rate": float(nsa.sum() / T), "nonstandard_ci95": [float(np.percentile(bns, 2.5)), float(np.percentile(bns, 97.5))],
            "minority_by_class_per_1000": {c: 1000 * v / T for c, v in by_class.most_common()},
            "nonstandard_by_class_per_1000": {c: 1000 * v / T for c, v in ns_by_class.most_common()},
            "examples": examples, "_line_minority": dict(zip([l for l, _ in lines], per_line_min))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    samples = load_samples()
    sets = defaultdict(list)
    for k, s in samples.items():
        sets[s["dataset"]].append((k, s["gt_raw"]))
    for pool, gl in TRAIN_TEXT.items():
        col = TEXT_COL.get(pool, "text")
        sets[pool] = [(f"{pool}/{i}", t or "") for i, t in enumerate(t for f in files(gl) for t in pq.read_table(f, columns=[col]).column(col).to_pylist())]
    need = set()
    for L in sets.values():
        for _, t in L:
            need.update(tokens(t))
    lexf, lex = {}, set()
    with gzip.open(LEX, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in need:
                lex.add(p[0])
                lexf[p[0]] = max(lexf.get(p[0], 0), int(p[4]) if len(p) > 4 and p[4].isdigit() else 0)
    print("lexicon hits", len(lex), "of", len(need), flush=True)
    res = {ds: analyse(L, lexf, lex, a.reps) for ds, L in sorted(sets.items())}
    line_min = {ds: r.pop("_line_minority") for ds, r in res.items()}
    for ds, r in res.items():
        r["decision"] = "orthographic rung required in headline" if r["minority_ci95"][0] >= 0.01 else "canonical rung may stand alone"
        print(ds, r["tokens"], "keys inc %.3f" % r["keys_inconsistent_share"], "minority %.4f [%.4f,%.4f]" % (r["minority_token_rate"], *r["minority_ci95"]),
              "nonstd %.4f" % r["nonstandard_token_rate"], r["minority_by_class_per_1000"])

    # ---- R4.4 conventions table
    conv = json.load(open(W / "results/analysis/B/conventions.json"))
    fp, merged = conv.get("fingerprint", {}), conv.get("merged_spellings", {})
    dia = json.load(open(a.out / "diacritics.json"))["R1.3_profile"] if (a.out / "diacritics.json").exists() else {}
    table = {}
    for ds in res:
        f = fp.get(ds, {})
        table[ds] = {"source": SOURCE.get(ds, ds), "documented_conventions": "not yet extracted from the dataset paper",
                     "hamza_on_alif": f.get("hamza_on_alif"), "ta_marbuta_share": f.get("ta_marbuta_share"), "alif_maqsura_share": f.get("alif_maqsura_share"),
                     "digits": f.get("digits"), "double_space_lines": f.get("double_space_lines"), "space_before_punct": f.get("space_before_punct"),
                     "tatweel_lines": f.get("tatweel_lines"),
                     "lines_with_harakat": (dia.get(ds) or {}).get("share_lines_with_marks"), "harakat_policy": (dia.get(ds) or {}).get("policy"),
                     "dialect_merger_per_1000": (merged.get(ds) or {}).get("per_1000_tokens"),
                     "spelling_minority_rate": res[ds]["minority_token_rate"], "spelling_nonstandard_rate": res[ds]["nonstandard_token_rate"]}
    # split differences inside one source
    splits = {}
    for x, y in (("nakba_lines", "nakba_blind"), ("kitab_khatt", "kitab_khatt_full"), ("rasam1", "rasam2"), ("muharaf_lines", "muharaf_train"), ("nakba_lines", "nakba_train")):
        if x in table and y in table:
            d = {}
            for k in ("hamza_on_alif", "ta_marbuta_share", "alif_maqsura_share", "double_space_lines", "space_before_punct", "lines_with_harakat", "spelling_minority_rate"):
                if table[x][k] is not None and table[y][k] is not None:
                    d[k] = table[x][k] - table[y][k]
            splits[f"{x} - {y}"] = {"diff": d, "differs_by_gt_5pt": [k for k, v in d.items() if abs(v) > 0.05]}
    json.dump({"decision_rule": __doc__.split("Decision rule (written before the run):")[1].split("Outputs:")[0].strip(), "R4.3": res}, open(a.out / "spelling.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"table": table, "split_differences": splits}, open(a.out / "conventions_table.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(splits, indent=1))

    # ---- gallery: the lines carrying the most minority spellings per test set, plus random ones
    import random
    rnd = random.Random(0)
    pick = {}
    for ds in ("kitab_khatt", "kitab_khatt_full", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"):
        lm = line_min.get(ds, {})
        top = sorted(lm, key=lambda k: -lm[k])[:4]
        pick[ds] = [(k, "most minority spellings") for k in top if lm[k]] + [(k, "random") for k in rnd.sample(sorted(lm), min(3, len(lm)))]
    imgs = panel_images({k for v in pick.values() for k, _ in v})
    html = [HTML_HEAD.format(title="Spelling and conventions"), "<h1>Spelling inconsistency (R4.3) and conventions (R4.4), checklist 2.10</h1>",
            "<table><tr><th>set</th><th>tokens</th><th>recurring keys written 2+ ways</th><th>minority-variant tokens [95% CI]</th><th>non-standard tokens [95% CI]</th><th>by class (/1000 tokens)</th><th>decision</th></tr>"]
    for ds, r in res.items():
        html.append(f"<tr><td>{ds}</td><td>{r['tokens']}</td><td>{r['keys_inconsistent_share']:.1%}</td><td>{r['minority_token_rate']:.2%} [{r['minority_ci95'][0]:.2%}, {r['minority_ci95'][1]:.2%}]</td>"
                    f"<td>{r['nonstandard_token_rate']:.2%} [{r['nonstandard_ci95'][0]:.2%}, {r['nonstandard_ci95'][1]:.2%}]</td><td>{', '.join(f'{c} {v:.1f}' for c, v in r['minority_by_class_per_1000'].items())}</td><td>{r['decision']}</td></tr>")
    html.append("</table><h2>Conventions table (R4.4)</h2><table><tr><th>set</th><th>hamza on alif</th><th>tāʾ marbūṭa share</th><th>alif maqṣūra share</th><th>digits</th><th>double-space lines</th><th>space before punct.</th><th>lines with harakat</th><th>policy</th><th>dialect mergers /1000</th><th>source</th></tr>")
    pct = lambda v: "–" if v is None else f"{v:.1%}"
    for ds, t in table.items():
        html.append(f"<tr><td>{ds}</td><td>{pct(t['hamza_on_alif'])}</td><td>{pct(t['ta_marbuta_share'])}</td><td>{pct(t['alif_maqsura_share'])}</td><td>{json.dumps(t['digits'])}</td>"
                    f"<td>{pct(t['double_space_lines'])}</td><td>{pct(t['space_before_punct'])}</td><td>{pct(t['lines_with_harakat'])}</td><td>{t['harakat_policy']}</td>"
                    f"<td>{'–' if t['dialect_merger_per_1000'] is None else round(t['dialect_merger_per_1000'], 2)}</td><td>{t['source']}</td></tr>")
    html.append("</table><h2>Examples of inconsistent spellings</h2>")
    for ds, r in res.items():
        html.append(f"<h3>{ds}</h3><table><tr><th>variants (count)</th><th>class</th><th>in lexicon (arwiki freq)</th></tr>")
        for e in r["examples"][:10]:
            html.append(f"<tr><td class=ar>{' · '.join(f'{v} ({n})' for v, n in e['variants'])}</td><td>{e['class']}</td><td class=ar>{' · '.join(f'{v}: {int(ok)}/{fq}' for v, (ok, fq) in e['lexicon'].items())}</td></tr>")
        html.append("</table>")
    for ds, L in pick.items():
        html.append(f"<h2>{ds}: lines</h2>")
        for k, why in L:
            if k in imgs:
                html.append(f"<div class=card><div class=mut>{why} · {k} · {line_min[ds][k]} minority tokens</div><img src='{png_b64(imgs[k])}'><div class=ar>{samples[k]['gt_raw']}</div></div>")
    html.append("</body></html>")
    (a.out / "spelling_gallery.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()

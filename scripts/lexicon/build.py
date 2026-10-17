"""Build the lexicon (checklist 1.7) and the modern-text sample (1.8) from the raw downloads on the SSD.

Run with `python -I` (downloads are untrusted data; nothing from them is imported or executed):

    python -I scripts/lexicon/build.py all            # or: ayaspell aspell pecina anetac geonames wikidata arwiki merge

Inputs : $DATA/lexicon/<source>/... and $DATA/text/arwiki/*.parquet  (see fetch.sh)
Outputs: $DATA/lexicon/build/
    <source>.tsv.gz        form \t class [\t extra]   (key-normalised, unique)
    ayaspell_stems.tsv.gz  stem \t flags \t class      (affix lexicon; use affix.Checker for membership)
    aspell_stems.tsv.gz    stem \t flags
    arwiki_vocab.tsv.gz    form \t count               (Arabic-letter tokens of the Wikipedia sample)
    lexicon.tsv.gz         form \t variant_key \t sources \t classes \t wiki_count   (merged finite list)
    build_stats.json
         $DATA/text/arwiki/arwiki_20231101_shard2.txt.gz   one paragraph per line (the LM sample, 1.8)

Normalisation (lexicon.md, Assembly 1): `key` = encoding clean-up (NFC, bidi/ZW, presentation forms) + lookalike
letters folded + tatweel and harakat stripped. Hamza / alif / yā' / tā' marbūṭa variants are NOT merged; the
`variant_key` column (the repo's diac_insensitive profile) links them.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from affix import AffixSet, parse_dic_line  # noqa: E402
from arabic_ocr_bench.normalize import (canonicalize_encoding, normalize, normalize_lookalikes,  # noqa: E402
                                        strip_harakat, strip_tatweel)

DATA = Path(os.environ.get("LEXICON_DATA", "/Volumes/PortableSSD/data"))
LEX = DATA / "lexicon"
OUT = LEX / "build"
ARWIKI = DATA / "text" / "arwiki"

# Arabic letters (incl. extended letters) with combining marks and tatweel; digits and punctuation split tokens.
TOKEN_RE = re.compile("[ء-غـ-ٰٟ-ۓۺ-ۿ]+")
LETTER_RE = re.compile("[ء-غف-يٱ-ۓۺ-ۿ]")


def key(s: str) -> str:
    return strip_harakat(strip_tatweel(normalize_lookalikes(canonicalize_encoding(s)))).strip()


def vkey(s: str) -> str:
    return normalize(s, "diac_insensitive")


def tokens(text: str):
    for m in TOKEN_RE.finditer(canonicalize_encoding(text)):
        k = key(m.group())
        if k and LETTER_RE.search(k):
            yield k


def write_rows(path: Path, rows) -> int:
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------- Ayaspell
AYA_PERSON = {6, 7, 8, 21, 23}
AYA_PLACE = {9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22}
AYA_ORG = {18}


def build_ayaspell(stats):
    d = LEX / "ayaspell"
    aff = AffixSet(str(d / "ar.aff"))
    section, rows = "", []
    for line in open(d / "ar.dic", encoding="utf-8", errors="replace"):
        m = re.match(r"#\s*(\S+\.dic)\s*#", line)
        if m:
            section = m.group(1)
            continue
        p = parse_dic_line(line, aff)
        if not p:
            continue
        w, _, morph = p
        flags = line.split("\t")[0].strip().partition("/")[2]
        am = int(morph) if morph.isdigit() else 0
        cls = ("person" if am in AYA_PERSON else "place" if am in AYA_PLACE else "organisation" if am in AYA_ORG
               else "name" if section == "names.dic" else "stopword" if section == "stopwords.dic" else "word")
        rows.append((key(w), flags, cls))
    n = write_rows(OUT / "ayaspell_stems.tsv.gz", rows)
    stats["ayaspell"] = {"stems": n, "classes": dict(Counter(r[2] for r in rows)), "rules": sum(len(v) for v in aff.rules.values())}


# ---------------------------------------------------------------- aspell-ar
def read_prezip(path: Path):
    """Decode aspell's prezip word-list compression: <shared-prefix-byte-count><suffix bytes>..., header 0x02."""
    b = path.read_bytes()
    if b[0] != 2:
        raise ValueError("not a prezip v2 file")
    prev, i, n = b"", 1, len(b)
    while i < n:
        c = b[i]
        i += 1
        if c == 31:
            break
        j = i
        while j < n and b[j] >= 32:
            j += 1
        cur = prev[:c] + b[i:j]
        yield cur.decode("utf-8")
        prev, i = cur, j


def aspell_dir() -> Path:
    return LEX / "aspell_ar" / "aspell6-ar-1.2-0"


def build_aspell(stats):
    rows = []
    for e in read_prezip(aspell_dir() / "ar.cwl"):
        w, _, f = e.partition("/")
        rows.append((key(w), f))
    n = write_rows(OUT / "aspell_stems.tsv.gz", rows)
    stats["aspell"] = {"stems": n, "stems_with_flags": sum(1 for r in rows if r[1])}


def load_checker(name: str):
    """Membership checker for the two affix lexicons (expanded forms are ~4e8 / ~2.6e7, not materialised)."""
    from affix import Checker
    if name == "ayaspell":
        aff = AffixSet(str(LEX / "ayaspell" / "ar.aff"))
        ents = []
        with gzip.open(OUT / "ayaspell_stems.tsv.gz", "rt", encoding="utf-8") as fh:
            for ln in fh:
                w, f, _ = ln.rstrip("\n").split("\t")
                ents.append((w, aff.flags(f) if f else frozenset()))
        return Checker(aff, ents)
    aff = AffixSet(str(aspell_dir() / "ar_affix.dat"), flag_mode="char")
    ents = []
    with gzip.open(OUT / "aspell_stems.tsv.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            w, _, f = ln.rstrip("\n").partition("\t")
            ents.append((w, frozenset(f)))
    return Checker(aff, ents)


# ---------------------------------------------------------------- Pecina / Attia word list
def build_pecina(stats):
    z = zipfile.ZipFile(LEX / "pecina" / "Arabic-Wordlist-1.6.zip")
    members = [i for i in z.infolist() if not i.is_dir()]
    stats["pecina"] = {"members": [(i.filename, i.file_size) for i in members]}
    forms = set()
    for i in members:
        if not i.filename.lower().endswith((".txt", ".dic", ".lst", ".wl")) and i.file_size < 10_000_000:
            continue
        with z.open(i) as fh:
            for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                w = raw.strip().split("\t")[0].split("/")[0]
                if w and not w.startswith("#"):
                    k = key(w)
                    if k and LETTER_RE.search(k):
                        forms.add(k)
    n = write_rows(OUT / "pecina.tsv.gz", ((f, "word") for f in sorted(forms)))
    stats["pecina"]["forms"] = n


# ---------------------------------------------------------------- ANETAC
def build_anetac(stats):
    forms = Counter()
    for s in ("train", "dev", "test"):
        p = LEX / "anetac" / f"{s}.ar"
        if p.exists():
            for ln in open(p, encoding="utf-8", errors="replace"):
                for t in tokens(ln):
                    forms[t] += 1
    n = write_rows(OUT / "anetac.tsv.gz", ((f, "name", c) for f, c in sorted(forms.items())))
    stats["anetac"] = {"tokens": n, "note": "transliteration release only; no class labels in the public repo"}


# ---------------------------------------------------------------- GeoNames
def build_geonames(stats):
    z = zipfile.ZipFile(LEX / "geonames" / "alternateNamesV2.zip")
    full, toks = Counter(), Counter()
    with z.open("alternateNamesV2.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            c = raw.split("\t")
            if len(c) > 3 and c[2] == "ar":
                full[key(c[3])] += 1
                for t in tokens(c[3]):
                    toks[t] += 1
    write_rows(OUT / "geonames_names.tsv.gz", ((f, "place", n) for f, n in sorted(full.items())))
    n = write_rows(OUT / "geonames.tsv.gz", ((f, "place", n) for f, n in sorted(toks.items())))
    stats["geonames"] = {"ar_names": sum(full.values()), "distinct_names": len(full), "tokens": n}


# ---------------------------------------------------------------- Wikidata
WD_CLASS = {"person": "person", "given_name": "person", "family_name": "person", "country": "place",
            "admin_area": "place", "settlement": "place", "organisation": "organisation"}


def build_wikidata(stats):
    toks: dict[tuple, int] = Counter()
    full = Counter()
    st = {}
    for f in sorted((LEX / "wikidata").glob("*.tsv.gz")):
        name = f.name.split(".")[0]
        cls = WD_CLASS.get(name, "name")
        n = 0
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                n += 1
                full[(key(parts[2]), cls)] += 1
                for t in tokens(parts[2]):
                    toks[(t, cls)] += 1
        st[name] = n
    write_rows(OUT / "wikidata_names.tsv.gz", ((f, c, n) for (f, c), n in sorted(full.items())))
    n = write_rows(OUT / "wikidata.tsv.gz", ((f, c, n) for (f, c), n in sorted(toks.items())))
    stats["wikidata"] = {"rows_by_class": st, "distinct_labels": len(full), "token_class_pairs": n}


# ---------------------------------------------------------------- Arabic Wikipedia sample (1.8) + vocabulary
def build_arwiki(stats):
    import pyarrow.parquet as pq
    vocab = Counter()
    n_art = n_par = n_chars = 0
    files = sorted(ARWIKI.glob("*.parquet"))
    out = ARWIKI / "arwiki_20231101_shard2.txt.gz"
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fo:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["text"], batch_size=2000):
                for text in batch.column(0).to_pylist():
                    n_art += 1
                    for par in canonicalize_encoding(text or "").split("\n"):
                        par = par.strip()
                        if len(par) < 20 or not LETTER_RE.search(par):
                            continue
                        fo.write(par + "\n")
                        n_par += 1
                        n_chars += len(par)
                        vocab.update(tokens(par))
    n = write_rows(OUT / "arwiki_vocab.tsv.gz", ((w, c) for w, c in sorted(vocab.items(), key=lambda t: -t[1])))
    stats["arwiki"] = {"parquet_files": [f.name for f in files], "articles": n_art, "paragraphs": n_par,
                       "characters": n_chars, "tokens": sum(vocab.values()), "types": n,
                       "types_count_ge3": sum(1 for c in vocab.values() if c >= 3), "sample_file": str(out)}


# ---------------------------------------------------------------- merge
FINITE = ("pecina", "anetac", "geonames", "wikidata")


def merge(stats):
    """Finite merged list: Pecina + name tokens + Ayaspell/aspell stems + Wikipedia types (count >= 3)."""
    tmp = OUT / "merge_unsorted.tsv"
    with open(tmp, "w", encoding="utf-8") as fo:
        for src in FINITE:
            p = OUT / f"{src}.tsv.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for ln in fh:
                    c = ln.rstrip("\n").split("\t")
                    fo.write(f"{c[0]}\t{src}\t{c[1]}\t0\n")
        for src in ("ayaspell", "aspell"):
            p = OUT / f"{src}_stems.tsv.gz"
            if p.exists():
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    for ln in fh:
                        c = ln.rstrip("\n").split("\t")
                        fo.write(f"{c[0]}\t{src}\t{c[2] if len(c) > 2 else 'word'}\t0\n")
        p = OUT / "arwiki_vocab.tsv.gz"
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for ln in fh:
                    w, c = ln.rstrip("\n").split("\t")
                    if int(c) >= 3:
                        fo.write(f"{w}\tarwiki\tword\t{c}\n")
    env = dict(os.environ, LC_ALL="C", TMPDIR=str(OUT))
    srt = OUT / "merge_sorted.tsv"
    subprocess.run(["sort", "-t", "\t", "-k1,1", "-S", "1G", "-o", str(srt), str(tmp)], check=True, env=env)
    tmp.unlink()
    n = 0
    by_src = Counter()

    def flush(fo, form, srcs, clss, cnt):
        nonlocal n
        if not form:
            return
        fo.write(f"{form}\t{vkey(form)}\t{','.join(sorted(srcs))}\t{','.join(sorted(clss))}\t{cnt}\n")
        n += 1
        for s in srcs:
            by_src[s] += 1

    with open(srt, encoding="utf-8") as fh, gzip.open(OUT / "lexicon.tsv.gz", "wt", encoding="utf-8") as fo:
        cur, srcs, clss, cnt = None, set(), set(), 0
        for ln in fh:
            w, s, c, k = ln.rstrip("\n").split("\t")
            if w != cur:
                flush(fo, cur, srcs, clss, cnt)
                cur, srcs, clss, cnt = w, set(), set(), 0
            srcs.add(s)
            clss.add(c)
            cnt = max(cnt, int(k))
        flush(fo, cur, srcs, clss, cnt)
    srt.unlink()
    stats["merged"] = {"forms": n, "forms_by_source": dict(by_src)}


SOURCES = {
    "ayaspell": ("https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ar/ (Ayaspell 3.x via LibreOffice)",
                 "GPL-2.0-or-later / LGPL-2.1-or-later / MPL-1.1 tri-licence"),
    "aspell_ar": ("https://ftp.gnu.org/gnu/aspell/dict/ar/aspell6-ar-1.2-0.tar.bz2", "GPL-2.0-or-later"),
    "pecina": ("https://sourceforge.net/projects/arabic-wordlist/ (Attia, Pecina, Samih, Shaalan, van Genabith, LREC 2012)",
               "GPL-3.0-or-later (COPYING in the zip)"),
    "anetac": ("https://github.com/HadjAmeur/ANETAC-Dataset (Hadj Ameur, Meziane, Guessoum 2019)",
               "no licence file; 'freely available' per the paper; cite arXiv:1907.03110"),
    "geonames": ("https://download.geonames.org/export/dump/alternateNamesV2.zip", "CC BY 4.0"),
    "wikidata": ("https://qlever.dev/api/wikidata (QLever mirror of Wikidata)", "CC0 1.0"),
    "arwiki": ("https://huggingface.co/datasets/wikimedia/wikipedia 20231101.ar", "CC BY-SA 4.0 (and GFDL)"),
}


def manifest(stats):
    """Sizes and SHA-256 of every raw download and build output -> results/lexicon/manifest.json (repo)."""
    import hashlib
    from datetime import datetime, timezone

    def entry(p: Path):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}

    m = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "sources": {}, "build": []}
    for name, (url, lic) in SOURCES.items():
        d = ARWIKI if name == "arwiki" else LEX / name
        files = sorted(p for p in d.rglob("*") if p.is_file() and not p.name.startswith("."))
        if name == "aspell_ar":
            files = [p for p in files if p.suffix == ".bz2"]
        if name == "arwiki":
            files = [p for p in files if p.suffix == ".parquet" or p.name.endswith(".txt.gz")]
        m["sources"][name] = {"url": url, "licence": lic, "files": [entry(p) for p in files]}
    m["build"] = [entry(p) for p in sorted(OUT.glob("*")) if p.is_file()]
    m["stats"] = stats
    dest = HERE.parents[1] / "results" / "lexicon" / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


STEPS = {"ayaspell": build_ayaspell, "aspell": build_aspell, "pecina": build_pecina, "anetac": build_anetac,
         "geonames": build_geonames, "wikidata": build_wikidata, "arwiki": build_arwiki, "merge": merge, "manifest": manifest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "build_stats.json"
    stats = json.loads(sp.read_text()) if sp.exists() else {}
    for s in (list(STEPS) if "all" in a.steps else a.steps):
        print("step", s, flush=True)
        STEPS[s](stats)
        sp.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()

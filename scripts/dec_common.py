"""Shared helpers for the decoding side studies (checklist 2.14 tokenizer view, 2.15 root-sharing).

Runs in the decoding venv (/Volumes/PortableSSD/venvs/decoding: camel-tools, tokenizers, pandas, rapidfuzz); the text
normaliser is imported straight from src/arabic_ocr_bench/normalize.py (stdlib only) so the package is not installed there.

Readings: results/store/lines.parquet, one reading per (model, sample) (first stored), transform "none", test splits only,
reversed_suspect lines dropped. Text view: canonical encoding, lookalikes folded, harakat and tatweel stripped (the
lexicon `key`, = the letters view plus Persian/Urdu lookalike folding).
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "scripts"))
from eval_stats import bh, boot_p  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("aocr_normalize", W / "src/arabic_ocr_bench/normalize.py")
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)

OUT = W / "results/decoding"
LEXICON = Path("/Volumes/PortableSSD/data/lexicon/build/lexicon.tsv.gz")
COMMON = {"ayaspell", "aspell", "pecina", "arwiki"}
DOT_GROUPS = ["بتثنيى", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة"]
DOT_OF = {c: g for g in DOT_GROUPS for c in g}
LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
# skeleton (rasm) class per letter: dot-group letters collapse to one class
RASM = {c: g[0] for g in DOT_GROUPS for c in g}

# model -> tokenizer repo. All Qwen-lineage VLMs (Qwen2/2.5/3-VL, AIN, Fanar-2-Oryx, Baseer, Qari) share one BPE
# (verified: identical ids on Arabic text); PaliGemma2 uses the Gemma-2 SentencePiece vocabulary.
QWEN_TOK = "Qwen/Qwen3-VL-4B-Instruct"
GEMMA_TOK = "unsloth/gemma-2-2b-it"
TOKENIZERS = {"qwen": QWEN_TOK, "gemma2": GEMMA_TOK}
MODELS = {  # (family, model): tokenizer key (None = CRNN control, no tokenizer)
    ("vlm-base", "qwen3-vl-4b-bf16"): "qwen",
    ("vlm-base", "qwen3-vl-8b-bf16"): "qwen",
    ("vlm-base", "ain-7b-bf16"): "qwen",
    ("vlm-base", "fanar-oryx-7b-bf16"): "qwen",
    ("vlm-base", "paligemma2-3b-bf16"): "gemma2",
    ("vlm-ft", "suite-v1-qwen3vl4b-ft-bf16"): "qwen",
    ("vlm-ft", "suite-v1-qwen3vl8b-ft-bf16"): "qwen",
    ("vlm-ft", "suite-v1-ain7b-ft-bf16"): "qwen",
    ("vlm-ft", "suite-v1-fanar2-oryx-ft-bf16"): "qwen",
    ("vlm-ft", "baseer-nakba-bf16"): "qwen",
    ("vlm-ft", "exp3090-paligemma2-3b-ft-bf16"): "gemma2",
    ("ocr-engine", "qari-ocr-0.2"): "qwen",
    ("crnn", "base"): None,
    ("crnn", "combo"): None,
}


def key(s: str) -> str:
    return N.strip_harakat(N.strip_tatweel(N.normalize_lookalikes(N.canonicalize_encoding(s or "")))).strip()


def is_arabic_word(w: str) -> bool:
    return len(w) > 0 and all(c in LETTERS for c in w)


def rasm(w: str) -> str:
    return "".join(RASM.get(c, c) for c in w)


def load_samples() -> pd.DataFrame:
    s = pd.read_parquet(W / "results/store/samples.parquet")
    s = s.copy()  # kitab_khatt (KITAB 200) is stored under split "train" but is an evaluation set
    pages = [page_of(m) for m in s.meta]
    s["page"] = pages
    s["cluster"] = [p if p else k for p, k in zip(pages, s.sample_key)]  # a set without page ids: each line its own cluster
    return s.set_index("sample_key")


def page_of(meta: str):
    """Same clusters as analyze_lines.page_of: NAKBA doc-page / blind year-page, RASAM manuscript-page."""
    try:
        m = json.loads(meta or "{}")
    except ValueError:
        return None
    if "filename" in m:
        f = m["filename"]
        return f.rsplit("-", 1)[0] if f.count("-") >= 2 else f.rsplit("_", 1)[0]
    if "manuscript_name" in m:
        return f"{m['manuscript_name']}/{m.get('image_name')}"
    return None


def load_readings(samples: pd.DataFrame) -> pd.DataFrame:
    cols = ["family", "model", "transform", "pack", "sample_key", "dataset", "hyp", "reversed_suspect", "in_train",
            "e_no_diacritics", "n_no_diacritics"]
    l = pd.read_parquet(W / "results/store/lines.parquet", columns=cols)
    want = set(MODELS)
    l = l[[(f, m) in want for f, m in zip(l.family, l.model)]]
    l = l[(l["transform"] == "none") & ~l.reversed_suspect & l.sample_key.isin(samples.index) & (l.in_train != "yes")]
    crnn = l.family == "crnn"
    l = l[~crnn | (l.pack == "panel-2026-10-13")]
    l = l.drop_duplicates(["family", "model", "sample_key"]).copy()
    l["gt"] = samples.loc[l.sample_key, "gt_clean"].values
    l["cluster"] = samples.loc[l.sample_key, "cluster"].values
    l["tok"] = [MODELS[(f, m)] for f, m in zip(l.family, l.model)]
    return l.reset_index(drop=True)


def align_words(gt: str, hyp: str):
    """Word alignment on the key view. Yields (gt_word, hyp_word or None) for every GT word: equal -> same word,
    1:1 inside a replace block of equal size -> substitution, otherwise None (deleted / split / merged)."""
    g, h = key(gt).split(), key(hyp).split()
    out = []
    for op in Levenshtein.opcodes(g, h):
        a = g[op.src_start:op.src_end]
        b = h[op.dest_start:op.dest_end]
        if op.tag == "equal":
            out += list(zip(a, a))
        elif op.tag == "replace" and len(a) == len(b):
            out += list(zip(a, b))
        else:
            out += [(x, None) for x in a]
    return out


_LEX = None


def lexicon() -> set:
    """Key-normalised forms present in a Common source (Ayaspell ∪ aspell-ar ∪ Pecina ∪ Arabic-Wikipedia sample)."""
    global _LEX
    if _LEX is None:
        s = set()
        with gzip.open(LEXICON, "rt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and COMMON.intersection(p[2].split(",")):
                    s.add(p[0])
        _LEX = s
    return _LEX


def cluster_boot(clusters: np.ndarray, stat, reps: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap a statistic resampling whole clusters (page where the set has page ids, else line).
    `stat(idx)` takes an index array into the item arrays."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    gl = [np.array(v) for v in groups.values()]
    out = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(gl), len(gl))
        out[r] = stat(np.concatenate([gl[j] for j in pick]))
    return out


def ci(boots) -> list:
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [math.nan, math.nan]


def gallery_css() -> str:
    return """<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--bd:#e3e3e8;--hi:#c0392b;--ok:#1e7f4f;--card:#fafafc}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa3;--bd:#2c2c31;--hi:#ff7b6b;--ok:#5fd39a;--card:#1b1b1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px}
h1{font-size:22px}h2{font-size:18px;margin-top:28px}.mut{color:var(--mut)}
.item{border:1px solid var(--bd);background:var(--card);border-radius:8px;padding:10px;margin:10px 0}
.item img{max-width:100%;display:block;margin:6px 0;image-rendering:auto}
.ar{direction:rtl;font-size:19px;font-family:'Noto Naskh Arabic','Geeza Pro',serif}
.hi{color:var(--hi);font-weight:600}.ok{color:var(--ok);font-weight:600}
table{border-collapse:collapse;font-size:14px}td,th{border-bottom:1px solid var(--bd);padding:4px 8px;text-align:left}
code{font-size:13px}
</style>"""


def load_images(keys: set) -> dict:
    """Line images (64 px high, full width) from the panel pickle, for the galleries."""
    import pickle
    blob = pickle.load(open(W / "results/panel/panel.pkl", "rb"))
    return {r["id"]: r["img"] for r in blob["rows"] if r["id"] in keys and not r["dataset"].endswith("__inkcrop")}


def img_b64(img) -> str:
    import base64
    import io

    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

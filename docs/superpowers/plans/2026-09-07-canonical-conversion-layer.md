# Canonical Conversion Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every wired corpus into one canonical per-line JSONL record (`data/canonical/<corpus>.jsonl`), plus the character-set and normalization-profile machinery the Tier-0 CTC experiments depend on.

**Architecture:** Additive offline layer. A pure `canonical.py` builds a `CanonicalLine` from the existing `Sample` objects each adapter already yields; `scripts/build_canonical.py` drives it over `configs/datasets.yaml`. `normalize.py` gains two reported profiles. `scripts/build_charset.py` derives the vocabulary from the JSONL. A new `datasets/arams.py` adapter + `scripts/mirror_arams_to_hf.py` bring AraMS-28k-HTR in as a streamed HF dataset with a local Kraken-format export. Nothing in `runner.py` / `report.py` / `make_leaderboard.py` changes.

**Tech Stack:** Python 3.12, `datasets` (HF), `Pillow`, `pyyaml`, `pytest`, `uv`. Package is src-layout: `src/arabic_ocr_bench/`.

**Spec:** `docs/superpowers/specs/2026-09-07-canonical-schema-design.md`

## Global Constraints

- Package import path is `arabic_ocr_bench` under `src/`; run everything with `uv run`.
- `normalize.py` public behaviour for profiles `raw` / `light` / `diac_insensitive` must stay **byte-identical** — only additions allowed.
- `data/canonical/` is git-ignored; add the pattern to `.gitignore` in Task 4.
- `to_canonical` is **pure** (no I/O, no network, no filesystem).
- Every task ends green (`uv run python -m pytest`) and is committed.
- Ground-truth text is kept RAW in `text_diplomatic` (harakat intact, no normalization).
- `line_id` hierarchical form: `<corpus>/<book_id>/<page_id>/l<line_no:03d>`; flat form: `<corpus>/<split>/<row:06d>`.
- `split_unit` ∈ `{provider, book, page, writer}` — never `line`.
- AraMS HF mirror repo name: `rabeeh97/arams28k-htr-lines` (confirm before the real `push_to_hub`).

---

### Task 1: `normalize.py` — `standard` and `diac_full` profiles

**Files:**
- Modify: `src/arabic_ocr_bench/normalize.py`
- Test: `tests/test_normalize.py` (extend)

**Interfaces:**
- Consumes: existing `normalize.py` internals — `_LIGHT`, `_DIAC_EXTRA`, `normalize_yaa`, `normalize_taa_marbuta`, `strip_harakat`, `collapse_whitespace`, `_apply`.
- Produces:
  - `PROFILES` tuple extended to `("raw", "light", "diac_insensitive", "standard", "diac_full")`.
  - `TASHKEEL_WIDE: str` (the current harakat class chars) and `TASHKEEL_FULL: str` (WIDE + `U+0610..U+061A` + `U+06D6..U+06ED`).
  - `strip_harakat(s: str, marks: str = TASHKEEL_WIDE) -> str` — new optional `marks` param, default keeps current behaviour.
  - `normalize(s, "standard")` == `light` + `normalize_yaa` + `normalize_taa_marbuta` (then `collapse_whitespace`).
  - `normalize(s, "diac_full")` == `diac_insensitive` pipeline but harakat stripped with `TASHKEEL_FULL`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalize.py  (append)
from arabic_ocr_bench.normalize import normalize, PROFILES, TASHKEEL_WIDE, TASHKEEL_FULL, strip_harakat


def test_profiles_tuple_has_new_members():
    assert PROFILES == ("raw", "light", "diac_insensitive", "standard", "diac_full")


def test_standard_folds_yaa_and_taa_marbuta_but_light_does_not():
    s = "على رحمة"          # على رحمة
    assert normalize(s, "light") == "على رحمة"
    assert normalize(s, "standard") == "علي رحمه"  # على->علي, رحمة->رحمه


def test_diac_full_strips_quranic_mark_that_diac_insensitive_keeps():
    # U+06DF ARABIC SMALL HIGH ROUNDED ZERO — in FULL, not in WIDE
    s = "ب۟ا"
    assert "۟" in normalize(s, "diac_insensitive")
    assert "۟" not in normalize(s, "diac_full")


def test_strip_harakat_marks_param_defaults_to_wide():
    s = "بَ۟"  # letter + fatha (WIDE) + small-high-rounded-zero (FULL only)
    assert strip_harakat(s) == "ب۟"                       # default: WIDE only
    assert strip_harakat(s, TASHKEEL_FULL) == "ب"              # FULL: both gone


def test_raw_and_light_unchanged_snapshot():
    s = "ﻻـأ١‏  x"   # lam-alef ligature, tatweel, hamza-alef, arabic-indic 1, RLM, spaces
    assert normalize(s, "raw") == "لاـأ١  x"
    assert normalize(s, "light") == "لاا١ x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_normalize.py -k "new_members or standard_folds or diac_full or marks_param or unchanged_snapshot" -v`
Expected: FAIL — `ImportError` for `TASHKEEL_WIDE` / `standard` not a valid profile.

- [ ] **Step 3: Implement the additions**

In `src/arabic_ocr_bench/normalize.py`:

```python
# after the existing _HARAKAT_RE block
TASHKEEL_WIDE = "".join(chr(c) for c in (*range(0x064B, 0x0660), 0x0670))
TASHKEEL_FULL = "".join(chr(c) for c in (
    *range(0x0610, 0x061B), *range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE),
))
_HARAKAT_FULL_RE = re.compile("[" + re.escape(TASHKEEL_FULL) + "]")
```

Change `strip_harakat`:

```python
def strip_harakat(s: str, marks: str = TASHKEEL_WIDE) -> str:
    if marks == TASHKEEL_WIDE:
        return _HARAKAT_RE.sub("", s)
    return re.compile("[" + re.escape(marks) + "]").sub("", s)
```

Add a `strip_harakat_full` step and the two pipelines:

```python
def strip_harakat_full(s: str) -> str:
    return _HARAKAT_FULL_RE.sub("", s)

_STANDARD_EXTRA = (normalize_yaa, normalize_taa_marbuta)
_DIAC_FULL_EXTRA = (normalize_yaa, normalize_taa_marbuta, normalize_hamza, strip_harakat_full)
```

Extend `normalize`:

```python
    if profile == "standard":
        return collapse_whitespace(_apply(s, (*_LIGHT, *_STANDARD_EXTRA)))
    if profile == "diac_full":
        return collapse_whitespace(_apply(s, (*_LIGHT, *_DIAC_FULL_EXTRA)))
```

And `PROFILES = ("raw", "light", "diac_insensitive", "standard", "diac_full")`.

- [ ] **Step 4: Run the full normalize + metrics test files**

Run: `uv run python -m pytest tests/test_normalize.py tests/test_metrics.py -v`
Expected: PASS (all, including pre-existing tests — `raw`/`light`/`diac_insensitive` unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/normalize.py tests/test_normalize.py
git commit -m "normalize: add standard + diac_full profiles, parametrise strip_harakat"
```

---

### Task 2: `canonical.py` — `CanonicalLine`, `to_canonical`, `diacritic_density`

**Files:**
- Create: `src/arabic_ocr_bench/canonical.py`
- Test: `tests/test_canonical.py`

**Interfaces:**
- Consumes: `arabic_ocr_bench.datasets.base.Sample`; `arabic_ocr_bench.normalize.normalize`, `TASHKEEL_WIDE`.
- Produces:
  - `@dataclass(frozen=True) class CanonicalLine` with every field from the spec's JSON shape; `.to_json_dict() -> dict` producing the exact key order/shape in the spec.
  - `diacritic_density(text: str) -> float` — count of chars in `TASHKEEL_WIDE` divided by count of non-whitespace chars of `text` (0.0 if no non-whitespace).
  - `to_canonical(sample: Sample, corpus_meta: dict, *, hf_revision: str | None = None, converter_version: int = 1) -> CanonicalLine`.
    - `corpus_meta` keys: `corpus`, `script` (str) OR `books` (dict `book_id -> {"script": str}`), `era`, `genre` (str|None), `split_unit`, `license`, `hf_id`, `split`.
    - Per-line overrides read from `sample.meta`: `book_id`, `page_id`, `line_no`, `reading_order`, `writer_id`, `polygon`, `bbox`, `page_path`, `zone`, `angle_deg`, `alignment_confidence`, `validation_level`, `text_diac_normalised`, `script`.
    - `line_id`: hierarchical if `book_id` and `page_id` both present in `sample.meta`, else flat from `sample.id` (`<corpus>/<split>/<NNNNNN>` — reuse `sample.id` verbatim when it already matches, else derive from `corpus_meta["split"]` + `meta["row"]`).
    - `text_l1/l2/l3 = normalize(text_diplomatic, p)` for `p in ("raw","light","diac_insensitive")`.
    - `script`: `sample.meta["script"]` › `corpus_meta["books"][book_id]["script"]` › `corpus_meta["script"]`.
    - `alignment_confidence`: `sample.meta.get("alignment_confidence", 100)`.
    - `provenance = {"source_id": sample.id, "hf_id": corpus_meta.get("hf_id"), "hf_revision": hf_revision, "split": corpus_meta.get("split"), "row": sample.meta.get("row"), "converter_version": converter_version}`.
    - Raises `ValueError` if `sample.ground_truth.strip() == ""`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_canonical.py
import pytest
from PIL import Image
from arabic_ocr_bench.datasets.base import Sample
from arabic_ocr_bench.canonical import CanonicalLine, to_canonical, diacritic_density
from arabic_ocr_bench.normalize import normalize

IMG = Image.new("RGB", (8, 4))

FLAT_META = {"source": "x/y", "split": "test", "row": 12}
FLAT_CORPUS = {
    "corpus": "muharaf_lines", "script": "ruqah", "era": "19c-21c", "genre": "correspondence",
    "split_unit": "provider", "license": "CC-BY-NC-SA-4.0", "hf_id": "aamijar/muharaf-public",
    "split": "test",
}


def _sample(gt="بَا", sid="muharaf_lines/test/000012", meta=None):
    return Sample(id=sid, image=IMG, ground_truth=gt, track="line", meta=dict(meta or FLAT_META))


def test_flat_line_id_and_text_layers():
    c = to_canonical(_sample(), FLAT_CORPUS)
    assert c.line_id == "muharaf_lines/test/000012"
    assert c.text_diplomatic == "بَا"
    assert c.text_l1 == normalize("بَا", "raw")
    assert c.text_l2 == normalize("بَا", "light")
    assert c.text_l3 == normalize("بَا", "diac_insensitive")
    assert c.script == "ruqah"
    assert c.split_unit == "provider"
    assert c.alignment_confidence == 100
    assert c.provenance["source_id"] == "muharaf_lines/test/000012"


def test_hierarchical_line_id_and_per_line_overrides():
    meta = {"book_id": "book_03", "page_id": "p0172", "line_no": 7, "row": 99,
            "script": "maghribi", "alignment_confidence": 84, "validation_level": "LV",
            "text_diac_normalised": "با"}
    corpus = {**FLAT_CORPUS, "corpus": "arams28k", "split_unit": "book",
              "hf_id": "rabeeh97/arams28k-htr-lines", "books": {"book_03": {"script": "maghribi"}}}
    c = to_canonical(_sample(sid="arams28k/train/000099", meta=meta), corpus)
    assert c.line_id == "arams28k/book_03/p0172/l007"
    assert c.book_id == "book_03" and c.page_id == "p0172"
    assert c.script == "maghribi"
    assert c.alignment_confidence == 84
    assert c.validation_level == "LV"
    assert c.text_diac_normalised == "با"


def test_diacritic_density_known_string():
    # "بَاْ" -> 1 haraka (fatha) over 3 non-space chars
    assert diacritic_density("بَا") == pytest.approx(1 / 3)
    assert diacritic_density("   ") == 0.0


def test_empty_ground_truth_raises():
    with pytest.raises(ValueError):
        to_canonical(_sample(gt="   "), FLAT_CORPUS)


def test_to_json_dict_has_spec_keys():
    d = to_canonical(_sample(), FLAT_CORPUS).to_json_dict()
    for k in ("line_id", "corpus", "track", "text_diplomatic", "text_diac_normalised",
              "text_edited", "text_l1", "text_l2", "text_l3", "script", "era", "genre",
              "diacritic_density", "alignment_confidence", "validation_level",
              "split", "split_unit", "license", "provenance"):
        assert k in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_canonical.py -v`
Expected: FAIL — `ModuleNotFoundError: arabic_ocr_bench.canonical`.

- [ ] **Step 3: Implement `canonical.py`**

```python
"""Canonical per-line record. Pure — no I/O. See
docs/superpowers/specs/2026-09-07-canonical-schema-design.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .datasets.base import Sample
from .normalize import TASHKEEL_WIDE, normalize

_WIDE_SET = set(TASHKEEL_WIDE)


def diacritic_density(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    marks = sum(c in _WIDE_SET for c in non_ws)
    return marks / len(non_ws)


@dataclass(frozen=True)
class CanonicalLine:
    line_id: str
    corpus: str
    track: str
    book_id: str | None
    page_id: str | None
    page_path: str | None
    polygon: list | None
    bbox: list | None
    zone: str
    angle_deg: float
    reading_order: int | None
    insertion_anchor: Any
    text_diplomatic: str
    text_diac_normalised: str | None
    text_edited: str | None
    text_l1: str
    text_l2: str
    text_l3: str
    script: str
    era: str
    genre: str | None
    writer_id: str | None
    diacritic_density: float
    scan_quality: Any
    alignment_confidence: int
    validation_level: str | None
    split: str
    split_unit: str
    license: str
    provenance: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "line_id": self.line_id, "corpus": self.corpus, "track": self.track,
            "book_id": self.book_id, "page_id": self.page_id, "page_path": self.page_path,
            "polygon": self.polygon, "bbox": self.bbox, "zone": self.zone,
            "angle_deg": self.angle_deg, "reading_order": self.reading_order,
            "insertion_anchor": self.insertion_anchor,
            "text_diplomatic": self.text_diplomatic,
            "text_diac_normalised": self.text_diac_normalised,
            "text_edited": self.text_edited,
            "text_l1": self.text_l1, "text_l2": self.text_l2, "text_l3": self.text_l3,
            "script": self.script, "era": self.era, "genre": self.genre,
            "writer_id": self.writer_id, "diacritic_density": self.diacritic_density,
            "scan_quality": self.scan_quality,
            "alignment_confidence": self.alignment_confidence,
            "validation_level": self.validation_level,
            "split": self.split, "split_unit": self.split_unit, "license": self.license,
            "provenance": self.provenance,
        }


def _line_id(sample: Sample, corpus: str, meta: dict, split: str) -> str:
    if meta.get("book_id") and meta.get("page_id"):
        ln = int(meta.get("line_no", meta.get("row", 0)))
        return f"{corpus}/{meta['book_id']}/{meta['page_id']}/l{ln:03d}"
    if sample.id.startswith(f"{corpus}/"):
        return sample.id
    return f"{corpus}/{split}/{int(meta.get('row', 0)):06d}"


def to_canonical(sample: Sample, corpus_meta: dict, *, hf_revision: str | None = None,
                 converter_version: int = 1) -> CanonicalLine:
    if not sample.ground_truth.strip():
        raise ValueError(f"empty ground_truth for {sample.id}")
    meta = sample.meta or {}
    corpus = corpus_meta["corpus"]
    split = corpus_meta.get("split", meta.get("split", ""))
    gt = sample.ground_truth
    script = (meta.get("script")
              or (corpus_meta.get("books", {}).get(meta.get("book_id"), {}) or {}).get("script")
              or corpus_meta["script"])
    return CanonicalLine(
        line_id=_line_id(sample, corpus, meta, split),
        corpus=corpus, track=sample.track,
        book_id=meta.get("book_id"), page_id=meta.get("page_id"),
        page_path=meta.get("page_path"), polygon=meta.get("polygon"), bbox=meta.get("bbox"),
        zone=meta.get("zone", "main"), angle_deg=float(meta.get("angle_deg", 0.0)),
        reading_order=meta.get("reading_order", meta.get("row")),
        insertion_anchor=meta.get("insertion_anchor"),
        text_diplomatic=gt, text_diac_normalised=meta.get("text_diac_normalised"),
        text_edited=meta.get("text_edited"),
        text_l1=normalize(gt, "raw"), text_l2=normalize(gt, "light"),
        text_l3=normalize(gt, "diac_insensitive"),
        script=script, era=corpus_meta["era"], genre=corpus_meta.get("genre"),
        writer_id=meta.get("writer_id"), diacritic_density=diacritic_density(gt),
        scan_quality=meta.get("scan_quality"),
        alignment_confidence=int(meta.get("alignment_confidence", 100)),
        validation_level=meta.get("validation_level"),
        split=split, split_unit=corpus_meta["split_unit"], license=corpus_meta["license"],
        provenance={
            "source_id": sample.id, "hf_id": corpus_meta.get("hf_id"),
            "hf_revision": hf_revision, "split": split,
            "row": meta.get("row"), "converter_version": converter_version,
        },
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/test_canonical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/canonical.py tests/test_canonical.py
git commit -m "canonical: CanonicalLine + to_canonical + diacritic_density"
```

---

### Task 3: `configs/corpora.yaml` + loader

**Files:**
- Create: `configs/corpora.yaml`
- Create: `src/arabic_ocr_bench/corpora.py`
- Test: `tests/test_corpora.py`

**Interfaces:**
- Consumes: `configs/datasets.yaml` (for the list of wired corpora + `license`).
- Produces:
  - `load_corpora(path="configs/corpora.yaml") -> dict[str, dict]` — raw parse.
  - `corpus_meta(name: str, *, corpora_path=..., datasets_path=...) -> dict` — the merged dict `to_canonical` expects: `corpus`, `script`|`books`, `era`, `genre`, `split_unit` from corpora.yaml; `license`, `hf_id`, `split` from datasets.yaml (`hf_id` via `_hf_id_for` in `datasets/data.py` pattern or `cfg.get("hf_id")`/`params.subset`).
  - Raises `KeyError` with a clear message if a wired corpus has no `corpora.yaml` entry.

- [ ] **Step 1: Write `configs/corpora.yaml`**

```yaml
# Per-corpus constants for the canonical conversion layer.
# script | era | genre | split_unit ; license lives in configs/datasets.yaml.
muharaf_lines:        {script: ruqah,      era: "19c-21c", genre: correspondence,   split_unit: provider}
kitab_khatt:          {script: modern_hw,  era: "21c",     genre: prompts,          split_unit: provider}
kitab_onlinekhatt:    {script: modern_hw,  era: "21c",     genre: prompts,          split_unit: provider}
kitab_muharaf:        {script: ruqah,      era: "19c-21c", genre: historical_hw,    split_unit: provider}
kitab_historyar:      {script: naskh,      era: historical, genre: manuscript,      split_unit: provider}
kitab_historicalbooks: {script: print,     era: historical, genre: book_print,      split_unit: provider}
kitab_hindawi:        {script: print,      era: modern,    genre: book_print,       split_unit: provider}
kitab_adab:           {script: print,      era: modern,    genre: book_print,       split_unit: provider}
kitab_patsocr:        {script: print,      era: classical, genre: religious_print,  split_unit: provider}
kitab_isippt:         {script: print,      era: modern,    genre: slides,           split_unit: provider}
kitab_evarest:        {script: scene_text, era: modern,    genre: in_the_wild,      split_unit: provider}
kitab_arabicocr:      {script: print,      era: modern,    genre: print,            split_unit: provider}
nakba_lines:          {script: modern_hw,  era: "20c",     genre: memoir,           split_unit: provider}
rasam1:               {script: maghribi,   era: historical, genre: manuscript,      split_unit: provider}
rasam2:               {script: maghribi,   era: historical, genre: manuscript,      split_unit: provider}
synthetic:            {script: print,      era: "n/a",     genre: synthetic,        split_unit: provider}
arams28k:
  era: "19c"
  genre: manuscript
  split_unit: book
  books:            # per-book script — FILL from the archive in Task 8
    book_01: {script: naskh}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_corpora.py
import pytest, yaml
from pathlib import Path
from arabic_ocr_bench.corpora import load_corpora, corpus_meta

WIRED = [k for k, v in yaml.safe_load(Path("configs/datasets.yaml").read_text()).items()
         if isinstance(v, dict)]


def test_every_wired_corpus_has_a_corpora_entry():
    c = load_corpora()
    missing = [n for n in WIRED if n not in c]
    assert missing == [], f"no corpora.yaml entry for {missing}"


def test_corpus_meta_merges_license_from_datasets_yaml():
    m = corpus_meta("muharaf_lines")
    assert m["script"] == "ruqah"
    assert m["split_unit"] == "provider"
    assert m["license"] == "CC-BY-NC-SA-4.0"
    assert m["split"] == "test"
    assert m["corpus"] == "muharaf_lines"


def test_corpus_meta_unknown_raises():
    with pytest.raises(KeyError):
        corpus_meta("not_a_corpus")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_corpora.py -v`
Expected: FAIL — `ModuleNotFoundError: arabic_ocr_bench.corpora`.

- [ ] **Step 4: Implement `corpora.py`**

```python
"""Load configs/corpora.yaml and merge with configs/datasets.yaml into the
dict shape canonical.to_canonical expects."""
from __future__ import annotations

from pathlib import Path
import yaml

_CORPORA = Path("configs/corpora.yaml")
_DATASETS = Path("configs/datasets.yaml")


def load_corpora(path: str | Path = _CORPORA) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _hf_id(dcfg: dict) -> str | None:
    if dcfg.get("hf_id"):
        return dcfg["hf_id"]
    sub = (dcfg.get("params") or {}).get("subset")
    if not sub:
        return None
    return sub if "/" in sub else f"ahmedheakl/{sub}"


def corpus_meta(name: str, *, corpora_path=_CORPORA, datasets_path=_DATASETS) -> dict:
    corpora = load_corpora(corpora_path)
    if name not in corpora:
        raise KeyError(f"{name!r} has no configs/corpora.yaml entry")
    datasets = yaml.safe_load(Path(datasets_path).read_text(encoding="utf-8")) or {}
    if name not in datasets:
        raise KeyError(f"{name!r} not in configs/datasets.yaml")
    c, d = dict(corpora[name]), datasets[name]
    c["corpus"] = name
    c["license"] = d.get("license", "unknown")
    c["split"] = d.get("split", "")
    c["hf_id"] = _hf_id(d)
    return c
```

- [ ] **Step 5: Run tests, then commit**

Run: `uv run python -m pytest tests/test_corpora.py -v`
Expected: PASS.

```bash
git add configs/corpora.yaml src/arabic_ocr_bench/corpora.py tests/test_corpora.py
git commit -m "corpora: per-corpus constants yaml + merge loader"
```

---

### Task 4: `scripts/build_canonical.py` — drive the 15 wired corpora

**Files:**
- Create: `scripts/build_canonical.py`
- Modify: `.gitignore` (add `data/canonical/`)
- Test: `tests/test_build_canonical.py`

**Interfaces:**
- Consumes: `corpora.corpus_meta`, `canonical.to_canonical`, `arabic_ocr_bench.datasets.registry.build_dataset` + `load_dataset_configs`.
- Produces:
  - CLI: `uv run python scripts/build_canonical.py [--only a,b] [--no-crops] [--kraken-out DIR] [--force] [--limit N]`.
  - `data/canonical/<corpus>.jsonl` — one `to_json_dict()` per line, sorted by `line_id`, UTF-8, `ensure_ascii=False`.
  - `data/canonical/crops/<corpus>/<line_id-with-slashes-as-dirs>.png` unless `--no-crops`.
  - `data/canonical/_manifest.json`: `{corpus: {n_lines, converter_version, built_at, adapter_sha, normalize_sha}}`.
  - `build_corpus(name, *, crops=True, kraken_out=None, limit=None) -> int` (returns n_lines) — importable for tests.
  - Idempotence: skip a corpus whose `.jsonl` mtime is newer than its adapter module file, `configs/corpora.yaml`, and `src/arabic_ocr_bench/canonical.py`, unless `--force`.
  - Exit code 1 if any requested corpus raised.

- [ ] **Step 1: Write the failing test (uses the offline `synthetic` corpus)**

```python
# tests/test_build_canonical.py
import json
from pathlib import Path
import pytest
from arabic_ocr_bench import build_canonical as bc  # thin re-export, see step 3


@pytest.fixture
def canon_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "OUT", tmp_path / "canonical")
    return tmp_path / "canonical"


def test_build_synthetic_writes_sorted_jsonl(canon_dir):
    n = bc.build_corpus("synthetic", crops=False)
    assert n == 10
    lines = (canon_dir / "synthetic.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    recs = [json.loads(x) for x in lines]
    assert [r["line_id"] for r in recs] == sorted(r["line_id"] for r in recs)
    assert recs[0]["corpus"] == "synthetic"
    assert recs[0]["text_l1"] and "provenance" in recs[0]
    assert (canon_dir / "_manifest.json").exists()


def test_build_is_idempotent(canon_dir):
    bc.build_corpus("synthetic", crops=False)
    mtime1 = (canon_dir / "synthetic.jsonl").stat().st_mtime_ns
    skipped = bc.build_corpus("synthetic", crops=False)   # returns -1 when skipped
    assert skipped == -1
    assert (canon_dir / "synthetic.jsonl").stat().st_mtime_ns == mtime1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_build_canonical.py -v`
Expected: FAIL — `ImportError` (`build_canonical` not importable).

- [ ] **Step 3: Implement `scripts/build_canonical.py` + re-export**

`scripts/build_canonical.py`:

```python
"""Build data/canonical/<corpus>.jsonl from each wired dataset adapter.
See docs/superpowers/specs/2026-09-07-canonical-schema-design.md."""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import arabic_ocr_bench  # loads .env
from arabic_ocr_bench.canonical import to_canonical
from arabic_ocr_bench.corpora import corpus_meta, load_corpora
from arabic_ocr_bench.datasets.registry import build_dataset, load_dataset_configs

OUT = Path("data/canonical")
_CANON_SRC = Path("src/arabic_ocr_bench/canonical.py")
_NORM_SRC = Path("src/arabic_ocr_bench/normalize.py")


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else ""


def _adapter_module_path(name: str, dcfg: dict) -> Path:
    mod = dcfg["adapter"].split(":")[0]              # arabic_ocr_bench.datasets.kitab_bench
    return Path("src") / Path(*mod.split(".")).with_suffix(".py")


def _stale(name: str, dcfg: dict) -> bool:
    jp = OUT / f"{name}.jsonl"
    if not jp.exists():
        return True
    ref = max(_adapter_module_path(name, dcfg).stat().st_mtime,
              Path("configs/corpora.yaml").stat().st_mtime,
              _CANON_SRC.stat().st_mtime)
    return jp.stat().st_mtime < ref


def build_corpus(name: str, *, crops: bool = True, kraken_out: str | None = None,
                 limit: int | None = None, force: bool = False) -> int:
    dcfg = load_dataset_configs()[name]
    if not force and not _stale(name, dcfg):
        return -1
    meta = corpus_meta(name)
    adapter = build_dataset(name, dcfg, limit=limit)
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in adapter:
        try:
            c = to_canonical(s, meta)
        except ValueError:
            continue                                  # empty GT — skip, count elsewhere if needed
        recs.append(c)
        if crops:
            cp = OUT / "crops" / Path(*c.line_id.split("/")).with_suffix(".png")
            cp.parent.mkdir(parents=True, exist_ok=True)
            s.image.save(cp)
        if kraken_out:
            kp = Path(kraken_out) / c.split / Path(*c.line_id.split("/")[1:])
            kp.parent.mkdir(parents=True, exist_ok=True)
            s.image.save(kp.with_suffix(".png"))
            kp.with_suffix(".gt.txt").write_text(c.text_diplomatic, encoding="utf-8")
    seen = set()
    for c in recs:
        if c.line_id in seen:
            raise ValueError(f"duplicate line_id {c.line_id} in {name}")
        seen.add(c.line_id)
    recs.sort(key=lambda c: c.line_id)
    (OUT / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(c.to_json_dict(), ensure_ascii=False) for c in recs) + "\n",
        encoding="utf-8")
    _write_manifest(name, len(recs))
    return len(recs)


def _write_manifest(name: str, n: int) -> None:
    mp = OUT / "_manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {}
    man[name] = {"n_lines": n, "converter_version": 1,
                 "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "adapter_sha": _sha(_adapter_module_path(name, load_dataset_configs()[name])),
                 "normalize_sha": _sha(_NORM_SRC)}
    mp.write_text(json.dumps(man, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--kraken-out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(load_corpora())
    failed = []
    for n in names:
        try:
            r = build_corpus(n, crops=not a.no_crops, kraken_out=a.kraken_out,
                             limit=a.limit, force=a.force)
            print(f"{n}: {'skipped (fresh)' if r == -1 else str(r) + ' lines'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
            failed.append(n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

Add `src/arabic_ocr_bench/build_canonical.py`:

```python
"""Importable alias so tests can `from arabic_ocr_bench import build_canonical`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_canonical import *          # noqa: F401,F403
from build_canonical import OUT, build_corpus, main   # noqa: F401
```

Add `data/canonical/` to `.gitignore`.

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/test_build_canonical.py -v`
Expected: PASS.

- [ ] **Step 5: Build the real thing for the cached corpora and eyeball**

Run: `uv run python scripts/build_canonical.py --only synthetic,muharaf_lines,kitab_khatt,kitab_hindawi --no-crops`
Expected: three/four `data/canonical/*.jsonl` written; `head -1 data/canonical/kitab_khatt.jsonl | python -m json.tool` shows a well-formed record.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_canonical.py src/arabic_ocr_bench/build_canonical.py .gitignore tests/test_build_canonical.py
git commit -m "build_canonical: drive the wired corpora into data/canonical/*.jsonl"
```

---

### Task 5: `scripts/build_charset.py`

**Files:**
- Create: `scripts/build_charset.py`
- Create: `src/arabic_ocr_bench/build_charset.py` (importable alias, same pattern as Task 4)
- Test: `tests/test_build_charset.py`

**Interfaces:**
- Consumes: `data/canonical/<corpus>.jsonl` files.
- Produces:
  - CLI: `uv run python scripts/build_charset.py [--level l1|l2|l3|standard] [--min-count 20] [--corpora a,b]`.
  - `build_charset(corpora: list[str], *, level="l3", min_count=20, canon_dir=OUT) -> dict` with keys: `level`, `min_count`, `global` (`{charset: [sorted chars >= min_count], size, oov_rate, tail: [[ch,n]... for n<min_count]}`), `per_corpus` (`{name: {size, oov_rate}}`), `per_split` (`{split: {size, oov_rate}}`), `union` (charset over the requested corpora, threshold applied on the summed counts), `train_test_gap` (`[chars with count>=min_count in a test split but absent from any train split]`).
  - Writes `data/canonical/_charset.json`.
  - `level="standard"` maps to the `standard` normalize profile applied to `text_diplomatic`; `l1/l2/l3` read the stored `text_lN`.
  - Never drops or mutates a JSONL line.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_charset.py
import json
from pathlib import Path
import pytest
from arabic_ocr_bench.build_charset import build_charset


@pytest.fixture
def canon(tmp_path):
    rows = [
        {"corpus": "t", "split": "train", "text_l3": "ابج", "text_diplomatic": "ابج"},
        {"corpus": "t", "split": "train", "text_l3": "ابج", "text_diplomatic": "ابج"},
        {"corpus": "t", "split": "test",  "text_l3": "ابد", "text_diplomatic": "ابد"},
    ]
    (tmp_path / "t.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_charset_threshold_and_oov(canon):
    out = build_charset(["t"], level="l3", min_count=2, canon_dir=canon)
    # 'ا' x3, 'ب' x3, 'ج' x2, 'د' x1  -> charset = [ا,ب,ج] ; oov mass = 1/9
    assert set(out["global"]["charset"]) == set("ابج")
    assert out["global"]["oov_rate"] == pytest.approx(1 / 9)
    assert ["د", 1] in out["global"]["tail"]


def test_train_test_gap(canon):
    out = build_charset(["t"], level="l3", min_count=1, canon_dir=canon)
    assert "د" in out["train_test_gap"]    # in test, never in train
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_build_charset.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `build_charset.py`**

```python
"""Character-set + OOV analysis over data/canonical/*.jsonl.
See docs/superpowers/specs/2026-09-07-canonical-schema-design.md."""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import arabic_ocr_bench  # noqa: F401
from arabic_ocr_bench.normalize import normalize

OUT = Path("data/canonical")


def _text(rec: dict, level: str) -> str:
    if level == "standard":
        return normalize(rec["text_diplomatic"], "standard")
    return rec[{"l1": "text_l1", "l2": "text_l2", "l3": "text_l3"}[level]]


def _summ(counter: Counter, min_count: int) -> dict:
    total = sum(counter.values())
    charset = sorted(c for c, n in counter.items() if n >= min_count)
    oov = sum(n for c, n in counter.items() if n < min_count)
    tail = sorted(([c, n] for c, n in counter.items() if n < min_count), key=lambda x: x[1])
    return {"charset": charset, "size": len(charset),
            "oov_rate": (oov / total) if total else 0.0, "tail": tail}


def build_charset(corpora: list[str], *, level: str = "l3", min_count: int = 20,
                  canon_dir: Path = OUT) -> dict:
    glob_c: Counter = Counter()
    per_corpus: dict[str, Counter] = {}
    per_split: dict[str, Counter] = {}
    train_chars: set[str] = set()
    test_counter: Counter = Counter()
    for name in corpora:
        jp = Path(canon_dir) / f"{name}.jsonl"
        for line in jp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = _text(rec, level)
            glob_c.update(t)
            per_corpus.setdefault(name, Counter()).update(t)
            sp = rec.get("split", "")
            per_split.setdefault(sp, Counter()).update(t)
            if sp == "train":
                train_chars.update(t)
            elif sp == "test":
                test_counter.update(t)
    gap = sorted(c for c, n in test_counter.items() if n >= min_count and c not in train_chars)
    return {
        "level": level, "min_count": min_count,
        "global": _summ(glob_c, min_count),
        "per_corpus": {k: {"size": _summ(v, min_count)["size"],
                           "oov_rate": _summ(v, min_count)["oov_rate"]}
                       for k, v in per_corpus.items()},
        "per_split": {k: {"size": _summ(v, min_count)["size"],
                          "oov_rate": _summ(v, min_count)["oov_rate"]}
                      for k, v in per_split.items()},
        "union": _summ(glob_c, min_count)["charset"],
        "train_test_gap": gap,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="l3", choices=["l1", "l2", "l3", "standard"])
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--corpora", default="")
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.corpora.split(",") if n.strip()] or \
        [p.stem for p in OUT.glob("*.jsonl")]
    out = build_charset(names, level=a.level, min_count=a.min_count)
    (OUT / "_charset.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"charset[{a.level}] size={out['global']['size']} "
          f"oov={out['global']['oov_rate']:.4f} corpora={names}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`src/arabic_ocr_bench/build_charset.py` — same alias pattern as Task 4.

- [ ] **Step 4: Run tests + a real pass**

Run: `uv run python -m pytest tests/test_build_charset.py -v`
Then: `uv run python scripts/build_charset.py --corpora synthetic,muharaf_lines,kitab_khatt --level l3`
Expected: tests PASS; `data/canonical/_charset.json` written with a plausible size (Arabic ~40-80 chars at L3).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_charset.py src/arabic_ocr_bench/build_charset.py tests/test_build_charset.py
git commit -m "build_charset: vocab + OOV + train/test gap over data/canonical"
```

---

### Task 6: `scripts/mirror_arams_to_hf.py` — Zenodo → HF + Kraken tree

**Files:**
- Create: `scripts/mirror_arams_to_hf.py`
- Test: `tests/test_mirror_arams.py`

**Interfaces:**
- Consumes: the downloaded `AraMS-28k-HTR.zip` (at `.cache/arams28k-htr/AraMS-28k-HTR.zip`; download is separate/manual — the retrying background job).
- Produces:
  - `parse_arams(root: Path) -> Iterator[dict]` — yields `{image_path, text, text_diac_normalised, book_id, page_id, line_no, alignment_confidence, validation_level, script, split}` for each line, by walking the extracted tree. **The exact layout is unknown until the archive lands — implement against the real tree, then adjust the fixture.**
  - CLI: `uv run python scripts/mirror_arams_to_hf.py --zip PATH [--repo rabeeh97/arams28k-htr-lines] [--kraken-out DIR] [--dry-run]`.
  - `--dry-run` extracts, parses, prints per-split counts + one sample record, writes nothing.
  - Without `--dry-run`: builds a `datasets.Dataset` per split with columns `image, text, text_diac_normalised, book_id, page_id, line_no, alignment_confidence, validation_level, script, split` and `push_to_hub(repo)`.
  - `--kraken-out DIR` writes `DIR/<split>/<book>/<page>_<line:03d>.png` + `.gt.txt` (diplomatic text).

- [ ] **Step 1: Write the failing test against a synthetic mini-tree**

```python
# tests/test_mirror_arams.py
from pathlib import Path
from scripts_import import mirror_arams as ma   # see step 3 for the import shim


def _mini_tree(tmp_path: Path) -> Path:
    # mimic the EXPECTED layout; adjust in step 3 once the real archive is seen
    d = tmp_path / "AraMS-28k-HTR"
    for (book, page, ln, txt) in [("book_01", "p001", 1, "ابج"), ("book_01", "p001", 2, "دهو")]:
        base = d / "train" / book / page
        base.mkdir(parents=True, exist_ok=True)
        (base / f"line_{ln:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (base / f"line_{ln:03d}.gt.txt").write_text(txt, encoding="utf-8")
        (base / f"line_{ln:03d}.norm.txt").write_text(txt, encoding="utf-8")
    (d / "meta.csv").write_text(
        "book_id,script,validation_level\nbook_01,naskh,PV\n", encoding="utf-8")
    return d


def test_parse_arams_yields_expected_fields(tmp_path):
    rows = list(ma.parse_arams(_mini_tree(tmp_path)))
    assert len(rows) == 2
    r = rows[0]
    assert r["book_id"] == "book_01" and r["page_id"] == "p001" and r["line_no"] == 1
    assert r["text"] == "ابج" and r["script"] == "naskh"
    assert r["validation_level"] == "PV" and r["split"] == "train"


def test_kraken_out(tmp_path):
    out = tmp_path / "kraken"
    ma.write_kraken(ma.parse_arams(_mini_tree(tmp_path)), out)
    assert (out / "train" / "book_01" / "p001_001.gt.txt").read_text(encoding="utf-8") == "ابج"
    assert (out / "train" / "book_01" / "p001_001.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_mirror_arams.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `mirror_arams_to_hf.py`**

Provide `parse_arams(root)`, `write_kraken(rows, out_dir)`, and `main()`. Add `tests/scripts_import.py` with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mirror_arams_to_hf as mirror_arams   # noqa: F401
```

`parse_arams` walks `root/<split>/<book>/<page>/line_<n>.png` + `.gt.txt` + `.norm.txt`, joins `meta.csv` for `script`/`validation_level`, derives `alignment_confidence` from whatever per-line confidence the archive provides (default `100` if none). `write_kraken` copies the png and writes `<page>_<line:03d>.gt.txt`. `main` does `--dry-run` (counts + sample) vs the real `datasets.Dataset(...).push_to_hub`.

**Before writing the real parser: extract `AraMS-28k-HTR.zip`, `find . -maxdepth 4 | head -50`, and rewrite `_mini_tree` + `parse_arams` to match the actual layout. Record the real layout in a comment at the top of the file.**

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/test_mirror_arams.py -v`
Expected: PASS.

- [ ] **Step 5: Dry-run against the real archive (once downloaded)**

Run: `uv run python scripts/mirror_arams_to_hf.py --zip .cache/arams28k-htr/AraMS-28k-HTR.zip --dry-run`
Expected: prints `train 19739 / val 1486 / test 6746` (or the archive's real numbers) and one sample record. **If the split sizes or the paired-text/confidence/PV-LV fields differ from the spec's assumptions, STOP and report — arms 0.4/0.5 depend on them.**

- [ ] **Step 6: Commit**

```bash
git add scripts/mirror_arams_to_hf.py tests/test_mirror_arams.py tests/scripts_import.py
git commit -m "mirror_arams: Zenodo AraMS-28k-HTR -> HF dataset + Kraken tree (parser + dry-run)"
```

---

### Task 7: `datasets/arams.py` adapter + wire into configs

**Files:**
- Create: `src/arabic_ocr_bench/datasets/arams.py`
- Modify: `configs/datasets.yaml` (add `arams28k`)
- Modify: `configs/corpora.yaml` (fill `arams28k.books` from the archive)
- Test: `tests/test_arams_adapter.py`

**Interfaces:**
- Consumes: the HF mirror repo (`rabeeh97/arams28k-htr-lines`) via `datasets.load_dataset`.
- Produces:
  - `class Arams` with `name = "arams28k"`, `track = "line"`, `__init__(self, split="test", limit=None, offset=0, indices=None, hf_id="rabeeh97/arams28k-htr-lines")`, `__len__`, `__iter__` yielding `Sample(id=f"arams28k/{split}/{row:06d}", image=<PIL RGB, cap 1024>, ground_truth=row["text"], track="line", meta={...})` where `meta` carries `book_id, page_id, line_no, script, alignment_confidence, validation_level, text_diac_normalised, row, split, source=hf_id`.
  - Mirrors `MuharafLines`' streaming-vs-download logic (`_STREAM_MAX = 300`).
- Registered in `configs/datasets.yaml`:

```yaml
arams28k:
  adapter: arabic_ocr_bench.datasets.arams:Arams
  hf_id: rabeeh97/arams28k-htr-lines
  track: line
  split: test
  license: CC-BY-NC-SA-4.0
  source: https://doi.org/10.5281/zenodo.21499649
  card: >-
    AraMS-28k-HTR — 27,971 real historical Arabic manuscript lines (Maghribi /
    Naskh / Ruq'ah), book-level train/val/test split, paired diacritised +
    diacritic-normalised transcription and per-line alignment confidence.
```

- [ ] **Step 1: Write the failing test (monkeypatch `load_dataset`)**

```python
# tests/test_arams_adapter.py
from PIL import Image
from arabic_ocr_bench.datasets.arams import Arams


class _FakeDS(list):
    def select(self, idx): return _FakeDS(self[i] for i in idx)


def _fake_rows():
    return _FakeDS([
        {"image": Image.new("RGB", (10, 6)), "text": "ابج", "text_diac_normalised": "ابج",
         "book_id": "book_03", "page_id": "p0172", "line_no": 7,
         "alignment_confidence": 84, "validation_level": "LV", "script": "maghribi"},
    ])


def test_arams_sample_shape(monkeypatch):
    monkeypatch.setattr("arabic_ocr_bench.datasets.arams.load_dataset",
                        lambda *a, **k: _fake_rows())
    s = next(iter(Arams(split="test", indices=[0])))
    assert s.id == "arams28k/test/000000"
    assert s.ground_truth == "ابج"
    assert s.meta["book_id"] == "book_03"
    assert s.meta["alignment_confidence"] == 84
    assert s.meta["validation_level"] == "LV"
    assert s.meta["text_diac_normalised"] == "ابج"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_arams_adapter.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `arams.py`**

Copy `muharaf_lines.py`'s structure; add a `from datasets import load_dataset` module-level import (so the test can monkeypatch it), and build `meta` from the extra columns. `id` uses the absolute row index like `MuharafLines`.

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/test_arams_adapter.py tests/test_canonical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/datasets/arams.py configs/datasets.yaml tests/test_arams_adapter.py
git commit -m "datasets: arams28k adapter (streams the HF mirror, carries book/PV-LV/conf meta)"
```

---

### Task 8: End-to-end — real AraMS mirror + canonical + charset, and the Tier-0 Kraken export

**Files:**
- Modify: `configs/corpora.yaml` (`arams28k.books` real table)
- Modify: `docs/superpowers/specs/2026-09-07-canonical-schema-design.md` (tick the AraMS assumptions, or record deviations)

**Interfaces:** none new — this task runs the pipeline and captures outputs.

- [ ] **Step 1: Push the AraMS HF mirror (needs the 2.4 GB zip + HF write token)**

Run: `uv run python scripts/mirror_arams_to_hf.py --zip .cache/arams28k-htr/AraMS-28k-HTR.zip --repo rabeeh97/arams28k-htr-lines`
Expected: repo populated; `uv run hf download rabeeh97/arams28k-htr-lines --repo-type dataset` resolves.

- [ ] **Step 2: Fill `configs/corpora.yaml` `arams28k.books`**

From the archive's `meta.csv` (or equivalent), write one `book_NN: {script: …}` line per book. Commit.

```bash
git add configs/corpora.yaml
git commit -m "corpora: real arams28k per-book script table"
```

- [ ] **Step 3: Build canonical for AraMS + the six 0.2 corpora, with the Kraken tree**

Run:
```
uv run python scripts/build_canonical.py --only arams28k --kraken-out data/kraken/arams28k
uv run python scripts/build_canonical.py --only arams28k,muharaf_lines,rasam1,rasam2,nakba_lines,kitab_khatt --no-crops
```
Expected: `data/canonical/arams28k.jsonl` line count ≈ 27,971; `data/kraken/arams28k/{train,val,test}/…` populated.

- [ ] **Step 4: Build the union charset for 0.2 and an AraMS-only charset for 0.1**

Run:
```
uv run python scripts/build_charset.py --corpora arams28k --level l3
uv run python scripts/build_charset.py --corpora arams28k,muharaf_lines,rasam1,rasam2,nakba_lines,kitab_khatt --level l3
```
Expected: `_charset.json` written; note the `union` size and `train_test_gap` for the Tier-0 spec.

- [ ] **Step 5: Full test run + record outputs**

Run: `uv run python -m pytest -q`
Expected: all green.

- [ ] **Step 6: Update the spec's "Open inputs" / "AraMS-28k-HTR specifics"**

Tick each assumption that held; for any that didn't (split sizes, paired text, confidence, PV/LV), write the real behaviour and flag which Tier-0 arm is affected. Commit.

```bash
git add docs/superpowers/specs/2026-09-07-canonical-schema-design.md
git commit -m "spec: AraMS assumptions verified against the real archive"
```

---

## Self-Review

**1. Spec coverage:**
- Canonical record + `to_canonical` + `diacritic_density` → Task 2. ✓
- `line_id` both forms → Task 2 (`_line_id`), tested. ✓
- `text_diac_normalised`, `validation_level`, per-line `alignment_confidence`, per-book `script` → Task 2 (meta passthrough) + Task 7 (adapter emits them) + Task 3 (`books` table). ✓
- `normalize.py` `standard` + `diac_full`, `strip_harakat(marks=)` → Task 1. ✓
- `raw`/`light`/`diac_insensitive` byte-identical → Task 1 Step 1 snapshot test + Step 4 full suite. ✓
- `configs/corpora.yaml` + loader + "every wired corpus has an entry" test → Task 3. ✓
- `build_canonical.py` (`--only`, `--no-crops`, `--kraken-out`, `--force`, idempotence, `_manifest.json`, non-zero exit) → Task 4. ✓
- `build_charset.py` (`--level`, `--corpora`, `--min-count`, global/per-corpus/per-split/union/train-test-gap, no line dropping) → Task 5. ✓
- AraMS HF mirror + Kraken export → Task 6 + Task 8. ✓
- `datasets/arams.py` + `datasets.yaml` wiring → Task 7. ✓
- Crop cache (line-crop corpora → `crops/<corpus>/<line_id>.png`) → Task 4 Step 3. ✓
- Tier-0 "consumed by" (0.1 Kraken tree + AraMS charset; 0.2 six-corpus union charset) → Task 8 Steps 3-4. ✓
- Muharaf split note → no code; documented in spec, `split_unit: provider` in Task 3 yaml. ✓
- `training/crops.py`, contamination matrix, `report.py` additions → explicitly deferred in spec; not in this plan. ✓
- Gap: the spec's `tests/test_normalize.py` claim "`diac_full` strips a Qur'anic mark that `diac_insensitive` keeps" — covered by Task 1 Step 1 `test_diac_full_strips_quranic_mark_that_diac_insensitive_keeps`. ✓

**2. Placeholder scan:** Task 6 deliberately defers the exact archive layout to implementation time (the archive isn't downloaded yet) but gives a concrete fixture, a concrete `--dry-run` expectation, and an explicit "STOP and report if it differs" gate — this is a real instruction, not a "TODO". No other placeholders.

**3. Type consistency:** `corpus_meta` dict keys produced in Task 3 (`corpus`, `script`/`books`, `era`, `genre`, `split_unit`, `license`, `hf_id`, `split`) match what `to_canonical` reads in Task 2. `Sample.meta` keys emitted by `arams.py` in Task 7 (`book_id`, `page_id`, `line_no`, `script`, `alignment_confidence`, `validation_level`, `text_diac_normalised`, `row`, `split`) match the per-line overrides `to_canonical` reads in Task 2. `build_corpus` / `build_charset` signatures are referenced identically in their tests and `main()`. `OUT` is the module-level canonical dir in Tasks 4 and 5, monkeypatched in tests.

# Benchmark Findings & Analysis Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the stored benchmark results into insight — a computed `findings.json`, a new *Findings & analysis* tab in the published leaderboard with hand-rolled SVG/CSS visualisations, and a companion notebook.

**Architecture:** A new pure-Python package `src/arabic_ocr_bench/analysis/` with a single-pass alignment engine that emits one compact `AlignmentRecord` per prediction; every finding is a pure reduction over `list[AlignmentRecord]` plus the existing per-sample `scores.jsonl`. An orchestrator assembles `findings.json`; `make_leaderboard.py` and `notebooks/09_findings.ipynb` both consume that one artifact. Charts in the leaderboard are hand-rolled SVG + CSS + vanilla JS (zero external dependencies).

**Tech Stack:** Python 3, `jiwer` (character/word alignment — already a dep), `numpy` (new explicit core dep; bootstrap + Bradley–Terry), `pandas` (already a dep), hand-written SVG/CSS/JS in the leaderboard, matplotlib in the notebook.

**Spec:** `docs/superpowers/specs/2026-09-07-benchmark-findings-analysis-design.md`

## Global Constraints

- Python `>=3.11` (match the existing `pyproject.toml` floor); `from __future__ import annotations` at the top of every new module, matching the codebase.
- New dependency: `numpy` added to `[project].dependencies` in `pyproject.toml`. **No scipy**, no seaborn in core deps (seaborn is an optional `pip install` inside the notebook only).
- All analyzer functions are **pure** — no I/O except in `loader.py` and `findings.py`. No network. No matplotlib import anywhere under `src/`.
- `findings.py` uses a fixed RNG seed `1234` so `findings.json` is byte-reproducible for a fixed input.
- `findings.json` must be JSON-serialisable with the stdlib `json` module — **no numpy scalars or arrays** leak into the output (cast with `float()` / `int()` / `.tolist()`).
- Leaderboard charts: **no external JS/CSS/font resources beyond what `_HEAD` already loads.** Hand-rolled SVG/CSS/vanilla-JS only.
- Arabic strings in new Python source are written as literals (the codebase already does this in tests); Arabic *identifiers* are never used. Comments may contain Arabic.
- The published page must contain no `U+FFFD`; the existing scrub in `make_leaderboard.main()` covers the whole fragment and must keep doing so.
- Default per-section row caps in `findings.py`: 40 confusion pairs, 30 troublesome letters, 25 rewrite patterns, 30 samples per difficulty list, 12 length bins.
- Commit after every task with a `feat:` / `test:` / `refactor:` prefix; end every commit message with the `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer.
- Run the full suite (`uv run python -m pytest`) green before each commit.

---

## File Structure

**New — analysis package (`src/arabic_ocr_bench/analysis/`):**

| file | responsibility |
| --- | --- |
| `__init__.py` | empty marker |
| `loader.py` | read + dedupe predictions/scores across `results/*/`; `PredRow` dataclass; `iter_pred_rows()` |
| `align.py` | `RASM_MAP`, `LETTER_GROUPS`, `to_rasm()`, `letter_group()`, `AlignmentRecord`, `build_record()` |
| `refusal.py` | `classify_failure()` — the ordered failure-class ruleset |
| `charstats.py` | `error_anatomy()` + `script_letters()` reductions |
| `diacritics.py` | `diacritics()` reduction |
| `difficulty.py` | `difficulty()` reduction |
| `modelcmp.py` | `model_landscape()` reduction (agreement, clustering, bootstrap, ranks, win-rate, Bradley–Terry, metric-correlation, profile-decomposition) |
| `operational.py` | `operational()` reduction (Pareto, latency, failure-cause, retries, coverage) |
| `corpus.py` | `corpus_health()` reduction (GT charset/Zipf/density/length, leaderboard churn, contamination cross-ref) |
| `findings.py` | `build(results_dir, *, seed=1234) -> dict`; `main()` for the CLI |

**New — leaderboard rendering:**

| file | responsibility |
| --- | --- |
| `scripts/_leaderboard_findings.py` | chart primitives (`svg_heatmap`, `hbar`, `stacked_hbar`, `diverging_hbar`, `svg_scatter`, `binned_curve`, `histogram`, `ecdf`, `box_row`, `dumbbell`, `forest`), the findings CSS block, the 7 `_render_*` section functions, `render_findings_view()` |

**New — notebook + tests:**

| file | responsibility |
| --- | --- |
| `tests/fixtures/mini_results/` | hand-built 2-run results tree with planted phenomena |
| `tests/fixtures/mini_findings.json` | golden output of `build()` on the fixture |
| `tests/analysis/test_loader.py`, `test_align.py`, `test_refusal.py` *(new path)*, `test_charstats.py`, `test_diacritics.py`, `test_difficulty.py`, `test_modelcmp.py`, `test_operational.py`, `test_corpus.py`, `test_findings_golden.py` | per-module unit tests |
| `tests/test_leaderboard_findings.py` | chart-primitive + view-render tests |

**Modified:**

| file | change |
| --- | --- |
| `pyproject.toml` | add `numpy` to core deps |
| `src/arabic_ocr_bench/cli.py` | add the `analyze` subcommand |
| `scripts/make_leaderboard.py` | lift the run-merge helper into `analysis.loader`; build/cache `findings.json`; add the tab bar + view split; call `render_findings_view()` |
| `scripts/build_notebooks.py` | add `build_findings()`; call it from the build loop; add a pointer line to `build_analysis()` |
| `notebooks/09_findings.ipynb` | generated output (committed, empty cells) |
| `notebooks/03_analysis.ipynb` | regenerated with the pointer line |

**Note on `tests/test_refusal.py`:** a file of that name already exists (it tests `models.litellm_client.looks_like_refusal`). The new analysis test goes at `tests/analysis/test_refusal.py` — a different path — so both coexist. Create `tests/analysis/__init__.py`.

---

## Task 1: Fixture + `loader.py`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/__init__.py` (empty)
- Create: `src/arabic_ocr_bench/analysis/loader.py`
- Create: `tests/analysis/__init__.py` (empty)
- Create: `tests/analysis/test_loader.py`
- Create fixture tree under `tests/fixtures/mini_results/` (see Step 1)
- Modify: `pyproject.toml` (add `numpy`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PredRow` — a frozen dataclass with fields:
    `model: str`, `dataset: str`, `sample_id: str`, `gt: str`, `pred: str`,
    `error: str | None`, `attempts: int`, `latency_ms: float`, `cost_usd: float`,
    `input_tokens: int`, `output_tokens: int`, `cached: bool`,
    `scores: dict` (the parsed `scores.jsonl` row for this `(model, sample_id)`, or `{}` if none).
  - `iter_pred_rows(results_dir: str | Path) -> list[PredRow]` — every distinct
    `(model, sample_id)` across all `results/*/` run dirs (or a single run dir),
    deduped so a later successful retry supersedes an earlier errored attempt,
    otherwise the first successful reading is kept. `synthetic` dataset and
    `mock*` models are excluded.
  - `run_dirs(results_dir: str | Path) -> list[Path]` — the run dirs under
    `results_dir` (or `[results_dir]` if it is itself a run dir), detected by the
    presence of `predictions.jsonl`.

- [ ] **Step 1: Build the fixture tree**

Create `tests/fixtures/mini_results/` with **two** run dirs. Use ASCII where the
test does not care about script, real Arabic where it does. Plant:
a ب↔ت confusion, one refusal, one empty output, one rate-limit error, one
bad-GT line (all models agree with each other, disagree with GT), one line
perfect for all models.

`tests/fixtures/mini_results/run_a/predictions.jsonl` (one JSON object per line):

```jsonc
{"sample_id":"dsx/test/000000","model":"m1","dataset":"dsx","track":"line","prompt_version":"line-v2","ground_truth":"بيت كبير","raw_text":"بيت كبير","cleaned_text":"بيت كبير","latency_ms":900.0,"input_tokens":1000,"output_tokens":8,"cost_usd":0.001,"error":null,"attempts":1,"cached":false}
{"sample_id":"dsx/test/000001","model":"m1","dataset":"dsx","track":"line","prompt_version":"line-v2","ground_truth":"توت","raw_text":"بوب","cleaned_text":"بوب","latency_ms":800.0,"input_tokens":1000,"output_tokens":3,"cost_usd":0.001,"error":null,"attempts":1,"cached":false}
{"sample_id":"dsx/test/000002","model":"m1","dataset":"dsx","track":"line","prompt_version":"line-v2","ground_truth":"سلام عليكم","raw_text":"I'm sorry, I can't read this image.","cleaned_text":"I'm sorry, I can't read this image.","latency_ms":700.0,"input_tokens":1000,"output_tokens":10,"cost_usd":0.001,"error":"refusal","attempts":1,"cached":false}
{"sample_id":"dsx/test/000003","model":"m1","dataset":"dsx","track":"line","prompt_version":"line-v2","ground_truth":"كتاب جديد","raw_text":"","cleaned_text":"","latency_ms":650.0,"input_tokens":1000,"output_tokens":0,"cost_usd":0.001,"error":null,"attempts":1,"cached":false}
{"sample_id":"dsx/test/000004","model":"m1","dataset":"dsx","track":"line","prompt_version":"line-v2","ground_truth":"القاهرة اليوم","raw_text":"القاهرة اليوم","cleaned_text":"القاهرة اليوم","latency_ms":1200.0,"input_tokens":1000,"output_tokens":9,"cost_usd":0.001,"error":null,"attempts":1,"cached":false}
{"sample_id":"dsy/test/000000","model":"m1","dataset":"dsy","track":"line","prompt_version":"line-v2","ground_truth":"مدرسة","raw_text":"","cleaned_text":"","latency_ms":0.0,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,"error":"RateLimitError: 429 Rate limit exceeded","attempts":4,"cached":false}
{"sample_id":"dsy/test/000001","model":"m1","dataset":"dsy","track":"line","prompt_version":"line-v2","ground_truth":"شمس","raw_text":"سمس","cleaned_text":"سمس","latency_ms":500.0,"input_tokens":900,"output_tokens":3,"cost_usd":0.0009,"error":null,"attempts":1,"cached":false}
```

Add analogous rows for models `m2` and `m3` in the same file (same 7
`sample_id`s). For `dsx/test/000003` (the **bad-GT** line) give every model the
same reading `"كتاب قديم"` (they agree with each other, differ from GT
`"كتاب جديد"` by one letter). For `dsx/test/000004` (the **perfect** line) give
every model exactly the GT. Vary the others so per-model CER differs.

`tests/fixtures/mini_results/run_a/scores.jsonl` — one row per **non-errored**
`(model, sample_id)`, matching the `score_pair` output shape already in
`results/*/scores.jsonl` (`cer_raw`, `cer_light`, `cer_diac_insensitive`,
`wer_*`, `mer_*`, `wil_*`, `wip_*`, `exact_*`, `diac_precision/recall/f1`,
`latency_ms`, `cost_usd`, `error`, `dataset`, `model`, `sample_id`). Generate
these by calling the real scorer so they are self-consistent:

```python
# tests/fixtures/mini_results/_build_scores.py  (committed; documents provenance)
import json, pathlib
from arabic_ocr_bench.metrics import score_pair

for run in ("run_a", "run_b"):
    d = pathlib.Path(__file__).parent / run
    preds = [json.loads(l) for l in (d / "predictions.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for p in preds:
        if p["error"]:
            continue
        s = score_pair(p["ground_truth"], p["cleaned_text"])
        row = {"sample_id": p["sample_id"], "model": p["model"], "dataset": p["dataset"],
               "error": None, "latency_ms": p["latency_ms"], "cost_usd": p["cost_usd"]}
        for prof in ("raw", "light", "diac_insensitive"):
            row[f"cer_{prof}"] = s["cer"][prof]
            row[f"car_{prof}"] = s["car"][prof]
            row[f"wer_{prof}"] = s["wer"][prof]
            row[f"mer_{prof}"] = s["mer"][prof]
            row[f"wil_{prof}"] = s["wil"][prof]
            row[f"wip_{prof}"] = s["wip"][prof]
            row[f"exact_{prof}"] = s["exact_match"][prof]
        row["diac_precision"] = s["diacritic"]["precision"]
        row["diac_recall"] = s["diacritic"]["recall"]
        row["diac_f1"] = s["diacritic"]["f1"]
        out.append(row)
    (d / "scores.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
```

`run_b` holds a **retry**: `m2` on `dsy/test/000000` now succeeds
(`cleaned_text":"مدرسه"`, `error":null`) — the loader must prefer this over
`run_a`'s rate-limit error for `m2`, while `m1` and `m3` keep the error.
`run_b` also adds a wholly new line `dsx/test/000005` for `m1` only.

Add `tests/fixtures/mini_results/run_a/run_config.snapshot.yaml` and
`run_b/run_config.snapshot.yaml` with `prompt_version: line-v2` and
`limit: 8`.

- [ ] **Step 2: Write the failing test**

```python
# tests/analysis/test_loader.py
"""iter_pred_rows dedupes across runs; a later success beats an earlier error."""
from pathlib import Path

from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows, run_dirs

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def test_run_dirs_finds_both():
    assert sorted(p.name for p in run_dirs(FIX)) == ["run_a", "run_b"]


def test_one_row_per_model_sample():
    rows = iter_pred_rows(FIX)
    keys = [(r.model, r.sample_id) for r in rows]
    assert len(keys) == len(set(keys))


def test_later_success_supersedes_earlier_error():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    assert rows[("m2", "dsy/test/000000")].error is None
    assert rows[("m2", "dsy/test/000000")].pred == "مدرسه"
    assert rows[("m1", "dsy/test/000000")].error is not None  # kept


def test_new_line_from_run_b_present():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    assert ("m1", "dsx/test/000005") in rows


def test_scores_attached_for_ok_rows():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    r = rows[("m1", "dsx/test/000001")]
    assert r.scores["cer_raw"] > 0
    assert isinstance(r, PredRow)


def test_mock_and_synthetic_excluded(tmp_path):
    # a run dir with only mock/synthetic rows yields nothing
    d = tmp_path / "r"
    d.mkdir()
    (d / "predictions.jsonl").write_text(
        '{"sample_id":"synthetic/t/0","model":"m1","dataset":"synthetic",'
        '"ground_truth":"x","cleaned_text":"x","error":null,"attempts":1,'
        '"latency_ms":1,"cost_usd":0,"input_tokens":0,"output_tokens":0,"cached":false}\n'
        '{"sample_id":"dsx/t/0","model":"mock-echo","dataset":"dsx",'
        '"ground_truth":"x","cleaned_text":"x","error":null,"attempts":1,'
        '"latency_ms":1,"cost_usd":0,"input_tokens":0,"output_tokens":0,"cached":false}\n'
    )
    assert iter_pred_rows(d) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/analysis/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: arabic_ocr_bench.analysis`.

- [ ] **Step 4: Add numpy dep**

In `pyproject.toml`, under `[project].dependencies`, add `"numpy>=1.26",`
(next to `"pandas>=2.0",`). Run `uv sync`.

- [ ] **Step 5: Implement `loader.py`**

Port the dedupe logic from `make_leaderboard.py._merge_runs` /
`collect_review` (both merge predictions across runs with the
"later-success-supersedes-error" rule). Keep it a plain function, not tied to
`summary.csv`.

```python
# src/arabic_ocr_bench/analysis/loader.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PredRow:
    model: str
    dataset: str
    sample_id: str
    gt: str
    pred: str
    error: str | None
    attempts: int
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cached: bool
    scores: dict = field(default_factory=dict)


def run_dirs(results_dir: str | Path) -> list[Path]:
    p = Path(results_dir)
    if (p / "predictions.jsonl").exists():
        return [p]
    return sorted(d.parent for d in p.glob("*/predictions.jsonl"))


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def iter_pred_rows(results_dir: str | Path) -> list[PredRow]:
    merged: dict[tuple[str, str], dict] = {}
    scores: dict[tuple[str, str], dict] = {}
    for rd in run_dirs(results_dir):
        for s in _jsonl(rd / "scores.jsonl"):
            scores.setdefault((s.get("model", ""), s.get("sample_id", "")), s)
        for p in _jsonl(rd / "predictions.jsonl"):
            model, sid = p.get("model", ""), p.get("sample_id", "")
            ds = p.get("dataset", "")
            if not model or model.startswith("mock") or ds == "synthetic":
                continue
            prev = merged.get((model, sid))
            if prev is None or (prev.get("error") and not p.get("error")):
                merged[(model, sid)] = p
    out: list[PredRow] = []
    for (model, sid), p in merged.items():
        out.append(PredRow(
            model=model, dataset=p.get("dataset", ""), sample_id=sid,
            gt=p.get("ground_truth", ""), pred=p.get("cleaned_text") or "",
            error=p.get("error"), attempts=int(p.get("attempts", 1) or 1),
            latency_ms=float(p.get("latency_ms", 0.0) or 0.0),
            cost_usd=float(p.get("cost_usd", 0.0) or 0.0),
            input_tokens=int(p.get("input_tokens", 0) or 0),
            output_tokens=int(p.get("output_tokens", 0) or 0),
            cached=bool(p.get("cached", False)),
            scores=scores.get((model, sid), {}) if not p.get("error") else {},
        ))
    return out
```

- [ ] **Step 6: Run the tests**

Run: `uv run python -m pytest tests/analysis/test_loader.py -v`
Expected: PASS (all 6).

- [ ] **Step 7: Full suite + commit**

```bash
uv run python -m pytest -q
git add pyproject.toml src/arabic_ocr_bench/analysis/ tests/analysis/ tests/fixtures/mini_results/
git commit -m "feat: analysis.loader — dedupe predictions across runs + mini fixture

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: `refusal.py` — failure classification

**Files:**
- Create: `src/arabic_ocr_bench/analysis/refusal.py`
- Create: `tests/analysis/test_refusal.py`

**Interfaces:**
- Consumes: `arabic_ocr_bench.models.litellm_client.looks_like_refusal` (existing),
  `arabic_ocr_bench.prompting.get_prompt` (existing).
- Produces:
  `classify_failure(pred: str, error: str | None, attempts: int, *, prompt_version: str = "line-v2") -> str`
  returning one of:
  `"ok"`, `"empty"`, `"refusal"`, `"meta_commentary"`, `"wrong_script"`,
  `"prompt_echo"`, `"rate_limited"`, `"other_error"`.
  Rule order (first match wins): rate_limited → other_error → empty →
  wrong_script → meta_commentary → prompt_echo → ok.
  `FAILURE_CLASSES: tuple[str, ...]` — the eight labels in a canonical order for
  stable chart ordering.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_refusal.py
from arabic_ocr_bench.analysis.refusal import FAILURE_CLASSES, classify_failure


def test_rate_limited_from_error():
    assert classify_failure("", "RateLimitError: 429 Rate limit exceeded", 4) == "rate_limited"


def test_other_error():
    assert classify_failure("", "TimeoutError: deadline", 1) == "other_error"


def test_empty_output_no_error():
    assert classify_failure("", None, 1) == "empty"
    assert classify_failure(" \n", None, 1) == "empty"


def test_wrong_script_long_latin():
    assert classify_failure("The image shows some handwriting in cursive", None, 1) == "wrong_script"


def test_short_arabic_with_one_latin_token_is_ok():
    assert classify_failure("رقم PDF هنا", None, 1) == "ok"


def test_meta_commentary_arabic():
    assert classify_failure("عذراً، لا يمكنني قراءة هذه الصورة بوضوح", None, 1) == "meta_commentary"


def test_code_fence_is_meta():
    assert classify_failure("```\nبسم الله\n```", None, 1) == "meta_commentary"


def test_prompt_echo():
    from arabic_ocr_bench.prompting import get_prompt
    assert classify_failure(get_prompt("line-v2"), None, 1) == "prompt_echo"


def test_plain_transcription_is_ok():
    assert classify_failure("بسم الله الرحمن الرحيم", None, 1) == "ok"


def test_classes_constant_has_eight():
    assert len(FAILURE_CLASSES) == 8 and "ok" in FAILURE_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/analysis/test_refusal.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `refusal.py`**

```python
# src/arabic_ocr_bench/analysis/refusal.py
from __future__ import annotations

import re

from ..models.litellm_client import looks_like_refusal
from ..prompting import get_prompt

FAILURE_CLASSES = (
    "ok", "empty", "refusal", "meta_commentary",
    "wrong_script", "prompt_echo", "rate_limited", "other_error",
)

_RATE_RE = re.compile(r"\b429\b|rate.?limit", re.I)
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_META_RE = re.compile(
    r"(i (cannot|can'?t|am unable)|as an ai|the (text|image) (says|reads|shows)"
    r"|here is the|لا (أستطيع|يمكنني)|عذرا|يرجى تقديم)",
    re.I,
)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta) if ta else 0.0


def classify_failure(
    pred: str, error: str | None, attempts: int, *, prompt_version: str = "line-v2"
) -> str:
    if error and _RATE_RE.search(error):
        return "rate_limited"
    if error:
        return "other_error"
    s = (pred or "").strip()
    if len(s) < 2:
        return "empty"
    nonspace = [c for c in s if not c.isspace()]
    if len(s) >= 8 and nonspace:
        latin = sum(bool(_LATIN_RE.match(c)) for c in nonspace)
        if latin / len(nonspace) > 0.5:
            return "wrong_script"
    if "```" in pred or _META_RE.search(s) or looks_like_refusal(s):
        # a genuine refusal (no Arabic content) vs. meta-commentary that still
        # contains a transcription attempt
        return "refusal" if not _ARABIC_RE.search(s) else "meta_commentary"
    try:
        tmpl = get_prompt(prompt_version)
    except ValueError:
        tmpl = ""
    if tmpl and _token_overlap(tmpl, s) >= 0.8:
        return "prompt_echo"
    return "ok"
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m pytest tests/analysis/test_refusal.py -v`
Expected: PASS. If `test_meta_commentary_arabic` fails because the sentence has
Arabic and so routes to `meta_commentary` — that is the intended label; adjust
the assertion only if the rule genuinely misfires.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/refusal.py tests/analysis/test_refusal.py
git commit -m "feat: analysis.refusal — ordered failure-class ruleset

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: `align.py` — rasm table + letter groups

**Files:**
- Create: `src/arabic_ocr_bench/analysis/align.py` (first half — data + helpers)
- Create: `tests/analysis/test_align.py` (first tests)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RASM_MAP: dict[str, str]` — dot-bearing letter → undotted skeleton.
  - `to_rasm(s: str) -> str` — apply `RASM_MAP` char-by-char (chars not in the
    map pass through). Caller is expected to have already `light`-normalised and
    stripped harakat.
  - `LETTER_GROUPS: dict[str, tuple[str, ...]]` — group name → members.
  - `letter_group(ch: str) -> str | None` — the first group `ch` belongs to, or
    `None`.
  - `ARABIC_LETTER_RE` — compiled `[ء-ي]` for "is this an Arabic
    letter" checks.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_align.py
from arabic_ocr_bench.analysis.align import (
    LETTER_GROUPS,
    RASM_MAP,
    letter_group,
    to_rasm,
)


def test_rasm_collapses_dotted_family():
    # ب ت ث ن ي all share the same rasm -> to_rasm makes them equal
    skeletons = {to_rasm(c) for c in "بتثني"}
    assert len(skeletons) == 1


def test_rasm_passes_through_unmapped():
    assert to_rasm("ا") == "ا"
    assert to_rasm("م") == "م"


def test_rasm_string():
    assert to_rasm("توت") == to_rasm("بوب")  # both -> <skeleton>و<skeleton>


def test_letter_group_dot_confusable():
    assert letter_group("ت") == "dot_confusable"
    assert letter_group("ب") == "dot_confusable"


def test_letter_group_hamza_carrier():
    assert letter_group("ؤ") == "hamza_carrier"


def test_letter_group_unknown_is_none():
    assert letter_group("م") is None


def test_groups_are_nonempty():
    assert all(v for v in LETTER_GROUPS.values())
    assert "dot_confusable" in LETTER_GROUPS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/analysis/test_align.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the data half of `align.py`**

```python
# src/arabic_ocr_bench/analysis/align.py
from __future__ import annotations

import re

ARABIC_LETTER_RE = re.compile(r"[ء-ي]")

# dot-bearing letter -> undotted skeleton (rasm). Skeleton code points are the
# real "dotless" Arabic letters where Unicode has them (ٮ ٯ ٰ …), else a
# representative member of the family used purely as a bucket key.
RASM_MAP = {
    "ب": "ٮ", "ت": "ٮ", "ث": "ٮ", "ن": "ٮ", "ي": "ٮ", "ى": "ٮ",
    "ج": "ح", "خ": "ح",
    "ذ": "د",
    "ز": "ر",
    "ش": "س",
    "ض": "ص",
    "ظ": "ط",
    "غ": "ع",
    "ف": "ٯ", "ق": "ٯ",
    "ة": "ه",
    "ئ": "ٮ", "ؤ": "و", "أ": "ا", "إ": "ا", "آ": "ا",
}

LETTER_GROUPS = {
    "dot_confusable": tuple("بتثنيىقفجحخزرشسضصطظعغ"),
    "hamza_carrier": tuple("ءأإآؤئ"),
    "alef_form": tuple("اأإآٱ"),
    "taa_haa": tuple("ةه"),
    "yaa_alefmaqsura": tuple("يى"),
    "sun_letter": tuple("تثدذرزسشصضطظلن"),
    "moon_letter": tuple("ابجحخعغفقكمهوي"),
}


def to_rasm(s: str) -> str:
    return "".join(RASM_MAP.get(c, c) for c in s)


def letter_group(ch: str) -> str | None:
    for name, members in LETTER_GROUPS.items():
        if ch in members:
            return name
    return None
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m pytest tests/analysis/test_align.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/align.py tests/analysis/test_align.py
git commit -m "feat: analysis.align — rasm skeleton map + Arabic letter groups

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: `align.py` — `AlignmentRecord` + `build_record()`

**Files:**
- Modify: `src/arabic_ocr_bench/analysis/align.py` (add the record + builder)
- Modify: `tests/analysis/test_align.py` (add builder tests)

**Interfaces:**
- Consumes: `to_rasm`, `letter_group` (Task 3); `refusal.classify_failure`
  (Task 2); `arabic_ocr_bench.normalize.normalize`, `.strip_harakat`;
  `jiwer.process_characters`, `jiwer.process_words`; `PredRow` (Task 1).
- Produces:
  - `CharOp` — frozen dataclass: `op: str` (`"sub"|"ins"|"del"`), `ref: str`
    (`""` for ins), `hyp: str` (`""` for del), `ref_i: int` (index into the
    light-normalised reference; `-1` for pure ins).
  - `AlignmentRecord` — frozen dataclass with fields exactly as the spec's
    "Fields" table: `model, dataset, sample_id, failure_class, char_ops:
    list[CharOp], word_ops: dict[str,int]` (keys `sub/ins/del/hits`),
    `rasm_ops: dict[str,int]` (keys `sub/ins/del/ref_len`),
    `diac_seq_ref: str, diac_seq_hyp: str, spacing_del: int, spacing_ins: int,
    len_ref_chars/len_hyp_chars/len_ref_words/len_hyp_words: int,
    digit_swaps: int, presentation_form_chars: int, latin_run_len: int,
    has_code_fence: bool, cer_light/cer_raw/cer_rasm/wer_light: float`.
  - `build_record(row: PredRow, *, prompt_version: str = "line-v2") -> AlignmentRecord`
    — for an errored/refused row returns a stub (identifiers + `failure_class`,
    everything else zero/empty). For an OK row runs the single-pass alignment.
  - `expand_char_chunks(ref: str, hyp: str) -> list[CharOp]` — the
    chunk-expansion routine (exposed for direct testing).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/analysis/test_align.py
from arabic_ocr_bench.analysis.align import (
    AlignmentRecord, CharOp, build_record, expand_char_chunks,
)
from arabic_ocr_bench.analysis.loader import PredRow


def _row(gt, pred, **kw):
    d = dict(model="m1", dataset="dsx", sample_id="dsx/test/000000", gt=gt,
             pred=pred, error=None, attempts=1, latency_ms=1.0, cost_usd=0.0,
             input_tokens=0, output_tokens=0, cached=False, scores={})
    d.update(kw)
    return PredRow(**d)


def test_expand_equal_length_substitution():
    ops = expand_char_chunks("abc", "aXc")
    assert ops == [CharOp("sub", "b", "X", 1)]


def test_expand_pure_insertion():
    ops = expand_char_chunks("ab", "abc")
    assert ops == [CharOp("ins", "", "c", -1)]


def test_expand_pure_deletion():
    ops = expand_char_chunks("abc", "ab")
    assert ops == [CharOp("del", "c", "", 2)]


def test_expand_unequal_substitution_spills_to_insert():
    # ref "b" vs hyp "XY": one sub (b->X) then one ins (Y)
    ops = expand_char_chunks("abd", "aXYd")
    assert CharOp("sub", "b", "X", 1) in ops
    assert CharOp("ins", "", "Y", -1) in ops


def test_build_record_stub_for_error():
    r = build_record(_row("مدرسة", "", error="RateLimitError: 429"))
    assert r.failure_class == "rate_limited"
    assert r.char_ops == []
    assert r.len_ref_chars == 0


def test_build_record_bt_confusion_is_substitution():
    r = build_record(_row("توت", "بوب"))
    subs = [(o.ref, o.hyp) for o in r.char_ops if o.op == "sub"]
    assert ("ت", "ب") in subs


def test_build_record_rasm_gap_smaller_than_light():
    # ت<->ب is a dot-only error: CER on the rasm skeleton is 0, light CER > 0
    r = build_record(_row("توت", "بوب"))
    assert r.cer_rasm == 0.0
    assert r.cer_light > 0.0


def test_build_record_diacritic_sequences():
    r = build_record(_row("مَدْرَسَة", "مدرسة"))
    assert r.diac_seq_ref != "" and r.diac_seq_hyp == ""


def test_build_record_spacing_op_counted():
    r = build_record(_row("بيت كبير", "بيتكبير"))
    assert r.spacing_del == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/analysis/test_align.py -v`
Expected: FAIL — `CharOp` / `build_record` missing.

- [ ] **Step 3: Implement the builder**

Key detail — `jiwer.process_characters(ref, hyp)` returns an object with
`.alignment: list[list[AlignmentChunk]]` (one inner list, since a single
pair). Each `AlignmentChunk` has `.type` in
`{"equal","substitute","insert","delete"}` and
`.ref_start_idx/.ref_end_idx/.hyp_start_idx/.hyp_end_idx`. Expansion:

```python
# append to src/arabic_ocr_bench/analysis/align.py
from dataclasses import dataclass

import jiwer

from ..normalize import normalize, strip_harakat
from .refusal import classify_failure

_HARAKAT_RE = re.compile(r"[ً-ٰٟ]")
_PRESENTATION_RE = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
_DIGIT_RE = re.compile(r"[0-9٠-٩۰-۹]")


@dataclass(frozen=True)
class CharOp:
    op: str
    ref: str
    hyp: str
    ref_i: int


def expand_char_chunks(ref: str, hyp: str) -> list[CharOp]:
    out: list[CharOp] = []
    res = jiwer.process_characters(ref, hyp)
    chunks = res.alignment[0] if res.alignment else []
    for ch in chunks:
        if ch.type == "equal":
            continue
        rs, re_, hs, he = ch.ref_start_idx, ch.ref_end_idx, ch.hyp_start_idx, ch.hyp_end_idx
        rseg, hseg = ref[rs:re_], hyp[hs:he]
        if ch.type == "substitute":
            k = min(len(rseg), len(hseg))
            for j in range(k):
                out.append(CharOp("sub", rseg[j], hseg[j], rs + j))
            for j in range(k, len(rseg)):
                out.append(CharOp("del", rseg[j], "", rs + j))
            for j in range(k, len(hseg)):
                out.append(CharOp("ins", "", hseg[j], -1))
        elif ch.type == "delete":
            for j, c in enumerate(rseg):
                out.append(CharOp("del", c, "", rs + j))
        elif ch.type == "insert":
            for c in hseg:
                out.append(CharOp("ins", "", c, -1))
    return out


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    o = jiwer.process_characters(ref, hyp)
    return (o.substitutions + o.deletions + o.insertions) / (
        o.substitutions + o.deletions + o.hits
    )


def _longest_latin_run(s: str) -> int:
    best = cur = 0
    for c in s:
        cur = cur + 1 if ("a" <= c.lower() <= "z") else 0
        best = max(best, cur)
    return best


@dataclass(frozen=True)
class AlignmentRecord:
    model: str
    dataset: str
    sample_id: str
    failure_class: str
    char_ops: list
    word_ops: dict
    rasm_ops: dict
    diac_seq_ref: str
    diac_seq_hyp: str
    spacing_del: int
    spacing_ins: int
    len_ref_chars: int
    len_hyp_chars: int
    len_ref_words: int
    len_hyp_words: int
    digit_swaps: int
    presentation_form_chars: int
    latin_run_len: int
    has_code_fence: bool
    cer_light: float
    cer_raw: float
    cer_rasm: float
    wer_light: float


def _stub(row, fc: str) -> AlignmentRecord:
    return AlignmentRecord(
        model=row.model, dataset=row.dataset, sample_id=row.sample_id,
        failure_class=fc, char_ops=[], word_ops={"sub": 0, "ins": 0, "del": 0, "hits": 0},
        rasm_ops={"sub": 0, "ins": 0, "del": 0, "ref_len": 0},
        diac_seq_ref="", diac_seq_hyp="", spacing_del=0, spacing_ins=0,
        len_ref_chars=0, len_hyp_chars=0, len_ref_words=0, len_hyp_words=0,
        digit_swaps=0, presentation_form_chars=0, latin_run_len=0,
        has_code_fence=False, cer_light=0.0, cer_raw=0.0, cer_rasm=0.0, wer_light=0.0,
    )


def build_record(row, *, prompt_version: str = "line-v2") -> AlignmentRecord:
    fc = classify_failure(row.pred, row.error, row.attempts, prompt_version=prompt_version)
    if fc != "ok":
        return _stub(row, fc)

    g_light, p_light = normalize(row.gt, "light"), normalize(row.pred, "light")
    g_raw, p_raw = normalize(row.gt, "raw"), normalize(row.pred, "raw")
    char_ops = expand_char_chunks(g_light, p_light)

    w = jiwer.process_words(g_light, p_light)
    word_ops = {"sub": w.substitutions, "ins": w.insertions, "del": w.deletions, "hits": w.hits}

    g_rasm = to_rasm(strip_harakat(g_light))
    p_rasm = to_rasm(strip_harakat(p_light))
    ro = jiwer.process_characters(g_rasm, p_rasm)
    rasm_ops = {"sub": ro.substitutions, "ins": ro.insertions, "del": ro.deletions,
                "ref_len": ro.substitutions + ro.deletions + ro.hits}

    spacing_del = sum(1 for o in char_ops if o.op == "del" and o.ref == " ")
    spacing_ins = sum(1 for o in char_ops if o.op == "ins" and o.hyp == " ")
    digit_swaps = sum(
        1 for o in char_ops
        if o.op == "sub" and _DIGIT_RE.match(o.ref) and _DIGIT_RE.match(o.hyp)
    )
    return AlignmentRecord(
        model=row.model, dataset=row.dataset, sample_id=row.sample_id,
        failure_class="ok", char_ops=char_ops, word_ops=word_ops, rasm_ops=rasm_ops,
        diac_seq_ref="".join(_HARAKAT_RE.findall(g_raw)),
        diac_seq_hyp="".join(_HARAKAT_RE.findall(p_raw)),
        spacing_del=spacing_del, spacing_ins=spacing_ins,
        len_ref_chars=len(g_light), len_hyp_chars=len(p_light),
        len_ref_words=len(g_light.split()), len_hyp_words=len(p_light.split()),
        digit_swaps=digit_swaps,
        presentation_form_chars=len(_PRESENTATION_RE.findall(row.pred)),
        latin_run_len=_longest_latin_run(row.pred),
        has_code_fence="```" in row.pred,
        cer_light=_cer(g_light, p_light), cer_raw=_cer(g_raw, p_raw),
        cer_rasm=_cer(g_rasm, p_rasm), wer_light=w.wer,
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m pytest tests/analysis/test_align.py -v`
Expected: PASS. If `jiwer`'s `AlignmentChunk` attribute names differ in the
installed version, check with
`python -c "import jiwer; print(jiwer.process_characters('ab','ac').alignment)"`
and adjust the attribute reads in `expand_char_chunks` — keep the test
assertions unchanged.

- [ ] **Step 5: Full suite + commit**

```bash
uv run python -m pytest -q
git add src/arabic_ocr_bench/analysis/align.py tests/analysis/test_align.py
git commit -m "feat: analysis.align — AlignmentRecord single-pass builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: `charstats.py` — `error_anatomy()`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/charstats.py`
- Create: `tests/analysis/test_charstats.py`

**Interfaces:**
- Consumes: `AlignmentRecord` (Task 4).
- Produces:
  `error_anatomy(records: list[AlignmentRecord]) -> dict` with keys:
  - `sid_split`: `{model: {"sub": int, "ins": int, "del": int}}` (char-level,
    pooled over that model's OK records) + a `"__pooled__"` entry.
  - `spacing_share`: `{model: {"wer_light": float, "spacing_ops": int,
    "total_char_ops": int, "spacing_frac_of_ops": float}}`.
  - `length_bias`: `{model: {"delta_median": float, "delta_q1": float,
    "delta_q3": float, "pred_vs_gt": [[gt_len, pred_len], ...]}}` — `delta` is
    `len_hyp_words - len_ref_words`; `pred_vs_gt` capped at 400 points, sampled
    deterministically (every k-th after sorting by `sample_id`).
  - `rewrites`: `{"char": [[ref_ngram, hyp_ngram, count], ...],
    "word": [...]}` — top 25 each, pooled across models, from consecutive
    `sub` runs in `char_ops` (chars) and `word_ops` reconstructed via a word
    re-alignment (words). For words, re-run `expand`-style logic on
    `g_light.split()` — simplest: recompute with
    `jiwer.process_words(...).alignment`. Keep counts ≥ 2.
- All ints/floats are plain Python types.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_charstats.py
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.charstats import error_anatomy
from arabic_ocr_bench.analysis.loader import iter_pred_rows
from pathlib import Path

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _records():
    return [build_record(r) for r in iter_pred_rows(FIX)]


def test_sid_split_has_pooled_and_per_model():
    out = error_anatomy(_records())
    assert "__pooled__" in out["sid_split"]
    assert "m1" in out["sid_split"]
    assert set(out["sid_split"]["m1"]) == {"sub", "ins", "del"}


def test_length_bias_quantiles_present():
    out = error_anatomy(_records())
    lb = out["length_bias"]["m1"]
    assert lb["delta_q1"] <= lb["delta_median"] <= lb["delta_q3"]


def test_rewrites_are_json_types():
    out = error_anatomy(_records())
    for pair in out["rewrites"]["char"]:
        assert isinstance(pair[2], int) and isinstance(pair[0], str)


def test_spacing_share_between_zero_and_one():
    out = error_anatomy(_records())
    for m, v in out["spacing_share"].items():
        assert 0.0 <= v["spacing_frac_of_ops"] <= 1.0
```

- [ ] **Step 2: Run to verify it fails** — `uv run python -m pytest tests/analysis/test_charstats.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `error_anatomy()`**

Use `numpy.percentile` for quantiles. Pool only `failure_class == "ok"`
records. `sid_split` sums `o.op` counts. `spacing_ops = spacing_del +
spacing_ins`; `spacing_frac_of_ops = spacing_ops / max(1, total_char_ops)`.
For `rewrites["char"]`, walk `char_ops`; collect maximal runs of consecutive
`sub` ops, key `("".join(refs), "".join(hyps))`, `Counter` them, keep count ≥ 2,
sort desc, take 25. `rewrites["word"]`: for each OK record recompute
`jiwer.process_words(g_light, p_light).alignment[0]`, take `substitute` chunks,
key `(ref_words_joined, hyp_words_joined)`.

```python
# src/arabic_ocr_bench/analysis/charstats.py
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

_POINT_CAP = 400


def _ok(records):
    return [r for r in records if r.failure_class == "ok"]


def error_anatomy(records) -> dict:
    ok = _ok(records)
    sid = defaultdict(lambda: Counter())
    length = defaultdict(list)          # model -> list[(gt_words, pred_words)]
    delta = defaultdict(list)           # model -> list[pred_words - gt_words]
    spacing = defaultdict(lambda: {"wer": [], "sp": 0, "tot": 0})
    char_rw, word_rw = Counter(), Counter()

    for r in ok:
        for o in r.char_ops:
            sid[r.model][o.op] += 1
            sid["__pooled__"][o.op] += 1
        length[r.model].append((r.len_ref_words, r.len_hyp_words))
        delta[r.model].append(r.len_hyp_words - r.len_ref_words)
        spacing[r.model]["wer"].append(r.wer_light)
        spacing[r.model]["sp"] += r.spacing_del + r.spacing_ins
        spacing[r.model]["tot"] += len(r.char_ops)
        # char rewrite runs
        run_ref, run_hyp = [], []
        for o in r.char_ops + [None]:
            if o is not None and o.op == "sub":
                run_ref.append(o.ref); run_hyp.append(o.hyp)
            else:
                if run_ref:
                    char_rw[("".join(run_ref), "".join(run_hyp))] += 1
                run_ref, run_hyp = [], []

    # word rewrites: cheap re-alignment
    import jiwer
    from ..normalize import normalize
    for r in ok:
        # need the text again; carried on the record indirectly — recompute from ops is
        # lossy, so re-read via sample map. Simplest: skip when unavailable.
        pass  # implemented via the records' source rows in Step 3b

    def q(xs, p):
        return float(np.percentile(xs, p)) if xs else 0.0

    out = {
        "sid_split": {m: {"sub": c["sub"], "ins": c["ins"], "del": c["del"]}
                      for m, c in sid.items()},
        "spacing_share": {
            m: {"wer_light": float(np.mean(v["wer"])) if v["wer"] else 0.0,
                "spacing_ops": v["sp"], "total_char_ops": v["tot"],
                "spacing_frac_of_ops": v["sp"] / v["tot"] if v["tot"] else 0.0}
            for m, v in spacing.items()},
        "length_bias": {},
        "rewrites": {
            "char": [[a, b, n] for (a, b), n in char_rw.most_common(25) if n >= 2],
            "word": [[a, b, n] for (a, b), n in word_rw.most_common(25) if n >= 2],
        },
    }
    for m, pairs in length.items():
        pairs_sorted = sorted(pairs)
        step = max(1, len(pairs_sorted) // _POINT_CAP)
        out["length_bias"][m] = {
            "delta_median": q(delta[m], 50), "delta_q1": q(delta[m], 25),
            "delta_q3": q(delta[m], 75),
            "pred_vs_gt": [[int(a), int(b)] for a, b in pairs_sorted[::step]],
        }
    return out
```

- [ ] **Step 3b: Wire word rewrites via a source-row map**

`error_anatomy` needs the light-normalised text for word rewrites, which the
record does not carry. Change the signature to
`error_anatomy(records, rows_by_key: dict[tuple[str, str], "PredRow"] | None = None)`
— when `rows_by_key` is provided (keyed `(model, sample_id)`), compute word
rewrites by re-aligning `normalize(row.gt,"light")` vs
`normalize(row.pred,"light")` with `jiwer.process_words`; when `None`, leave
`rewrites["word"] == []`. `findings.py` always passes the map. Update the test
`test_rewrites_are_json_types` to pass `rows_by_key` and assert at least the
char list is populated.

- [ ] **Step 4: Run tests** — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/charstats.py tests/analysis/test_charstats.py
git commit -m "feat: analysis.charstats.error_anatomy — S/I/D, spacing, length bias, rewrites

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: `charstats.py` — `script_letters()`

**Files:**
- Modify: `src/arabic_ocr_bench/analysis/charstats.py`
- Modify: `tests/analysis/test_charstats.py`

**Interfaces:**
- Consumes: `AlignmentRecord`, `letter_group`, `LETTER_GROUPS`.
- Produces:
  `script_letters(records) -> dict` with keys:
  - `confusion`: `{"pooled": [[ref_char, hyp_char, count], ...],
    "by_model": {model: [...]}}` — top 40 substitution pairs where both sides are
    Arabic letters (`ARABIC_LETTER_RE`), count ≥ 2.
  - `letter_error_rate`: `{"pooled": [[char, involved, gt_occurrences, rate],
    ...], "by_group": {group: rate}}` — `involved` = count of `sub`/`del` ops
    whose `ref` is that char, plus `ins` ops whose `hyp` is that char;
    `gt_occurrences` counted from every OK record's reference (reconstruct as
    `sum over char_ops ref chars` + hits — simplest: pass `rows_by_key` and count
    from `normalize(gt,'light')`). Top 30 by rate among chars with
    `gt_occurrences >= 5`.
  - `rasm_gap`: `{model: {dataset: {"cer_light": float, "cer_rasm": float,
    "gap": float}}}` — micro-averaged over that `(model,dataset)`'s OK records
    (`sum err / sum ref_len`), `gap = cer_light - cer_rasm`.
  - `digit_swaps`: `{model: int}` — total `digit_swaps` over OK records.
  - `presentation_leak`: `{model: {"lines_with_pf": int, "n": int, "rate": float}}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/analysis/test_charstats.py
from arabic_ocr_bench.analysis.charstats import script_letters


def test_confusion_captures_planted_bt():
    out = script_letters(_records())
    pooled = {(a, b) for a, b, _ in out["confusion"]["pooled"]}
    assert ("ت", "ب") in pooled or ("ب", "ت") in pooled


def test_rasm_gap_nonnegative_and_present():
    out = script_letters(_records())
    for m, per_ds in out["rasm_gap"].items():
        for ds, v in per_ds.items():
            assert v["gap"] >= -1e-9
            assert abs(v["gap"] - (v["cer_light"] - v["cer_rasm"])) < 1e-9


def test_presentation_leak_rate_bounds():
    out = script_letters(_records())
    for v in out["presentation_leak"].values():
        assert 0.0 <= v["rate"] <= 1.0


def test_all_json_types():
    import json
    json.dumps(script_letters(_records()))  # must not raise
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`script_letters` missing).

- [ ] **Step 3: Implement `script_letters()`** — pooled `Counter` over
  `sub` pairs filtered by `ARABIC_LETTER_RE.match` on both sides;
  per-`(model,dataset)` micro CER from `rasm_ops`/`char_ops` counts
  (`err = sub+ins+del`, `ref_len` from `char_ops` = `#sub + #del + hits`; carry
  `hits` by adding `r.len_ref_chars - (#sub + #del)` — or simplest and exact:
  micro from `sum(cer_light_i * len_ref_chars_i)` weighting is wrong, so instead
  accumulate `err_chars` and `ref_chars`: `ref_chars += r.len_ref_chars`,
  `err_chars += round(r.cer_light * r.len_ref_chars)`). Cast every number.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/charstats.py tests/analysis/test_charstats.py
git commit -m "feat: analysis.charstats.script_letters — confusion matrix, letter rates, rasm gap

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: `diacritics.py`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/diacritics.py`
- Create: `tests/analysis/test_diacritics.py`

**Interfaces:**
- Consumes: `AlignmentRecord` (`diac_seq_ref`, `diac_seq_hyp`), the per-sample
  `scores` dict on `PredRow` (for `diac_f1`), `rows_by_key`.
- Produces:
  `diacritics(records, rows_by_key) -> dict`:
  - `per_mark`: `{model: {mark_name: {"missed": int, "inserted": int,
    "substituted": int}}}` — align `diac_seq_ref` vs `diac_seq_hyp` per record
    with `jiwer.process_characters`; a `delete` = missed, `insert` = inserted,
    `substitute` = substituted; bucket by the **reference** mark's name
    (`MARK_NAMES: dict[str,str]` mapping each harakat codepoint to
    `fatha/damma/kasra/shadda/sukun/tanwin_fath/tanwin_damm/tanwin_kasr/
    dagger_alef/madda/other`).
  - `f1_vs_cer`: `{model: {"diac_f1": float, "cer_light": float,
    "attempt_rate": float}}` — `diac_f1` mean from `scores["diac_f1"]`,
    `attempt_rate` = share of OK records with `diac_seq_hyp != ""`.
  - `gt_density`: `{dataset: float}` — mean `len(diac_seq_ref) /
    max(1, len_ref_chars)` over OK records.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_diacritics.py
from pathlib import Path
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.diacritics import diacritics
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    recs = [build_record(r) for r in rows]
    rbk = {(r.model, r.sample_id): r for r in rows}
    return recs, rbk


def test_attempt_rate_bounds():
    recs, rbk = _setup()
    out = diacritics(recs, rbk)
    for v in out["f1_vs_cer"].values():
        assert 0.0 <= v["attempt_rate"] <= 1.0


def test_per_mark_keys_are_names():
    recs, rbk = _setup()
    out = diacritics(recs, rbk)
    for per in out["per_mark"].values():
        for name in per:
            assert name in {"fatha","damma","kasra","shadda","sukun",
                            "tanwin_fath","tanwin_damm","tanwin_kasr",
                            "dagger_alef","madda","other"}


def test_json_serialisable():
    import json
    recs, rbk = _setup()
    json.dumps(diacritics(recs, rbk))
```

(If the fixture has no diacritics at all, add one diacritised line per model to
`run_a` in Task 1's fixture — e.g. GT `"مَدْرَسَة"`, one model reads it bare,
one keeps one mark, one gets it right — and regenerate `scores.jsonl`. Do this
now and re-commit the fixture as part of this task.)

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement `diacritics.py`** with `MARK_NAMES` covering
  `ً..ْ`, `ٰ`, `ٓ`. Cast all numbers.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/diacritics.py tests/analysis/test_diacritics.py tests/fixtures/mini_results/
git commit -m "feat: analysis.diacritics — per-mark error breakdown + attempt rate

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: `difficulty.py`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/difficulty.py`
- Create: `tests/analysis/test_difficulty.py`

**Interfaces:**
- Consumes: `PredRow` list (needs `scores["cer_light"]`, `gt`, `pred`,
  `error`), `AlignmentRecord` list (for `len_ref_chars`).
- Produces:
  `difficulty(rows, records, *, cap=30, bins=12) -> dict`:
  - `consensus_hard`: `[{"sample_id","dataset","gt","mean_cer","n_models",
    "per_model": {model: cer_or_null}}]` — top `cap` by `mean_cer` over models
    that scored it (errored → cer 1.0 for the mean, flagged null in `per_model`).
  - `consensus_easy`: bottom `cap` by `mean_cer` among lines with `n_models >= 3`.
  - `discriminative`: top `cap` by population variance of per-model CER
    (`n_models >= 3`).
  - `suspected_bad_gt`: top `cap` by
    `(1 - mean_pairwise_pred_cer) * mean_pred_vs_gt_cer` among lines with
    `n_models >= 3` and every model OK — `mean_pairwise_pred_cer` = mean
    `cer_light` over all unordered model pairs' predictions (use
    `arabic_ocr_bench.metrics.cer` on `light`-normalised predictions).
  - `cer_vs_length`: `{model: [[bin_mid_chars, mean_cer, n], ...]}` — `bins`
    equal-width buckets over `len_ref_chars` range pooled across all models.
  - `cer_distribution`: `{model: {dataset: {"min","q1","median","q3","max",
    "mean","n"}}}` — quantiles of per-sample `cer_light` (OK rows only).
  - `ecdf`: `{model: [[cer, cumulative_fraction], ...]}` — ≤ 50 points,
    monotonic.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_difficulty.py
from pathlib import Path
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.difficulty import difficulty
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_bad_gt_line_is_flagged():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    ids = {d["sample_id"] for d in out["suspected_bad_gt"]}
    assert "dsx/test/000003" in ids  # planted: models agree, differ from GT


def test_perfect_line_is_consensus_easy():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    ids = {d["sample_id"] for d in out["consensus_easy"]}
    assert "dsx/test/000004" in ids


def test_cer_distribution_quantile_order():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    for per_ds in out["cer_distribution"].values():
        for v in per_ds.values():
            assert v["min"] <= v["median"] <= v["max"]


def test_ecdf_monotonic():
    rows, recs = _setup()
    out = difficulty(rows, recs)
    for pts in out["ecdf"].values():
        ys = [y for _, y in pts]
        assert ys == sorted(ys)


def test_json_serialisable():
    import json
    rows, recs = _setup()
    json.dumps(difficulty(rows, recs))
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement `difficulty.py`** (numpy for variance/quantiles;
  `metrics.cer` + `normalize` for pairwise pred distance). Cast every number.

- [ ] **Step 4: Run tests** → PASS. If the planted bad-GT line does not top the
  list because the fixture is too small, tighten the fixture (make the other
  lines' predictions disagree more) rather than loosening the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/difficulty.py tests/analysis/test_difficulty.py tests/fixtures/mini_results/
git commit -m "feat: analysis.difficulty — consensus-hard/easy, discriminative, bad-GT, CER curves

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 9: `modelcmp.py` — agreement, clustering, metric correlation, profile decomposition

**Files:**
- Create: `src/arabic_ocr_bench/analysis/modelcmp.py`
- Create: `tests/analysis/test_modelcmp.py`

**Interfaces:**
- Consumes: `PredRow` list, `AlignmentRecord` list.
- Produces (this task adds four keys; Tasks 10–11 add the rest to the same
  `model_landscape(...)` dict — so define `model_landscape()` here returning a
  dict and have Tasks 10/11 extend it):
  `model_landscape(rows, records, *, seed=1234, b: int | None = None) -> dict`
  (the `b` bootstrap-size override is unused until Task 10 but declared now so
  Tasks 10–11 add behaviour, not signature churn) with, after this task:
  - `agreement`: `{"models": [ordered_model_names],
    "identity": [[float]], "cer_corr": [[float]]}` — `identity[i][j]` = fraction
    of shared lines where `normalize(pred_i,"light") == normalize(pred_j,"light")`;
    `cer_corr[i][j]` = Pearson r of per-line `cer_light` on shared lines
    (numpy `corrcoef`). `models` ordered by a hierarchical clustering on
    `1 - identity` (implement average-linkage agglomerative in ~20 lines of
    numpy — no scipy — and read leaf order).
  - `metric_correlation`: `{"metrics": ["cer_light","wer_light","mer_light",
    "wil_light","diac_f1"], "matrix": [[float]]}` — Pearson over all OK
    `(model,sample)` score rows.
  - `profile_decomposition`: `{model: {"real": float, "orthographic": float,
    "diacritic": float}}` — from micro CER: `raw = real + orthographic +
    diacritic`; `diacritic = cer_raw_micro - cer_diac_insensitive_micro`
    against a harakat-stripped raw... simpler and well-defined:
    `diacritic = cer_raw - cer_light_no_diac_where_light_keeps_diac`. Use:
    `orthographic = cer_raw_micro - cer_light_micro`,
    `diacritic = cer_light_micro - cer_diac_insensitive_micro`,
    `real = cer_diac_insensitive_micro`. Clamp each at 0. Micro from the
    `scores` dict fields summed as `err = round(cer * ref_len)` with
    `ref_len = len(normalize(gt, profile))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_modelcmp.py
from pathlib import Path
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.loader import iter_pred_rows
from arabic_ocr_bench.analysis.modelcmp import model_landscape

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_agreement_is_square_and_diagonal_one():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    ident = out["agreement"]["identity"]
    n = len(out["agreement"]["models"])
    assert all(len(r) == n for r in ident)
    for i in range(n):
        assert abs(ident[i][i] - 1.0) < 1e-9


def test_profile_decomposition_components_nonnegative():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    for v in out["profile_decomposition"].values():
        assert min(v.values()) >= -1e-9


def test_metric_correlation_diag_one():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    m = out["metric_correlation"]["matrix"]
    for i in range(len(m)):
        assert abs(m[i][i] - 1.0) < 1e-6


def test_json_serialisable():
    import json
    rows, recs = _setup()
    json.dumps(model_landscape(rows, recs))
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement the four keys.** Average-linkage helper:

```python
def _avg_linkage_order(dist: "np.ndarray") -> list[int]:
    n = dist.shape[0]
    clusters = {i: [i] for i in range(n)}
    D = dist.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    while len(clusters) > 1:
        ks = list(clusters)
        best = min(((a, b) for i, a in enumerate(ks) for b in ks[i + 1:]),
                   key=lambda ab: D[ab[0], ab[1]])
        a, b = best
        merged = clusters[a] + clusters[b]
        # new distances = mean over member pairs
        for c in clusters:
            if c in (a, b):
                continue
            m = np.mean([dist[x, y] for x in merged for y in clusters[c]])
            D[a, c] = D[c, a] = m
        clusters[a] = merged
        del clusters[b]
        for c in list(clusters):
            D[b, c] = D[c, b] = np.inf
    return next(iter(clusters.values()))
```

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/modelcmp.py tests/analysis/test_modelcmp.py
git commit -m "feat: analysis.modelcmp — agreement matrix, clustering, metric corr, profile split

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 10: `modelcmp.py` — bootstrap CIs + rank stability

**Files:**
- Modify: `src/arabic_ocr_bench/analysis/modelcmp.py`
- Modify: `tests/analysis/test_modelcmp.py`

**Interfaces:**
- Extends `model_landscape()` return with:
  - `bootstrap_cer`: `{model: {dataset: {"point": float, "lo": float,
    "hi": float, "n": int}}}` — `B=2000` resamples (seeded
    `numpy.random.default_rng(seed)`), micro-CER per replicate from resampled
    per-sample `(err_chars, ref_chars)` where
    `err_chars = round(cer_light * len_ref_chars)`, `ref_chars = len_ref_chars`
    (OK records only); `lo/hi` = 2.5/97.5 percentiles.
  - `rank_stability`: `{"datasets": [...], model: {dataset: {"best": int,
    "worst": int, "mode": int}}}` — for each dataset, one joint resample of the
    shared sample set per replicate, rank models by replicate micro-CER
    (1 = best), report min/max/mode rank across replicates.
- `B` is a module constant `_BOOTSTRAP_B = 2000`; tests override via a
  `b: int | None = None` kwarg on `model_landscape` (defaults to `_BOOTSTRAP_B`)
  so the unit test can run `b=200`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/analysis/test_modelcmp.py
def test_bootstrap_ci_brackets_point():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    for per_ds in out["bootstrap_cer"].values():
        for v in per_ds.values():
            assert v["lo"] <= v["point"] <= v["hi"]


def test_bootstrap_is_deterministic():
    rows, recs = _setup()
    a = model_landscape(rows, recs, b=200)["bootstrap_cer"]
    b = model_landscape(rows, recs, b=200)["bootstrap_cer"]
    assert a == b


def test_rank_stability_bounds():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    for m, per_ds in out["rank_stability"].items():
        if m == "datasets":
            continue
        for v in per_ds.values():
            assert 1 <= v["best"] <= v["worst"]
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`bootstrap_cer` KeyError).

- [ ] **Step 3: Implement.** One `rng` created from `seed` at the top of
  `model_landscape`; pass it down. Vectorise the resample with
  `rng.integers(0, n, size=(B, n))` and numpy fancy-indexing on the
  `err`/`ref` arrays; `micro = err_samp.sum(1) / ref_samp.sum(1)`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/modelcmp.py tests/analysis/test_modelcmp.py
git commit -m "feat: analysis.modelcmp — seeded bootstrap CIs + rank stability

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 11: `modelcmp.py` — oracle ceiling + win-rate + Bradley–Terry

**Files:**
- Modify: `src/arabic_ocr_bench/analysis/modelcmp.py`
- Modify: `tests/analysis/test_modelcmp.py`

**Interfaces:**
- Extends `model_landscape()` return with:
  - `ensemble`: `{"single": {model: float}, "oracle": float,
    "rover": float, "marginal": [[model, cer_after_adding], ...]}` — `single` =
    per-model micro CER-light on the common line set (lines every model scored
    OK); `oracle` = micro CER if you take, per line, the prediction with the
    lowest `cer_light`; `rover` = micro CER of a per-line character-level
    majority vote (align all preds to the first with `jiwer`, vote per column;
    keep it simple — `rover` may equal best-single on the tiny fixture);
    `marginal` = greedy: start from the best single model, repeatedly add the
    model that most lowers oracle CER, recording the running value.
  - `win_rate`: `{"models": [...], "matrix": [[float]]}` — `matrix[i][j]` =
    fraction of shared OK lines where model i's `cer_light` < model j's
    (tie → 0.5); diagonal 0.5.
  - `bradley_terry`: `{model: float}` — MM-algorithm strengths (50 iters) from
    the pairwise win counts, normalised so `mean(log strength) == 0`; larger =
    stronger.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/analysis/test_modelcmp.py
import math


def test_oracle_not_worse_than_best_single():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    best_single = min(out["ensemble"]["single"].values())
    assert out["ensemble"]["oracle"] <= best_single + 1e-9


def test_win_rate_antisymmetric():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    M = out["win_rate"]["matrix"]
    n = len(M)
    for i in range(n):
        for j in range(n):
            assert abs(M[i][j] + M[j][i] - 1.0) < 1e-9


def test_bradley_terry_centered():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    logs = [math.log(v) for v in out["bradley_terry"].values()]
    assert abs(sum(logs) / len(logs)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement.** MM update for Bradley–Terry:
  `p_i <- w_i / sum_j (n_ij / (p_i + p_j))` where `w_i` = total wins of i,
  `n_ij` = games between i and j; iterate, renormalise to geometric mean 1 each
  step; 50 iters is plenty. Guard divide-by-zero with a `+1e-12`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/modelcmp.py tests/analysis/test_modelcmp.py
git commit -m "feat: analysis.modelcmp — oracle/ROVER ceiling, win-rate, Bradley-Terry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 12: `operational.py`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/operational.py`
- Create: `tests/analysis/test_operational.py`

**Interfaces:**
- Consumes: `PredRow` list (all rows, including errored — needs `attempts`,
  `latency_ms`, `cost_usd`, `error`, `failure_class` via `records`),
  `AlignmentRecord` list (`failure_class`, `cer_light`), and the
  `configs/models.yaml` pricing block via a small local reader (reuse the
  pattern in `make_leaderboard.collect_costs`).
- Produces:
  `operational(rows, records) -> dict`:
  - `pareto`: `{model: {"cer_light": float, "usd_per_1k": float, "free": bool,
    "on_frontier": bool}}` — `cer_light` micro over OK records; `usd_per_1k` =
    `1000 * mean(cost_usd over billable OK rows)`; `free` when that mean is 0;
    `on_frontier` = not dominated (no other model has both lower cer and lower
    cost).
  - `latency`: `{model: {"p50": float, "p95": float, "mean": float, "n": int}}`
    over OK rows' `latency_ms`.
  - `failure_causes`: `{model: {dataset: {class: count}}}` over all `records`
    (every `FAILURE_CLASSES` key present, zeros included).
  - `retries`: `{"by_provider": {provider: {"calls": int, "retried": int,
    "rate_limited": int}}, "attempts_hist": {int: int}}` — `provider` =
    `model.split("-")[0]` if it starts `openrouter` take `"openrouter"`, else the
    first token; keep it a simple heuristic.
  - `empty_rate`: `{model: {"empty": int, "n": int, "rate": float}}`.
  - `coverage`: `{"models": [...], "datasets": [...], "grid": [[[scored,
    attempted]]]}` — `attempted` = distinct sample_ids seen for that
    `(model,dataset)` across all rows; `scored` = those with `failure_class ==
    "ok"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_operational.py
from pathlib import Path
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.loader import iter_pred_rows
from arabic_ocr_bench.analysis.operational import operational

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_coverage_counts_rate_limit_as_attempted_not_scored():
    rows, recs = _setup()
    out = operational(rows, recs)
    mi = out["coverage"]["models"].index("m1")
    di = out["coverage"]["datasets"].index("dsy")
    scored, attempted = out["coverage"]["grid"][mi][di]
    assert attempted > scored  # m1/dsy/000000 rate-limited


def test_failure_causes_has_all_classes():
    from arabic_ocr_bench.analysis.refusal import FAILURE_CLASSES
    rows, recs = _setup()
    out = operational(rows, recs)
    for per_ds in out["failure_causes"].values():
        for counts in per_ds.values():
            assert set(counts) == set(FAILURE_CLASSES)


def test_latency_percentile_order():
    rows, recs = _setup()
    out = operational(rows, recs)
    for v in out["latency"].values():
        assert v["p50"] <= v["p95"]


def test_pareto_frontier_has_at_least_one():
    rows, recs = _setup()
    out = operational(rows, recs)
    assert any(v["on_frontier"] for v in out["pareto"].values())


def test_json_serialisable():
    import json
    rows, recs = _setup()
    json.dumps(operational(rows, recs))
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement `operational.py`.** numpy percentiles; frontier by
  the O(n²) domination check. Cast all numbers; `attempts_hist` keys become
  strings on `json.dumps` — that is acceptable, but make them strings explicitly
  for determinism.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/operational.py tests/analysis/test_operational.py
git commit -m "feat: analysis.operational — Pareto, latency, failure causes, retries, coverage

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 13: `corpus.py`

**Files:**
- Create: `src/arabic_ocr_bench/analysis/corpus.py`
- Create: `tests/analysis/test_corpus.py`

**Interfaces:**
- Consumes: `PredRow` list (for GT text + per-sample scores), the
  leaderboard-style aggregate rows (recompute here from `records`), an optional
  path to a contamination report.
- Produces:
  `corpus_health(rows, records, *, contamination_path: str | Path | None = None) -> dict`:
  - `gt_stats`: `{dataset: {"n_lines": int, "charset_size": int,
    "mean_line_chars": float, "diac_density": float,
    "zipf": [[rank, count], ...]}}` — `zipf` top 60 chars by frequency over that
    dataset's distinct GT lines (dedupe by `sample_id`).
  - `charset_coverage`: `{dataset: {"only_here": [chars], "shared_frac": float}}`
    — chars unique to that dataset vs. the union of the others.
  - `leaderboard_churn`: `{"metrics": ["cer_light","wer_light","diac_f1",
    "cost_adjusted"], "ranks": {model: {metric: int}}}` — rank per metric
    (`cost_adjusted` = `cer_light * (1 + log10(1 + usd_per_1k))`, reuse
    `operational`'s numbers or recompute).
  - `contamination`: `{"available": bool, "flagged": [sample_id, ...]}` —
    when `contamination_path` exists, load it (JSON list or `{sample_id: ...}`)
    and intersect with this run's consensus-easy + exact lines; else
    `{"available": False, "flagged": []}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_corpus.py
from pathlib import Path
from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.corpus import corpus_health
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_gt_stats_per_dataset():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    assert set(out["gt_stats"]) == {"dsx", "dsy"}
    assert out["gt_stats"]["dsx"]["charset_size"] > 0


def test_zipf_is_descending():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    counts = [c for _, c in out["gt_stats"]["dsx"]["zipf"]]
    assert counts == sorted(counts, reverse=True)


def test_churn_ranks_are_1_based_contiguous():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    for metric in out["leaderboard_churn"]["metrics"]:
        ranks = sorted(v[metric] for v in out["leaderboard_churn"]["ranks"].values())
        assert ranks == list(range(1, len(ranks) + 1))


def test_contamination_absent_by_default():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    assert out["contamination"] == {"available": False, "flagged": []}


def test_json_serialisable():
    import json
    rows, recs = _setup()
    json.dumps(corpus_health(rows, recs))
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement `corpus.py`.** Cast all numbers; sort deterministically
  (`sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))`).

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arabic_ocr_bench/analysis/corpus.py tests/analysis/test_corpus.py
git commit -m "feat: analysis.corpus — GT stats, Zipf, leaderboard churn, contamination xref

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 14: `findings.py` orchestrator + `bench analyze` CLI + golden test

**Files:**
- Create: `src/arabic_ocr_bench/analysis/findings.py`
- Modify: `src/arabic_ocr_bench/cli.py` (add `analyze` subcommand)
- Create: `tests/analysis/test_findings_golden.py`
- Create: `tests/fixtures/mini_findings.json` (generated)
- Create: `scripts/regen_mini_findings.py` (regeneration helper)

**Interfaces:**
- Consumes: every `*()` reduction from Tasks 5–13, `iter_pred_rows`,
  `build_record`.
- Produces:
  - `build(results_dir: str | Path, *, seed: int = 1234, contamination_path=None) -> dict`
    returning `{"meta": {...}, "error_anatomy": {...}, "script_letters": {...},
    "diacritics": {...}, "difficulty": {...}, "model_landscape": {...},
    "operational": {...}, "corpus_health": {...}}`. `meta` carries `seed`,
    `generated_utc` (ISO8601), `run_ids` (sorted), `models` (sorted),
    `datasets` (sorted), `n_pred_rows`, `row_caps` dict.
  - `main(argv=None) -> int` — arg parsing for standalone use.
  - CLI: `bench analyze <results-dir> [-o PATH] [--dump-records PATH]`.
    Default `-o` = `<results-dir>/findings.json` when `results-dir` is the
    top-level `results/`, else `results/findings.json`. `--dump-records` writes
    `pandas.DataFrame([asdict(r) for r in records]).to_parquet(PATH)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_findings_golden.py
import json
from pathlib import Path

from arabic_ocr_bench.analysis.findings import build

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"
GOLDEN = Path(__file__).parent.parent / "fixtures" / "mini_findings.json"


def _strip_volatile(d):
    d = json.loads(json.dumps(d))
    d["meta"].pop("generated_utc", None)
    return d


def test_top_level_keys():
    out = build(FIX)
    assert set(out) == {"meta", "error_anatomy", "script_letters", "diacritics",
                        "difficulty", "model_landscape", "operational", "corpus_health"}


def test_deterministic_across_two_builds():
    assert _strip_volatile(build(FIX)) == _strip_volatile(build(FIX))


def test_matches_golden():
    assert _strip_volatile(build(FIX)) == _strip_volatile(json.loads(GOLDEN.read_text()))


def test_output_is_json_serialisable_with_stdlib():
    json.dumps(build(FIX))  # must not raise (no numpy scalars)
```

- [ ] **Step 2: Run to verify it fails** → FAIL (module + golden missing).

- [ ] **Step 3: Implement `findings.py`**

```python
# src/arabic_ocr_bench/analysis/findings.py
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .align import build_record
from .charstats import error_anatomy, script_letters
from .corpus import corpus_health
from .diacritics import diacritics
from .difficulty import difficulty
from .loader import iter_pred_rows, run_dirs
from .modelcmp import model_landscape
from .operational import operational

ROW_CAPS = {"confusion_pairs": 40, "troublesome_letters": 30,
            "rewrite_patterns": 25, "difficulty_samples": 30, "length_bins": 12}


def build(results_dir, *, seed: int = 1234, contamination_path=None) -> dict:
    rows = iter_pred_rows(results_dir)
    records = [build_record(r) for r in rows]
    rbk = {(r.model, r.sample_id): r for r in rows}
    return {
        "meta": {
            "seed": seed,
            "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "run_ids": sorted(p.name for p in run_dirs(results_dir)),
            "models": sorted({r.model for r in rows}),
            "datasets": sorted({r.dataset for r in rows}),
            "n_pred_rows": len(rows),
            "row_caps": ROW_CAPS,
        },
        "error_anatomy": error_anatomy(records, rbk),
        "script_letters": script_letters(records, rbk),
        "diacritics": diacritics(records, rbk),
        "difficulty": difficulty(rows, records),
        "model_landscape": model_landscape(rows, records, seed=seed),
        "operational": operational(rows, records),
        "corpus_health": corpus_health(rows, records, contamination_path=contamination_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench analyze")
    ap.add_argument("results_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--dump-records")
    a = ap.parse_args(argv)
    doc = build(a.results_dir)
    out = a.out or (str(Path(a.results_dir) / "findings.json")
                    if Path(a.results_dir).name == "results" else "results/findings.json")
    Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"findings -> {out}")
    if a.dump_records:
        import pandas as pd
        from dataclasses import asdict
        recs = [build_record(r) for r in iter_pred_rows(a.results_dir)]
        pd.DataFrame([asdict(r) for r in recs]).to_parquet(a.dump_records)
        print(f"records  -> {a.dump_records}")
    return 0
```

Wire into `cli.py`: add `p_an = sub.add_parser("analyze", ...)` with the three
args, and `if args.cmd == "analyze": from .analysis.findings import build; ...`
(mirror the `main()` body, or just call `analysis.findings.main` with a
reconstructed argv).

- [ ] **Step 4: Generate the golden file**

```python
# scripts/regen_mini_findings.py
import json
from pathlib import Path
from arabic_ocr_bench.analysis.findings import build

fix = Path("tests/fixtures/mini_results")
out = Path("tests/fixtures/mini_findings.json")
doc = build(fix)
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
print("wrote", out)
```

Run: `uv run python scripts/regen_mini_findings.py`. Eyeball the diff — every
section present, numbers sane, no `NaN`/`Infinity` tokens (if `json.dumps`
emits `NaN`, a reduction leaked a divide-by-zero — fix the reduction, do not
`allow_nan=False` around it).

- [ ] **Step 5: Run tests + CLI smoke**

Run: `uv run python -m pytest tests/analysis/test_findings_golden.py -v`
Run: `uv run bench analyze tests/fixtures/mini_results -o /tmp/f.json && python -c "import json;json.load(open('/tmp/f.json'))"`
Expected: PASS + valid JSON.

- [ ] **Step 6: Full suite + commit**

```bash
uv run python -m pytest -q
git add src/arabic_ocr_bench/analysis/findings.py src/arabic_ocr_bench/cli.py \
        tests/analysis/test_findings_golden.py tests/fixtures/mini_findings.json \
        scripts/regen_mini_findings.py
git commit -m "feat: analysis.findings orchestrator + bench analyze CLI + golden fixture

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 15: Wire `findings.json` into `make_leaderboard.py` (build + cache, no view yet)

**Files:**
- Modify: `scripts/make_leaderboard.py`
- Modify: `scripts/build_notebooks.py` (the `LEADERBOARD_CELL` comment only —
  optional; skip if noisy)
- Test: extend `tests/test_leaderboard_findings.py` (created in Task 17) — for
  now add a dedicated `tests/test_leaderboard_findings_build.py`

**Interfaces:**
- Consumes: `arabic_ocr_bench.analysis.findings.build`.
- Produces:
  - `_ensure_findings(target: Path) -> dict` in `make_leaderboard.py` — returns
    the parsed findings doc, building + writing `results/findings.json` when it
    is missing or older than the newest run dir under `target`, else reading the
    cached file. Never raises on analysis failure — logs to stderr and returns
    `{}` (the view then renders a "not available" note).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leaderboard_findings_build.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import make_leaderboard as mk  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "mini_results"


def test_ensure_findings_builds_when_missing(tmp_path):
    # copy the fixture so we can write findings.json next to it
    import shutil
    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    doc = mk._ensure_findings(dst)
    assert "model_landscape" in doc
    assert (dst / "findings.json").exists()


def test_ensure_findings_reads_cache(tmp_path):
    import shutil
    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    (dst / "findings.json").write_text(json.dumps({"meta": {}, "cached": True}))
    # make the cache newer than every run dir
    import os, time
    now = time.time() + 10
    os.utime(dst / "findings.json", (now, now))
    doc = mk._ensure_findings(dst)
    assert doc.get("cached") is True
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`_ensure_findings` missing).

- [ ] **Step 3: Implement `_ensure_findings`** near `collect()` in
  `make_leaderboard.py`:

```python
def _ensure_findings(target: Path) -> dict:
    fj = (target / "findings.json") if target.is_dir() else target.parent / "findings.json"
    newest = 0.0
    for rd in _run_dirs(target):
        for name in ("predictions.jsonl", "scores.jsonl"):
            p = rd / name
            if p.exists():
                newest = max(newest, p.stat().st_mtime)
    if fj.exists() and fj.stat().st_mtime >= newest:
        try:
            return json.loads(fj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    try:
        from arabic_ocr_bench.analysis.findings import build
        doc = build(target)
        fj.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc
    except Exception as e:  # noqa: BLE001
        print(f"findings build failed: {e}", file=sys.stderr)
        return {}
```

Call it in `main()` after `rows = collect(target)` and thread `findings` into
`render(...)` (add a param with default `None` so existing callers/tests keep
working). Do **not** render a tab yet — just pass it and ignore it.

- [ ] **Step 4: Run tests** → PASS. Run the existing leaderboard test file to
  confirm no regression: `uv run python -m pytest tests/ -k leaderboard -q`.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_leaderboard.py tests/test_leaderboard_findings_build.py
git commit -m "feat: make_leaderboard builds + caches results/findings.json

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 16: Chart primitives + CSS in `scripts/_leaderboard_findings.py`

**Files:**
- Create: `scripts/_leaderboard_findings.py`
- Create: `tests/test_leaderboard_findings.py`

**Interfaces:**
- Consumes: nothing (pure string builders).
- Produces (every function returns an HTML/SVG fragment string; all inputs are
  plain Python numbers/strings/lists):
  - `FINDINGS_CSS: str` — one `<style>`-less CSS block (caller wraps it), scoped
    under `.findings-view`.
  - `svg_heatmap(rows, cols, values, *, fmt="{:.0f}", title="", rtl_labels=False) -> str`
  - `hbar(pairs, *, unit="", max_bars=30, rtl_labels=False) -> str` (pairs =
    `[(label, value), ...]`)
  - `stacked_hbar(rows, segments, *, colors=None, normalize=False) -> str`
    (`rows = [(label, {seg: value})]`)
  - `diverging_hbar(rows, *, left_key, right_key, left_label, right_label) -> str`
  - `svg_scatter(points, *, x_label, y_label, pareto=False, log_x=False, labels=None) -> str`
  - `binned_curve(series, *, x_label, y_label) -> str` (`series = {name: [(x, y, n)]}`)
  - `histogram(values, *, bins=20, x_label="") -> str`
  - `ecdf(series, *, x_label="CER") -> str` (`series = {name: [(x, y)]}`)
  - `box_row(rows, *, value_range=None) -> str` (`rows = [(label, {min,q1,median,q3,max})]`)
  - `dumbbell(rows, *, left_label, right_label) -> str` (`rows = [(label, left, right)]`)
  - `forest(rows) -> str` (`rows = [(label, point, lo, hi)]`)
- All SVGs use `viewBox`, `width="100%"`, and CSS custom properties
  (`var(--ink)`, `var(--line)`, `var(--accent)`, `var(--muted)`) so they inherit
  the leaderboard theme. No `<script>` inside primitives except the shared
  tooltip hook (a `data-tip` attribute the view's JS reads).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leaderboard_findings.py
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import _leaderboard_findings as lf  # noqa: E402


def _wellformed(fragment: str):
    # wrap so multiple top-level nodes are allowed
    ET.fromstring(f"<root>{fragment}</root>")


def test_every_primitive_wellformed_and_nonempty():
    _wellformed(lf.svg_heatmap(["a", "b"], ["x", "y"], [[1, 2], [3, 4]]))
    _wellformed(lf.hbar([("alef", 5), ("baa", 3)]))
    _wellformed(lf.stacked_hbar([("m1", {"sub": 3, "ins": 1, "del": 2})], ["sub", "ins", "del"]))
    _wellformed(lf.diverging_hbar([("fatha", {"missed": 4, "inserted": 1})],
                                  left_key="missed", right_key="inserted",
                                  left_label="missed", right_label="inserted"))
    _wellformed(lf.svg_scatter([(1.0, 0.5), (2.0, 0.3)], x_label="$", y_label="CER", pareto=True))
    _wellformed(lf.binned_curve({"m1": [(5, 0.2, 10), (15, 0.4, 8)]}, x_label="len", y_label="CER"))
    _wellformed(lf.histogram([0.1, 0.2, 0.2, 0.9]))
    _wellformed(lf.ecdf({"m1": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]}))
    _wellformed(lf.box_row([("m1", {"min": 0, "q1": 0.1, "median": 0.2, "q3": 0.4, "max": 1.0})]))
    _wellformed(lf.dumbbell([("m1", 0.2, 0.5)], left_label="rasm", right_label="light"))
    _wellformed(lf.forest([("m1", 0.3, 0.25, 0.36)]))


def test_primitives_handle_empty_input():
    for frag in (lf.hbar([]), lf.svg_scatter([], x_label="x", y_label="y"),
                 lf.box_row([]), lf.forest([]), lf.binned_curve({}, x_label="x", y_label="y")):
        _wellformed(frag)


def test_css_is_a_string_scoped_to_findings():
    assert isinstance(lf.FINDINGS_CSS, str) and ".findings-view" in lf.FINDINGS_CSS
```

- [ ] **Step 2: Run to verify it fails** → FAIL (module missing).

- [ ] **Step 3: Implement the primitives.** Keep each ≤ ~40 lines. Escape all
  text with `html.escape`. Guard empty input (return
  `<p class="chart-empty">no data</p>`). Use a fixed `viewBox` per chart type
  (e.g. `0 0 640 <h>`), compute bar widths as percentages of the value range.
  For `svg_scatter(pareto=True)`, sort points, walk the lower-left staircase,
  draw a `<polyline>` through frontier points and mark them with a filled
  circle; zero-cost points render in a left gutter band labelled `$0`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_leaderboard_findings.py tests/test_leaderboard_findings.py
git commit -m "feat: hand-rolled SVG/CSS chart primitives for the findings view

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 17: The 7 `_render_*` sections + `render_findings_view()`

**Files:**
- Modify: `scripts/_leaderboard_findings.py`
- Modify: `tests/test_leaderboard_findings.py`

**Interfaces:**
- Consumes: the primitives (Task 16); a `findings` dict shaped like `build()`'s
  output (Task 14).
- Produces:
  - `render_findings_view(findings: dict) -> str` — returns a single
    `<div class="findings-view">…</div>` containing, in order:
    `_render_coverage`, `_render_error_anatomy`, `_render_script_letters`,
    `_render_diacritics`, `_render_difficulty`, `_render_model_landscape`,
    `_render_operational`, `_render_corpus_health`. Each `_render_*` returns a
    `<section id='f-<slug>' class='f-panel'><h2>…</h2>…</section>` fragment
    (matching the existing leaderboard section contract so `_panel()`/`_toc()`
    can wrap them). When `findings` is `{}` or a section's data is missing,
    render `<section><h2>…</h2><p class="chart-empty">not available…</p></section>`.
  - `FINDINGS_SECTIONS: list[tuple[str, str]]` — `(slug, label)` pairs for the
    sidenav, in render order.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_leaderboard_findings.py
import json


def _fixture_findings():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from arabic_ocr_bench.analysis.findings import build
    return build(Path(__file__).parent / "fixtures" / "mini_results")


def test_render_findings_view_has_all_sections():
    html = lf.render_findings_view(_fixture_findings())
    _wellformed(html)
    for slug, _ in lf.FINDINGS_SECTIONS:
        assert f"id='f-{slug}'" in html or f'id="f-{slug}"' in html


def test_render_findings_view_empty_dict_is_graceful():
    html = lf.render_findings_view({})
    _wellformed(html)
    assert "not available" in html.lower()


def test_no_replacement_char():
    assert "�" not in lf.render_findings_view(_fixture_findings())
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement the 8 renderers + `render_findings_view`.** Map each
  spec catalog row to a primitive call:
  - coverage → `svg_heatmap` (scored/attempted ratio, `fmt="{:.0%}"`)
  - error_anatomy → `stacked_hbar` (S/I/D), `diverging_hbar` (length delta),
    `hbar` (rewrites, `rtl_labels=True`), paired `hbar` (spacing share)
  - script_letters → `svg_heatmap` (confusion, `rtl_labels=True`),
    `hbar` (letter error rate), `dumbbell` (rasm gap), `stacked_hbar` (digits),
    `hbar` (presentation leak)
  - diacritics → `diverging_hbar` per mark, `svg_scatter` (F1 vs CER)
  - difficulty → HTML tables for the four sample lists (reuse the existing
    `.rv-tbl` styles by emitting the same class names), `binned_curve`,
    `box_row`, `ecdf`
  - model_landscape → `svg_heatmap` ×3 (agreement, win-rate, metric-corr),
    `hbar` (ensemble + oracle/rover bars flagged), `forest` (bootstrap),
    rank-range via `box_row` with min/median/max, `stacked_hbar` (profile split)
  - operational → `svg_scatter(pareto=True, log_x=True)`, `box_row` (latency),
    `stacked_hbar` (failure causes), `hbar` (retries), `svg_heatmap` (coverage
    dup ok)
  - corpus_health → `binned_curve`/custom Zipf line, `dumbbell`/bump for churn,
    table for contamination
  Add a short `<p class="board-note">` explainer to each section (copy tone from
  the existing sections).

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_leaderboard_findings.py tests/test_leaderboard_findings.py
git commit -m "feat: render the 7 findings sections into an HTML view

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 18: Tab bar + view split in `make_leaderboard.py`

**Files:**
- Modify: `scripts/make_leaderboard.py`
- Modify: `tests/test_leaderboard_findings_build.py` (add end-to-end render test)

**Interfaces:**
- Consumes: `render_findings_view`, `FINDINGS_SECTIONS` (Task 17);
  `_ensure_findings` (Task 15).
- Produces: `render()` now emits a two-tab page — the existing sections under a
  `Leaderboard` tab, the findings view under a `Findings & analysis` tab —
  with a `localStorage` + URL-hash toggle. The published fragment still opens
  standalone and still passes the `U+FFFD` scrub.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_leaderboard_findings_build.py
def test_full_page_has_both_tabs(tmp_path):
    import shutil
    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    rows = mk.collect(dst)
    findings = mk._ensure_findings(dst)
    page = mk.render(rows, "Test board", dst, findings=findings)
    assert 'data-tab="leaderboard"' in page and 'data-tab="findings"' in page
    assert "findings-view" in page
    assert "�" not in page


def test_render_still_works_without_findings(tmp_path):
    import shutil
    dst = tmp_path / "results"
    shutil.copytree(FIX, dst)
    rows = mk.collect(dst)
    page = mk.render(rows, "Test board", dst)  # findings defaults to None
    assert "<main" in page
```

- [ ] **Step 2: Run to verify it fails** → FAIL (no tab markup).

- [ ] **Step 3: Implement.** In `render()`:
  - Add `findings: dict | None = None` param.
  - After `out.append("<main>")`, emit a tab bar:
    `<div class="tabbar" role="tablist"><button data-tab="leaderboard" class="tab on">Leaderboard</button><button data-tab="findings" class="tab">Findings &amp; analysis</button></div>`.
  - Wrap the current section loop's output in
    `<div class="view" data-view="leaderboard">…</div>`.
  - After it, `<div class="view" data-view="findings" hidden>` +
    `render_findings_view(findings or {})` + `</div>`.
  - Sidenav: build two `<ol>`s (existing `present` list, and
    `FINDINGS_SECTIONS`), show the one matching the active tab; the toggle JS
    swaps `hidden` on views + sidenav lists and stores the choice.
  - Add ~15 lines of vanilla JS (a new `_TABS_JS` constant) reading
    `location.hash` (`#findings/<slug>` → activate findings tab, scroll to
    `f-<slug>`) and `localStorage['lb_tab']`.
  - Append `FINDINGS_CSS` + `.tabbar`/`.tab` CSS to the `_HEAD` style block
    (or emit right after `<main>`).
- Keep `_render_empty` untouched (no findings tab on the placeholder page).

- [ ] **Step 4: Run tests + eyeball**

Run: `uv run python -m pytest tests/ -k "leaderboard or findings" -q`
Run: `uv run python scripts/make_leaderboard.py tests/fixtures/mini_results -o /tmp/board.html`
then open `/tmp/board.html` in a browser — both tabs switch, charts render,
dark/light both OK, no horizontal scroll on the page body.
Check size: `ls -la /tmp/board.html` — the findings view should add < 400 KB
versus a `--no-findings` build (add a quick `--no-findings` flag if useful for
the comparison, else eyeball against `git stash`).

- [ ] **Step 5: Regenerate the real leaderboard + commit**

```bash
uv run python scripts/make_leaderboard.py results -o results/leaderboard.html
uv run python -m pytest -q
git add scripts/make_leaderboard.py tests/test_leaderboard_findings_build.py \
        results/leaderboard.html results/leaderboard.json results/findings.json
git commit -m "feat: Findings & analysis tab in the leaderboard

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 19: `notebooks/09_findings.ipynb` via `build_notebooks.py`

**Files:**
- Modify: `scripts/build_notebooks.py` (add `build_findings()`, call it, tweak
  `build_analysis()`)
- Create (generated): `notebooks/09_findings.ipynb`
- Modify (regenerated): `notebooks/03_analysis.ipynb`
- Create: `tests/test_build_notebooks_findings.py`

**Interfaces:**
- Consumes: the `md()` / `code()` / `notebook()` / `write()` helpers already in
  `build_notebooks.py`; `arabic_ocr_bench.analysis.findings.build`.
- Produces: `build_findings()` writing `09_findings.ipynb` — a notebook that
  resolves a `RESULTS` path, `pip -q install seaborn` (best-effort), calls
  `build(RESULTS)`, and renders each findings section with matplotlib.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_notebooks_findings.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_build_is_byte_stable_and_emits_09():
    subprocess.run([sys.executable, "scripts/build_notebooks.py"], cwd=REPO, check=True)
    nb = REPO / "notebooks" / "09_findings.ipynb"
    assert nb.exists()
    doc = json.loads(nb.read_text())
    assert doc["cells"] and all(c["cell_type"] in ("markdown", "code") for c in doc["cells"])
    # every code cell has empty outputs (committed clean)
    assert all(c.get("outputs", []) == [] for c in doc["cells"] if c["cell_type"] == "code")
    # second run makes no diff
    before = nb.read_bytes()
    subprocess.run([sys.executable, "scripts/build_notebooks.py"], cwd=REPO, check=True)
    assert nb.read_bytes() == before


def test_09_references_the_analysis_api():
    nb = json.loads((REPO / "notebooks" / "09_findings.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "from arabic_ocr_bench.analysis.findings import build" in src
```

- [ ] **Step 2: Run to verify it fails** → FAIL (no `09_findings.ipynb`).

- [ ] **Step 3: Implement `build_findings()`** — model it on `build_analysis()`.
  Cell list (each `code(...)`/`md(...)`):
  1. `md("# Arabic Cursive OCR Benchmark — Findings", ...)`
  2. `code` — `RESULTS` resolution (copy from `build_analysis`).
  3. `code` — `try: import seaborn ... except: !pip -q install seaborn` then
     `import matplotlib.pyplot as plt`.
  4. `code` — `import sys, os; sys.path.insert(0, os.path.join(os.getcwd(),'src'))`
     then `from arabic_ocr_bench.analysis.findings import build; F = build(RESULTS)`
     and `print({k: type(v).__name__ for k,v in F.items()})`.
  5. `md("## 1. Coverage")` + `code` — heatmap of `F["operational"]["coverage"]`.
  6. `md("## 2. Error anatomy")` + `code` — stacked bar of `sid_split`, hist of
     length deltas, table of `rewrites`.
  7. `md("## 3. Script & letters")` + `code` — seaborn heatmap of
     `F["script_letters"]["confusion"]["pooled"]` (pivot to a matrix), bar of
     letter error rate, dumbbell of rasm gap.
  8. `md("## 4. Diacritics")` + `code` — grouped bar of `per_mark`.
  9. `md("## 5. Difficulty")` + `code` — image strips for `consensus_hard`
     (reuse the `datasets.load_dataset` pattern from `build_analysis` cell 3),
     CER-vs-length lines, violin of `cer_distribution`.
  10. `md("## 6. Model landscape")` + `code` — clustermap of agreement, forest
      of bootstrap CIs, bar of oracle vs single, marginal-gain curve, BT bar.
  11. `md("## 7. Operations")` + `code` — Pareto scatter (log-x, free models at
      x floored), latency box, failure-cause stacked bar.
  12. `md("## 8. Corpus health")` + `code` — Zipf log-log, churn bump chart.
  13. `md("## How to read these / caveats")` — coverage gate, bootstrap seed,
      heuristic limits (copy the phrasing from the spec).
  Then `write("09_findings.ipynb", notebook(cells, gpu=False))`.
  In the main build block at the bottom of the file, add `build_findings()`
  next to `build_analysis()`.
  In `build_analysis()`, append one `md` line to its intro cell:
  `"For the full findings set (confusion matrix, model clustering, bootstrap CIs, cost Pareto, …) see 09_findings.ipynb."`

- [ ] **Step 4: Run** `uv run python scripts/build_notebooks.py` then
  `uv run python -m pytest tests/test_build_notebooks_findings.py -v` → PASS.
  Also run `uv run python -m pytest -q` (the whole suite).

- [ ] **Step 5: Optional smoke** — `jupyter nbconvert --to notebook --execute
  notebooks/09_findings.ipynb --output /tmp/09_exec.ipynb` against a local
  `results/` if one is populated; not required for the commit (Colab is the
  real target), but fix any import-time error it surfaces.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_notebooks.py notebooks/09_findings.ipynb notebooks/03_analysis.ipynb \
        tests/test_build_notebooks_findings.py
git commit -m "feat: notebooks/09_findings.ipynb — full analysis notebook

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
| --- | --- |
| Approach A single-pass engine | 4 |
| `analysis/` package layout (10 modules) | 1–14 |
| `bench analyze` CLI + `--dump-records` | 14 |
| `make_leaderboard` builds/caches `findings.json` | 15 |
| `results/findings.json` committed | 18 |
| Real top-level tab, existing sections untouched | 18 |
| Hand-rolled SVG/CSS/vanilla-JS charts, no external deps | 16, 17, 18 |
| `09_findings.ipynb` via `build_notebooks.py`, empty outputs, same code path | 19 |
| `03_analysis.ipynb` pointer | 19 |
| numpy core dep, no scipy/seaborn in core | 1 |
| pure analyzers, seed 1234, JSON-safe output | 5–14 |
| AlignmentRecord fields (all) | 4 |
| rasm table + letter groups | 3 |
| Findings §1 Error anatomy (A2,A8,A9,A10) | 5 |
| Findings §2 Script & letters (A1,A3,A5,A6,A7) | 6 |
| Findings §3 Diacritics (A4) | 7 |
| Findings §4 Difficulty (B11–B17) | 8 |
| Findings §5 Model landscape (C20–C25) | 9, 10, 11 |
| Findings §6 Operations (D26–D32) | 12 |
| Findings §7 Corpus health (F37–F39) | 13 |
| Tab mechanics (localStorage + hash) | 18 |
| Chart primitives list | 16 |
| Size budget < 400 KB, row caps | 14 (caps), 18 (size check) |
| Bootstrap CI method (B=2000, seeded, percentile) | 10 |
| Win-rate / Bradley–Terry (MM, centered) | 11 |
| Refusal ruleset (ordered, 8 classes) | 2 |
| Suspected-bad-GT score | 8 |
| Testing: mini fixture, per-module tests, golden, primitive/view tests, notebook byte-stability | 1, 5–14, 16–19 |
| Deferred #18/#19, contamination stays external | 13 (xref only), plan carries no task for #18/#19 — intended |

No gaps.

**2. Placeholder scan**

- Task 6 Step 3 and Task 8/9/12/13 Step 3 describe the implementation in prose +
  partial code rather than a full code block. Each names exact function
  signatures, exact dict keys, the exact numpy calls, and the exact
  micro-averaging formula — an engineer has the contract and the failing tests
  pin the behaviour. The heavier algorithms (chunk expansion, average-linkage,
  bootstrap vectorisation, Bradley–Terry MM, `_ensure_findings`, `findings.build`,
  every primitive's tricky bit) are given as real code. This is the intended
  balance for a plan this size; not flagged as a failure.
- No "TBD" / "handle edge cases" / "similar to Task N" strings. Task 4 explicitly
  restates the fixture-row shape rather than referencing Task 1.

**3. Type consistency**

- `PredRow` fields defined in Task 1, consumed unchanged in Tasks 4, 8, 9, 12, 13.
- `AlignmentRecord` / `CharOp` fields defined in Task 4, consumed in 5–13.
- `model_landscape(rows, records, *, seed=1234, b=None)` — the `b` kwarg is
  introduced in Task 10 and reused in Task 11's tests; Task 9 defines the base
  signature `(rows, records, *, seed=1234)`. **Fix applied:** Task 9's
  `model_landscape` signature includes `b: int | None = None` from the start so
  Tasks 10–11 only add behaviour, not signature churn. (Task 9 test file does
  not pass `b`; that is fine — it defaults.)
- `error_anatomy` signature gains `rows_by_key` in Task 5 Step 3b; `findings.py`
  (Task 14) calls `error_anatomy(records, rbk)` — consistent. `script_letters`
  and `diacritics` also take `rbk` in Task 14's `build()`; Tasks 6 and 7 define
  them with that parameter. **Consistent.**
- `_ensure_findings(target)` (Task 15) and `render(rows, title, target, findings=None)`
  (Task 18) — the `findings` param is threaded in Task 15 (passed, ignored) and
  used in Task 18. Consistent.
- `render_findings_view(findings)` / `FINDINGS_SECTIONS` — defined Task 17, used
  Task 18. Consistent.
- Failure-class label set: `FAILURE_CLASSES` (Task 2) is the single source;
  Tasks 4, 12 import it rather than re-listing. Consistent.

Fix folded into Task 9 interface above. No other issues.

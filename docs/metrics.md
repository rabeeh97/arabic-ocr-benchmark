# Metrics & Arabic text normalization

Every model output is scored against the reference line with the same pipeline:
**normalize both strings, align them, count edits, aggregate.** All of it is
pure functions in `src/arabic_ocr_bench/metrics.py` and `normalize.py`; the
alignment is [`jiwer`](https://github.com/jitsi/jiwer) 4.x.

Rows that came back with a transport/API error (`error` set on the prediction)
are **quarantined** — not scored, not counted in `n`. A genuinely empty answer
(`error` null, text `""`) *is* a real prediction and is scored (CER ≈ 1).

---

## 1. Normalization profiles

Both the reference and the hypothesis pass through one of three profiles before
they are compared. Reported numbers always name the profile (`cer_micro_light`,
`wer_micro_raw`, …).

| profile | steps | keeps | use |
| --- | --- | --- | --- |
| **raw** | **encoding canonicalisation only** — strip bidi / zero-width controls → fold **Arabic Presentation Forms** (contextual glyph codepoints + ligatures) to their nominal letters → **NFC** | everything orthographic: alef forms, tatweel, digits, harakat, spacing | strictest — every *orthographic* variant is an error, but a different *encoding* of the same text is not |
| **light** | *raw*, then strip **tatweel** `ـ` → unify **alef** `أ إ آ ٱ → ا` → fold **Persian/Urdu look-alikes** `ک→ك`, `ی→ي`, `ہ ھ ە → ه` → map **Arabic-Indic digits** `٠-٩` / `۰-۹` → `0-9` → collapse **whitespace** | harakat, **taa-marbuta** `ة`, **alef-maqsura** `ى`, hamza carriers `ؤ ئ` | **the headline profile** |
| **diac_insensitive** ("no-diac") | *light*, plus **alef-maqsura → yaa** `ى → ي`, **taa-marbuta → haa** `ة → ه`, **hamza carriers folded** `ؤ → و`, `ئ → ي`, and **strip all harakat** ` َ ً ُ ٌ ِ ٍ ّ ْ ٰ …` | the bare consonant skeleton | "did it read the letters", ignoring spelling conventions and vowelling |

Harakat = the combining marks `U+064B–U+065F` plus the superscript (dagger) alef
`U+0670`.

### The "raw" encoding canonicalisation (why "raw" isn't literally raw)

Different models emit the *same* Arabic text in different Unicode encodings, and
so — it turns out — do some of the reference transcriptions (PDF-extraction
noise). Scoring those as errors measures the encoder, not the reader. So even
`raw` first applies three **lossless, non-orthographic** fixes:

1. **Strip bidi / zero-width controls** (`LRM/RLM/ALM`, the embeddings and
   isolates, `ZWSP/ZWJ/ZWNJ`, `BOM`) — invisible formatting, never content.
2. **Fold Arabic Presentation Forms** (`U+FB50–U+FDFF`, `U+FE70–U+FEFF`) to their
   nominal letters via a per-character compatibility decomposition: an
   "initial-form jeem" codepoint → `ج`, the lam-alef ligature → `ل` + `ا`, the
   `ﷺ` ligature → its spelled-out words.
3. **NFC** — canonical composition, which also *reorders combining marks*: a
   model that writes `ـّ` as `SHADDA, FATHA` and one that writes `FATHA, SHADDA`
   now compare equal (they render identically and NFC says they're the same
   string).

Measured effect of adding this (2026-09-06 rescore of all stored runs):
`cer_micro_raw` drops by ≤ 0.002 for most models (never rises), and **`diac_f1`
rises materially** for models that emitted non-canonical mark order — up to
+0.20 for `gpt-4o`, +0.09 for `claude-opus`, +0.05 for `claude-fable` — because
the diacritic metric no longer counts a reordering as a dropped-plus-invented
mark.

### Why this design

`light` is the level most Arabic OCR/NLP work compares at, and it mirrors the
normalizers in **CAMeL Tools** (`normalize_unicode`, `normalize_alef_ar`,
`normalize_alef_maksura_ar`, `normalize_teh_marbuta_ar`) and the **Lucene Arabic
analyzer**: Unicode-normalize, drop tatweel, unify alef, fold the Persian/Urdu
letter look-alikes, and (for the diacritic-insensitive variant) unify
alef-maqsura / taa-marbuta and remove tashkeel. Digit-script unification is
added because OCR models legitimately disagree on `١٢٣` vs `123` for the same
glyphs. Alef-maqsura `ى` is **not** folded at `light` — it is a real
final-position orthographic distinction that the references use ~8.5k times, so
it belongs only in `diac_insensitive`.

`light` deliberately **keeps** harakat, taa-marbuta and alef-maqsura so that a
model that drops every vowel or writes `ه` for `ة` is still marked down at the
headline level; `diac_insensitive` is the separate, more forgiving view, and the
**diacritic F1** metric (below) is the targeted view of vowelling specifically.

Individual steps are exported from `normalize.py`, so the profiles are auditable
and composable in tests.

---

## 2. Metrics

`S` = substitutions, `D` = deletions, `I` = insertions, `H` = hits (correct
tokens) from the alignment. `N_ref = S+D+H`, `N_hyp = S+I+H`. Tokens are
**characters** for CER/CAR and **whitespace-split words** for the rest.

| metric | formula | range | better | notes |
| --- | --- | --- | --- | --- |
| **CER** — character error rate | `(S+D+I) / N_ref` | ≥ 0, can exceed 1 | lower | the primary metric; sensitive to diacritics and long spurious output |
| **CAR** — character accuracy rate | `max(0, 1 − CER)` | 0–1 | higher | same information as CER, framed as "fraction right"; clamped at 0 because CER can exceed 1 |
| **WER** — word error rate | `(S+D+I) / N_ref` | ≥ 0, can exceed 1 | lower | word-level CER |
| **MER** — match error rate | `(S+D+I) / (S+D+I+H)` | 0–1 | lower | like WER but bounded — hits are in the denominator, so a very long wrong output can't push it past 1 |
| **WIL** — word information lost | `1 − (H/N_ref)·(H/N_hyp)` | 0–1 | lower | penalises **both** missed reference words (`H/N_ref`) and invented extra words (`H/N_hyp`); harsher than WER on hallucinated text |
| **WIP** — word information preserved | `(H/N_ref)·(H/N_hyp)` | 0–1 | higher | `1 − WIL` |
| **exact %** | share of lines with `ref == hyp` after normalization | 0–1 | higher | |
| **diacritic P / R / F1** | precision/recall/F1 over the ordered sequence of harakat marks, computed on **raw** text | 0–1 | higher | still sees dropped/invented vowels even when `diac_insensitive` CER is 0 |

WER can be misleading on its own — it exceeds 1 whenever a model rambles, which
makes "which rambler is worse" hard to read. **MER** and **WIL** are the bounded
companions: MER for a straight error fraction, WIL when you care specifically
about hallucinated words.

### micro vs macro

Every rate is reported both ways:

- **micro** — sum `S/D/I/H` across the whole corpus, *then* divide. Long lines
  contribute proportionally more. This is the corpus-level number.
- **macro** — compute the rate per line, then take the plain mean. Every line
  weighs the same, so short lines (where one wrong character is a big fraction)
  pull it around.

They diverge most when line lengths are uneven. The board's headline column is
**micro**; `CER·macro` is shown alongside as a spread indicator.

### degenerate inputs (fixed conventions, so results reproduce)

| reference | hypothesis | error rates | CAR / WIP |
| --- | --- | --- | --- |
| empty | empty | 0.0 | 1.0 |
| empty | non-empty | 1.0 | 0.0 |
| any | — (diacritics: neither side has harakat) | — | P = R = F1 = 1.0 |

---

## 3. Where the numbers live

- `results/<run_id>/scores.jsonl` — one row per (line, model): `cer_<profile>`,
  `car_<profile>`, `wer_<profile>`, `mer_<profile>`, `wil_<profile>`,
  `wip_<profile>`, `exact_<profile>`, `diac_{precision,recall,f1}`.
- `results/<run_id>/summary.csv` — per (model, dataset): the `*_micro_<profile>`
  and `*_macro_<profile>` aggregates, `exact_<profile>`, `diac_f1`,
  `refusal_rate`, `latency_ms_mean`, `cost_usd_total`.
- `results/leaderboard.json` — the merged, deduped board rows (same keys).
- The **leaderboard** page: the matrix uses `CER·light / WER·light`; the
  *by-dataset* panel's sortable table shows `CER·{light,raw,no-diac,macro}`,
  `CAR·light`, `WER·light`, `MER·light`, `WIL·light`, `exact %`, `diac F1`. Hover
  any metric name for its one-line definition; the full glossary is the
  "Definitions" section at the foot of the page.

## 4. Recomputing

Metrics are derived entirely from the stored `predictions.jsonl` (ground truth +
cleaned model text), so adding or changing a metric never needs the models
re-run — just rescore:

```bash
uv run python - <<'PY'
from pathlib import Path
from arabic_ocr_bench.report import score_run
for d in sorted(Path("results").glob("*/")):
    if (d / "predictions.jsonl").exists():
        score_run(d, progress=False)
PY
uv run python scripts/make_leaderboard.py results -o results/leaderboard.html
```

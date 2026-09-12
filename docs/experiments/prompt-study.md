# Prompt-sensitivity study — does the OCR prompt matter, and how?

- **Driver:** `scripts/run_prompt_study.py`
- **Model:** `gemini-flash-lite` (cheapest/fastest Gemini, 3-account key pool),
  `gemini-flash-lite-640tok` (same model/keys, higher token cap) for `page-v2` only
- **Prompts:** `src/arabic_ocr_bench/prompting.py` — see "Variants" below
- **Phases:** `pilot` (10-20 samples/dataset, this doc's first pass) → `full`
  (the rest of each dataset), same matrix, same cache — re-running only fills
  in whatever's missing

## Research questions

Production OCR calls (`src/arabic_ocr_bench/models/litellm_client.py`) are one
user-role chat turn: prompt text + image, `temperature: 0`, no system prompt,
no history. The prompt has always been a single fixed template (`line-v2`).
This study asks, holding the model and images fixed:

1. **Modality disclosure.** `line-v2` always claims the image is
   *handwritten* — including on `kitab_hindawi` / `kitab_historicalbooks`,
   which are printed. Does telling the model the correct modality help? Does
   the wrong claim actively hurt (vs. saying nothing)? This is the highest-
   priority question: it's a live confound in every existing benchmark number
   on the two printed KITAB-Bench subsets, not just a hypothetical.
2. **Language of the instructions.** English (current) vs. Arabic — does
   native-language prompting help the model align to the target script?
3. **Persona framing.** Neutral technical instruction (current) vs. an
   "expert paleographer" persona prefix.
4. **Diacritics instruction.** "Preserve what's there, don't invent" (current)
   vs. an explicit "omit all diacritics, skeleton letters only" — does
   suppressing the invitation to guess harakat reduce noise, cross-checked
   against the `diac_insensitive` CER profile already tracked?
5. **Granularity / task-scale match.** `kitab_hindawi` and
   `kitab_historicalbooks` are paragraph/page-scale ground truth
   (~350+ tokens) but have only ever been scored with a single-*line* prompt
   (see `findings-and-notes.md`, "kitab_hindawi / kitab_historicalbooks are
   page-scale, not lines"). Does a page-appropriate prompt (`page-v2`) close
   that gap for a model that's actually asked to use it correctly?

## Variants (`src/arabic_ocr_bench/prompting.py`)

| version | axis | applied to |
|---|---|---|
| `line-v2` | baseline (claims handwritten) | all 8 datasets |
| `line-v2-no-modality` | drops the modality claim entirely | all 8 |
| `line-v2-claims-printed` | claims printed (correct on the 2 printed sets, wrong on the 6 handwritten ones — one prompt string covers both control directions) | all 8 |
| `line-v2-arabic` | same content, Arabic instructions | all 8 |
| `line-v2-persona` | expert-paleographer persona prefix | all 8 |
| `line-v2-no-diac` | explicit skeleton-only (no harakat) | all 8 |
| `page-v2` | page/paragraph-appropriate + correctly "printed" | `kitab_hindawi`, `kitab_historicalbooks` only |

`line-v2` × handwritten datasets and `line-v2-claims-printed` × printed
datasets are the two "correct-modality-claim" cells; `line-v2` × printed and
`line-v2-claims-printed` × handwritten are the two "wrong-claim" cells;
`line-v2-no-modality` is the neutral control across all 8. No separate
"correct" / "wrong" prompt strings were needed — the 2×2 falls out of
applying the same two claim-strings to both dataset groups.

## Datasets (picked to look as different from each other as possible)

| dataset | modality | script / condition | pilot N |
|---|---|---|---|
| `muharaf_lines` | handwritten | Ruq'ah, degraded | 15 (cache-only — see Gotchas) |
| `nakba_lines` | handwritten | degraded 20th-c. (gated) | 15 |
| `rasam1` | handwritten | Maghrebi script | 15 |
| `kitab_khatt` | handwritten | modern, clean | 15 |
| `kitab_historyar` | handwritten | historical, degraded | 15 |
| `kitab_onlinekhatt` | handwritten | stylus-captured, cleaner | 15 |
| `kitab_hindawi` | printed | clean digitised books | 15 |
| `kitab_historicalbooks` | printed | degraded historical print | 10 (all it has) |

## Gotchas hit while building this

- **`muharaf_lines`, `nakba_lines`, `rasam1`, and now `rasam2` are
  cache-only.** All four have never been *fully* downloaded on this machine
  (unlike the other 5-6 datasets here, which read from a local HF cache from
  earlier sweeps — no network needed; a partial cache dir existing on disk,
  as `rasam2`'s did, is NOT proof the download completed — it crashed the
  same way anyway). `muharaf_lines`'s ~36MB parquet drops mid-download via the
  classic HTTP path (repeated `IncompleteRead` from the HF CDN — the same
  reason `configs/run.free-accumulate.yaml` excludes it). `nakba_lines` /
  `rasam1` stalled identically via both HF's "xet" transfer backend
  ("connection struggling", stuck near 100% for 10+ minutes) and, after
  setting `HF_HUB_DISABLE_XET=1` (now set unconditionally at the top of
  `run_prompt_study.py`, but NOT in the general `bench run` CLI path — that's
  presumably why `rasam2` hit it too, via a plain `bench run`), the classic
  downloader too (dead stop at exactly 20MB, no further progress after 20+
  seconds). `rasam2`, hit through `configs/run.paid-sweep-small.yaml` (plain
  `bench run`, no `HF_HUB_DISABLE_XET`), raised an *unhandled* `OSError` from
  `hf_hub_download`'s xet path deep enough in the call stack that it crashed
  the entire multi-model sweep rather than failing just that one cell — a
  real robustness gap worth hardening later (a dataset load failure shouldn't
  be able to take down every other model/dataset queued behind it in the same
  run). This is a genuine network reliability problem on this machine for
  first-time large HF downloads, not an xet-specific bug — cache-only drivers
  never attempt a fresh fetch for any of the four; they only report whatever's
  cached. All four need retrying from a more reliable network before they can
  contribute real numbers to this
  study.
- **`scored_indices` needed a `prompt_version` filter.** It only filtered by
  (model, dataset) before this study — a sample scored under any prompt
  counted as "done." Extended (backward-compatibly) in `runner.py` so each
  prompt variant tracks its own completion independently.
- **`gemini-flash-lite`'s default `max_tokens` (256) truncates `page-v2`.**
  Registered a sibling row `gemini-flash-lite-640tok` (same model/keys/pool,
  `max_tokens: 640`) rather than raise the shared row's cap and risk affecting
  the concurrently-running full Gemini crawl.

## Results — pilot (N=10-15/dataset)

`muharaf_lines` / `nakba_lines` / `rasam1` couldn't get fresh data this pass
(see Gotchas) — the numbers below cover the 5 datasets that could:
`kitab_khatt`, `kitab_historyar`, `kitab_onlinekhatt`, `kitab_hindawi`,
`kitab_historicalbooks`. CER·light, mean across those 5 datasets' per-dataset
means:

| variant | khatt | historyar | onlinekhatt | hindawi | historicalbooks | **mean** |
|---|---|---|---|---|---|---|
| `line-v2` (baseline) | 46.9% | 39.1% | 9.1% | 63.7% | 72.9% | **46.3%** |
| `line-v2-no-modality` | 36.4% | 39.2% | 9.0% | 71.4% | 83.1% | 47.8% |
| `line-v2-claims-printed` | 33.6% | 37.0% | 11.4% | 78.5% | 66.5% | 45.4% |
| `line-v2-arabic` | 32.2% | 40.2% | 10.8% | 73.7% | 72.3% | 45.8% |
| `line-v2-persona` | 35.3% | 38.0% | 10.1% | 79.1% | 74.0% | 47.3% |
| `line-v2-no-diac` | 33.3% | 37.9% | 9.0% | 58.5% | 65.7% | **40.9%** |

`page-v2` vs `line-v2`, the two page-scale datasets only (also mean predicted
output length, to show what's actually driving the gap):

| | `line-v2` (256 tok cap) | `page-v2` (640 tok cap) |
|---|---|---|
| `kitab_hindawi` CER·light | 63.7% | **35.3%** |
| `kitab_hindawi` mean pred length | 268 chars | 513 chars |
| `kitab_historicalbooks` CER·light | 72.9% | **8.2%** |
| `kitab_historicalbooks` mean pred length | 114 chars | 379 chars |

### Reading these

1. **Granularity match is the single biggest lever found in this study, by
   far.** Telling the model it may be looking at a paragraph/page (not one
   line) and giving it room to answer (640 vs 256 tokens) very nearly
   *quarters* CER on `kitab_historicalbooks` (72.9%→8.2%) and roughly halves
   it on `kitab_hindawi` (63.7%→35.3%). This directly confirms the standing
   open question from `findings-and-notes.md` ("kitab_hindawi /
   kitab_historicalbooks are page-scale, not lines") — the gap was
   overwhelmingly a *prompt/token-budget* bug, not a model-capability
   ceiling. **This is the one finding from this pilot worth acting on in
   production immediately** (see below), independent of the fuller sweep.
2. **Suppressing diacritic-guessing helps more than it hurts.** `line-v2-no-diac`
   is the best-performing *text* variant (40.9% mean CER·light vs 46.3%
   baseline) even though it necessarily tanks diac-F1 (e.g. 0.079 vs 0.346 on
   `kitab_hindawi` — it emits none). Read together: the baseline's "preserve
   what's there, don't invent" instruction is failing to stop invented
   harakat, and on these particular datasets/scripts the cost of that
   invention (counted against CER·light, which is diacritic-sensitive)
   outweighs the cost of periodically omitting a real one. Suggests the
   current instruction wording isn't doing its job, not that diacritics don't
   matter.
3. **Modality-claim results are inconsistent, not a clean confirmation of the
   hypothesis.** On the 3 handwritten sets with data (`khatt`, `historyar`,
   `onlinekhatt`), the *wrong* claim (`claims-printed`) beat the baseline's
   correct claim on 2 of 3. On the 2 printed sets, the *correct* claim helped
   on `historicalbooks` (72.9%→66.5%) but *hurt* on `hindawi` (63.7%→78.5%).
   `line-v2-no-modality` (the neutral control) was the worst variant on
   average (47.8%). At N=10-15/dataset this reads as noise-dominated, not
   settled — the modality-disclosure question needs the full sweep's larger N
   before drawing a real conclusion, and right now there is **no pilot
   evidence that fixing the "always claims handwritten" bug helps** (it might
   still be worth fixing for correctness/fairness reasons even if it's not a
   CER lever).
4. **Persona framing and Arabic-language instructions are both washes**
   (47.3% and 45.8% mean vs 46.3% baseline) — no signal either way at this N.

## Results — full sweep

*(pending — phase 2, after these pilot findings are reviewed)*

## What this suggests for production

- **Register a page-appropriate prompt + higher token cap for `kitab_hindawi`
  / `kitab_historicalbooks` across every model in the main benchmark**, not
  just this study's `gemini-flash-lite`. This is the highest-confidence,
  highest-impact change this pilot surfaced — worth doing independent of
  whether the rest of the prompt-variant study continues.
- Revisit the `line-v2` diacritics instruction wording — `line-v2-no-diac`'s
  win suggests "preserve what's there, don't invent" isn't working as
  intended; a full sweep should confirm whether a clearer instruction (rather
  than the blunt "never output diacritics" tested here, which sacrifices
  diac-F1 entirely) can get the same CER win without giving up diacritics.
- Not yet acting on the modality-claim, persona, or Arabic-instruction axes —
  none showed a consistent enough pilot signal to justify a production
  change; the full sweep will say whether any of them hold up at scale.

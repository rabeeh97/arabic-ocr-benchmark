# 03 · Toolkit specification (v0.1, 2026-11-02)

The open-source engine. The paper describes and evaluates it, and the platform runs it. It starts life inside this repo
(`src/arabic_ocr_bench/dataqa.py`) and moves to its own package `tadqiq` once the API settles (roadmap phase 1).

## 1. Principles

1. **Never mutate data.** Detectors emit *flags*. Fixers emit *patches*. A separate, explicit step applies a patch, and it writes a new
   version with provenance.
2. **Evidence, not verdicts.** Every flag carries the numbers that produced it (scores, spans, the matched line, the readings) so a
   person or a later model can judge it.
3. **Tiers by need.** A detector declares what it needs (`gt`, `img`, `page`, `read`, `lm`, `lex`, `ds`, `meta`). The runner
   skips what cannot run and says so. A dataset with text only still gets the C, O and D checks.
4. **Offline and CPU first.** Everything except the reader models runs on a laptop CPU. Readers are pluggable and may run elsewhere;
   only their outputs (`reads.json`) are needed.
5. **Calibrated per dataset.** Thresholds have defaults, from RASAM and later the paper benchmark. Dataset-relative checks use robust
   statistics. Precision estimates per check are stored and updated from votes.

## 2. Package layout (target)

```
tadqiq/
  core/        sample.py (Sample, Page, Dataset views) · flag.py (Flag, Evidence, Span) · patch.py · registry.py
  io/          hf_parquet.py · hf_datasets.py · folder.py (img + txt pairs) · page_xml.py · alto.py · csv_jsonl.py · tsv_speech.py (future)
  text/        unicode.py (C*) · ortho.py (O01–O04) · lexicon.py (O05–O07: clitics, rasm, variant folding) · charlm.py (O08, O09)
  image/       ink.py (padding, ink, deskew, core) · lines.py (I04/I05) · edges.py (I06) · quality.py (I08) · cc.py (J02/J03 components)
  joint/       length.py (J01) · paws.py (J02) · dots.py (J03) · marks.py (J04)
  reader/      interface.py (Reader protocol) · align.py (edit ops, span alignment) · checks.py (R01–R10) · ensemble.py (R06 fusion)
  dataset/     dups.py (D01/D02) · overlap.py (D03/D04) · shift.py (D05) · schema.py (D06, M*)
  fix/         a0.py (lossless) · a1.py (confirmed) · suggest.py (S patches) · apply.py
  eval/        corrupt.py (synthetic error injection) · bench.py (detector precision/recall) · gold.py (verified real errors)
  report/      card.py (quality card JSON + HTML) · gallery.py · summary.py
  cli.py       tadqiq scan | overlap | fix | apply | corrupt | bench | card
```

Mapping from today's code:
- `dataqa.py` splits into `text/`, `image/`, `joint/`, `reader/` and `dataset/`.
- `dataqa_scan.py` becomes `cli scan` plus `io/hf_parquet.py`.
- `dataqa_overlap.py` becomes `dataset/overlap.py`.
- `rasam_clean_lists.py` (garbled, LM) becomes `text/charlm.py`.
- `clean_rerun_local.py` (re-cuts) becomes `fix/a1.py` re-cut.

## 3. Data model

```python
Sample(id: str, gt: str | None, image: PIL.Image | None, page: PageRef | None, meta: dict)
PageRef(page_id, image_path, polygon, baseline, line_index)       # PAGE-XML / ALTO geometry when known
Read(sample_id, reader_id, text, char_conf: list[float] | None)    # any recogniser
Flag(sample_id, error_id: "R03", detector: "doubled_gt@0.2", score: float in [0,1],
     severity: "info"|"look"|"likely"|"certain", span: (start, end) | None,
     evidence: dict, message: str)
Patch(sample_id, field: "gt"|"image"|"meta", op: "replace"|"recut"|"drop"|"normalize",
      before, after, fix_class: "A0"|"A1"|"S", flags: [ids], evidence: dict, author: detector|user)
```

**Formats on disk.**
- `flags.jsonl` (one flag per line).
- `patches.jsonl`.
- `card.json`: the dataset quality card.
- `reads/<reader>.jsonl`.

Line IDs are `<dataset>/<split>/<row>`, stable against the source (HF row index or file stem).

## 4. Detector interface

```python
@detector(id="O05", name="dotting_error", needs={"gt", "lex"}, tier="text", version="0.1")
def dotting_error(sample: Sample, ctx: Context) -> Iterable[Flag]: ...

@dataset_detector(id="J01", needs={"img", "gt", "ds"})
def length_mismatch(samples: Iterable[Sample], ctx: Context) -> Iterable[Flag]: ...
```

- `Context` holds shared resources: the lexicon, the char LM, the readers' outputs, the dataset statistics cache, the config and its thresholds.
- Detectors are pure and deterministic given the context, and each has unit tests built from synthetic positives and negatives.
- Third-party detectors register through an entry point (`tadqiq.detectors`). The platform scores them on the benchmark before
  they are listed (see the product spec, "detector leaderboard").

## 5. Evidence fusion and line status

For each line, combine its flags into one status:
- **clean**: no flags above `look`;
- **suspect GT**: GT-side flags (C/O/R01/R03/R04/R06-agree) dominate;
- **suspect image**: I/J/R02 flags dominate;
- **hard**: readers disagree with each other (R06), and there is no GT-side evidence;
- **certain defect**: an A0/A1 condition is met.

**Score.** v0 uses a calibrated noisy-OR over flag scores, with per-detector weights equal to the estimated precision. v1 uses a small learned
model on flag features, trained on platform votes (logistic regression; keep it interpretable).

**R06 in detail** (our "ensemble agrees to disagree" idea):
- **Inputs:** readings from k ≥ 2 readers. The readers should be independent: different architectures, or cross-fitted folds that never saw the line.
- **Measures:**
  - CER between readers: `d_rr`, the mean over pairs;
  - CER from each reader to the GT: `d_rg`.
- **Likely GT error:** `d_rr` is low and `d_rg` is high.
  - The suspect span is where the aligned readings agree with each other but not with the GT.
- **Hard sample:** `d_rr` is high.
- **Weighting:** each reader counts by its in-domain accuracy.

## 6. Fix policy

| class | when it is applied automatically | examples |
|---|---|---|
| A0 | always (lossless) | remove BOM or invisible characters (C02), NFC (C05), presentation forms (C03), EXIF bake (M02), alpha compositing (M03) |
| A1 | only when the policy enables it and every condition holds | remove a pasted copy confirmed by ≥ 2 independent readers (R03); re-cut from the PAGE polygon when the re-cut's reading matches the GT at CER ≤ the dataset median (I01/I04); reverse a visual-order line when every word of the reversal is in the lexicon (O09); shift a whole page when ≥ 80 % of its lines match one-to-one better shifted (P01); fix mojibake when the round trip yields valid Arabic (C12) |
| S | never; a person accepts it (single click, or a vote) | the dotting error's lexicon candidate (O05), the GT completion a reader proposes (R01), span fixes (R06/R08) |
| H | no patch, only a flag | everything else |

Every applied patch is listed in a changelog for the new version, with before and after, the evidence and the policy that allowed it.

**"100 % sure" is only claimed for A0.** For A1, the paper reports measured precision on verified examples. The product shows it next to the policy switch.

## 7. Evaluation harness (the paper depends on it)

1. **Synthetic corruption** (`eval/corrupt.py`). Inject known errors into clean lines, at controlled rates, with ground-truth labels:
   - **GT-side:**
     - paste a span (O13/R03);
     - truncate the start or end (R01);
     - swap a GT with a page neighbour (R04);
     - change dots (O05);
     - one-edit typo (O06);
     - merge or split words (O03/O04);
     - reverse the order (O09);
     - insert invisible or presentation-form characters (C02/C03);
     - mojibake (C12);
     - convert to Persian look-alike letters (C04);
     - add or strip harakat (J04).
   - **Image-side:**
     - crop to 1–7 px (I01);
     - extend the crop to the neighbour line (I04);
     - cut the left or right end (I06);
     - rotate 180° or mirror (I07);
     - blank or wrong-place crop (I03);
     - downscale or blur (I08);
     - add a border (I09).

   Corrupting real clean lines (not synthetic images) keeps the distribution realistic.
2. **Verified real errors** (`eval/gold.py`). Lines whose error type a person confirmed. Sources:
   - RASAM: our by-eye list, the drop/re-cut lists and PR #7 labels;
   - Muharaf: the published cleaned-Muharaf flags (`alazzawi2026labelerr`, check the licence);
   - new votes.
3. **Metrics.**
   - Per error type: recall on synthetic, recall and precision on verified real, precision at k on unverified real (by-eye samples).
   - Per dataset: flag rates.
   - Cost: seconds per 1k lines on CPU.
   - Fixes: correctness of A1 patches.
4. **Downstream effect.**
   - Retrain and score on the official vs cleaned views. Our RASAM clean benchmark is the template: page-bootstrap confidence intervals, and checking that rankings are stable.

## 8. Resources

- **Lexicon.**
  - Word types from OpenITI (cached: `MLM_classical_arabic_openiti`) plus the training GTs of the dataset being scanned (never its test).
  - Light clitic stripping, then folding: hamza and alif, ى/ي, ة/ه, harakat.
  - Keep frequencies. Store as a compact trie or a set of folded keys.
  - The rasm skeleton map is built from the same list.
  - Optional plugins: CAMeL Tools / Farasa morphological analysers (check licences).
- **Char LM.** The OpenITI char 8-gram we already built (RS4f) for classical text. A modern-text LM for MSA datasets.
- **Readers.** Ours, in order of preference:
  1. CRNN-CTC (E60) + pretrained (PT), cross-fitted;
  2. Kraken / TariMa;
  3. HATFormer;
  4. VLMs via API or Colab.

  The interface takes any `reads.jsonl`.

## 9. CLI (target)

```
tadqiq scan    --hf calfa-ai/RASAM-1:test [--reads r1.jsonl r2.jsonl] [--lex openiti] [--lm openiti8] -o out/
tadqiq overlap calfa-ai/RASAM-1:test calfa-ai/AMIDDA:train
tadqiq fix     out/flags.jsonl --policy a0,a1 -o out/patches.jsonl
tadqiq apply   out/patches.jsonl --to new_version/
tadqiq corrupt --hf calfa-ai/RASAM-1:test --rate 0.05 --types all -o corrupted/
tadqiq bench   --corrupted corrupted/ --gold gold.jsonl
tadqiq card    out/ -o card.html
```

## 10. Licences

- **Code:** Apache-2.0 (proposed; matches RASAM and most HF tooling).
- **Flags and patches:** published per dataset, under a licence compatible with the source. A flag file keyed by line ID is generally publishable even when the data is not.

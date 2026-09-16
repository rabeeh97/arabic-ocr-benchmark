# Research memo: how Misraj built Baseer/Baseer-Nakba, and what it means for us

Date: 2026-09-16. Motivation: understand whether Baseer-Nakba (the NAKBA-2026
Systems-Track winner, 0.079 CER) is a generally superior Arabic OCR model or a
narrow specialist, whether its approach is reproducible, and what it tells us
about how to build our own competitive Arabic OCR/HTR model. This combines our
own experimental evidence (`findings-and-notes.md`, exp3090 A/B/C) with
external research on Misraj AI's published work.

**Primary sources:**
- Hennara et al., "Baseer-Nakba" system paper, NAKBA-NLP 2026 shared task —
  [ACL Anthology 2026.nakbanlp-1.27](https://aclanthology.org/2026.nakbanlp-1.27/)
- Hennara et al., "Baseer" (general model), 2025 —
  [arXiv:2509.18174](https://huggingface.co/papers/2509.18174)
- Ketaba-OCR system paper —
  [ACL Anthology 2026.nakbanlp-1.21](https://aclanthology.org/2026.nakbanlp-1.21/)
- Misraj AI — [company site](https://discover.misraj.ai/), [HF org](https://huggingface.co/Misraj), inference pipeline [github.com/misraj-ai/Nakba-pipeline](https://github.com/misraj-ai/Nakba-pipeline)
- Qari-OCR synthetic pipeline — [arXiv:2506.02295](https://arxiv.org/abs/2506.02295)
- AtlasOCR / OCRSmith synthetic tooling — [HF blog](https://huggingface.co/blog/imomayiz/atlasocr)

## 1. How Baseer-Nakba was actually built

It is **not** a single fine-tune of raw Qwen2.5-VL, and it is **not** a SLERP
merge across model lineages (our `docs/datasets.md` said "SLERP-merged" —
that was a mislabeling, now corrected there).

The real lineage is two stacked fine-tunes:

1. **General Baseer 3B** (arXiv:2509.18174): Qwen2.5-VL-3B-Instruct,
   full-parameter fine-tune with the vision encoder frozen, trained on 500K
   image-text pairs for Arabic document-to-Markdown OCR — 300K synthetic
   (Markdown → HTML → Word → PDF → image, 39 fonts, 29 augmentation types) +
   200K real scanned books/magazines/academic documents.
2. **Baseer-Nakba** (the HTR specialist, 2026.nakbanlp-1.27): a second
   fine-tune stacked on top of general Baseer, using only two real
   handwritten-manuscript corpora — Muharaf (21,196 lines, Ruq'ah calligraphy)
   as a domain-adaptation intermediary (converted to grayscale to match
   NAKBA's tonal profile), then the NAKBA competition data itself (15,962
   train / 2,095 dev lines of the Omar Al-Saleh memoirs — the same corpus as
   our `nakba_lines`). No extra synthetic data at this stage.

Their disclosed ablation ladder (their Table 3):

| stage | recipe | CER | WER |
|---|---|---|---|
| baseline | NAKBA-only, 3 epochs, decoder SFT | 0.19 | 0.42 |
| zero-shot control | Gemini 3.1 Pro, zero-shot on NAKBA dev | 0.22 | 0.47 |
| + Muharaf pretraining | Muharaf → NAKBA, decoder-only SFT | 0.16 | — |
| + vision unfrozen, differential LR | decoder 1e-4 / vision encoder 9e-6, full encoder-decoder, Muharaf → NAKBA | 0.10 | — |
| + final weight averaging | average epoch-1 and epoch-5 weights of the above run | **0.079** | **0.244** |

Their own zero-shot Gemini 3.1 Pro control (CER 0.22) is worse than even
their un-adapted NAKBA-only baseline (0.19) — the paper's point is that
dedicated fine-tuning, not just a strong general VLM, was necessary to win.

**They explicitly disclose the narrowness.** Their Ethics Statement says the
model's "capabilities are intrinsically tied to the specific calligraphic
styles and historical periods present in the training distribution" and
warns of bias risk "if applied to documents from different demographic
groups, regions, or eras." This matches our own exp3090 finding almost
exactly (see below) — Misraj isn't hiding this, we just hadn't read their
paper yet.

## 2. Is Baseer-Nakba just overfit to NAKBA, or is Misraj's Baseer actually good?

Both, in different senses — this is the core answer to "is it real or a
fluke":

- **General Baseer 3B is a legitimately strong, fairly broad Arabic
  document-OCR model.** On Misraj's own Misraj-DocOCR benchmark (400 diverse
  document images) it clearly beats untuned Qwen2.5-VL-3B/7B and GPT-4o-mini
  (WER 0.25 vs 0.87 / 0.92 / 1.36) and beats GPT-4o-mini on TEDS (structure
  fidelity) by a wide margin. On a corrected KITAB-Bench subsample it wins
  structural metrics but loses text-accuracy to a model called Dots.ocr — so
  it's strong, not universally SOTA, but a real, general-purpose result.
- **Baseer-Nakba is a narrow specialist stacked on top of that**, and our own
  measurements (exp3090, `findings-and-notes.md`) independently confirm it:
  - Home domain (`nakba_lines`): ~1.0% CER pre-fine-tune — dramatically
    better than generic Qwen3-VL-4B (76%) or PaliGemma-2-3B (88%) even before
    *their* fine-tuning.
  - Out-of-domain Maghrebi manuscripts (RASAM1/2): ~49-51% CER — comparable
    to or barely better than what *generic* bases reach only after
    fine-tuning on 8k lines of Muharaf.
  - Fine-tuning it further with our generic Muharaf-LoRA recipe made it
    **worse** on its own best domain (1.0% → 3.4% CER; paired per-sample
    diagnosis: 35/63 samples got worse, confirmed as real catastrophic
    forgetting, not noise).

So: the *general* Baseer is genuinely good and broadly Arabic-OCR-competent;
the *Nakba* variant trades that generality for a large win in one narrow
domain, exactly as its own ethics statement predicts and as our numbers show.

## 3. Misraj AI as a lab

Saudi (Khobar)-based; the AI research division of Misraj Technology, an
enterprise-software group. Mission is explicitly "Arabic-first" AI. Product
line under the "Kawn" ecosystem: Kawn LLM, Kawn Document OCR (descended from
general Baseer). Other published work: **Mutarjim**, bidirectional
Arabic-English translation ([arXiv:2505.17894](https://arxiv.org/abs/2505.17894));
**Kuwain 1.5B**, a small Arabic LM via language injection
([arXiv:2504.15120](https://arxiv.org/abs/2504.15120)). They publish openly —
arXiv papers, ACL Anthology system papers, open HF weights, and an open
inference pipeline repo — unusually transparent for a commercial Arabic AI
shop, which is exactly why the recipe above was reconstructable rather than
guessed at.

## 4. What other NAKBA-2026 competitors did (competing strategies)

Full published leaderboard (Hamoud et al., shared-task overview — not
independently fetched, taken from the Ketaba/Baseer papers' own tables):
Misraj 0.079, Oblevit 0.0925, 3reeq 0.0938, Latent Narratives 0.105,
Al-Warraq 0.114, Not Gemma 0.122, NAMAA-Qari 0.195, Fahras 0.227. Only
Misraj, Ketaba, Moankhaled10, and Al-Warraq released loadable checkpoints
(per our `docs/datasets.md`); the rest have no public weights.

- **Ketaba-OCR** (2nd/3rd, CER 0.082-0.094,
  [2026.nakbanlp-1.21](https://aclanthology.org/2026.nakbanlp-1.21/)):
  parameter-efficient route — QLoRA (4-bit) + DoRA/rsLoRA on top of
  `sherif1313/Arabic-English-handwritten-OCR-v3` (itself already an
  Arabic-English HTR specialist pretrained on KHATT + IAM among others),
  plus a "Linear+Boost" ensemble/voting scheme across adapters. Much cheaper
  than Misraj's full encoder-decoder route, lands close in score — evidence
  that a well-chosen HTR-pretrained base plus efficient adapters gets most of
  the way there for a fraction of the compute.
- **Al-Warraq** (5th, CER 0.114): QLoRA r=16 on QCRI's `Fanar-2-Oryx-IVU`
  (an Arabic-first Qwen2.5-VL-7B). Notably worse than the 3B-based leaders —
  parameter count wasn't the deciding factor; base-model Arabic-domain fit
  and fine-tune recipe mattered more than scale.
- **None of the published systems report cross-domain generalization outside
  NAKBA.** As far as this research could find, we appear to be the only ones
  who've run these checkpoints against Maghrebi RASAM / KHATT — that's a real
  differentiator for this repo's benchmark work, not just an internal
  curiosity.

## 5. What this means for building our own competitive Arabic OCR model

Takeaways that are directly actionable, not just descriptive:

1. **Baseer's edge is a disclosed, cheap, reproducible recipe — not
   inaccessible IP or scale.** The winning ingredients were: (a) full/near-full
   parameter fine-tuning with a **differential learning rate between the
   vision encoder and text decoder** (much lower LR on vision), (b) a
   **domain-adaptation intermediary corpus** (a different calligraphic style,
   Muharaf) before the true target domain, not training on the target domain
   alone, and (c) **final-epoch weight averaging** (epoch 1 + epoch 5) as a
   cheap regularizer against overfitting/forgetting. None of this requires
   more compute than what we're already running on the 3090 rig.
2. **Curriculum beats naive mixing.** Misraj's own ablation shows sequential
   Muharaf→NAKBA fine-tuning (CER 0.19→0.16) beat a naive joint Muharaf+NAKBA
   mix (0.19→0.18, barely moved). This directly supports trying a staged
   curriculum in our own recipe rather than dumping all datasets into one
   training mix — worth testing against our current single-mix approach.
3. **Vision-encoder unfreezing with a differential LR is the biggest lever
   left untested in our own experiments** (`findings-and-notes.md` already
   flags `train_vision: true` as untested and "the most promising lever for
   the still-weak Maghrebi scores" — this external result corroborates that
   priority directly; Misraj's own biggest single jump, 0.16→0.10 CER, came
   from exactly this change).
4. **Don't re-fine-tune an already-specialized base with a generic recipe
   without domain replay.** Our regression on Baseer-Nakba is the same
   failure mode Misraj's own ethics section warns about. If we want a model
   that's strong on NAKBA *and* generalizes, the fix (already noted in
   `findings-and-notes.md` item 2) is a replay mix that includes some
   NAKBA-domain lines alongside new data, not a clean fine-tune on new data
   alone.
5. **Synthetic text-rendering pipelines are the standard way to scale
   printed/modern Arabic OCR data cheaply**, and multiple independent groups
   converge on the same recipe: render text corpora through diverse fonts and
   layouts (Misraj: 39 fonts via Markdown→HTML→Word→PDF→image; Qari-OCR:
   12 fonts via HTML rendering, three-tier clean/moderate/heavy degradation,
   [arXiv:2506.02295](https://arxiv.org/abs/2506.02295)) then apply
   degradation augmentations. **AtlasOCR's OCRSmith** is an open-sourced
   library that does exactly this (fonts + layouts + backgrounds + distortion
   pipelines, generates labeled images at scale —
   [HF blog](https://huggingface.co/blog/imomayiz/atlasocr)) — worth
   evaluating as reusable tooling rather than building a synthetic generator
   from scratch, if/when we want to scale up printed-text coverage alongside
   our real handwritten-manuscript sets (Muharaf, NAKBA, KHATT, RASAM).
6. **Real historical handwriting corpora worth combining, each a different
   calligraphic/regional style**: Muharaf (Ruq'ah, already in our pipeline),
   NAKBA/Omar-Al-Saleh (already in ours), KHATT, RASAM (Maghrebi). The
   generalizable lesson from Misraj's ablation is diversity-then-specialize,
   not one giant undifferentiated pool.

## Open gaps / unconfirmed

- Exact Misraj-DocOCR benchmark composition beyond "400 diverse images" — no
  public dataset link found.
- Whether the Muharaf training stage used any synthetic augmentation beyond
  grayscale conversion — the paper doesn't say; likely no.
- Oblevit, 3reeq, Latent Narratives, Not Gemma (NAKBA's other top-4
  finishers) released no loadable checkpoints, so their methodology wasn't
  independently verifiable beyond the shared-task overview paper (not
  fetched in this pass).

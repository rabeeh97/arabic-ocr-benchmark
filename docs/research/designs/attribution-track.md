# Design: the attribution and localisation track (open items)

Map items: `X-A2b`, `X-A3`, `X-A4-A5-A7`, `X-A6`, `X-D0`, `X-D1`, `X-C2`, `grounding-lit`, `pointing-loss` (dropped). The full
plan and the done items (X-A1, X-B1, X-B2, X-C1) are in [`../xai-attribution-plan.md`](../xai-attribution-plan.md) and the
ledger's attribution section. Status: parked behind the characterisation work, because errors are misreadings of the right
region, not misdirected looking (X-B2).

## Order when resumed
1. **`X-A3` image reliance.** Per-token log-probability with and without the image, per VLM. It tells whether maps mean anything
   for a given model. Shared with `decoding-and-restoration.md` §6.
2. **`X-A6` maps on dots.** Gradient mass on dot components against other ink.
3. **`X-A7` other models.** Attribution on the CRNN against its own CTC spans (cheap), then other VLMs.
4. **`X-A2b` letter-family sample.** At least 50 tokens per family (the خ lead: 7 of 10 wrong), error rate with intervals.
5. **`X-D0` per-module gradient probe.** Gradient size per LoRA module and layer for tokens of ا ل against ج ح خ. No training.
6. **`X-D1` freeze-by-letter** (wild, gated on X-D0 showing family-specific modules). Protect weights that detect each letter
   group; ΔCER.
7. **`X-A4` Chefer relevance maps** and **`X-A5` map sensitivity**, before any method comparison is written up.
8. **`X-C2` LocateAnything fine-tune** on word and PAW pseudo-labels from CTC spans. Gated: span quality (shape-library
   extraction will tell) and a baseline of no usable word box.

## Literature to read or check (`grounding-lit`)
- `qmask2026`: read in full; does it work below word level?
- `positionocr2026`: compare with CTC alignment as the spotter.
- `instructocr2025`: background only.
- `chinesesegfree2022`: an option if CTC spans are too coarse on cursive joins.
- `vmlhd`: check licence and format. It is the validation set for X-B1 letter spans; otherwise hand-check 15 lines
  (`bullinger2025ctcalign`).
- `niss2023charseg`: read the full text; is the data public?

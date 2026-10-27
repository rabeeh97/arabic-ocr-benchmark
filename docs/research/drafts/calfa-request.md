# Draft (NOT sent): request to Calfa about the AMIDDA per-corpus CRNNs

To: the AMIDDA authors (Chahan Vidal-Gorène, Calfa; contact address to be taken from the paper or calfa.fr, chosen by the user)
Subject: Question about the per-corpus CRNN baselines in "Lessons from AMIDDA" (HIP 2026)

Dear Dr Vidal-Gorène,

We are working on Maghribi handwritten text recognition with your RASAM-1, RASAM-2 and TARIMA datasets, and we evaluate on the official
Hugging Face test splits, which (as we checked) are exactly the AMIDDA test lines for these three corpora. Thank you for releasing the
datasets, the splits and the AMIDDA model; we were able to reproduce the Qwen3.5-4B LoRA results closely (21.7 / 28.3 / 24.8 % CER on
RASAM-1 / RASAM-2 / TARIMA with the released weights and the card's settings).

Your per-corpus CRNN baselines (strategy vi: 3.6 / 4.9 / 3.9 %) are the strongest numbers on these lines, and we would like to compare
against them properly. Could you tell us:

1. Were these models trained only on the train (and validation) part of the current Hugging Face / AMIDDA splits, or on an earlier release
   of the corpora that may include lines now in the test split? (They are described as pre-existing production models.)
2. Training details beyond Section 4 (vi): number of epochs or stopping rule, optimiser and learning rate, augmentation, and whether a
   language model or lexicon is used at decoding.
3. Whether the models, or their predictions on the test lines, could be shared for research comparison.

For context, our current open baseline (a 2.4 M-parameter character CRNN with a character n-gram LM, trained on RASAM-1 + RASAM-2 + TARIMA)
reaches 4.12 / 8.92 / 7.04 % on the same lines. We also found 80 degenerate MS 1936 crops (1-7 px tall; 7 of them in the RASAM-2 test
split) that we re-cut from the page images; we would be glad to share the list and the re-cut crops.

With thanks and best regards,
[name, affiliation]

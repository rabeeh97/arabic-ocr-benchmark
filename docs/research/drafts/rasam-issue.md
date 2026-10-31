# Draft (NOT sent): GitHub issue for calfa-co/rasam-dataset

To be opened by the user at https://github.com/calfa-co/rasam-dataset/issues/new. Nothing has been posted.

---

**Title:** Broken line crops and mismatched references in the Hugging Face release (RASAM-1, RASAM-2, TariMa): automatic fix list

Hello, and thank you for releasing RASAM and TariMa together with the AMIDDA splits.

While benchmarking line recognition on the Hugging Face configs `RASAM-1`, `RASAM-2` and `tarima`, we found two kinds of defects. We
built a fix list for them automatically, from the page images, the PAGE XML and model readings. We hope it is useful for the issues
already open here: #6 (corrupted values) and PR #7 (manual validation of RASAM-1), and the comments on #2.

**1. Broken crops (250 lines, 25 of them in test).** The line is intact on the page image, but the published crop is not. Most are MS 1936
crops in RASAM-2 that are 1–7 px tall. We re-cut them from the page along the PAGE XML polygon. In the test split this affects 23
RASAM-2 lines and 2 RASAM-1 lines. These 23 lines alone add about 3 CER points to a RASAM-2 test score: for our CRNN, 10.79 on the
official crops and 7.79 on the re-cuts with the dropped lines removed.

**2. References that do not match the image (202 lines, 21 of them in test).** These are found by rules over two cross-fitted readers and
a character LM, plus the invalid lines of PR #7. The rules are: blank crop, reference of a neighbouring line, vertical or marginal strip,
truncated crop, reference far less probable than the reading, and whole pages of uncorrected-looking references. The page-level rule
independently flags BULAC MS ARA 1977 pages 0017, 0071 and 0158, the pages called "uncorrected HTR" in #2.

Everything is in one folder, with a manifest giving the Hugging Face index, the action and the reasons for each line, the re-cut PNGs and
a small scorer:
https://github.com/rabeeh97/arabic-ocr-benchmark/tree/worktree-xai-attribution-spike/release/rasam-clean

Two notes:
- **Not a replacement for a manual check.** Our list and PR #7 overlap only partly (precision 0.24, recall 0.30 against PR #7), so each
  catches defects the other misses.
- **Test scores barely change apart from the re-cuts.** Dropping the 21 test lines lowers CER by 0.0–0.6 points for every system we score,
  and the ranking of the systems does not change.

Would you consider:
1. replacing the 250 crops in the Hugging Face release, and
2. marking or removing the dropped lines, or a corrected list after your own check?

We are happy to open a PR in whatever format suits you, e.g. a CSV like PR #7's.

Best regards,
[name, affiliation]

# Why every RASAM paper has its own split, and which Maghrebi datasets exist (2026-10-25)

Sources were read directly unless marked (snippet). Literature sweep by OpenAlex citations and web search; three paywalled Springer papers that cite RASAM
could not be read.

## 1. The splits

| Work | How it split | Reason given | Split published? | Same as the HF split? |
|---|---|---|---|---|
| [RASAM, ICDAR 2021](https://halshs.archives-ouvertes.fr/halshs-03430697) | "80 % of data for training and 20 % for testing" in its transfer experiment; unit (line or page) not stated; the 4.8 % has no split described. Pages were "randomly selected" for the corpus | none | no (the [GitHub release](https://github.com/calfa-co/rasam-dataset) has no split files; its README even says 450 images vs 300 in the paper) | not established |
| [Lucas et al. 2022](https://arxiv.org/abs/2211.16147) | page-level test: 10 pages of MS 1977, 7 of MS 609, 15 of MS 417 (32 pages), labels standardised from 54 to 38 classes; 3.23 % is the ARA-mWB "All" row of Table 3 | not stated (it standardises data across 5 architectures) | page counts only, no page IDs | no: HF test lines come from pages that are also in train |
| [RASAM 2, CHR 2024](https://ceur-ws.org/Vol-3834/paper35.pdf) | no split defined: random image selection, then in-domain and out-of-domain "samples"; footnote 4 says Lucas used "a different split" | none | no | not defined |
| [HAFITH](https://huggingface.co/datasets/mdnaseif/hafith-combined-benchmark) | 4,560 RASAM lines, 2,739 / 915 / 906 (about 60/20/20); unit and method undocumented; no paper yet | none | yes, the split itself is released as an HF dataset (only `image`, `text`) | no (counts do not match 7,331 / 3,720) |
| [Phoenix / Athar](https://arxiv.org/abs/2608.19385) | 1,789-line RASAM development guard, separated by "manuscript/page manifests" | "official splits of several datasets are not manuscript-independent", so they would not show generalisation to unseen manuscripts | manifests said to be fixed; no public link found | no |

Calfa's Hugging Face mirrors (RASAM-1, RASAM-2, TariMa) were created on **2026-05-20** (last modified 2026-09-11); their cards do not say how the splits
were made, and our own check shows they are random **by line** (every test page also has lines in train).

**Why the papers use their own splits.** Every RASAM paper (2021, 2022, 2024) predates the Hugging Face splits by two to five years, and the original GitHub
release has no split at all, so each paper had to make one; none describes it well enough to reproduce. The two papers written after the HF release (HAFITH,
Phoenix) still made their own: Phoenix explicitly because a random line split mixes lines of the same page and manuscript into train and test, which overstates
how well a model reads a new manuscript. So the published numbers are on unknown, mutually different lines, and the HF split is, as far as we can tell, the first
common one (that it is "official" is our inference; Calfa has not said so).

**What this means for comparisons.** Only systems scored on the HF test lines are strictly comparable. To compare with the 4.8 % / 6.79 % models properly we would
need either their exact test lists or their models (not public), and then a check that their training lines do not include our test lines. Our B2 experiment
(section 8 of `designs/rasam-crnn-and-replication.md`) suggests that for RASAM-2 the line-level mixing barely changes the CER of our models; for RASAM-1 it is untested.

## 2. Maghrebi datasets beyond RASAM

| Dataset | Size | Content | Licence | Where | Best CER (source) |
|---|---|---|---|---|---|
| [TariMa](https://huggingface.co/datasets/calfa-ai/tarima) | 2,285 lines (1,828 / 228 / 229), 26 documents | Maghrebi manuscripts and lithographs (BULAC, BnF, MMSH), same Calfa guidelines as RASAM | Apache-2.0 | HF, cached on the SSD | 9.70 % out-of-domain with the RASAM-2 model (CHR 2024 Table 4); Phoenix 10.72 % (abstract) |
| [AraMS-28k](https://arxiv.org/abs/2608.26921) | 28,600 lines, 14 books; **Maghrebi: 4 books, 8,890 lines** | Naskh, Ruq'ah, Maghrebi | CC BY-NC-SA 4.0 | Zenodo 10.5281/zenodo.21499649 (zip verified on the SSD) | book-level split; the Maghrebi test book (book_03): Kraken 32.71 %, HATFormer 37.88 % |
| Maouloud & Ba, [Springer 2026](https://doi.org/10.1007/978-3-032-22032-5_11) | "more than 50,000" line images from 5 books (snippet) | Maghrebi hands | unknown | dataset link not found | "up to 0.70 %" CER (snippet; split and leakage unknown, unverified) |
| [IRHAS](https://zenodo.org/records/21824878) | 10,825 lines, 19 manuscripts | Aljamiado (Old Spanish / Aragonese in Arabic script, likely Maghrebi-type hands) | CC BY-NC-SA 4.0 | Zenodo, 5.7 GB | no baseline given |
| [Wolof Ajami (LREC 2026)](https://aclanthology.org/2026.lrec-1.253/) | not stated | Wolof in Ajami script | not stated | no link | Arabic HTR models 61 to 81 % |
| OpenITI MAKHZAN | about 1,500 pages | Arabic, Persian, Ottoman, Urdu; Maghribi share not given | CC BY 4.0 | Zenodo | none |

No Timbuktu, Mauritanian, Moroccan or Tunisian line-level HTR set was found, and the Hugging Face hub has no other Maghrebi HTR dataset.

**Verdict.** RASAM (with TariMa) is still the only established Maghrebi line-level benchmark. The two datasets most worth adding are TariMa (same conventions,
already in training) and the four Maghrebi AraMS books (a book-level split, so a leakage-free test of reading unseen books). The 50,000-line Springer dataset could
change the picture if it is released; we could ask the authors.

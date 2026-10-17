# Lexicon and modern-text sources: licences and use notes

Checklist items 1.7 (lexicon) and 1.8 (modern Arabic text sample). Raw downloads live on the SSD under
`/Volumes/PortableSSD/data/lexicon/<source>/` and `/Volumes/PortableSSD/data/text/arwiki/`; built files under
`/Volumes/PortableSSD/data/lexicon/build/`. Nothing from the data is redistributed in this repo; only scripts
(`scripts/lexicon/`), this note, `manifest.json` (URLs, sizes, SHA-256) and the coverage report.

| Source | What we kept | Licence | Obligations / notes |
|---|---|---|---|
| **Ayaspell** (Hunspell `ar`, via LibreOffice/dictionaries) | `ar.dic` (465,885 stems, Ayaspell's own class tags for names, places, organisations) + `ar.aff` (1,586 affix rules) | GPL-2.0-or-later / LGPL-2.1-or-later / MPL-1.1 tri-licence (`COPYING.txt`) | Keep the licence text with any copy. Full affix expansion is ~4×10⁸ forms, so it is used through `scripts/lexicon/affix.py` (`Checker`), not as a list. |
| **aspell-ar** 1.2-0 (GNU aspell6, Buckwalter-derived, Google 2006) | `ar.cwl` (938,957 stems, prezip-decoded) + `ar_affix.dat` | GPL-2.0-or-later (`Copyright`, `COPYING`) | Expansion ~2.6×10⁷ forms; used through the same checker. |
| **Arabic Wordlist for Spellchecking** (Attia, Pecina, Samih, Shaalan, van Genabith, LREC 2012), v1.6 | `arabic-wordlist-1.6.txt` (fully inflected surface forms) | GPL-3.0-or-later (`COPYING` in the zip) | Cite the LREC 2012 paper. |
| **Wikidata** via QLever | Arabic labels and aliases of persons, given and family names, countries, administrative areas, settlements, organisations | CC0 1.0 | No obligations; record the retrieval date (`*.done` files). |
| **GeoNames** `alternateNamesV2` | rows with `isolanguage = ar` | CC BY 4.0 | Attribute GeoNames (geonames.org) in any derived publication or release. |
| **ANETAC** (Hadj Ameur, Meziane, Guessoum 2019) | Arabic side of the EN–AR transliteration pairs (train/dev/test) | No licence file in the repository (`HadjAmeur/ANETAC-Dataset`); the paper calls it freely available | Treat as research-only; cite arXiv:1907.03110. The public release has **no class labels** (person / location / organisation), so ANETAC entries are tagged `name` only. |
| **Arabic Wikipedia** 2023-11-01 (`wikimedia/wikipedia`, `20231101.ar`, shard 2 of 7) | parquet shard; paragraph text sample; vocabulary with counts | CC BY-SA 4.0 (and GFDL) | Text derived from it (LM training text, word lists with counts) is share-alike: attribute Wikipedia and keep CC BY-SA on redistributed derivatives. Models trained on it are not themselves restricted by most readings, but say so in any release. |

Not obtained: the **Alotaibi & Lee** named-entity gazetteer (optional in `designs/lexicon.md`; no stable public URL found).
Our own training texts (Muharaf, NAKBA, KHATT train vocabularies) are not copied here; they stay under their dataset
licences and can be added as frequency anchors from the existing data packs.

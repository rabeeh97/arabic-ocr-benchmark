# Design: a comprehensive Arabic lexicon with proper names (اسم علم)

Map item: `lexicon`. Origin: the user's idea of 2026-10-15. Used by: synthesis text (`shape-programme.md` step 3),
lexicon-constrained decoding and dot restoration (`decoding-and-restoration.md`), OOV analysis, the calibration sentence.

## Sources (download to the SSD; record each licence)

| Source | Content | Size | Licence | Notes |
|---|---|---|---|---|
| Ayaspell (Hunspell ar) | inflected word forms | ~300k | GPL / LGPL / MPL | via SourceForge; expand affixes with Hunspell's `unmunch` |
| aspell-ar | inflected forms (Buckwalter-based) | ~900k | GPL | Linux packages |
| Pecina et al., LREC 2012 | fully inflected surface forms | ~9M | to check | find the release location in the paper |
| Arabic Wikipedia dump | running text: vocabulary and frequencies, names in context | large | CC BY-SA | also the modern-text LM source |
| Wikidata (SPARQL) | persons, cities, countries, regions, organisations with Arabic labels and aliases | ~10⁵–10⁶ | CC0 | filter by Arabic label |
| GeoNames | place names with Arabic alternate names | ~10⁵ Arabic | CC BY | alternateNames table |
| Alotaibi & Lee NE corpora / gazetteer | fine-grained named entities from Wikipedia | — | research | KAU / SourceForge |
| ANETAC | English–Arabic named entities with class | 79,924 | free | person, location, organisation |
| Our training texts | Muharaf, NAKBA, KHATT train vocabularies | ~50k each | dataset licences | frequency anchors |

## Assembly
1. Normalise every source the same way as our labels (NFC, harakat stripped for the main list; a diacritised side list kept).
2. Tag each entry with source, class (common word, person, place, organisation) and frequency (Wikipedia counts where available).
3. Keep orthographic variants (hamza, yāʾ / alif maqṣūra, tāʾ marbūṭa) as linked forms, not merged. Our conventions analysis
   shows label sets differ in exactly these (ledger §17).
4. Deduplicate; build a dotless-skeleton index (for restoration) and a PAW index (for PAW-aware spacing and P-L3).
5. **Coverage report:** share of tokens in each test set covered (known, variant, unseen); proper-name share of OOV.

## Uses and checks
- **Synthesis:** draw text with controlled shares of common words, rare words and names, so the recogniser sees "all kinds of
  possibilities".
- **Decoding:** a lexicon-constrained beam; report the false-correction rate on proper names and on genuine OOV words.
- **Risk:** a large lexicon makes constrained decoding permissive, so measure coverage against false matches.

## Done when
- The lexicon is on the SSD with a manifest (sources, licences, counts).
- The coverage report is in the ledger.
- The decoding and synthesis code can load it.

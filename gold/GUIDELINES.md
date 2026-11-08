# Gold set labeling guidelines (v0.1 draft, 2026-11-08)

Purpose: every labeler who sees the same line should end up with the same text. Read the quick card, keep this page
open while you work, and when a case is not covered, write what you decided in the notes box (that is how these
guidelines get better). Changing a rule means a new version number; every label records the version it was made under.

## Quick card

- **Transcribe what the image shows, letter by letter, in the scribe's spelling.** Never what the word "should" be.
- **Each letter is written as its standard Unicode letter.** Missing, stray or smudged dots do not create a different letter: write the letter the word needs.
- **Do not modernise spelling.** Hamza only if written. Do not complete words from knowledge.
- **Diacritics: only marks you can clearly see.** Never add from knowledge, remove marks that are not in the image.
- **Spaces follow standard word boundaries**, not the gaps between connected pieces (PAW gaps).
- **`#` for each unreadable word.** One `#` per word, never a guess.
- **Do not transcribe crossed-out text.** Transcribe what stands after the correction.
- **Digits as written** (Arabic-Indic stays Arabic-Indic, Western stays Western). **Punctuation as drawn**, none added.
- **Logical reading order**, no direction marks, no line breaks. Multi-line crops: top line first, one space between lines.
- **No OCR or AI help for the transcription.** Your eyes decide. The toolkit's notes appear after you save a verdict, on purpose.

## 1. The principle

The verified text is *image-grounded*: someone who sees only the image could reproduce it. It is not the dataset's
convention, not a dictionary spelling, not a model's reading. Differences that are purely a scoring convention
(tatweel, direction marks, presentation-form characters, Unicode order of marks, extra spaces) are removed
automatically by the tool and by the scoring profiles; you do not need to think about them.

## 2. Choosing the verdict

1. No valid text in the image (blank, drawing, hopeless crop, nothing readable): **unusable**, and say why (image issue or notes).
2. You cannot decide between readings and would like a second opinion: **unsure**, and say why in the notes.
3. The label already equals what you would transcribe, letter for letter, including dots, diacritics, spaces, digits and punctuation: **correct**.
4. Anything differs: **corrected**. Type the full transcription and tick what was wrong in the original (letters, dots, missing text, extra text, diacritics, spacing, punctuation or digits, order, convention, other).
   - Use **convention** when the only difference is a transcription convention the original dataset follows, for example the label has no marks and the image has visible ones.

An illegible line cannot be correct or corrected; mark it unusable. Use legibility "partly_illegible" when some words are `#`.

## 3. Letters, dots, spelling

- Write the standard letter for each shape, so a scribe's undotted ب that must be ت for the word becomes ت. If the dots are absent and several words are possible, choose the reading that makes sense in the context and mark legibility **hard**.
- Keep the scribe's spelling, including old, dialectal or mistaken spellings. Do not correct them.
- Hamza: write it only where it is written. Do not add أ إ ؤ ئ from knowledge.
- Final ى or ي, and ة or ه: use the standard spelling of the word, because scribes dot them unreliably (على، إلى، حتى take ى).
- Lam-alef: type ل then ا. Never type the ligature code points.
- Tatweel (kashida): do not type it.
- Persian or Urdu letters inside Arabic text (ک ی ہ): write the Arabic letters (ك ي ه) unless the line itself is Persian or Urdu.
- Maghrebi script: qāf has one dot above and fā one dot below. Transcribe the intended letter (ق or ف), not the Mashriqi look-alike.

## 4. Diacritics (harakat)

- Transcribe a mark only if you can clearly see it. If it is a doubtful stroke, leave it out.
- Never add marks because the word "should" have them. Remove label marks that are not in the image.
- Use proper marks: fatha, damma, kasra, shadda, sukun, tanwin (fath, damm, kasr). The order you type shadda and the vowel does not matter.
- Quranic annotation signs, verse-end ornaments and page decorations are not transcribed. Mention them in the notes.
- Set **diacritics visible** from the image: none (no marks), sparse (a few), partial (many letters), full (almost every letter).

## 5. Spaces and word boundaries

- Use the spaces of standard Arabic orthography: a gap inside a word between non-connecting letters is **not** a space.
- Attached particles stay attached (وقال، بالكتاب، لل...).
- One space between words. No leading or trailing space. No line breaks.

## 6. Unreadable, damaged and edited text

- **Unreadable word: `#`**, one per word. If a letter or two is damaged but the visible ink clearly identifies it, write the letter. If you would have to guess the word from context, use `#`.
- **Crossed-out, erased or overwritten text:** do not transcribe the struck text; transcribe the final reading on top. Tick **scratched**.
- **Interlinear addition or caret insertion inside the crop:** transcribe it at the position the caret shows. If the position is unclear, leave it out. Tick **marginal_or_interlinear**.
- **Stains, holes, show-through:** transcribe what is readable, `#` for the rest, and tick the image issue.

## 7. Crop problems

- **A word cut by the crop at the start (right edge) or end (left edge):** write the letters that are visibly present (more or less half of the letter body must be there). Do not complete the word from knowledge. Tick **cropped_start** or **cropped_end**. The verdict is still correct or corrected: cropping is an image issue we count separately.
- **Ink of the neighbouring lines** (ascenders, descenders, parts of letters at the edge): ignore it and tick **neighbour_text**.
- **A second complete line inside the crop:** tick **multi_line** and transcribe all complete lines in reading order (top to bottom), separated by one space.
- **Text rotated or strongly skewed:** transcribe it and tick **skewed_or_rotated**.

## 8. Numbers, punctuation, symbols, mixed script

- Digits as written. Arabic-Indic (٠١٢) stays Arabic-Indic, Western (012) stays Western, Persian (۰۱۲) stays Persian. Scoring folds them.
- Abbreviations and abbreviation marks: transcribe as written, do not expand.
- Punctuation: transcribe what is drawn, using Arabic marks (، ؛ ؟) in Arabic text. Do not add punctuation that is not visible. Quotation marks: « » if guillemets, otherwise ".
- Latin letters or words inside Arabic text: write them as written (keep case), in logical order, without direction marks. Tick **mixed_script** in image issues.
- Verse: separate hemistichs with one space unless a separator mark is drawn.

## 9. Line kind, legibility, notes

- **Line kind:** main_text, heading, marginal, catchword, number_only (folio number, date, digits only), signature_or_stamp, poetry, other. A signature or stamp you can read is transcribed; one that is only a scrawl is unusable.
- **Legibility:** clear (every letter readable), hard (readable with effort or context), partly_illegible (some `#`), illegible (unusable).
- **Script style seen (optional):** write what you see (maghrebi, ruq'ah, naskh, thuluth...). Leave empty if you cannot tell.
- **Notes:** anything unusual, any rule you had to improvise. Short is fine.

## 10. Working rules

- Use your own labeler ID. Never share a login: the agreement check needs independent labelers.
- **Calibration lines (marked ★):** the first five lines of every dataset in tranche 1 are labelled by every labeler, blind. Do them before the rest of that dataset. After that, `scripts/gold_agreement.py` lists the disagreements and we update this document (new version) before labeling continues.
- Label dataset by dataset. Take breaks: tired labelers drift (the second-hour labels are the ones that disagree).
- Do not look up the line in the source dataset, in other models, or in a search engine.
- The toolkit's notes are revealed only after you save. Read them, but do not change a decision for a note unless you can see the problem in the image yourself.

## 11. English lines (IAM)

The same principles apply. Keep IAM's own tokenised spacing around punctuation exactly as in the original label (it is a property of IAM's ground truth; scoring folds it). Transcribe capitalisation as written, keep the writer's spelling mistakes, omit crossed-out words. Diacritics visible: n/a.

## 12. Recorded hard cases (draft for the future recorded set)

For lines we record ourselves the text is prompted, so the ground truth is the prompt and the labeler's job is to check that the writer wrote what was asked. When designing prompts, cover the known hard cases of handwriting recognition in any language and the Arabic-specific ones:
similar-letter minimal pairs (dots), diacritics-heavy words, ligatures and lam-alef, numbers and dates, abbreviations, Latin names and mixed script, crossed-out words, ruled and underlined paper, very small and very large writing, slant and curved baselines, overlapping lines, shadows and folds, low light, left-handed and child writers, rare words and names, nonsense strings (to catch models that lean on language priors), and text that contradicts the picture or common sense. Record capture conditions with every line.

## Version log

- **v0.1 (2026-11-08):** first draft, based on the RASAM conventions (scribe spelling kept, hamza only if written, `#` for illegible text, spaces restored), the findings in `docs/experiments/findings-and-notes.md`, and the label-review work on CER-HV flags. To be validated by the calibration lines.

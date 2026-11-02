# 06 · Open questions for the user

The answers go here; an agent starting phase 1 reads this first. ★ marks the recommended default if no answer is given.

1. **Name.** Tadqiq (تدقيق) ★, Musahhih (مصحح), Mizan (ميزان), something else? Check that PyPI, GitHub and the domain are free before committing.
2. **Code licence.** Apache-2.0 ★ (permissive; good for adoption and government use), or MIT, or AGPL (protects a hosted product from clones but scares companies).
3. **Repo split.**
   - ★ The toolkit and benchmark go in a new public repo; the platform goes in a separate private repo.
   - Alternatively, keep everything in `arabic-ocr-benchmark` for now.
4. **Product ownership.** Open-core (free toolkit, paid hosted platform and on-prem), a non-profit or academic service, or still undecided? This affects licence, branding and funding choices.
5. **Affiliation.** Is there a university, lab or PI to put on the paper and on any future applications?
6. **Venue and timeline.** Is the target ICDAR 2027 (verify the deadline), ArabicNLP 2027, or a journal first? Is there a hard date?
7. **v1 scope.** ★ Line-level OCR/HTR only, for the paper. Should speech or NLP-label datasets be in v1 (more impact, more work), or in a later version?
8. **Human verification.** Can you (and maybe 1–2 others) label about 800 flags for TadqiqBench-real? It takes about 4–6 hours each with the keyboard UI. Arabic-literate annotators for the Maghrebi lines are the bottleneck.
9. **Compute and money.** Is a small budget for VLM readings OK for E3 (about 500 lines × 2 models; we review the cost first)? Is the 3090 available again for the HATFormer/Kraken readings?
10. **Dataset maintainers.** Should we contact Calfa (RASAM), the KHATT and Muharaf authors before the audit is public? ★ Yes: send them findings as issues first. You send them.
11. **Privacy and sensitive texts.** Should religious texts (Qur'an and hadith verification) be in the public use-case list, or kept to private pitches only? ★ Private pitches only, until there is a partner institution.

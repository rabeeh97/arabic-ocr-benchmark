# Research notes — understanding Arabic script, text and data

The academic layer of the project: what the literature says about Arabic script
and text as a recognition problem, measured against our own data, and turned
into experiment hypotheses and requirements for our own dataset.

**Start here:** [`GUIDELINES.md`](GUIDELINES.md) (direction, rules, what we know) and [`NEXT_STEPS.md`](NEXT_STEPS.md) (the backlog).

**Live view:** the Arabic Script Atlas (https://claude.ai/artifact/8Z6dsabo5FDFPW11XQn7yf, private, built from `atlas/template.html` by `scripts/build_atlas.py`). Tracker data: [`progress.json`](progress.json). Rebuild after any change:

```bash
PYTHONPATH=src .venv/bin/python scripts/profile_datasets.py --muharaf-train   # ~4 min, CPU
PYTHONPATH=src .venv/bin/python scripts/build_atlas.py
```

Plan & priorities: Wave 0 (script & language characteristics) → Wave A
(diacritics, annotation schema, script types) → Wave B (architectures,
datasets, cross-lingual transfer). Every topic file follows
[`_TEMPLATE.md`](_TEMPLATE.md) and ends in **Implications for us**.

| wave | topic | file | status |
|---|---|---|---|
| 0 | Script & language characteristics (positional forms, dots, PAWs, ligatures, statistics, writer variability) | [`script-characteristics.md`](script-characteristics.md) | in progress |
| 0 | **Characterisation plan** (dataset + language) | [`characterization-plan.md`](characterization-plan.md) | proposed |
| 0 | Empirical text profile of our datasets (P1) | Atlas page + `results/characterize/<stamp>/` | first pass done |
| 0 | Error taxonomy of existing model predictions | [`error-taxonomy.md`](error-taxonomy.md) | planned |
| A1 | Diacritics (harakat, Unicode ambiguity, metrics) | `diacritics.md` | planned |
| A2 | Annotation & input schema | `annotation-schema.md` | planned |
| A3 | HTR vs print vs historical fonts vs layouts | `script-types.md` | planned |
| B1 | Architectures (CTC → seq2seq → VLM; ViT freezing; tokenizers) | `architectures.md` | planned |
| B2 | Arabic datasets taxonomy | `arabic-datasets.md` | planned |
| B3 | Other languages' datasets & transferability | `cross-lingual-transfer.md` | planned |
| — | Our own dataset: living spec | [`dataset-design.md`](dataset-design.md) | started |

Related existing notes (not duplicated here):
[`../datasets-catalog.md`](../datasets-catalog.md) §5,
[`../experiments/misraj-baseer-research.md`](../experiments/misraj-baseer-research.md),
[`../models.md`](../models.md),
[`../experiments/findings-and-notes.md`](../experiments/findings-and-notes.md).

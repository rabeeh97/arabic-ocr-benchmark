"""Gold set: a seeded, de-duplicated, verified sample of lines from the datasets we benchmark on.

Pipeline (see gold/README.md):
    pool.build_pool      rank every eligible test line per dataset with one seed (dedup by image)
    tranche.export       take ranks 1..50, 51..100, ... of each dataset, attach tags and toolkit notes
    labeler_app          local web tool: verify / correct each line, one file per labeler ID
    agreement            how closely two labelers landed on the same text
"""
GUIDELINE_VERSION = "0.1"

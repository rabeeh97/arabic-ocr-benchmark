#!/bin/zsh
# RS2: full RASAM test sets, both published Kraken models, binarised input.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike
M=/Volumes/PortableSSD/models/rasam-replication
for ds in RASAM-1 RASAM-2; do
  PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_kraken_score.py --model $M/phoenix/model.mlmodel --name phoenix --dataset $ds --split test --variant bin 2>&1 | grep -v -i "warn\|Recognizers"
  PYTHONPATH=src ~/.venvs/kraken-rasam/bin/python scripts/rasam_kraken_score.py --model $M/tarima/model-manuscript_best.mlmodel --name tarima-manuscript --dataset $ds --split test --variant bin 2>&1 | grep -v -i "warn\|Recognizers"
done
echo ALL DONE

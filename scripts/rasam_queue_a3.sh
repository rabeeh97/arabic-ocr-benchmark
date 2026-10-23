#!/bin/zsh
# Wait for A2 to finish, then resume A3 (RASAM-1 + RASAM-2 train). A3 resumes from its last checkpoint (epoch 5 as of 2026-10-23).
# Run detached:  nohup caffeinate -i scripts/rasam_queue_a3.sh &
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike
R=/Volumes/PortableSSD/models/rasam-replication/crnn-runs
until grep -q -E "^done" $R/A2_w05_s0.log; do sleep 120; done
echo "A2 done $(date); starting A3" >> $R/queue.log
$HOME/.venvs/kraken-rasam/bin/python scripts/xai_train_aligner.py \
  --data /Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_raw.pkl --out $R/A3_w05_s0 \
  --epochs 30 --bs 16 --device cpu --width 0.5 --train-sets r1 r2 --eval-every 2 --seed 0 --geom-aug >> $R/A3_w05_s0.log 2>&1
echo "A3 finished $(date)" >> $R/queue.log

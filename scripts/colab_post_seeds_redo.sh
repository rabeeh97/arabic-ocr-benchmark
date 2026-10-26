#!/bin/zsh
# Redo the seed-2 LM decodes that colab_post_seeds.sh lost to an SSD disconnect on 2026-10-25 (RASAM-2, TariMa, repaired RASAM-2).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-seeds-t4-* | tail -1)/content/crnn_out/reptm_h96_w05_s2
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_seeds.log
echo "redo start $(date)" >> $LOG
for t in r2 tm; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --name RAW_reptm_s2_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --name RAW_reptm_s2_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
  --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
echo "REDO DONE $(date)" >> $LOG

#!/bin/zsh
# AraMS-28k Maghrebi books (2026-10-28): greedy + char-8-gram LM readings of our best model (60 epochs, model_best) on the four Maghrebi
# books, LM weight tuned on the RASAM-2 validation lines. Scored afterwards by scripts/arams_report.py (AraMS conventions).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-* | tail -1)/content/crnn_out/reptm_h96_w05_e60_ema
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/arams.log
echo "== arams $(date)" >> $LOG
$PY scripts/rasam_lm_decode.py --data $DD/arams_maghrebi_h96.pkl --run $RUN --ckpt model_best.pt --name AM_E60 --train-sets r1 r2 tm \
  --val-tag r2 --test-tags am03 am19 am20 am24 --order 8 --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
echo "ARAMS DONE $(date)" >> $LOG

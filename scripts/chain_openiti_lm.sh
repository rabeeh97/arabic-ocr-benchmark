#!/bin/zsh
# Idea 4 (2026-10-28): our 60-epoch model decoded with the in-domain char LM interpolated with the OpenITI char 8-gram (big_char_lm.py).
# lambda (in-domain share; 1.0 = in-domain LM alone), alpha and beta tuned on 200 validation lines per set; AraMS books use RASAM-2 validation.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-long-ema-* | tail -1)/content/crnn_out/reptm_h96_w05_e60_ema
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
BIG=/Volumes/PortableSSD/models/rasam-replication/lm/openiti_o8.npz
PY=/Users/rabeehjouni/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/openiti_lm.log
G=(--order 8 --alphas 0.4 0.6 --betas 1.0 1.5 --lambdas 0.3 0.5 0.7 1.0 --val-n 200 --big-lm $BIG)
echo "== openiti LM $(date)" >> $LOG
for t in r1 r2 tm; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --ckpt model_best.pt --name OI_E60_$t --train-sets r1 r2 tm \
    --val-tag $t --test-tags $t $G >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --ckpt model_best.pt --name OI_E60_r2rep --train-sets r1 r2 tm \
  --val-tag r2 --test-tags r2 $G >> $LOG 2>&1
$PY scripts/rasam_lm_decode.py --data $DD/arams_maghrebi_h96.pkl --run $RUN --ckpt model_best.pt --name OI_AM_E60 --train-sets r1 r2 tm \
  --val-tag r2 --test-tags am03 am19 am20 am24 $G >> $LOG 2>&1
echo "OPENITI DONE $(date)" >> $LOG

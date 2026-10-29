#!/bin/zsh
# RS4g (2026-10-29): decode the pretrained + fine-tuned CRNN (pre_B) like the 60-epoch baseline: in-domain char LM alone (PT_*) and
# mixed with the OpenITI LM (PTOI_*), on the official RASAM-1 / RASAM-2 / TariMa test sets, repaired RASAM-2 and the AraMS Maghrebi books.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-pretrain-resume-t4-1029* | grep -v 10291205 | tail -1)/content/crnn_out/pre_B
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
BIG=/Volumes/PortableSSD/models/rasam-replication/lm/openiti_o8.npz
PY=/Users/rabeehjouni/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/pretrain_decode.log
IN=(--order 8 --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300)
OI=(--order 8 --alphas 0.4 0.6 --betas 1.0 1.5 --lambdas 0.3 0.5 0.7 1.0 --val-n 200 --big-lm $BIG)
echo "== pretrain decode $(date) $RUN" >> $LOG
for t in r1 r2 tm; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --ckpt model_best.pt --name PT_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t $IN >> $LOG 2>&1
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --ckpt model_best.pt --name PTOI_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t $OI >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --ckpt model_best.pt --name PTOI_r2rep --train-sets r1 r2 tm --val-tag r2 --test-tags r2 $OI >> $LOG 2>&1
$PY scripts/rasam_lm_decode.py --data $DD/arams_maghrebi_h96.pkl --run $RUN --ckpt model_best.pt --name PTOI_AM --train-sets r1 r2 tm --val-tag r2 --test-tags am03 am19 am20 am24 $OI >> $LOG 2>&1
echo "PRETRAIN DECODE DONE $(date)" >> $LOG

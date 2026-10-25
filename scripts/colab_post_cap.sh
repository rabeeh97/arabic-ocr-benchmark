#!/bin/zsh
# Score the full-width 96 px model (cut by its unit cap at about epoch 26) on the Mac: official and repaired tests, greedy and with the LM.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-cap-h96-* | tail -1)/content/crnn_out/a3_h96_w10
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_cap.log
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_raw_h96.pkl --run $RUN --name CAP_h96_w10_best --ckpt model_best.pt --tags r1 r2 >> $LOG 2>&1
for t in r1 r2; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_raw_h96.pkl --run $RUN --name CAP_h96_lm_$t --train-sets r1 r2 --val-tag $t --test-tags $t --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_raw_h96_repaired.pkl --run $RUN --name CAP_h96_lm_repairedtest --train-sets r1 r2 --val-tag r2 --test-tags r2 --order 8 \
  --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
echo "POST DONE $(date)" >> $LOG

#!/bin/zsh
# Score the R2 models (Colab, verified weights in ~/colab_staging) on the Mac: data packs at 96 and 128 px (built once on the SSD), both published test
# sets, greedy and with the character LM, result summaries and error analysis. The 128 px run was cut by its unit cap at about epoch 21 (partial).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
J=/Users/rabeehjouni/colab_staging/jobs/job-r2height-t4-t4-10251410/content/crnn_out
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_r2.log
echo "start $(date)" >> $LOG
for h in 96 128; do
  [ -f $DD/rasam_raw_h$h.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_raw_h$h.pkl --labels raw --height $h >> $LOG 2>&1
  RUN=$J/a3_h${h}_w05_bs16
  NAME=R2_h${h}_w05_s0_best
  [ -f $RUN/model_best.pt ] || { echo "no model for h$h" >> $LOG; continue; }
  $PY scripts/rasam_crnn_eval.py --data $DD/rasam_raw_h$h.pkl --run $RUN --name $NAME --ckpt model_best.pt --tags r1 r2 >> $LOG 2>&1
  for t in r1 r2; do
    $PY scripts/rasam_result_summary.py results/rasam/crnn/${NAME}__${t}__test.json >> results/rasam/queue_summary.txt 2>&1
    $PY scripts/rasam_error_analysis.py results/rasam/crnn/${NAME}__${t}__test.json --out results/rasam/analysis/${NAME}_${t}_test.json >> $LOG 2>&1
    $PY scripts/rasam_lm_decode.py --data $DD/rasam_raw_h$h.pkl --run $RUN --name R2_h${h}_lm_$t --train-sets r1 r2 --val-tag $t --test-tags $t --order 8 \
      --alphas 0.2 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
done
echo "POST DONE $(date)" >> $LOG

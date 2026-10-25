#!/bin/zsh
# Score the Colab A3 model (verified weights in ~/colab_staging) on the Mac: both published test sets, greedy and with the character LM,
# plus the result summary and the error analysis. Everything is written under results/rasam/ on the SSD.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
RUN=/Users/rabeehjouni/colab_staging/jobs/job-a3redo-t4-t4-10251408/content/crnn_out/a3_w05_bs16
D=/Volumes/PortableSSD/models/rasam-replication/crnn-data/rasam_raw.pkl
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_a3.log
echo "start $(date)" >> $LOG
$PY scripts/rasam_crnn_eval.py --data $D --run $RUN --name A3colab_w05_s0_best --ckpt model_best.pt --tags r1 r2 >> $LOG 2>&1
for t in r1 r2; do
  $PY scripts/rasam_result_summary.py results/rasam/crnn/A3colab_w05_s0_best__${t}__test.json >> results/rasam/queue_summary.txt 2>&1
  $PY scripts/rasam_error_analysis.py results/rasam/crnn/A3colab_w05_s0_best__${t}__test.json --out results/rasam/analysis/A3colab_${t}_test.json >> $LOG 2>&1
done
for t in r1 r2; do
  $PY scripts/rasam_lm_decode.py --data $D --run $RUN --name A3colab_lm_$t --train-sets r1 r2 --val-tag $t --test-tags $t --order 8 \
    --alphas 0.2 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
echo "POST DONE $(date)" >> $LOG

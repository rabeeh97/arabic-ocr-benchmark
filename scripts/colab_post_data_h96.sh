#!/bin/zsh
# Score the data-h96 Colab job on the Mac: (a) RASAM-1+2+TariMa model on all three test sets, (b) all-of-RASAM-1 model on RASAM-2 test only;
# both on the official and on the repaired RASAM-2 test; greedy and with the character LM (tuned on validation only).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-data-h96-* | tail -1)/content/crnn_out
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_data_h96.log
echo "start $(date)" >> $LOG
[ -f $DD/rasam_tm_h96.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h96.pkl --sets RASAM-1 RASAM-2 tarima --labels raw --height 96 >> $LOG 2>&1
[ -f $DD/rasam_tm_h96_repaired.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h96_repaired.pkl --sets RASAM-1 RASAM-2 tarima --labels raw --height 96 \
  --repair results/rasam/repairs/rasam2_recut.pkl >> $LOG 2>&1
# (a) TariMa model
RUN=$J/a3tm_h96_w05
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h96.pkl --run $RUN --name TM_h96_w05_best --ckpt model_best.pt --tags r1 r2 tm >> $LOG 2>&1
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --name TM_h96_w05_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
for t in r1 r2 tm; do
  $PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96.pkl --run $RUN --name TM_h96_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
$PY scripts/rasam_lm_decode.py --data $DD/rasam_tm_h96_repaired.pkl --run $RUN --name TM_h96_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
  --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
# (b) all of RASAM-1 + RASAM-2 train: score RASAM-2 only (its RASAM-1 test lines were training data)
RUN=$J/r1all_h96_w05
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_raw_h96.pkl --run $RUN --name R1ALL_h96_w05_best --ckpt model_best.pt --tags r2 >> $LOG 2>&1
$PY scripts/rasam_crnn_eval.py --data $DD/rasam_raw_h96_repaired.pkl --run $RUN --name R1ALL_h96_w05_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
$PY scripts/rasam_lm_decode.py --data $DD/rasam_raw_h96.pkl --run $RUN --name R1ALL_h96_lm_r2 --train-sets r1 r2 --val-tag r2 --test-tags r2 --order 8 \
  --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
$PY scripts/rasam_lm_decode.py --data $DD/rasam_raw_h96_repaired.pkl --run $RUN --name R1ALL_h96_lm_repairedtest --train-sets r1 r2 --val-tag r2 --test-tags r2 --order 8 \
  --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
echo "POST DONE $(date)" >> $LOG

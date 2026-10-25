#!/bin/zsh
# Score the clean-data + augmentation job on the Mac: three models trained on RASAM-1 + repaired RASAM-2 (96 px, half width):
# rep (no extra augmentation), rep_aug (+ neighbour ink + fade), rep_aug_dark (+ darkest-channel input). Official tests (RASAM-1, RASAM-2) and repaired
# RASAM-2; greedy and with the character LM (tuned on validation only).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
J=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-cleanaug-h96-* | tail -1)/content/crnn_out
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_cleanaug.log
echo "start $(date)" >> $LOG
[ -f $DD/rasam_raw_h96_dark.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_raw_h96_dark.pkl --sets RASAM-1 RASAM-2 --labels raw --height 96 --channel darkest >> $LOG 2>&1
[ -f $DD/rasam_raw_h96_dark_repaired.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_raw_h96_dark_repaired.pkl --sets RASAM-1 RASAM-2 --labels raw --height 96 \
  --channel darkest --repair results/rasam/repairs/rasam2_recut.pkl >> $LOG 2>&1
for spec in "rep_h96_w05 rasam_raw_h96.pkl rasam_raw_h96_repaired.pkl CA_rep" "rep_aug_h96_w05 rasam_raw_h96.pkl rasam_raw_h96_repaired.pkl CA_aug" \
            "rep_aug_dark_h96_w05 rasam_raw_h96_dark.pkl rasam_raw_h96_dark_repaired.pkl CA_dark"; do
  set -- ${=spec}
  RUN=$J/$1
  $PY scripts/rasam_crnn_eval.py --data $DD/$2 --run $RUN --name ${4}_best --ckpt model_best.pt --tags r1 r2 >> $LOG 2>&1
  $PY scripts/rasam_crnn_eval.py --data $DD/$3 --run $RUN --name ${4}_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
  for t in r1 r2; do
    $PY scripts/rasam_lm_decode.py --data $DD/$2 --run $RUN --name ${4}_lm_$t --train-sets r1 r2 --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  $PY scripts/rasam_lm_decode.py --data $DD/$3 --run $RUN --name ${4}_lm_repairedtest --train-sets r1 r2 --val-tag r2 --test-tags r2 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
done
echo "POST DONE $(date)" >> $LOG

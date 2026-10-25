#!/bin/zsh
# Score the repaired + TariMa models (raw labels and standardised labels) on the Mac, after the previous scoring has finished (8.6 GB RAM: one at a time).
# Raw-label model: official RASAM-1 / RASAM-2 / TariMa tests + repaired RASAM-2. Standardised-label model: the same tests with standardised references.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
until grep -q "POST DONE" /Users/rabeehjouni/colab_staging/logs/post_cleanaug.log 2>/dev/null; do sleep 30; done
DD=/Volumes/PortableSSD/models/rasam-replication/crnn-data
PY=$HOME/.venvs/kraken-rasam/bin/python
export PYTHONPATH=src
LOG=/Users/rabeehjouni/colab_staging/logs/post_reptm.log
echo "start $(date)" >> $LOG
score() {  # $1 run dir, $2 name, $3 official pack, $4 repaired pack
  $PY scripts/rasam_crnn_eval.py --data $DD/$3 --run $1 --name ${2}_best --ckpt model_best.pt --tags r1 r2 tm >> $LOG 2>&1
  $PY scripts/rasam_crnn_eval.py --data $DD/$4 --run $1 --name ${2}_repairedtest --ckpt model_best.pt --tags r2 >> $LOG 2>&1
  for t in r1 r2 tm; do
    $PY scripts/rasam_lm_decode.py --data $DD/$3 --run $1 --name ${2}_lm_$t --train-sets r1 r2 tm --val-tag $t --test-tags $t --order 8 \
      --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
  done
  $PY scripts/rasam_lm_decode.py --data $DD/$4 --run $1 --name ${2}_lm_repairedtest --train-sets r1 r2 tm --val-tag r2 --test-tags r2 --order 8 \
    --alphas 0.3 0.4 0.5 --betas 0.5 1.0 1.5 --val-n 300 >> $LOG 2>&1
}
# standardised-label packs (references standardised the same way as the training labels)
[ -f $DD/rasam_tm_h96_std.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h96_std.pkl --sets RASAM-1 RASAM-2 tarima --labels standardised --height 96 >> $LOG 2>&1
[ -f $DD/rasam_tm_h96_std_repaired.pkl ] || $PY scripts/rasam_prepare_crnn_data.py $DD/rasam_tm_h96_std_repaired.pkl --sets RASAM-1 RASAM-2 tarima --labels standardised --height 96 \
  --repair results/rasam/repairs/rasam2_recut.pkl >> $LOG 2>&1
STD=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-std-* | tail -1)/content/crnn_out/reptm_std_h96_w05
score $STD STD_reptm rasam_tm_h96_std.pkl rasam_tm_h96_std_repaired.pkl
# raw-label model (when its job has finished)
until ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-raw-*/content/crnn_out/reptm_h96_w05/model_best.pt > /dev/null 2>&1 \
      && grep -q "=== finished" /Users/rabeehjouni/colab_staging/logs/reptm_raw2.log 2>/dev/null; do sleep 60; done
RAW=$(ls -d /Users/rabeehjouni/colab_staging/jobs/job-reptm-raw-* | tail -1)/content/crnn_out/reptm_h96_w05
score $RAW RAW_reptm rasam_tm_h96.pkl rasam_tm_h96_repaired.pkl
echo "POST DONE $(date)" >> $LOG

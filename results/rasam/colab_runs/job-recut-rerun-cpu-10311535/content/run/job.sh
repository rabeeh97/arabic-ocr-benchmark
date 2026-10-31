#!/bin/bash
cd /content
mkdir -p /content/run
stamp(){ echo "STAMP $1 $(date +%s)"; }
stamp start
nvidia-smi -L
HFP=
if [ -f /content/.hf_token ]; then chmod 600 /content/.hf_token; ( while true; do sleep 300; python /content/colab_hf_push.py SubSpring/arabic-ocr-colab-runs job-recut-rerun-cpu-10311535 >> /content/run/hf_push.log 2>&1; done ) & HFP=$!; fi
FAILS=0
stamp setup_start
git clone --depth 1 https://github.com/mdnaseif/hafith.git /content/hafith_src && pip install -q -e /content/hafith_src && pip install -q 'transformers>=4.45,<5' && echo HAFITH_INSTALLED
rc=$?; echo "RC setup $rc"; [ $rc -ne 0 ] && FAILS=$((FAILS+1))
stamp setup_end
stamp aragpt2_scores_start
cd /tmp && python -u /content/colab_remote_lmscore.py aubmindlab/aragpt2-base
rc=$?; echo "RC aragpt2_scores $rc"; [ $rc -ne 0 ] && FAILS=$((FAILS+1))
stamp aragpt2_scores_end
stamp hafith_recut_start
cd /tmp && python -u /content/colab_recut_rerun.py hafith
rc=$?; echo "RC hafith_recut $rc"; [ $rc -ne 0 ] && FAILS=$((FAILS+1))
stamp hafith_recut_end
stamp amidda_recut_start
pip install -q -U 'transformers>=5.9' accelerate && cd /tmp && python -u /content/colab_recut_rerun.py amidda
rc=$?; echo "RC amidda_recut $rc"; [ $rc -ne 0 ] && FAILS=$((FAILS+1))
stamp amidda_recut_end
[ -n "$HFP" ] && kill $HFP
python /content/colab_hf_push.py SubSpring/arabic-ocr-colab-runs job-recut-rerun-cpu-10311535 final >> /content/run/hf_push.log 2>&1
echo "RC hfpush $?"
echo "ALLDONE failed_steps=$FAILS"
exit $FAILS

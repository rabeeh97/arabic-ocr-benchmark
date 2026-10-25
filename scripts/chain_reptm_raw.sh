#!/bin/zsh
# The account allows 2 GPU sessions: start the repaired+TariMa raw-label job as soon as the clean-data + augmentation job has finished.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
until grep -q "=== finished" /Users/rabeehjouni/colab_staging/logs/cleanaug_h96.log 2>/dev/null; do sleep 30; done
sleep 30
scripts/colab_job_retry.sh /Users/rabeehjouni/colab_staging/logs/reptm_raw2.log configs/colab/reptm_raw_t4.json

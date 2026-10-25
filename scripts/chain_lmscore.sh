#!/bin/zsh
# Wait for the N-best lists, then run the LM-scoring job on a Colab CPU session (detached chain; logs in ~/colab_staging/logs).
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike || exit 1
until grep -q "NBEST DONE" /Users/rabeehjouni/colab_staging/logs/nbest.log 2>/dev/null; do sleep 30; done
scripts/colab_job_retry.sh /Users/rabeehjouni/colab_staging/logs/lmscore.log configs/colab/lmscore_cpu.json

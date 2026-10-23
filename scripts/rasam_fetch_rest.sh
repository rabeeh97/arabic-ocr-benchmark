#!/bin/zsh
# Sequential, retrying fetch of the remaining RASAM-study data (run detached: nohup caffeinate -i scripts/rasam_fetch_rest.sh).
# One download at a time on purpose: Zenodo answers with retry-after: 60 and drops concurrent connections.
cd /Volumes/PortableSSD/repos/arabic_ocr/.claude/worktrees/xai-attribution-spike
PY=$HOME/.venvs/kraken-rasam/bin/python
HF=$HOME/.venvs/kraken-rasam/bin/hf
LOG=/Volumes/PortableSSD/data/fetch_rest.log
Z=/Volumes/PortableSSD/data/arams_28k_htr/AraMS-28k-HTR.zip
echo "start $(date)" >> $LOG

# 1. AraMS-28k-HTR (2,423,023,171 bytes, md5 ff673b1378861b730fcc31a44a4618d5)
for i in {1..30}; do
  size=$(stat -f%z $Z 2>/dev/null || echo 0)
  [ "$size" -ge 2423023171 ] && break
  echo "aramS try $i at $size bytes" >> $LOG
  curl -sS -L -C - --max-time 1500 -o $Z "https://zenodo.org/api/records/21499649/files/AraMS-28k-HTR.zip/content" >> $LOG 2>&1
  sleep 65
done
echo "aramS md5: $(md5 -q $Z)" >> $LOG

# 2. RASAM page images (rerun-safe; skips what exists)
for i in {1..6}; do
  $PY scripts/rasam_fetch_pages.py --delay 1.0 >> $LOG 2>&1
  n=$(ls /Volumes/PortableSSD/data/rasam-pages/images | wc -l)
  [ "$n" -ge 547 ] && break
  sleep 30
done

# 3. AMIDDA (9.5 GB, resumable)
for i in {1..8}; do
  $HF download calfa-ai/AMIDDA --repo-type dataset >> $LOG 2>&1 && break
  sleep 60
done
echo "ALL DONE $(date)" >> $LOG

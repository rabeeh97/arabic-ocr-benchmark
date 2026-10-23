#!/bin/zsh
# Clean re-download of AraMS-28k-HTR (Zenodo 10.5281/zenodo.21499649) onto the SSD, with MD5 verification.
# The first attempt was corrupted by two overlapping curl runs appending to one file (2.84 GB, wrong md5).
# Everything stays on /Volumes/PortableSSD. The zip is NOT extracted: its ~28k PNGs would each take a 1 MB exFAT cluster (~28 GB).
# Read images straight from the zip with zipfile; only the small manifests are extracted.
# Run detached:  nohup caffeinate -i scripts/rasam_fetch_arams.sh &
D=/Volumes/PortableSSD/data/arams_28k_htr
Z=$D/AraMS-28k-HTR.zip
LOG=/Volumes/PortableSSD/data/fetch_arams.log
WANT=2423023171
MD5=ff673b1378861b730fcc31a44a4618d5
mkdir -p $D
if [ -f $Z ] && [ "$(stat -f%z $Z)" -ne $WANT ] && [ "$(md5 -q $Z)" != "$MD5" ]; then
  echo "removing corrupted zip $(date)" >> $LOG; rm -f $Z
fi
for i in {1..40}; do
  size=$(stat -f%z $Z 2>/dev/null || echo 0)
  [ "$size" -ge $WANT ] && break
  echo "try $i at $size bytes $(date +%H:%M)" >> $LOG
  curl -sS -L -C - --max-time 1500 -o $Z "https://zenodo.org/api/records/21499649/files/AraMS-28k-HTR.zip/content" >> $LOG 2>&1
  sleep 65
done
got=$(md5 -q $Z)
echo "size $(stat -f%z $Z) md5 $got (want $MD5)" >> $LOG
if [ "$got" = "$MD5" ]; then
  unzip -o -q $Z '*manifest*' -d $D >> $LOG 2>&1
  echo "VERIFIED, manifests extracted $(date)" >> $LOG
else
  echo "MD5 MISMATCH, do not use" >> $LOG
fi

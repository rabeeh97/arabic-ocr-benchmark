"""RS2: download the original RASAM page images (BULAC IIIF) named in calfa-co/rasam-dataset list-images.tsv.

    ~/.venvs/kraken-rasam/bin/python scripts/rasam_fetch_pages.py [--sets RASAM1 RASAM2] [--delay 0.5]

Reads /Volumes/PortableSSD/data/rasam-pages/rasam-dataset-main/list-images.tsv (the repo zip's page/ XML sits beside it) and writes
/Volumes/PortableSSD/data/rasam-pages/images/<FileName>.jpg, skipping files already present (rerun-safe). The IIIF "full/full"
request redirects to the original file; its pixel size is smaller than the size the PAGE-XML coordinates use (same aspect ratio),
so coordinates must be scaled by the ratio of the XML's imageWidth to the file's width.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

BASE = Path("/Volumes/PortableSSD/data/rasam-pages")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["RASAM1", "RASAM2"])
    ap.add_argument("--delay", type=float, default=0.5)
    a = ap.parse_args()
    out = BASE / "images"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(BASE / "rasam-dataset-main/list-images.tsv", encoding="utf-8"), delimiter="\t")
            if r["Dataset"] in a.sets]
    print(len(rows), "pages listed", flush=True)
    ok = fail = skipped = 0
    for i, r in enumerate(rows):
        dst = out / f"{r['FileName']}.jpg"
        if dst.exists() and dst.stat().st_size > 10_000:
            skipped += 1
            continue
        url = f"https://bina.bulac.fr/iiif/2/{r['IIIF image ID']}/full/full/0/default.jpg"
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                dst.write_bytes(resp.content)
                ok += 1
            else:
                fail += 1
                print("FAIL", r["FileName"], resp.status_code, flush=True)
        except Exception as e:  # network hiccup: keep going, a rerun fills the gaps
            fail += 1
            print("ERR", r["FileName"], repr(e)[:100], flush=True)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} ok={ok} fail={fail} skipped={skipped}", flush=True)
        time.sleep(a.delay)
    print(f"DONE ok={ok} fail={fail} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()

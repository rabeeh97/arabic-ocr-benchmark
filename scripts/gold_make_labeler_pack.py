#!/usr/bin/env python
"""Build a self-contained zip for another labeler: their share of a tranche + the labeling tool + instructions.

    PYTHONPATH=src python scripts/gold_make_labeler_pack.py --tranche 1 --name gold_part_b \
        --datasets baybars arams amidda onlinekhatt historyar isippt iam

The pack holds the lines of the listed datasets plus every calibration line (so both labelers do the same
calibration lines, blind, and gold_agreement.py can compare them). It needs only Python 3.10+ (standard library).
Writes gold/exports/<name>.zip and records the split in gold/tranches/tNN/assignments.json.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["src/arabic_ocr_bench/__init__.py", "src/arabic_ocr_bench/normalize.py",
        "src/arabic_ocr_bench/gold/__init__.py", "src/arabic_ocr_bench/gold/schema.py",
        "src/arabic_ocr_bench/gold/labels.py", "src/arabic_ocr_bench/gold/tranche.py",
        "src/arabic_ocr_bench/gold/labeler_app.py", "src/arabic_ocr_bench/gold/static/labeler.html",
        "scripts/gold_label_app.py", "gold/GUIDELINES.md"]

START = """# Gold set labeling: start here

You have a share of the gold set: {n} text-line images ({n_own} of your own, plus {n_cal} shared "calibration"
lines that every labeler does). Your job is to check each line's text against its image and fix it where it is
wrong, following the guidelines.

## Run it (Python 3.10 or newer, nothing to install)

- macOS / Linux: `./run.sh` (or `sh run.sh`)
- Windows: double-click `run.bat`

Then open http://127.0.0.1:8770/ in your browser. The first time, it asks for your **labeler ID**: use your own
(for example your first name in lower case, 2-24 characters of a-z 0-9 _ -). Use the same ID every time, and never
share it: the agreement check needs every labeler's labels to be independent.

## What to do

1. Read `gold/GUIDELINES.md` (also under the "Guidelines" link in the tool). The "Quick card" at the top is the summary.
2. **Do the calibration lines first.** They are marked with a star in the left list; the "calibration lines only" tick box shows just them. Label them without discussing them with anyone.
3. Then label your own lines, dataset by dataset (the dropdown at the top). Everything you save is kept; you can stop and continue later, and editing a saved line simply overwrites it.
4. Pick a verdict for each line (correct / corrected / unusable / unsure). For "corrected", type the full verified text and tick what was wrong. Always check the three drop-downs (line kind, legibility, diacritics visible).
5. Write anything unusual, or any rule you had to improvise, in the Notes box. That is how the guidelines improve.

Shortcuts: Ctrl+Enter saves (press it again to go to the next line), Ctrl+1 to Ctrl+4 pick the verdict, [ and ] go to the previous and next line.

Please do not use OCR or AI tools to read the lines, and do not look the lines up elsewhere. After you save a verdict, the tool shows
notes from our error-detection toolkit; read them, but decide from the image.

## When you are done (or want to hand in what you have so far)

Zip the folder `gold/labels/` (it contains one folder named after your ID) and send it back. Do not send the images.

## Practical

- Stop the tool with Ctrl+C in the terminal. If port 8770 is busy, change `8770` to another number in run.sh / run.bat.
- The images and texts come from third-party datasets with their own licences. Please keep this pack private.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="datasets that go to this pack (the rest stay with the owner)")
    a = ap.parse_args()
    tdir = ROOT / "gold" / "tranches" / f"t{a.tranche:02d}"
    lines = [json.loads(l) for l in open(tdir / "lines.jsonl", encoding="utf-8") if l.strip()]
    bad = set(a.datasets) - {x["dataset"] for x in lines}
    if bad:
        raise SystemExit(f"unknown datasets: {sorted(bad)}")
    sel = [x for x in lines if x["dataset"] in a.datasets or x["calibration"]]
    own = [x for x in sel if x["dataset"] in a.datasets and not x["calibration"]]
    cal = [x for x in sel if x["calibration"]]

    stage = ROOT / "gold" / "exports" / a.name
    if stage.exists():
        shutil.rmtree(stage)
    out = stage / "gold" / "tranches" / tdir.name
    (out / "images").mkdir(parents=True)
    for rel in CODE:
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, stage / rel)
    for x in sel:
        if x["image"]:
            (out / x["image"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tdir / x["image"], out / x["image"])
    with open(out / "lines.jsonl", "w", encoding="utf-8") as f:
        for x in sel:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    man = json.load(open(tdir / "manifest.json", encoding="utf-8"))
    man["datasets"] = {k: {"line_ids": [x["line_id"] for x in sel if x["dataset"] == k]} for k in sorted({x["dataset"] for x in sel})}
    man["n_lines"] = len(sel)
    man["pack"] = a.name
    json.dump(man, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (stage / "START_HERE.md").write_text(START.format(n=len(sel), n_own=len(own), n_cal=len(cal)), encoding="utf-8")
    (stage / "run.sh").write_text('#!/bin/sh\ncd "$(dirname "$0")" || exit 1\nPYTHONPATH=src exec python3 scripts/gold_label_app.py '
                                  f'--tranche gold/tranches/{tdir.name} --port 8770\n', encoding="utf-8")
    (stage / "run.sh").chmod(0o755)
    (stage / "run.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\nset PYTHONPATH=src\r\n'
                                   f'py scripts\\gold_label_app.py --tranche gold\\tranches\\{tdir.name} --port 8770\r\n'
                                   'pause\r\n', encoding="utf-8")
    zpath = ROOT / "gold" / "exports" / f"{a.name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo.from_file(p, str(Path(a.name) / p.relative_to(stage)))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
    asg = tdir / "assignments.json"
    d = json.load(open(asg, encoding="utf-8")) if asg.exists() else {"parts": {}}
    d["parts"][a.name] = {"datasets": a.datasets, "plus": "all calibration lines (shared by every labeler)"}
    json.dump(d, open(asg, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{zpath}\n  lines: {len(sel)} = {len(own)} own + {len(cal)} calibration; {zpath.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

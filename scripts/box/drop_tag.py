"""Remove one tag from crnn_eval_panel.py result files so a rerun scores it again (rerun skips finished tags).

    python drop_tag.py OUT_DIR TAG --pack NEW_PACK

Used after the 2026-10-18 inkcrop fix: the kitab_khatt_full__inkcrop rows of khatt_full_panel.pkl were uncropped copies.
Drops TAG from per_line / cer / conf of every OUT_DIR/*.json and sets "panel" to NEW_PACK (the rebuilt pickle's split
name), so store_build.py stops treating the file as the stale pack. The other tags are kept as they are: their images did
not change. Writes through a .tmp file; files without TAG are left alone.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--pack", required=True)
    a = ap.parse_args()
    for f in sorted(a.out_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if a.tag not in d.get("per_line", {}):
            print(f"{f.name}: no {a.tag}, left as is")
            continue
        for k in ("per_line", "cer", "conf"):
            d.get(k, {}).pop(a.tag, None)
        d["panel"] = a.pack
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)
        print(f"{f.name}: dropped {a.tag}")


if __name__ == "__main__":
    main()

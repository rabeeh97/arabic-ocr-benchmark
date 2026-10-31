"""Score RASAM-1 / RASAM-2 / TariMa test predictions on the official and the clean test view. Standard library only.

    python score.py --config RASAM-1 --pred my_preds.json [--recut-pred my_recut_preds.json] [--refs refs.json]

--pred        {"<test index>": "<reading>"} (or a list of readings in test-index order) for the official Hugging Face test crops
--recut-pred  optional {"<test index>": "<reading>"} for the re-cut crops in recut/<config>/test/ (read them with the same model);
              without it the clean view uses the official crops for those lines (the "drop-only" view)
--refs        optional {"<test index>": "<reference>"}; by default the references are taken from the Hugging Face dataset
              (needs `datasets`), loaded from --repo (default calfa-co/rasam-dataset, config as given)
Prints micro CER (sum of edits / sum of reference characters) on: official (all lines), drop-only, and clean (re-cut + drop).
Text is compared after stripping invisible format characters, folding Arabic presentation forms and NFC (no other normalisation).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent


def canon(s: str) -> str:
    s = "".join(unicodedata.normalize("NFKC", ch) if 0xFB50 <= ord(ch) <= 0xFDFF or 0xFE70 <= ord(ch) <= 0xFEFF else ch
                for ch in (s or "") if unicodedata.category(ch) != "Cf")
    return unicodedata.normalize("NFC", s).strip()


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {int(k): v for k, v in d.items()} if isinstance(d, dict) else dict(enumerate(d))


def cer(refs, hyps, idx):
    e = n = 0
    for i in idx:
        r = canon(refs[i])
        e += lev(r, canon(hyps.get(i, "")))
        n += len(r)
    return 100 * e / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["RASAM-1", "RASAM-2", "tarima"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--recut-pred")
    ap.add_argument("--refs")
    ap.add_argument("--repo", default="calfa-co/rasam-dataset")
    a = ap.parse_args()
    if a.refs:
        refs = load(a.refs)
    else:
        from datasets import load_dataset
        ds = load_dataset(a.repo, a.config, split="test")
        col = next(c for c in ("text", "label", "transcription") if c in ds.column_names)
        refs = dict(enumerate(ds[col]))
    hyp = load(a.pred)
    drop = {int(k.split("/")[2]) for k in json.loads((HERE / "drop.json").read_text()) if k.startswith(f"{a.config}/test/")}
    recut = {int(p.stem) for p in (HERE / "recut" / a.config / "test").glob("[0-9]*.png")}
    keep = [i for i in sorted(refs) if i not in drop]
    print(f"{a.config}: {len(refs)} test lines, {len(drop)} dropped, {len(recut)} re-cut")
    print(f"  official   CER {cer(refs, hyp, sorted(refs)):6.2f}  ({len(refs)} lines)")
    print(f"  drop-only  CER {cer(refs, hyp, keep):6.2f}  ({len(keep)} lines)")
    if a.recut_pred:
        rc = load(a.recut_pred)
        missing = sorted(i for i in recut if i in keep and i not in rc)
        if missing:
            print(f"  clean      missing re-cut readings for test lines {missing}")
        else:
            print(f"  clean      CER {cer(refs, {**hyp, **{i: rc[i] for i in recut}}, keep):6.2f}  ({len(keep)} lines, {len(recut & set(keep))} re-cut)")


if __name__ == "__main__":
    main()

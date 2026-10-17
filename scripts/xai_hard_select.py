"""Hard-example control for shape targeting (checklist 4.7, shape-programme "Related checks": `hard-example-control`).

    python xai_hard_select.py --data aligner_data.pkl --run aligner_run --match shape_select\\targeted.json --out shape_select\\hard.json

Scores every Arabic training line (role train or panel; at least 8 characters, at least half of them Arabic letters) of the pickle with the run's CRNN (no augmentation, fp32, one line at a
time, so no padding effects) and gives the lines with the highest CTC loss per label character the same number of extra
copies, on the same number of lines, as the --match file (X-G11's shape-targeted set: 12 % of lines, 3 extra copies).
The output has the --repeat-file format of xai_train_aligner.py, so a run with it differs from the shape-targeted runs only
in which lines are repeated: "similar shapes" against "hard lines". Also writes <out stem>_losses.json (row index -> loss).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from xai_train_aligner import CRNN, make_batch  # noqa: E402


def arabic_line(label: str) -> bool:
    """At least 8 characters, and Arabic letters are at least half of the non-space characters."""
    s = label.replace(" ", "")
    return len(label) >= 8 and sum("ء" <= c <= "ي" for c in s) >= 0.5 * max(1, len(s))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--match", type=Path, required=True, help="repeat file whose line count and copy count to match")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-lines", type=int, default=0, help="score at most this many lines (smoke tests)")
    a = ap.parse_args()
    match = json.loads(a.match.read_text())["extra"]
    n_sel, copies = len(match), max(match.values())
    rows = pickle.load(open(a.data, "rb"))["rows"]
    chars = json.loads((a.run / "charset.json").read_text(encoding="utf-8"))["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    cfg = json.loads((a.run / "config.json").read_text()) if (a.run / "config.json").exists() else {}
    model = CRNN(len(chars) + 1, width=cfg.get("width", 1.0)).to(a.device).eval()
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=a.device))
    ctc = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    # like-for-like with the shape-targeted lines (Arabic text lines): without this filter the per-character loss ranks Latin
    # snippets, shelf marks and short numerals first
    idx = [j for j, r in enumerate(rows) if r["role"] in ("train", "panel") and arabic_line(r["label"])]
    if a.max_lines:
        idx = idx[: a.max_lines]
    loss = {}
    for n, j in enumerate(idx):
        x, lens, tgt = make_batch([rows[j]], stoi, False, a.device)
        if not tgt[0]:
            continue
        with torch.no_grad():
            lp = model(x).float().log_softmax(-1)
            v = ctc(lp, torch.tensor(tgt[0], device=a.device), torch.clamp(lens // 4, min=1), torch.tensor([len(tgt[0])], device=a.device))
        loss[j] = float(v) / len(tgt[0])
        if n % 5000 == 0:
            print("scored", n, "of", len(idx), flush=True)
    ranked = sorted(loss, key=lambda j: -loss[j])[:n_sel]
    a.out.write_text(json.dumps({"extra": {str(j): copies for j in ranked}}))
    a.out.with_name(a.out.stem + "_losses.json").write_text(json.dumps({str(j): round(v, 4) for j, v in loss.items()}))
    overlap = len(set(ranked) & {int(k) for k in match})
    print(json.dumps({"scored": len(loss), "selected": len(ranked), "copies": copies, "min_loss_selected": round(loss[ranked[-1]], 4) if ranked else None,
                      "overlap_with_match": overlap, "examples": [rows[j]["label"] for j in ranked[:5]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

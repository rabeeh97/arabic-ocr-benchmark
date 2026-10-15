"""vlm_ink_extent groups by the model field (checklist 1.9) and change_gallery shows line breaks in readings."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
vie = pytest.importorskip("vlm_ink_extent")
cg = pytest.importorskip("change_gallery")


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def row(model, i, gt, hyp):
    return {"model": model, "sample_id": f"kitab_khatt/train/{i:06d}", "ground_truth": gt, "cleaned_text": hyp}


def write(p, rows):
    p.parent.mkdir(parents=True)
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_groups_by_model_field_and_keeps_latest_run(tmp_path):
    old = tmp_path / "suite/bench/full/kitab_khatt/20260901T000000Z/predictions.jsonl"
    new = tmp_path / "suite/bench/full/kitab_khatt/20260902T000000Z/predictions.jsonl"
    write(old, [row("base", 0, "abc", "xxx"), row("ft", 0, "abc", "xxx")])
    write(new, [row("base", 0, "abc", "abc"), row("base", 1, "abcd", "abcd")])
    got = vie.latest_rows_by_model([new, old])
    assert set(got) == {"base", "ft"}  # the two models in one folder are not pooled
    assert len(got["base"]) == 2 and got["base"]["kitab_khatt/train/000000"]["cleaned_text"] == "abc"
    m = vie.score(got["base"].values(), {0: 0.2, 1: 0.9}, "stored", lev)
    assert m["lines"] == 2 and m["cer_all"] == 0.0


def test_paddle_unreverse_only_touches_paddle():
    gt = "سلام عليكم"
    pad, other = row("arabic-paddleocr-v5", 0, gt, gt[::-1]), row("qwen", 0, gt, gt[::-1])
    assert vie.score([pad], {0: 0.9}, "stored", lev)["cer_all"] > 0.5
    assert vie.score([pad], {0: 0.9}, "unreverse", lev)["cer_all"] == 0.0
    assert vie.score([other], {0: 0.9}, "unreverse", lev)["cer_all"] > 0.5


def test_gallery_shows_newlines():
    out = cg.marked("ab c", "ab\nc")
    assert "\n" not in out and "⏎" in out
    assert cg.esc("a\r\n<b>") == "a" + cg.NL + "&lt;b&gt;"

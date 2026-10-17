"""Unit checks for detectors D4, D21, D23, D24 and the store's kitab_khatt_full handling
(run: .geomvenv/bin/python -m pytest tests/test_detectors_extra.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from detectors_extra import (d4_flag, intrusion, lam_alef, lam_alef_letters, lam_and_alef_elsewhere,  # noqa: E402
                             quirk_hits, quirk_words, ws_quirks)
from seed_instability import cell_unstable, seed_unstable  # noqa: E402
from store_build import LINE_SETS, Store  # noqa: E402


def _line(H=64, Wd=600):
    img = np.full((H, Wd), 255, np.uint8)
    for x in range(20, Wd - 20, 30):  # words on a baseline band in the middle
        img[24:40, x:x + 12] = 0
    return img


def test_d4_clean_line_not_flagged():
    m = intrusion(_line())
    assert m["ok"] and not d4_flag(m)


def test_d4_neighbour_descenders_flagged():
    img = _line()
    for x in range(30, 560, 16):  # thin slanted descenders of the line above, cut by the top edge
        for i in range(8):
            img[i, x + i:x + i + 2] = 0
    m = intrusion(img)
    assert m["n_edge"] >= 4 and d4_flag(m)


def test_d4_ruling_ignored():
    img = _line()
    img[:, 100:102] = 0  # a vertical rule through the whole height
    img[:, 300:302] = 0
    assert not d4_flag(intrusion(img))


def test_lam_alef():
    gt = "لا إله إلا الله"
    assert lam_alef(gt, gt) == (2, 0)
    assert lam_alef(gt, "ال إله إلا الله") == (2, 1)
    assert lam_alef("ﻻ", "لا") == (1, 0)  # the presentation-form ligature counts as lam + alef
    assert lam_alef_letters(gt, gt) == (4, 0)
    n, e = lam_and_alef_elsewhere(gt, gt)
    assert n > 0 and e == 0


def test_ws_quirks_and_hits():
    gt = "و هي التي جعلته كريما  على فقره ."
    kinds = [k for k, _ in ws_quirks(gt)]
    assert set(kinds) == {"double space", "space before punctuation", "detached و"}
    assert quirk_hits(ws_quirks(gt), gt) == (3, 3)
    opp, rep = quirk_hits(ws_quirks(gt), "وهي التي جعلته كريما على فقره.")
    assert opp == 3 and rep == 0


def test_quirk_words():
    refs = ["كلمة صحيحة هنا"] * 9 + ["كلمة صحيحه هنا"]
    assert quirk_words("كلمة صحيحه هنا", refs, [0.05] * 10) == {"صحيحه"}
    assert quirk_words("كلمة صحيحه هنا", refs[:4], [0.05] * 4) == set()  # too few references
    assert quirk_words("كلمة صحيحه هنا", refs, [0.9] * 10) == set()  # references cannot read the line


def test_seed_cells():
    assert cell_unstable([0, 1, 0], 40)  # a flip
    assert not cell_unstable([3, 4, 3], 40)  # small spread
    assert cell_unstable([2, 9, 3], 40)  # spread ≥ 10 % and ≥ 2 edits
    c = {("base", "none"): {"unstable": True}, ("shape", "none"): {"unstable": False}, ("combo", "none"): {"unstable": True}}
    assert seed_unstable(c) is True
    assert seed_unstable(c, {"base"}) is True  # 1 of 2 remaining cells
    assert seed_unstable(c, {"base", "combo"}) is None  # one cell left: undefined


def test_store_kitab_full_in_train():
    assert "kitab_khatt_full" in LINE_SETS
    st = Store({}, {})
    st.aligner_train = {"muharaf_lines/test/000001"}
    st.kitab_image_of = {"kitab_khatt_full/test/000001": "kitab_khatt/train/000091"}
    st.khatt_train_kitab = {"kitab_khatt/train/000007"}
    assert st.in_train("crnn/base", "kitab_khatt_full/test/000001") == "no"  # its KITAB twin is not aligner training data
    st.aligner_train.add("kitab_khatt/train/000091")
    assert st.in_train("crnn/base", "kitab_khatt_full/test/000001") == "yes"  # the twin rule
    assert st.in_train("crnn/std_s0", "kitab_khatt_full/test/000001") == "no"
    assert st.in_train("crnn/khatt_run_s0", "kitab_khatt/train/000007") == "yes"
    assert st.in_train("crnn/khatt_run_s0", "kitab_khatt_full/test/000001") == "no"
    assert st.in_train("vlm/ain-7b-bf16", "kitab_khatt_full/test/000001") == "unknown"

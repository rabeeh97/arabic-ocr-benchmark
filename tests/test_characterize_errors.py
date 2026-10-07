"""R0.3 canonical runs and R2.1–R2.4 error decomposition, on hand-built pairs."""
import json

from arabic_ocr_bench.characterize import errors as E
from arabic_ocr_bench.characterize.runs import canonical, iter_readings, manifest
from arabic_ocr_bench.normalize import normalize

KITAB = "كتاب"        # book
BAB = "باب"           # door
KATABA = "كَتَبَ"      # kataba, fully marked
SALAM = "سلام"        # contains lam-alef
DARS = "درس"          # د does not join forward -> PAW gap after it


# ---- R2.4 taxonomy: the four required hand-built pairs ----------------------

def test_dot_confusion_ba_to_ta():
    assert E.classify_ops(BAB, "تاب") == {"dot": 1}


def test_dropped_fatha_is_diacritic():
    assert E.classify_ops(KATABA, "كتَبَ") == {"diacritic": 1}


def test_split_lam_alef_is_ligature():
    # the lam-alef read in visual order (alef before lam)
    assert E.classify_ops(SALAM, "ساlم".replace("l", "ل")) == {"ligature": 2}


def test_space_inserted_at_paw_gap():
    pred = "د رس"
    assert E.classify_ops(DARS, pred) == {"space": 1}
    assert E.is_paw_gap_space(DARS, pred) == (1, 1)
    # a space after a joining letter is inside a PAW, not at a gap
    assert E.is_paw_gap_space(KITAB, "كت اب") == (1, 0)
    # كتاب: 3 internal boundaries, one PAW gap (after the alef)
    assert E.paw_gap_base_rate(KITAB) == (3, 1)


def test_orthographic_shape_and_whole_line():
    assert E.classify_ops("أحمد", "احمد") == {"orthographic": 1}
    assert E.classify_ops("دار", "رار") == {"shape": 1}
    assert E.classify_ops(KITAB, "the image shows text") == {"whole_line": 20}
    assert E.classify_ops(KITAB, KITAB) == {}


# ---- R2.1 / R2.2 ladder ------------------------------------------------------

def test_ladder_no_diacritics_rung_matches_diac_insensitive_profile():
    s = "  أَحْمَدُ  ـ  ١٢ كتابة "
    assert E.ladder_strings(s)[E.RUNGS.index("no_diacritics")] == normalize(s, "diac_insensitive")


def test_rasm_rung_removes_dot_errors_only():
    lines = [E.ladder_edits(BAB, "تاب"), E.ladder_edits(KITAB, "كتاب")]
    s = E.summarize_ladder(lines)
    assert s["edits"]["no_diacritics"] == 1 and s["edits"]["rasm"] == 0
    assert s["rasm_cer"] == 0 and s["dot_tax"] > 0
    assert abs(sum(s["share"].values()) - 1) < 1e-9


def test_ladder_attributes_marks_and_spaces():
    s = E.summarize_ladder([E.ladder_edits(KATABA, "كتب")])
    assert s["share"]["no_diacritics"] == 1 and s["diacritic_tax"] > 0
    s = E.summarize_ladder([E.ladder_edits(DARS, "د رس")])
    assert s["share"]["no_spaces"] == 1


# ---- R2.3 diacritic mismatch -------------------------------------------------

def test_diacritic_mismatch_counts():
    assert E.diacritic_mismatch(KITAB, "كِتَاب")["inserted"] == 2
    d = E.diacritic_mismatch(KATABA, "كتب")
    assert d["dropped"] == 3 and d["gt_marks"] == 3 and d["pred_marks"] == 0
    assert E.diacritic_mismatch("كَتب", "كُتب")["wrong"] == 1


# ---- R0.3 canonical runs -----------------------------------------------------

def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_canonical_latest_success_standard_prompt(tmp_path):
    base = {"model": "m", "dataset": "d", "ground_truth": "x"}
    _write(tmp_path / "results/20260101T000000Z/predictions.jsonl", [
        {**base, "sample_id": "d/t/1", "prompt_version": "line-v2", "cleaned_text": "old"},
        {**base, "sample_id": "d/t/2", "prompt_version": "line-v2", "cleaned_text": "ok"},
    ])
    _write(tmp_path / "results/20260201T000000Z/predictions.jsonl", [
        {**base, "sample_id": "d/t/1", "prompt_version": "line-v2", "cleaned_text": "new"},
        {**base, "sample_id": "d/t/2", "prompt_version": "line-v2", "cleaned_text": "", "error": "boom"},
        {**base, "sample_id": "d/t/1", "prompt_version": "line-v2-no-diac", "cleaned_text": "abl"},
    ])
    _write(tmp_path / "results/x/smoke/20260301T000000Z/predictions.jsonl", [
        {**base, "sample_id": "d/t/1", "prompt_version": "line-v2", "cleaned_text": "smoke"},
    ])
    c = canonical(iter_readings(tmp_path))[("m", "d")]
    assert c["prompt_version"] == "line-v2"
    assert c["readings"]["d/t/1"].pred == "new"        # latest wins
    assert c["readings"]["d/t/2"].pred == "ok"         # an error never displaces a success
    row = manifest(canonical(iter_readings(tmp_path)))[0]
    assert row["n"] == 2 and row["excluded_prompts"] == {"line-v2-no-diac": 1}

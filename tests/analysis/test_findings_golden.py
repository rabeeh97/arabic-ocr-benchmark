import json
from pathlib import Path

from arabic_ocr_bench.analysis.findings import build

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"
GOLDEN = Path(__file__).parent.parent / "fixtures" / "mini_findings.json"


def _strip_volatile(d):
    d = json.loads(json.dumps(d))
    d["meta"].pop("generated_utc", None)
    return d


def test_top_level_keys():
    out = build(FIX)
    assert set(out) == {"meta", "error_anatomy", "script_letters", "diacritics",
                        "difficulty", "model_landscape", "operational", "corpus_health"}


def test_deterministic_across_two_builds():
    assert _strip_volatile(build(FIX)) == _strip_volatile(build(FIX))


def test_matches_golden():
    assert _strip_volatile(build(FIX)) == _strip_volatile(json.loads(GOLDEN.read_text()))


def test_output_is_json_serialisable_with_stdlib():
    blob = json.dumps(build(FIX))  # must not raise; must contain no NaN/Infinity tokens
    assert "NaN" not in blob and "Infinity" not in blob

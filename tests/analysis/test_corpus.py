# tests/analysis/test_corpus.py
import json
from pathlib import Path

from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.corpus import corpus_health
from arabic_ocr_bench.analysis.loader import iter_pred_rows

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_gt_stats_per_dataset():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    assert set(out["gt_stats"]) == {"dsx", "dsy"}
    assert out["gt_stats"]["dsx"]["charset_size"] > 0
    assert isinstance(out["gt_stats"]["dsx"]["n_lines"], int)
    assert isinstance(out["gt_stats"]["dsx"]["mean_line_chars"], float)
    assert isinstance(out["gt_stats"]["dsx"]["diac_density"], float)


def test_zipf_is_descending():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    counts = [c for _, c in out["gt_stats"]["dsx"]["zipf"]]
    assert counts == sorted(counts, reverse=True)
    ranks = [r for r, _ in out["gt_stats"]["dsx"]["zipf"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_diac_density_detects_harakat():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    # dsx contains the harakat line "مَدْرَسَة"
    assert out["gt_stats"]["dsx"]["diac_density"] > 0.0


def test_charset_coverage_shape():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    assert set(out["charset_coverage"]) == {"dsx", "dsy"}
    for ds in ("dsx", "dsy"):
        cov = out["charset_coverage"][ds]
        assert cov["only_here"] == sorted(cov["only_here"])
        assert 0.0 <= cov["shared_frac"] <= 1.0


def test_churn_ranks_are_1_based_contiguous():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    for metric in out["leaderboard_churn"]["metrics"]:
        ranks = sorted(v[metric] for v in out["leaderboard_churn"]["ranks"].values())
        assert ranks == list(range(1, len(ranks) + 1))


def test_contamination_absent_by_default():
    rows, recs = _setup()
    out = corpus_health(rows, recs)
    assert out["contamination"] == {"available": False, "flagged": []}


def test_contamination_intersects_consensus_easy(tmp_path):
    rows, recs = _setup()
    # dsx/test/000004 is the line every model transcribed perfectly in the fixture
    report = tmp_path / "contam.json"
    report.write_text(json.dumps(["dsx/test/000004", "dsx/test/000001", "dsx/test/999999"]))
    out = corpus_health(rows, recs, contamination_path=report)
    assert out["contamination"]["available"] is True
    assert "dsx/test/000004" in out["contamination"]["flagged"]
    # 000001 was scored but not exact by every model; 999999 was never scored
    assert "dsx/test/000001" not in out["contamination"]["flagged"]
    assert "dsx/test/999999" not in out["contamination"]["flagged"]

    # object form (keys are sample_ids) is accepted too
    report.write_text(json.dumps({"dsx/test/000004": {"src": "x"}}))
    out2 = corpus_health(rows, recs, contamination_path=str(report))
    assert out2["contamination"]["flagged"] == ["dsx/test/000004"]


def test_contamination_missing_path():
    rows, recs = _setup()
    out = corpus_health(rows, recs, contamination_path="/no/such/file.json")
    assert out["contamination"] == {"available": False, "flagged": []}


def test_json_serialisable():
    rows, recs = _setup()
    json.dumps(corpus_health(rows, recs))

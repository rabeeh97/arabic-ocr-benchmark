# tests/analysis/test_loader.py
"""iter_pred_rows dedupes across runs; a later success beats an earlier error."""
from pathlib import Path

from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows, run_dirs

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def test_run_dirs_finds_both():
    assert sorted(p.name for p in run_dirs(FIX)) == ["run_a", "run_b"]


def test_one_row_per_model_sample():
    rows = iter_pred_rows(FIX)
    keys = [(r.model, r.sample_id) for r in rows]
    assert len(keys) == len(set(keys))


def test_later_success_supersedes_earlier_error():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    assert rows[("m2", "dsy/test/000000")].error is None
    assert rows[("m2", "dsy/test/000000")].pred == "مدرسه"
    assert rows[("m1", "dsy/test/000000")].error is not None  # kept


def test_new_line_from_run_b_present():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    assert ("m1", "dsx/test/000005") in rows


def test_scores_attached_for_ok_rows():
    rows = {(r.model, r.sample_id): r for r in iter_pred_rows(FIX)}
    r = rows[("m1", "dsx/test/000001")]
    assert r.scores["cer_raw"] > 0
    assert isinstance(r, PredRow)


def test_mock_and_synthetic_excluded(tmp_path):
    # a run dir with only mock/synthetic rows yields nothing
    d = tmp_path / "r"
    d.mkdir()
    (d / "predictions.jsonl").write_text(
        '{"sample_id":"synthetic/t/0","model":"m1","dataset":"synthetic",'
        '"ground_truth":"x","cleaned_text":"x","error":null,"attempts":1,'
        '"latency_ms":1,"cost_usd":0,"input_tokens":0,"output_tokens":0,"cached":false}\n'
        '{"sample_id":"dsx/t/0","model":"mock-echo","dataset":"dsx",'
        '"ground_truth":"x","cleaned_text":"x","error":null,"attempts":1,'
        '"latency_ms":1,"cost_usd":0,"input_tokens":0,"output_tokens":0,"cached":false}\n'
    )
    assert iter_pred_rows(d) == []

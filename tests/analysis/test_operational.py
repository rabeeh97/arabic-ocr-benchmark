# tests/analysis/test_operational.py
from pathlib import Path

import pytest

from arabic_ocr_bench.analysis.align import AlignmentRecord, build_record
from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows
from arabic_ocr_bench.analysis.operational import operational

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"


def _row(model, sample_id, *, cost_usd=0.0, error=None, dataset="dsx",
         attempts=1, latency_ms=100.0):
    return PredRow(
        model=model, dataset=dataset, sample_id=sample_id, gt="x", pred="y",
        error=error, attempts=attempts, latency_ms=latency_ms, cost_usd=cost_usd,
        input_tokens=0, output_tokens=0, cached=False, scores={},
    )


def _rec(model, sample_id, *, failure_class="ok", cer_light=0.1,
         len_ref_chars=10, dataset="dsx"):
    return AlignmentRecord(
        model=model, dataset=dataset, sample_id=sample_id,
        failure_class=failure_class, char_ops=[],
        word_ops={"sub": 0, "ins": 0, "del": 0, "hits": 0},
        rasm_ops={"sub": 0, "ins": 0, "del": 0, "ref_len": 0},
        diac_seq_ref="", diac_seq_hyp="", spacing_del=0, spacing_ins=0,
        len_ref_chars=len_ref_chars, len_hyp_chars=0, len_ref_words=0,
        len_hyp_words=0, digit_swaps=0, presentation_form_chars=0,
        latin_run_len=0, has_code_fence=False, cer_light=cer_light,
        cer_raw=0.0, cer_rasm=0.0, wer_light=0.0,
    )


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def test_coverage_counts_rate_limit_as_attempted_not_scored():
    rows, recs = _setup()
    out = operational(rows, recs)
    mi = out["coverage"]["models"].index("m1")
    di = out["coverage"]["datasets"].index("dsy")
    scored, attempted = out["coverage"]["grid"][mi][di]
    assert attempted > scored  # m1/dsy/000000 rate-limited


def test_failure_causes_has_all_classes():
    from arabic_ocr_bench.analysis.refusal import FAILURE_CLASSES
    rows, recs = _setup()
    out = operational(rows, recs)
    for per_ds in out["failure_causes"].values():
        for counts in per_ds.values():
            assert set(counts) == set(FAILURE_CLASSES)


def test_latency_percentile_order():
    rows, recs = _setup()
    out = operational(rows, recs)
    for v in out["latency"].values():
        assert v["p50"] <= v["p95"]


def test_pareto_frontier_has_at_least_one():
    rows, recs = _setup()
    out = operational(rows, recs)
    assert any(v["on_frontier"] for v in out["pareto"].values())


def test_json_serialisable():
    import json
    rows, recs = _setup()
    json.dumps(operational(rows, recs))


def test_pareto_billable_cost_and_cost_aware_frontier():
    # Real priced model names from configs/models.yaml (gpt-4o input 2.50,
    # gpt-4o-mini input 0.15) so _priced_models() actually reads the yaml.
    rows = [
        _row("gpt-4o", "s1", cost_usd=0.002),
        _row("gpt-4o", "s2", cost_usd=0.004),
        _row("gpt-4o", "s3", cost_usd=0.003),
        _row("gpt-4o", "s4", cost_usd=0.010, error="boom"),  # errored: not billable
        _row("gpt-4o-mini", "s1", cost_usd=0.0),
        _row("gpt-4o-mini", "s2", cost_usd=0.0),
    ]
    recs = [
        _rec("gpt-4o", "s1", cer_light=0.3, len_ref_chars=10),
        _rec("gpt-4o", "s2", cer_light=0.3, len_ref_chars=10),
        _rec("gpt-4o", "s3", cer_light=0.3, len_ref_chars=10),
        _rec("gpt-4o", "s4", failure_class="other_error", cer_light=0.0,
             len_ref_chars=0),
        _rec("gpt-4o-mini", "s1", cer_light=0.1, len_ref_chars=10),
        _rec("gpt-4o-mini", "s2", cer_light=0.1, len_ref_chars=10),
    ]
    p = operational(rows, recs)["pareto"]

    # mean over the 3 BILLABLE + OK rows only — the errored 0.010 row excluded.
    assert p["gpt-4o"]["usd_per_1k"] == pytest.approx(1000 * (0.009 / 3))
    assert p["gpt-4o"]["free"] is False
    # priced but zero cost -> still free.
    assert p["gpt-4o-mini"]["usd_per_1k"] == 0.0
    assert p["gpt-4o-mini"]["free"] is True
    # gpt-4o-mini: lower CER AND lower cost -> dominates gpt-4o.
    assert p["gpt-4o-mini"]["on_frontier"] is True
    assert p["gpt-4o"]["on_frontier"] is False


def test_pareto_excludes_model_with_only_errored_rows():
    rows = [
        _row("gpt-4o-mini", "s1", cost_usd=0.0),
        _row("deadmodel", "s1", cost_usd=0.0, error="boom"),
    ]
    recs = [
        _rec("gpt-4o-mini", "s1", cer_light=0.1, len_ref_chars=10),
        _rec("deadmodel", "s1", failure_class="other_error", cer_light=0.0,
             len_ref_chars=0),
    ]
    p = operational(rows, recs)["pareto"]
    assert "deadmodel" not in p
    assert "gpt-4o-mini" in p

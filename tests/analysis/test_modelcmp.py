import math
from pathlib import Path

from arabic_ocr_bench.analysis.align import AlignmentRecord, build_record
from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows
from arabic_ocr_bench.analysis.modelcmp import model_landscape


def _rec(model, dataset, sample_id, cer_light, len_ref_chars):
    """A minimal OK AlignmentRecord — only the fields model_landscape's
    bootstrap_cer / rank_stability blocks read carry meaning."""
    return AlignmentRecord(
        model=model, dataset=dataset, sample_id=sample_id, failure_class="ok",
        char_ops=[], word_ops={"sub": 0, "ins": 0, "del": 0, "hits": 0},
        rasm_ops={"sub": 0, "ins": 0, "del": 0, "ref_len": 0},
        diac_seq_ref="", diac_seq_hyp="", spacing_del=0, spacing_ins=0,
        len_ref_chars=len_ref_chars, len_hyp_chars=0, len_ref_words=0,
        len_hyp_words=0, digit_swaps=0, presentation_form_chars=0,
        latin_run_len=0, has_code_fence=False, cer_light=cer_light,
        cer_raw=0.0, cer_rasm=0.0, wer_light=0.0,
    )

FIX = Path(__file__).parent.parent / "fixtures" / "mini_results"

_METRICS = ["cer_light", "wer_light", "mer_light", "wil_light", "diac_f1"]


def _setup():
    rows = iter_pred_rows(FIX)
    return rows, [build_record(r) for r in rows]


def _ok_models(recs):
    return {r.model for r in recs if r.failure_class == "ok"}


def test_agreement_is_square_and_diagonal_one():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    ident = out["agreement"]["identity"]
    n = len(out["agreement"]["models"])
    assert all(len(r) == n for r in ident)
    for i in range(n):
        assert abs(ident[i][i] - 1.0) < 1e-9


def test_profile_decomposition_components_nonnegative():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    for v in out["profile_decomposition"].values():
        assert min(v.values()) >= -1e-9


def test_metric_correlation_diag_one():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    m = out["metric_correlation"]["matrix"]
    for i in range(len(m)):
        assert abs(m[i][i] - 1.0) < 1e-6


def test_json_serialisable():
    import json

    rows, recs = _setup()
    json.dumps(model_landscape(rows, recs))


def test_agreement_matrices_symmetric_and_unit_diagonal():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    n = len(out["agreement"]["models"])
    for key in ("identity", "cer_corr"):
        m = out["agreement"][key]
        assert len(m) == n and all(len(r) == n for r in m)
        for i in range(n):
            assert m[i][i] == 1.0
            for j in range(n):
                assert abs(m[i][j] - m[j][i]) < 1e-9


def test_agreement_models_are_exactly_the_ok_models():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    models = out["agreement"]["models"]
    assert set(models) == _ok_models(recs)
    assert len(models) == len(out["agreement"]["identity"])
    assert len(models) == len(out["agreement"]["cer_corr"])


def test_metric_correlation_shape_and_labels():
    rows, recs = _setup()
    out = model_landscape(rows, recs)
    mc = out["metric_correlation"]
    assert mc["metrics"] == _METRICS
    m = mc["matrix"]
    assert len(m) == len(_METRICS)
    assert all(len(r) == len(_METRICS) for r in m)
    for i in range(len(m)):
        assert abs(m[i][i] - 1.0) < 1e-6


def test_bootstrap_ci_brackets_point():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    for per_ds in out["bootstrap_cer"].values():
        for v in per_ds.values():
            assert v["lo"] <= v["point"] <= v["hi"]


def test_bootstrap_is_deterministic():
    rows, recs = _setup()
    a = model_landscape(rows, recs, b=200)["bootstrap_cer"]
    b = model_landscape(rows, recs, b=200)["bootstrap_cer"]
    assert a == b


def test_rank_stability_bounds():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    for m, per_ds in out["rank_stability"].items():
        if m == "datasets":
            continue
        for v in per_ds.values():
            assert 1 <= v["best"] <= v["worst"]


def test_rank_stability_is_deterministic():
    rows, recs = _setup()
    a = model_landscape(rows, recs, b=200)["rank_stability"]
    b = model_landscape(rows, recs, b=200)["rank_stability"]
    assert a == b


def test_rank_stability_zero_data_replicate_does_not_grant_rank_one():
    # Ruling 12: dsq has 4 samples. mgood covers all 4 (perfect); mbad covers
    # only sample 0 and is strictly worse there. Replicates that resample none
    # of sample 0 must drop mbad from the ranking, not score it rank 1 off a
    # no-data draw -- so mbad's best rank is 2 (behind mgood on their one
    # shared sample), never 1.
    recs = [
        _rec("mgood", "dsq", f"dsq/test/00000{i}", 0.0, 10) for i in range(4)
    ] + [_rec("mbad", "dsq", "dsq/test/000000", 0.9, 10)]
    rs = model_landscape([], recs, b=300)["rank_stability"]
    assert "mbad" in rs
    assert rs["mbad"]["dsq"]["best"] == 2
    assert rs["mbad"]["dsq"]["best"] <= rs["mbad"]["dsq"]["worst"]


def test_non_ok_rows_are_excluded_from_every_key():
    rows, recs = _setup()
    base = model_landscape(rows, recs)

    # A synthetic model whose only line classifies as ``empty`` (error is None,
    # but cleaned_text is blank) -> build_record yields a non-"ok" stub.
    zrow = PredRow(
        model="mZ", dataset="dsx", sample_id="dsx/test/000000",
        gt="بيت كبير", pred="", error=None, attempts=1, latency_ms=0.0,
        cost_usd=0.0, input_tokens=0, output_tokens=0, cached=False,
        scores={k: 0.5 for k in _METRICS}
        | {"cer_raw": 0.5, "cer_diac_insensitive": 0.5},
    )
    zrec = build_record(zrow)
    assert zrec.failure_class != "ok"

    out = model_landscape(rows + [zrow], recs + [zrec])
    assert "mZ" not in out["agreement"]["models"]
    assert "mZ" not in out["profile_decomposition"]
    assert out["agreement"]["models"] == base["agreement"]["models"]
    assert out["agreement"]["identity"] == base["agreement"]["identity"]
    assert out["agreement"]["cer_corr"] == base["agreement"]["cer_corr"]
    assert out["metric_correlation"] == base["metric_correlation"]
    assert out["profile_decomposition"] == base["profile_decomposition"]


def test_oracle_not_worse_than_best_single():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    best_single = min(out["ensemble"]["single"].values())
    assert out["ensemble"]["oracle"] <= best_single + 1e-9


def test_win_rate_antisymmetric():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    M = out["win_rate"]["matrix"]
    n = len(M)
    for i in range(n):
        for j in range(n):
            assert abs(M[i][j] + M[j][i] - 1.0) < 1e-9


def test_bradley_terry_centered():
    rows, recs = _setup()
    out = model_landscape(rows, recs, b=200)
    logs = [math.log(v) for v in out["bradley_terry"].values()]
    assert abs(sum(logs) / len(logs)) < 1e-6

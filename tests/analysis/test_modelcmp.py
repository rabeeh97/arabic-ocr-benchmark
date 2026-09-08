from pathlib import Path

from arabic_ocr_bench.analysis.align import build_record
from arabic_ocr_bench.analysis.loader import PredRow, iter_pred_rows
from arabic_ocr_bench.analysis.modelcmp import model_landscape

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

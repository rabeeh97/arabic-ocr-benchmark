"""X1: fine-tuned scores on a dataset that overlaps the training corpus
(`overlaps_source` in configs/datasets.yaml, e.g. kitab_muharaf) are never quoted."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import make_leaderboard as mk


def _row(model, ds, cer):
    return {"model": model, "dataset": ds, "cer_micro_light": str(cer),
            "wer_micro_light": str(cer), "n": "10", "run": "r"}


def test_kitab_muharaf_is_in_train():
    assert mk.in_train_datasets().get("kitab_muharaf") == "aamijar/muharaf-public"


def test_fine_tuned_models_from_manifest_and_name():
    fts = mk.fine_tuned_models(["qwen3-vl-4b-bf16", "foo-ft-bf16", "bar-ft", "draft-x"])
    assert "exp3090-qwen3vl4b-ft-bf16" in fts            # from experiments.yaml
    assert {"foo-ft-bf16", "bar-ft"} <= fts
    assert not {"qwen3-vl-4b-bf16", "draft-x"} & fts


def test_matrix_masks_fine_tune_cell_on_in_train_set():
    rows = [_row("qwen3-vl-4b-bf16", "kitab_muharaf", 0.5),
            _row("exp3090-qwen3vl4b-ft-bf16", "kitab_muharaf", 0.01),
            _row("exp3090-qwen3vl4b-ft-bf16", "kitab_khatt", 0.3)]
    html = mk._render_matrix(rows)
    blob = json.loads(re.search(r"id='mx-data' type='application/json'>(.*?)</script>", html).group(1))
    di, ft, base = blob["ds"].index("kitab_muharaf"), blob["m"].index("exp3090-qwen3vl4b-ft-bf16"), blob["m"].index("qwen3-vl-4b-bf16")
    assert blob["c"][di][ft] == "it"                       # leaked number not shipped
    assert blob["c"][di][base] == [0.5, 0.5]               # base still scored
    assert blob["c"][blob["ds"].index("kitab_khatt")][ft] == [0.3, 0.3]
    assert "kitab_muharaf" in blob["it"]
    assert "0.01" not in html


def test_finetuning_tab_never_counts_in_train_holdout(monkeypatch, tmp_path):
    manifest = {"x": {"title": "X", "base_model": "b", "ft_models": ["b-ft"],
                      "holdout_datasets": ["kitab_khatt", "kitab_muharaf"]}}
    real = mk.yaml.safe_load
    monkeypatch.setattr(mk.yaml, "safe_load",
                        lambda t: manifest if "holdout_datasets" in t and "exp3090" in t else real(t))
    rows = [_row("b", "kitab_khatt", 0.6), _row("b-ft", "kitab_khatt", 0.3),
            _row("b", "kitab_muharaf", 0.6), _row("b-ft", "kitab_muharaf", 0.02)]
    html, toc = mk._render_finetuning(rows, tmp_path)
    assert toc == [("ft-x", "X")]
    assert "across 1 dataset(s)" in html
    assert "<td>kitab_muharaf</td>" not in html
    assert "Not counted as held-out" in html and "2.0" not in html

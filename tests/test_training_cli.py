"""`bench-train` CLI: offline subcommands over the in-repo synthetic dataset."""

from __future__ import annotations

import json

from arabic_ocr_bench.training.cli import main

_CFG = """
experiment: cli-test
method: qlora
base_model: {hf_id: Qwen/Qwen2-VL-2B-Instruct, arch: qwen2_vl}
data:
  sources:
    - name: synthetic
      adapter: arabic_ocr_bench.datasets.synthetic:Synthetic
  val_fraction: 0.2
output_dir: results/training/cli-test
eval_datasets: [synthetic]
"""


def _write_cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_CFG)
    return p


def test_assemble_reports_source_and_split_counts(tmp_path, capsys):
    rc = main(["assemble", "--config", str(_write_cfg(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "synthetic: 10" in out
    assert "train: 8  val: 2" in out


def test_check_contam_flags_self_overlap_and_exits_nonzero(tmp_path, capsys):
    report = tmp_path / "r.json"
    rc = main(
        ["check-contam", "--config", str(_write_cfg(tmp_path)), "--out", str(report)]
    )
    assert rc == 1
    doc = json.loads(report.read_text())
    assert doc["n_flagged"] == 1
    assert doc["pairs"][0]["n_text_collisions"] == 10

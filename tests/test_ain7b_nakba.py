"""Stage-2 AIN LoRA on NAKBA train — config, leak guard, and launcher wiring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _experiment_bases import (  # noqa: E402
    BASES,
    is_local_checkpoint,
    pipeline_bench_args,
    round1_queue,
)
from eval_ain_nakba_ckpts import (  # noqa: E402
    OFFICIAL_REF,
    OFFICIAL_REF_CONTEST,
    OFFICIAL_STEPS,
)
from run_3090_benchmark import parse_datasets_block  # noqa: E402

from arabic_ocr_bench.prompting import get_prompt
from arabic_ocr_bench.training.config import DataSpec, SourceSpec, load_training_config
from arabic_ocr_bench.training.data import (
    check_source_not_scored,
    safe_eval_datasets,
)


def test_ain7b_nakba_is_wired_but_not_in_the_live_suite_queue():
    spec = BASES["ain7b_nakba"]
    assert spec["enabled"] is False
    assert "ain7b_nakba" not in round1_queue()
    assert spec["experiment"] == "ain7b-nakba-sft"
    assert spec["arch"] == "qwen2_vl"
    assert spec["base_model_row"] == "suite-v1-ain7b-ft-bf16"
    assert spec["no_full_accumulate"] is True
    assert spec["fetch_datasets"] == [
        ("U4RASD/omar-al-saleh-manuscripts-segments", "train"),
    ]


def test_ain7b_nakba_training_config_is_nakba_train_only():
    cfg = load_training_config("configs/training/ain7b_nakba.yaml")
    assert cfg.experiment == "ain7b-nakba-sft"
    assert cfg.method == "lora"
    assert cfg.base_model.arch == "qwen2_vl"
    assert cfg.eval_datasets == ["nakba_lines"]
    assert cfg.train.epochs == 2
    assert cfg.train.load_best_model_at_end is True
    assert cfg.train.save_strategy == "steps"
    assert cfg.train.save_steps == 400
    assert cfg.train.save_total_limit == 12
    assert len(cfg.data.sources) == 1
    src = cfg.data.sources[0]
    assert src.split == "train"
    assert src.params["subset"] == "U4RASD/omar-al-saleh-manuscripts-segments"
    assert src.limit is None  # full NAKBA train, not a round-1 cap


def test_nakba_train_is_allowed_and_test_stays_safe_to_score():
    check_source_not_scored(
        "U4RASD/omar-al-saleh-manuscripts-segments",
        "train",
        cfg_path="configs/datasets.yaml",
    )
    data = DataSpec.model_construct(
        sources=[
            SourceSpec(
                name="nakba_train",
                adapter="arabic_ocr_bench.datasets.kitab_bench:KitabBench",
                params={"subset": "U4RASD/omar-al-saleh-manuscripts-segments"},
                split="train",
            )
        ],
        val_fraction=0.0,
        prompt_version="line-v2",
    )
    safe = safe_eval_datasets(data, cfg_path="configs/datasets.yaml")
    assert "nakba_lines" in safe  # scored split is test, not train
    assert "nakba_blind" in safe  # official Systems-Track split, not train
    assert "muharaf_lines" in safe
    assert "rasam1" in safe


def test_ain7b_nakba_bench_yaml_is_forgetting_plus_nakba():
    text = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "training"
        / "ain7b_nakba.bench.yaml"
    ).read_text(encoding="utf-8")
    names = parse_datasets_block(text)
    assert names == [
        "nakba_lines",
        "kitab_khatt",
        "muharaf_lines",
        "rasam1",
        "rasam2",
    ]
    assert "kitab_hindawi" not in names


def test_pipeline_bench_args_pass_the_short_list():
    spec = BASES["ain7b_nakba"]
    flags = pipeline_bench_args(spec)
    assert "--datasets-file" in flags
    assert "--no-full-accumulate" in flags
    ds = flags[flags.index("--datasets-file") + 1]
    assert ds.endswith("ain7b_nakba.bench.yaml")

    assert pipeline_bench_args(BASES["ain7b"]) == []


def test_official_blind_scores_every_epoch1_checkpoint_and_contest_baseer():
    assert OFFICIAL_STEPS == {1200, 1600, 1936}
    assert OFFICIAL_REF == "nakba-misraj-baseer"
    assert OFFICIAL_REF_CONTEST == "nakba-misraj-baseer-contest"
    assert get_prompt("nakba-contest") == "Extract the text from the above document."
    models = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    text = models.read_text(encoding="utf-8")
    assert "nakba-misraj-baseer-contest:" in text
    assert "max_new_tokens: 128" in text.split("nakba-misraj-baseer-contest:")[1][:400]


def test_skip_official_400_unless_800_beats_1600():
    from eval_ain_nakba_ckpts import should_skip_ckpt400

    assert should_skip_ckpt400({}) is True
    assert should_skip_ckpt400({1600: 0.1078}) is True
    assert should_skip_ckpt400({800: 0.12, 1600: 0.1078}) is True
    assert should_skip_ckpt400({800: 0.10, 1600: 0.1078}) is False


def test_ain7b_nakba_visft_is_wired_unfrozen_and_not_in_suite_queue():
    spec = BASES["ain7b_nakba_visft"]
    assert spec["enabled"] is False
    assert "ain7b_nakba_visft" not in round1_queue()
    assert spec["official_all_ckpts"] is True
    assert spec["ckpt_eval_after_train"] is True
    cfg = load_training_config("configs/training/ain7b_nakba_visft.yaml")
    assert cfg.lora.train_vision is True
    assert cfg.train.vision_lr == 9.0e-6
    assert cfg.train.save_steps == 400
    assert cfg.train.epochs == 1
    assert cfg.train.load_best_model_at_end is False
    assert cfg.data.val_fraction == 0.0
    assert cfg.train.max_steps == 400
    assert cfg.base_model.adapter_id
    assert "checkpoint-1600" in cfg.base_model.adapter_id
    flags = pipeline_bench_args(spec)
    assert flags[flags.index("--datasets-file") + 1].endswith("ain7b_nakba_visft.bench.yaml")


def test_visft_staged_continues_only_on_strict_cer_improvement():
    from stage_ain_nakba_visft import BASELINE_CER, should_continue

    assert BASELINE_CER == 0.1078
    assert should_continue(0.1000, 0.1078) is True
    assert should_continue(0.1078, 0.1078) is False
    assert should_continue(0.1100, 0.1078) is False
    assert should_continue(0.0990, 0.1000) is True
    assert should_continue(0.1000, 0.0990) is False


def test_is_local_checkpoint_requires_config_json(tmp_path: Path):
    missing = tmp_path / "nope"
    assert is_local_checkpoint(str(missing)) is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_local_checkpoint(str(empty)) is False
    ckpt = tmp_path / "merged"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    assert is_local_checkpoint(str(ckpt)) is True

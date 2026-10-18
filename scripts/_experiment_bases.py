"""Single source of truth for '--base <name>' specs.

Shared by `run_3090_experiment.py`, `run_3090_pipeline.py`,
`run_3090_benchmark.py`, and `run_experiment_suite.py` — previously two of
those scripts each hand-maintained their own copy of this dict (and had
drifted: `run_3090_pipeline.py`'s copy hardcoded `arch: qwen2_5_vl` even for
the `qwen3vl4b`/`qwen3vl4b-visft` entries). One dict now, one place to add a
new model.

Deliberately plain Python, not YAML/JSON — `run_3090_experiment.py`'s
`bootstrap` phase runs before any venv or third-party package exists on a
fresh box (see `docs/experiments/3090-experiment-runbook.md` §9: "launcher
must stay stdlib-only, no pyyaml needed at launcher level"), so nothing
importable from this module may require a dependency beyond the stdlib.

Every entry:
    experiment          str        — must match the training config's `experiment:` field
    config              str        — path to the training YAML
    run_config          str        — path to the paired eval-only bench YAML
    hf_id               str        — base model repo id
    arch                str        — training arch tag (training/config.py's `Arch` literal)
    needs_transformers  str | None — minimum transformers version to pin, if any
    gated               bool       — needs an accepted license / HF_TOKEN with access
    enabled             bool       — included in `run_experiment_suite.py`'s round-1 queue
                                     (irrelevant to `--base <name>` on the other scripts,
                                     which resolve any entry here regardless of this flag)
    license_check       str | None — None (n/a), "pending", or "ok". A `gated` entry with
                                     anything other than "ok" is skipped by the suite
                                     orchestrator (logged, not queue-blocking) until
                                     someone flips it after actually checking the terms.
    base_model_row      str        — the untuned base's row name in configs/models.yaml,
                                     used by run_3090_pipeline.py / run_experiment_suite.py
                                     when invoking run_3090_benchmark.py's --base-name
    fetch_datasets      list[(str,str)] optional — Hub (repo, split) pairs fetch
                                     should download instead of the default Muharaf train
    bench_datasets_file str | None optional — curated YAML for run_3090_pipeline
                                     to pass as --datasets-file
    no_full_accumulate  bool       optional — pipeline passes --no-full-accumulate
"""

from __future__ import annotations

from pathlib import Path

BASES: dict[str, dict] = {
    # --- original exp3090 single-dataset experiments (unchanged) --------------
    "qwen3vl4b": {
        "experiment": "exp3090-qwen3vl4b",
        "config": "configs/training/exp3090_qwen3vl4b.yaml",
        "run_config": "configs/training/exp3090_qwen3vl4b.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "qwen3-vl-4b-bf16",
    },
    "qwen3vl4b-visft": {
        # Same as "qwen3vl4b" but with the ViT unfrozen (train_vision: true) —
        # tests whether the frozen encoder is the bottleneck on the Maghrebi
        # RASAM sets. See docs/experiments/findings-and-notes.md.
        "experiment": "exp3090-qwen3vl4b-visft",
        "config": "configs/training/exp3090_qwen3vl4b_visft.yaml",
        "run_config": "configs/training/exp3090_qwen3vl4b_visft.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "qwen3-vl-4b-bf16",
    },
    "baseer": {
        "experiment": "exp3090-baseer-nakba",
        "config": "configs/training/exp3090_baseer_nakba.yaml",
        "run_config": "configs/training/exp3090_baseer_nakba.run.yaml",
        "hf_id": "Misraj/Baseer__Nakba",
        "arch": "qwen2_5_vl",
        "needs_transformers": None,
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "baseer-nakba-bf16",
    },
    "paligemma": {
        "experiment": "exp3090-paligemma2-3b",
        "config": "configs/training/exp3090_paligemma2_3b.yaml",
        "run_config": "configs/training/exp3090_paligemma2_3b.run.yaml",
        "hf_id": "google/paligemma2-3b-pt-448",
        "arch": "paligemma",
        "needs_transformers": "4.49",
        "gated": True,
        # Not selected for the round-1 suite queue (kept out on purpose, see
        # docs/experiments/misraj-baseer-research.md discussion) — flip to
        # True to re-add it as a one-line change, no config work lost.
        "enabled": False,
        "license_check": "ok",  # Gemma license already accepted for the original exp3090 run
        "base_model_row": "paligemma2-3b-bf16",
    },
    # --- suite_v1 round-1 queue (2026-09-16) -----------------------------------
    # Shared multi-source data mix (Muharaf + AraMS-28k-HTR + RASAM-1/2), NAKBA
    # deliberately excluded so it stays an unseen-domain check. See
    # configs/training/suite_v1/*.yaml and docs/experiments/misraj-baseer-research.md.
    "qwen3vl4b_suite": {
        "experiment": "suite-v1-qwen3vl4b",
        "config": "configs/training/suite_v1/qwen3vl4b.yaml",
        "run_config": "configs/training/suite_v1/qwen3vl4b.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": True,
        "license_check": None,
        "base_model_row": "qwen3-vl-4b-bf16",
    },
    # --- E1: frozen vs unfrozen ViT, controlled (docs/research/NEXT_STEPS.md) ---
    # Same data/seed/recipe as qwen3vl4b_suite (the frozen arm, already run);
    # only train_vision / vision_lr differ. Launch only via run_3090_pipeline.py
    # --base <name> when asked. The difflr arm already trained (2026-09-24,
    # under the qwen3vl4b entry); it only needs the bench, see NEXT_STEPS E1.0.
    "qwen3vl4b_suite_visft": {
        "experiment": "suite-v1-qwen3vl4b-visft",
        "config": "configs/training/suite_v1/qwen3vl4b_visft.yaml",
        "run_config": "configs/training/suite_v1/qwen3vl4b_visft.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "qwen3-vl-4b-bf16",
        "bench_datasets_file": "configs/training/suite_v1/e1.bench.yaml",
        "no_full_accumulate": True,
    },
    "qwen3vl4b_suite_visft_difflr": {
        "experiment": "suite-v1-qwen3vl4b-visft-difflr",
        "config": "configs/training/suite_v1/qwen3vl4b_visft_difflr.yaml",
        "run_config": "configs/training/suite_v1/qwen3vl4b_visft_difflr.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "qwen3-vl-4b-bf16",
        "bench_datasets_file": "configs/training/suite_v1/e1.bench.yaml",
        "no_full_accumulate": True,
    },
    "qwen3vl8b": {
        "experiment": "suite-v1-qwen3vl8b",
        "config": "configs/training/suite_v1/qwen3vl8b.yaml",
        "run_config": "configs/training/suite_v1/qwen3vl8b.run.yaml",
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "arch": "qwen3_vl",
        "needs_transformers": "4.57",
        "gated": False,
        "enabled": True,
        "license_check": None,
        "base_model_row": "qwen3-vl-8b-bf16",
    },
    "ain7b": {
        "experiment": "suite-v1-ain7b",
        "config": "configs/training/suite_v1/ain7b.yaml",
        "run_config": "configs/training/suite_v1/ain7b.run.yaml",
        "hf_id": "MBZUAI/AIN",
        "arch": "qwen2_vl",
        "needs_transformers": None,
        "gated": False,
        "enabled": True,
        "license_check": None,
        "base_model_row": "ain-7b-bf16",
    },
    "fanar2_oryx": {
        "experiment": "suite-v1-fanar2-oryx",
        "config": "configs/training/suite_v1/fanar2_oryx.yaml",
        "run_config": "configs/training/suite_v1/fanar2_oryx.run.yaml",
        "hf_id": "QCRI/Fanar-2-Oryx-IVU",
        "arch": "qwen2_5_vl",
        "needs_transformers": None,
        "gated": True,
        "enabled": True,
        # HF repo is ungated Apache 2.0 (model card + Qwen2.5-VL-7B base).
        # Confirmed 2026-09-18: no click-through gate; fine-tune/redistribute
        # allowed with the usual Apache notices. Fanar's api.fanar.qa request
        # form is a different product and is not required for local training.
        "license_check": "ok",
        "base_model_row": "fanar-oryx-7b-bf16",
    },
    # --- stage-2: AIN suite-v1 FT → NAKBA train LoRA (not in round-1 queue) ---
    # Continues from the merged suite-v1-ain7b weights already on the 3090 box.
    # NAKBA test stays held out. Do not add this to ROUND1_ORDER — Fanar is
    # still the live suite job; launch only via run_3090_pipeline.py --base
    # ain7b_nakba when asked.
    "ain7b_nakba": {
        "experiment": "ain7b-nakba-sft",
        "config": "configs/training/ain7b_nakba.yaml",
        "run_config": "configs/training/ain7b_nakba.run.yaml",
        "hf_id": r"C:\Users\User\a-experiment\suite-v1\ain7b\merged\suite-v1-ain7b",
        "arch": "qwen2_vl",
        "needs_transformers": None,
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "suite-v1-ain7b-ft-bf16",
        "fetch_datasets": [
            ("U4RASD/omar-al-saleh-manuscripts-segments", "train"),
        ],
        "bench_datasets_file": "configs/training/ain7b_nakba.bench.yaml",
        "no_full_accumulate": True,
    },
    "ain7b_nakba_visft": {
        # Stage-3: unfreeze ViT. Prefer scripts/stage_ain_nakba_visft.py
        # (train 400 → official score → continue only if CER improves).
        "experiment": "ain7b-nakba-visft",
        "config": "configs/training/ain7b_nakba_visft.yaml",
        "run_config": "configs/training/ain7b_nakba_visft.run.yaml",
        "hf_id": r"C:\Users\User\a-experiment\suite-v1\ain7b\merged\suite-v1-ain7b",
        "arch": "qwen2_vl",
        "needs_transformers": None,
        "gated": False,
        "enabled": False,
        "license_check": None,
        "base_model_row": "suite-v1-ain7b-ft-bf16",
        "fetch_datasets": [
            ("U4RASD/omar-al-saleh-manuscripts-segments", "train"),
        ],
        "bench_datasets_file": "configs/training/ain7b_nakba_visft.bench.yaml",
        "no_full_accumulate": True,
        "official_all_ckpts": True,
        "ckpt_eval_after_train": True,
    },
}

# Queue order for run_experiment_suite.py's round-1 sweep. A name only runs if
# BASES[name]["enabled"] is also True (kept as a separate check, not baked
# into this list, so toggling `enabled` in one place is enough to add/drop it).
ROUND1_ORDER = ["qwen3vl4b_suite", "qwen3vl8b", "ain7b", "fanar2_oryx"]


def round1_queue() -> list[str]:
    """Enabled entry names, in queue order, for the round-1 suite sweep."""
    return [name for name in ROUND1_ORDER if BASES[name]["enabled"]]


def is_runnable(name: str) -> tuple[bool, str]:
    """(ok, reason) — False when a gated entry's license isn't confirmed yet."""
    spec = BASES[name]
    if spec["gated"] and spec.get("license_check") != "ok":
        state = spec.get("license_check") or "unset"
        return False, f"{name}: gated, license_check={state!r} (needs 'ok' to run)"
    return True, ""


def is_local_checkpoint(hf_id: str) -> bool:
    """True when ``hf_id`` is a merged-weights folder, not a Hub repo id."""
    p = Path(hf_id)
    try:
        return p.is_dir() and (p / "config.json").exists()
    except OSError:
        return False


def pipeline_bench_args(spec: dict) -> list[str]:
    """Extra ``run_3090_benchmark.py`` flags from optional BASES keys.

    Round-1 suite entries omit these (the suite runner hardcodes its own
    short list). Stage-2 jobs set ``bench_datasets_file`` /
    ``no_full_accumulate`` so ``run_3090_pipeline.py`` cannot fall back to
    the 24h accumulate-every-line-set sweep.
    """
    extra: list[str] = []
    ds = spec.get("bench_datasets_file")
    if ds:
        extra += ["--datasets-file", ds]
    if spec.get("no_full_accumulate"):
        extra.append("--no-full-accumulate")
    return extra

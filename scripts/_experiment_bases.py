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
"""

from __future__ import annotations

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
        # Fine-tuning/redistribution license unconfirmed as of 2026-09-16 (see
        # docs/experiments/misraj-baseer-research.md) — the suite orchestrator
        # skips this entry (logs why, doesn't block the queue) until someone
        # flips this to "ok" after actually checking QCRI's terms.
        "license_check": "pending",
        "base_model_row": "fanar-oryx-7b-bf16",
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

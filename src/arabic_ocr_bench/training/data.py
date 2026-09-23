"""Assemble line-pair training data — and refuse to touch a scored split.

The one hard rule of this whole subsystem: nothing the benchmark scores on may
enter a training mix. `scored_splits()` derives the forbidden `(hf_id, split)`
set straight from `configs/datasets.yaml` so it can never drift from the
registry, and `check_source_not_scored()` is the gate every loader calls.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..prompting import get_prompt

if TYPE_CHECKING:
    from .config import DataSpec

_DEFAULT_CFG = Path("configs/datasets.yaml")

# qwen_vl_utils.vision_process.smart_resize hard-rejects an image whose
# longest:shortest edge ratio is >= 200 ("absolute aspect ratio must be
# smaller than 200"). 190 leaves a small margin for cap_image()'s integer
# rounding to shift the ratio slightly without flipping a borderline sample
# from "accepted here" to "crashes the collator anyway".
_MAX_ASPECT_RATIO = 190


class ScoredSplitLeakError(RuntimeError):
    """Raised when a training source names a split the benchmark scores on."""


def _hf_id_for(cfg: dict) -> str | None:
    """The HF repo id a dataset entry pulls from, mirroring the adapters.

    KITAB-Bench entries carry it as `params.subset` (a bare suffix under
    `ahmedheakl/`, or a full `org/name`); everything else uses a top-level
    `hf_id`. In-repo datasets (synthetic) have neither.
    """
    if cfg.get("hf_id"):
        return cfg["hf_id"]
    subset = (cfg.get("params") or {}).get("subset")
    if subset:
        return subset if "/" in subset else f"ahmedheakl/{subset}"
    return None


def scored_splits(cfg_path: str | Path = _DEFAULT_CFG) -> set[tuple[str, str]]:
    """`(hf_id, split)` pairs the benchmark evaluates on — the do-not-train set."""
    registry = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    pairs: set[tuple[str, str]] = set()
    for entry in registry.values():
        if not isinstance(entry, dict):
            continue
        hf_id = _hf_id_for(entry)
        split = entry.get("split")
        if hf_id and split:
            pairs.add((hf_id, split))
    return pairs


def check_source_not_scored(
    hf_id: str, split: str, *, cfg_path: str | Path = _DEFAULT_CFG
) -> None:
    """Raise `ScoredSplitLeakError` if `(hf_id, split)` is a benchmark test set."""
    if (hf_id, split) in scored_splits(cfg_path):
        raise ScoredSplitLeakError(
            f"{hf_id}:{split} is scored by the benchmark and must not be used for training"
        )


def safe_eval_datasets(data: "DataSpec", *, cfg_path: str | Path = _DEFAULT_CFG) -> list[str]:
    """Every `track: line` benchmark dataset safe to score a model trained on
    ``data``'s sources against.

    Starts from every registered line-track dataset except ``synthetic``
    (not a real benchmark), then drops anything that overlaps a training
    source, by either signal:

    * an exact ``(hf_id, split)`` match — already structurally impossible
      given ``check_source_not_scored`` forbids training on a scored split in
      the first place, kept here as defense-in-depth;
    * a declared ``overlaps_source`` on the dataset entry — a same-corpus
      dataset repackaged under a *different* HF repo id (e.g. `kitab_muharaf`
      vs. a `muharaf_train` source's `aamijar/muharaf-public`), which the
      exact-tuple check can't see on its own.
    """
    registry = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}

    train_hf_ids: set[str] = set()
    train_pairs: set[tuple[str, str]] = set()
    for src in data.sources:
        hf_id = resolve_source_hf_id(src)
        split = getattr(src, "split", None) or (src.get("split") if isinstance(src, dict) else None)
        if hf_id:
            train_hf_ids.add(hf_id)
            if split:
                train_pairs.add((hf_id, split))

    safe = []
    for name, entry in registry.items():
        if not isinstance(entry, dict) or entry.get("track") != "line" or name == "synthetic":
            continue
        hf_id = _hf_id_for(entry)
        split = entry.get("split")
        if hf_id and split and (hf_id, split) in train_pairs:
            continue
        if entry.get("overlaps_source") in train_hf_ids:
            continue
        safe.append(name)
    return sorted(safe)


def build_conversation(prompt_text: str, target: str) -> list[dict]:
    """One SFT example in chat form: system instruction, user image+prompt, assistant target.

    The image itself is carried alongside in the row's ``images`` list; here the
    user turn just holds an ``{"type": "image"}`` placeholder, which is the shape
    TRL's vision SFT collators expect.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt_text}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


def _source_cfg(src) -> dict:
    """A SourceSpec (or plain dict) -> the dict shape registry.build_dataset wants."""
    get = src.get if isinstance(src, dict) else lambda k, d=None: getattr(src, k, d)
    cfg = {"adapter": get("adapter")}
    for k in ("hf_id", "split", "subset"):
        if get(k):
            cfg[k] = get(k)
    if get("params"):
        cfg["params"] = get("params")
    return cfg


def resolve_source_hf_id(src) -> str | None:
    """A training source's real HF repo id, however it's declared.

    A source can name its corpus three ways: a top-level ``hf_id`` (e.g.
    ``MuharafLines``), a top-level ``subset`` (e.g. ``KitabBench`` used
    directly), or ``params.subset`` (the same ``KitabBench`` convention used
    in ``configs/datasets.yaml``, e.g. a ``rasam1_train`` source). Checking
    only the literal ``.hf_id`` attribute (as this function's callers used to)
    silently skips the leak guard for the latter two — this resolves all
    three the same way `safe_eval_datasets` already does for the registry
    side, so the guard can't be bypassed just by choosing a different
    (equally valid) way to spell the same source.
    """
    return _hf_id_for(_source_cfg(src))


def iter_source_samples(src, *, cfg_path: str | Path = _DEFAULT_CFG):
    """Yield `Sample`s from one training source, after the no-leak check."""
    from ..datasets.registry import build_dataset

    get = src.get if isinstance(src, dict) else lambda k, d=None: getattr(src, k, d)
    hf_id, split = resolve_source_hf_id(src), get("split")
    if hf_id and split:
        check_source_not_scored(hf_id, split, cfg_path=cfg_path)
    adapter = build_dataset(get("name"), _source_cfg(src), limit=get("limit"))
    yield from adapter


def assemble_dataset(
    data: DataSpec, *, seed: int = 0, cfg_path: str | Path = _DEFAULT_CFG
) -> dict[str, list[dict]]:
    """Turn the configured sources into shuffled train / val row lists.

    Each row: ``{"messages": [...chat...], "images": [PIL], "source": str, "id": str}``.
    Returned as plain lists so this stays free of Arrow/PIL feature-inference
    quirks; the trainer wraps them in a ``datasets.Dataset``.
    """
    prompt_text = get_prompt(data.prompt_version)
    rows: list[dict] = []
    n_skipped = 0
    n_skipped_aspect = 0
    for src in data.sources:
        for sample in iter_source_samples(src, cfg_path=cfg_path):
            # Some dataset splits carry None/empty transcriptions (already
            # anticipated in contamination._match_key); an empty assistant
            # target isn't a trainable example and crashes the collators
            # (`"".join([None, ...])`), so drop it here instead.
            if not sample.ground_truth or not sample.ground_truth.strip():
                n_skipped += 1
                continue
            # A real crash found training on AraMS-28k-HTR: some real
            # manuscript-line crops are degenerate slivers with an extreme
            # aspect ratio (one observed at 389:1). qwen_vl_utils.smart_resize
            # hard-rejects anything >= 200:1 ("absolute aspect ratio must be
            # smaller than 200"), which crashes the collator mid-training —
            # not at data-assembly time, so it silently passes every earlier
            # check and only surfaces however many steps in it takes to draw
            # that specific sample. Filter it here instead, where every other
            # per-sample data-quality check already lives. cap_image() only
            # rescales proportionally (aspect ratio survives it unchanged
            # bar integer rounding), so checking the already-capped image is
            # equivalent to checking the source image.
            w, h = sample.image.size
            long_edge, short_edge = max(w, h), min(w, h)
            if short_edge == 0 or long_edge / short_edge >= _MAX_ASPECT_RATIO:
                n_skipped_aspect += 1
                continue
            rows.append(
                {
                    "messages": build_conversation(prompt_text, sample.ground_truth),
                    "images": [sample.image],
                    "source": getattr(src, "name", None)
                    or (src.get("name") if isinstance(src, dict) else None),
                    "id": sample.id,
                }
            )
    if n_skipped:
        print(f"[assemble_dataset] skipped {n_skipped} sample(s) with empty/missing ground truth")
    if n_skipped_aspect:
        print(
            f"[assemble_dataset] skipped {n_skipped_aspect} sample(s) with an aspect ratio "
            f">= {_MAX_ASPECT_RATIO}:1 (qwen_vl_utils.smart_resize hard-rejects >= 200:1)"
        )

    random.Random(seed).shuffle(rows)
    n_val = round(len(rows) * data.val_fraction)
    return {"val": rows[:n_val], "train": rows[n_val:]}

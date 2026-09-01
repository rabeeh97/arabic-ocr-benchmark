"""Check training sources for overlap with the benchmark's test sets.

Two signals, both deliberately cheap and dependency-light so this runs in the
core environment (no torch, no `imagehash`):

* **text collision** - the ground-truth string, aggressively normalized
  (diacritic-insensitive, whitespace removed), appears verbatim in a test set.
* **near-duplicate image** - a difference-hash (dHash) within a small Hamming
  radius of a test image.

`scan_overlap` is the pure core; `check_datasets` wires it to real dataset
configs and writes `results/contamination_report.json`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from ..normalize import normalize

_Item = tuple[str, Image.Image | None, str]  # (id, image_or_None, ground_truth)


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Row-wise difference hash. Grayscale, resize, compare adjacent pixels."""
    small = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    px = small.load()
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            bits = (bits << 1) | int(px[col, row] > px[col + 1, row])
    return bits


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def _match_key(text: str) -> str:
    """Normalized form used for exact text-collision comparison."""
    return normalize(text, "diac_insensitive").replace(" ", "")


def scan_overlap(
    train_items: Iterable[_Item],
    test_items: Iterable[_Item],
    *,
    near_dup_threshold: int = 4,
    min_text_len: int = 8,
    max_examples: int = 50,
) -> dict:
    """Compare one training source against one test set. Returns a report dict."""
    test_list = list(test_items)
    test_text_index: dict[str, str] = {}
    for tid, _img, gt in test_list:
        key = _match_key(gt)
        if len(key) >= min_text_len:
            test_text_index.setdefault(key, tid)
    test_hashes = [(tid, dhash(img)) for tid, img, _gt in test_list if img is not None]

    n_text_collisions = 0
    n_near_dup_images = 0
    text_collisions: list[dict] = []
    near_dup_images: list[dict] = []
    n_train = 0

    for trid, img, gt in train_items:
        n_train += 1

        key = _match_key(gt)
        if len(key) >= min_text_len and key in test_text_index:
            n_text_collisions += 1
            if len(text_collisions) < max_examples:
                text_collisions.append(
                    {"train_id": trid, "test_id": test_text_index[key], "text": gt}
                )

        if img is not None and test_hashes:
            h = dhash(img)
            best_tid, best_dist = min(
                ((tid, hamming(h, th)) for tid, th in test_hashes),
                key=lambda t: t[1],
            )
            if best_dist <= near_dup_threshold:
                n_near_dup_images += 1
                if len(near_dup_images) < max_examples:
                    near_dup_images.append(
                        {"train_id": trid, "test_id": best_tid, "hamming": best_dist}
                    )

    return {
        "n_train_items": n_train,
        "n_test_items": len(test_list),
        "n_text_collisions": n_text_collisions,
        "n_near_dup_images": n_near_dup_images,
        "text_collisions": text_collisions,
        "near_dup_images": near_dup_images,
    }


def check_datasets(
    train_specs: list[dict],
    test_dataset_names: list[str],
    *,
    datasets_cfg: str | Path = "configs/datasets.yaml",
    out_path: str | Path = "results/contamination_report.json",
    limit: int | None = None,
) -> dict:
    """Load real datasets, run `scan_overlap` for every (train x test) pair, and
    persist a combined report. Returns the report dict.

    train_specs: [{"name": ..., "adapter": "pkg.mod:Class", "params": {...},
                   "hf_id": ..., "split": ...}, ...]  (same shape as a
    configs/datasets.yaml entry, plus "name").
    """
    from ..datasets.registry import build_dataset, load_dataset_configs
    from .data import check_source_not_scored

    registry = load_dataset_configs(datasets_cfg)

    def _items(adapter) -> list[_Item]:
        return [(s.id, s.image, s.ground_truth) for s in adapter]

    test_sets = {
        name: _items(build_dataset(name, registry[name], limit=limit))
        for name in test_dataset_names
    }

    pairs = []
    for spec in train_specs:
        hf_id, split = spec.get("hf_id"), spec.get("split")
        if hf_id and split:
            check_source_not_scored(hf_id, split, cfg_path=datasets_cfg)
        adapter = build_dataset(
            spec["name"], {k: v for k, v in spec.items() if k != "name"}, limit=limit
        )
        train_items = _items(adapter)
        for test_name, test_items in test_sets.items():
            rep = scan_overlap(train_items, test_items)
            rep["train_source"] = spec["name"]
            rep["test_dataset"] = test_name
            pairs.append(rep)

    report = {
        "pairs": pairs,
        "n_flagged": sum(
            1 for p in pairs if p["n_text_collisions"] or p["n_near_dup_images"]
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

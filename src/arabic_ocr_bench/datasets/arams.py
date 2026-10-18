"""AraMS-28k-HTR adapter — real historical-manuscript line crops (Zenodo).

Source: Zenodo `10.5281/zenodo.21499649` ("AraMS-28k-HTR"), ~27,971 real,
pre-segmented handwritten-manuscript line crops. Not an HF dataset —
`scripts/fetch_arams.py` downloads and extracts it, and `root` here must be
the directory the archive extracts INTO (`data/arams_28k_htr` by default) —
NOT the nested `AraMs-28k-HTR/` folder the archive itself contains, since
the corpus's own manifest files use paths relative to that parent
(e.g. `./AraMs-28k-HTR/images/book_10_..._line000.png`).

CORRECTED 2026-09-17, verified against a real extraction: an earlier version
of this file assumed the corpus shipped no official split and invented a
seeded random per-line one. That was wrong. The archive ships its own
held-out split, **by book, not by line**
(`train_manifest.txt` / `val_manifest.txt` / `test_manifest.txt` — 19,739 /
1,486 / 6,744 lines respectively, confirmed different manuscript books
entirely in each file) — a materially better generalization test than a
per-line random split, since adjacent lines from the same page/book are
highly correlated in handwriting style. This adapter reads those manifests
directly; it does not invent a split.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ..imaging import cap_image
from .base import Sample

_MANIFEST_NAMES = {
    "train": "train_manifest.txt",
    "val": "val_manifest.txt",
    "test": "test_manifest.txt",
}


def _find_manifest(root: Path, split: str) -> Path:
    name = _MANIFEST_NAMES.get(split)
    if name is None:
        raise ValueError(f"AraMSHTR split must be one of {sorted(_MANIFEST_NAMES)}, got {split!r}")
    # The manifest lives inside the archive's own top-level folder (its exact
    # name isn't hardcoded here in case a future release renames it), or
    # directly under `root` if a caller already points there.
    for candidate in (*root.glob(f"*/{name}"), root / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {name} found under {root} (did scripts/fetch_arams.py finish?)")


def _manifest_rel_paths(root: Path, split: str) -> list[str]:
    manifest = _find_manifest(root, split)
    rel_paths = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Entries look like "./AraMs-28k-HTR/images/book_10_..._line000.png",
        # relative to `root` — drop the leading "./".
        rel_paths.append(line[2:] if line.startswith("./") else line)
    return rel_paths


class AraMSHTR:
    name = "arams_htr"
    track = "line"

    def __init__(
        self,
        root: str,
        split: str = "test",
        limit: int | None = None,
        offset: int = 0,
        indices: list[int] | None = None,
        name: str | None = None,
        max_dim: int = 1024,
    ):
        self.root = Path(root)
        self.split = split
        self.limit = limit
        self.offset = offset
        # explicit absolute row indices (accumulate / retry); wins over offset+limit
        self.indices = None if indices is None else list(indices)
        self.name = name or self.__class__.name
        self.max_dim = max_dim
        self._rel_paths: list[str] | None = None
        self._abs: list[int] = []

    def _load_rel_paths(self) -> list[str]:
        if self._rel_paths is not None:
            return self._rel_paths
        all_rel = _manifest_rel_paths(self.root, self.split)
        if self.indices is not None:
            sel = [i for i in self.indices if 0 <= i < len(all_rel)]
        else:
            start = min(self.offset, len(all_rel))
            end = (
                len(all_rel) if self.limit is None else min(self.offset + self.limit, len(all_rel))
            )
            sel = list(range(start, end))
        self._abs = sel
        self._rel_paths = [all_rel[i] for i in sel]
        return self._rel_paths

    def __len__(self) -> int:
        return len(self._load_rel_paths())

    def __iter__(self) -> Iterator[Sample]:
        rel_paths = self._load_rel_paths()
        for i, rel in enumerate(rel_paths):
            idx = self._abs[i] if i < len(self._abs) else self.offset + i
            img_path = self.root / rel
            gt_path = img_path.with_suffix(".gt.txt")
            img = Image.open(img_path).convert("RGB")
            gt = gt_path.read_text(encoding="utf-8").strip()
            yield Sample(
                id=f"{self.name}/{self.split}/{idx:06d}",
                image=cap_image(img, self.max_dim),
                ground_truth=gt,
                track=self.track,
                meta={
                    "source": "zenodo:10.5281/zenodo.21499649",
                    "split": self.split,
                    "row": idx,
                    "item_id": Path(rel).stem,
                },
            )

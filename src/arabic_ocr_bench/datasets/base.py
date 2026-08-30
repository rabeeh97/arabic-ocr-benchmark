"""The uniform sample type every dataset adapter yields."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class Sample:
    """One benchmark item.

    id:           stable, globally unique, e.g. "muharaf_lines/test/000123"
    image:        the line crop or full page
    ground_truth: reference transcription, kept RAW (no normalization, harakat intact)
    track:        "line" or "page"
    meta:         free-form: writer, era, script style, source page id, polygon, dpi, ...
    """

    id: str
    image: Image.Image
    ground_truth: str
    track: str
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str
    track: str

    def __iter__(self) -> Iterator[Sample]: ...

    def __len__(self) -> int: ...

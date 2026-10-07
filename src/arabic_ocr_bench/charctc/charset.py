"""Closed character vocabulary for CTC. Blank is index 0; unk is 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from arabic_ocr_bench.normalize import canonicalize_encoding

BLANK = "<blank>"
UNK = "<unk>"
BLANK_ID = 0
UNK_ID = 1


def canonicalize_label(text: str) -> str:
    """Lossless encoding cleanup + turn any leftover whitespace into a space.

    Line crops should not contain newlines; if a transcription does, it is a
    labelling artifact, not a character we want the CTC head to emit.
    """
    s = canonicalize_encoding(text)
    s = " ".join(s.split())
    return s


@dataclass(frozen=True)
class Charset:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != BLANK or self.tokens[1] != UNK:
            raise ValueError("Charset must start with <blank>, <unk>")
        object.__setattr__(self, "_stoi", {ch: i for i, ch in enumerate(self.tokens)})

    @property
    def n_classes(self) -> int:
        return len(self.tokens)

    def stoi(self, ch: str) -> int:
        return self._stoi.get(ch, UNK_ID)

    def encode(self, text: str) -> list[int]:
        return [self.stoi(ch) for ch in canonicalize_label(text)]

    def oov_chars(self, text: str) -> list[str]:
        known = set(self.tokens)
        return [ch for ch in canonicalize_label(text) if ch not in known]

    def to_json(self) -> dict:
        return {"tokens": list(self.tokens), "n_classes": self.n_classes}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, payload: dict) -> Charset:
        return cls(tokens=tuple(payload["tokens"]))

    @classmethod
    def load(cls, path: str | Path) -> Charset:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def build_charset(texts: list[str]) -> Charset:
    chars: set[str] = set()
    for text in texts:
        chars.update(canonicalize_label(text))
    chars.discard("")
    ordered = sorted(chars, key=lambda c: (c == " ", c))
    return Charset(tokens=(BLANK, UNK, *ordered))

"""Closed charset, CTC collapse, and line preprocess — no torch required."""

from __future__ import annotations

import json
import unicodedata

from PIL import Image

from arabic_ocr_bench.charctc.charset import (
    BLANK,
    UNK,
    UNK_ID,
    Charset,
    build_charset,
    canonicalize_label,
)
from arabic_ocr_bench.charctc.decode import ctc_greedy, ids_to_text
from arabic_ocr_bench.charctc.preprocess import TARGET_H, WIDTH_STRIDE, line_to_array, time_steps


def test_canonicalize_label_nfc_and_whitespace():
    raw = "  بَ \n\tالله  "
    out = canonicalize_label(raw)
    assert "\n" not in out
    assert out == unicodedata.normalize("NFC", "بَ الله")


def test_build_charset_blank_unk_first_and_roundtrip(tmp_path):
    cs = build_charset(["بسم الله", "بَسْمِ"])
    assert cs.tokens[0] == BLANK
    assert cs.tokens[1] == UNK
    assert "ب" in cs.tokens
    assert " " in cs.tokens
    assert "َ" in cs.tokens
    ids = cs.encode("بسم")
    assert UNK_ID not in ids
    path = tmp_path / "charset.json"
    cs.save(path)
    loaded = Charset.load(path)
    assert loaded.tokens == cs.tokens
    assert json.loads(path.read_text(encoding="utf-8"))["n_classes"] == cs.n_classes


def test_unknown_char_is_unk_and_reported_as_oov():
    cs = build_charset(["اب"])
    assert cs.stoi("س") == UNK_ID
    assert cs.oov_chars("بس") == ["س"]


def test_ctc_greedy_drops_blanks_and_collapses_repeats():
    # 0 = blank; 3 3 0 3 -> 3, 3
    assert ctc_greedy([0, 3, 3, 0, 3, 0]) == [3, 3]


def test_ids_to_text_skips_blank_and_unk():
    cs = build_charset(["اب"])
    # tokens: blank, unk, ا, ب  (space may also appear if in corpus — not here)
    a_id = cs.stoi("ا")
    b_id = cs.stoi("ب")
    assert ids_to_text([0, a_id, a_id, UNK_ID, b_id], cs) == "ااب"


def test_line_to_array_height_and_width_stride():
    img = Image.new("RGB", (200, 40), "white")
    arr = line_to_array(img)
    assert arr.shape[0] == 1
    assert arr.shape[1] == TARGET_H
    assert arr.shape[2] % WIDTH_STRIDE == 0
    assert time_steps(arr.shape[2]) * WIDTH_STRIDE == arr.shape[2]
    assert arr.min() >= -1.01 and arr.max() <= 1.01


def test_audit_markdown_renders_failures():
    from arabic_ocr_bench.charctc.audit import render_markdown

    md = render_markdown(
        {
            "n_texts": 12,
            "tokenizers": {
                "character": {
                    "ok": True,
                    "fertility": {
                        "tokens_per_word": {"mean": 4.2},
                        "chars_per_token": {"mean": 1.0},
                    },
                    "byte_fallback_rate": 0.0,
                    "diacritics": {"nfc_equals_nfd": True},
                    "charset_size": 80,
                },
                "trocr-english": {"ok": False, "error": "OSError: gated"},
            },
        }
    )
    assert "12" in md
    assert "`character`" in md
    assert "gated" in md


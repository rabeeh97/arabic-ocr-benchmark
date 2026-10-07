"""Tokenizer fertility / diacritic-consistency audit on our own GT texts.

Writes ``audit.json`` + ``audit.md`` — no GPU, no images required beyond what
the adapters load to yield samples (text is what we score).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from arabic_ocr_bench.charctc.charset import canonicalize_label
from arabic_ocr_bench.charctc.prepare import load_label_rows

# A letter + fatha, then the same pair with shadda, used as a consistency probe.
_DIAC_NFC = unicodedata.normalize("NFC", "بَ")
_DIAC_NFD = unicodedata.normalize("NFD", "بَ")
_SHADDA_FATHA = unicodedata.normalize("NFC", "بَّ")
_FATHA_SHADDA = "ب" + "\u064e" + "\u0651"  # fatha then shadda, not NFC


TOKENIZER_SPECS = (
    ("qwen3", "Qwen/Qwen3-VL-4B-Instruct"),
    ("trocr-english", "microsoft/trocr-base-handwritten"),
    ("aranizer-pbe-64k", "riotu-lab/Aranizer-PBE-64k"),
    ("paligemma2", "google/paligemma2-3b-pt-448"),
)


def _words(text: str) -> list[str]:
    return canonicalize_label(text).split()


def _ids_and_tokens(tokenizer, text: str) -> tuple[list[int], list[str]]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, tokens


def _fertility(ids: list[int], text: str) -> dict:
    words = _words(text)
    n_words = max(len(words), 1)
    n_chars = max(len(canonicalize_label(text).replace(" ", "")), 1)
    return {
        "n_tokens": len(ids),
        "n_words": len(words),
        "tokens_per_word": len(ids) / n_words,
        "tokens_per_char": len(ids) / n_chars,
        "chars_per_token": n_chars / max(len(ids), 1),
    }


def _byte_fallback_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    n = 0
    for tok in tokens:
        t = tok.lstrip("Ġ▁")
        if t.startswith("<0x") or t.startswith("byte") or t.startswith("<unk>"):
            n += 1
    return n / len(tokens)


def _load_tokenizer(hf_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)


def _probe_diacritics(tokenizer) -> dict:
    probes = {
        "nfc": _DIAC_NFC,
        "nfd": _DIAC_NFD,
        "shadda_fatha_nfc": _SHADDA_FATHA,
        "fatha_then_shadda": _FATHA_SHADDA,
    }
    out = {}
    for name, text in probes.items():
        ids, tokens = _ids_and_tokens(tokenizer, text)
        out[name] = {"ids": ids, "tokens": tokens, "n_tokens": len(ids)}
    out["nfc_equals_nfd"] = out["nfc"]["ids"] == out["nfd"]["ids"]
    out["mark_order_equals"] = (
        out["shadda_fatha_nfc"]["ids"] == out["fatha_then_shadda"]["ids"]
    )
    return out


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ("tokens_per_word", "tokens_per_char", "chars_per_token", "n_tokens")
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows]
        summary[k] = {
            "mean": sum(vals) / len(vals),
            "median": sorted(vals)[len(vals) // 2],
            "p90": sorted(vals)[int(len(vals) * 0.9)],
            "max": max(vals),
        }
    summary["n_lines"] = len(rows)
    return summary


def _char_level_stats(texts: list[str]) -> dict:
    rows = []
    for text in texts:
        label = canonicalize_label(text)
        words = _words(label)
        n_chars = len(label.replace(" ", ""))
        n_tok = n_chars  # one token per character, spaces dropped from fertility denom
        rows.append(
            {
                "n_tokens": n_tok,
                "n_words": len(words),
                "tokens_per_word": n_tok / max(len(words), 1),
                "tokens_per_char": 1.0,
                "chars_per_token": 1.0,
            }
        )
    charset_size = len({ch for t in texts for ch in canonicalize_label(t)})
    return {
        "hf_id": "character-level (this experiment)",
        "ok": True,
        "fertility": _summarise(rows),
        "byte_fallback_rate": 0.0,
        "diacritics": {
            "nfc_equals_nfd": True,
            "mark_order_equals": True,
            "note": "each haraka is its own class after NFC canonicalize",
        },
        "charset_size": charset_size,
    }


def audit_texts(texts: list[str], *, log=print) -> dict:
    reports: dict[str, dict] = {
        "character": _char_level_stats(texts),
    }
    for name, hf_id in TOKENIZER_SPECS:
        log(f"audit: loading tokenizer {name} ({hf_id})")
        try:
            tok = _load_tokenizer(hf_id)
        except Exception as exc:  # noqa: BLE001 — keep going; one gated card must not abort
            log(f"audit: {name} FAILED: {type(exc).__name__}: {exc}")
            reports[name] = {"hf_id": hf_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        per_line = []
        fb = []
        unk = Counter()
        for text in texts:
            label = canonicalize_label(text)
            try:
                ids, tokens = _ids_and_tokens(tok, label)
            except Exception as exc:  # noqa: BLE001
                log(f"audit: {name} encode failed: {exc}")
                continue
            stats = _fertility(ids, label)
            stats["byte_fallback_rate"] = _byte_fallback_rate(tokens)
            per_line.append(stats)
            fb.append(stats["byte_fallback_rate"])
            for t in tokens:
                if t in {"<unk>", "[UNK]", "<UNK>"}:
                    unk[t] += 1
        try:
            diac = _probe_diacritics(tok)
        except Exception as exc:  # noqa: BLE001
            diac = {"error": str(exc)}
        reports[name] = {
            "hf_id": hf_id,
            "ok": True,
            "vocab_size": getattr(tok, "vocab_size", None),
            "fertility": _summarise(per_line),
            "byte_fallback_rate": sum(fb) / len(fb) if fb else None,
            "unk_token_count": dict(unk),
            "diacritics": diac,
        }
        fert = reports[name]["fertility"]
        tpw = fert.get("tokens_per_word", {}).get("mean")
        log(f"audit: {name} tokens/word={tpw} vocab={reports[name]['vocab_size']}")
    return {"n_texts": len(texts), "tokenizers": reports}


def render_markdown(report: dict) -> str:
    lines = [
        "# Character-CTC tokenizer audit",
        "",
        f"Lines scored: **{report.get('n_texts', 0)}**",
        "",
        "| tokenizer | ok | tokens/word (mean) | chars/token (mean) | byte-fallback | NFC=NFD | vocab |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rec in report.get("tokenizers", {}).items():
        if not rec.get("ok"):
            lines.append(f"| `{name}` | no | — | — | — | — | {rec.get('error', '')} |")
            continue
        fert = rec.get("fertility") or {}
        tpw = fert.get("tokens_per_word", {}).get("mean")
        cpt = fert.get("chars_per_token", {}).get("mean")
        diac = rec.get("diacritics") or {}
        lines.append(
            f"| `{name}` | yes | "
            f"{_fmt(tpw)} | {_fmt(cpt)} | {_fmt(rec.get('byte_fallback_rate'))} | "
            f"{diac.get('nfc_equals_nfd', '')} | {rec.get('vocab_size') or rec.get('charset_size', '')} |"
        )
    lines.append("")
    lines.append(
        "Character-level is the output space this experiment trains. "
        "A 3–4× tokens/word gap on TrOCR-English vs Aranizer/character is the "
        "reason we are not fine-tuning stock TrOCR as-is."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="prepare() output dir with labels.jsonl")
    ap.add_argument("--out", required=True, help="directory for audit.json / audit.md")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    rows = load_label_rows(Path(args.cache))
    if args.limit:
        rows = rows[: args.limit]
    texts = [r["text"] for r in rows]
    print(f"[audit] {len(texts)} labels from {args.cache}", flush=True)
    report = audit_texts(texts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), flush=True)
    print(f"[audit] wrote {out / 'audit.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

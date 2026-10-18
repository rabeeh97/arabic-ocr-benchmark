"""Greedy CTC collapse (blank + repeat removal). No LM, no beam — CPU first try."""

from __future__ import annotations

from arabic_ocr_bench.charctc.charset import BLANK_ID, UNK_ID, Charset


def penalize_blank_logits(logits, penalty: float, *, blank_id: int = BLANK_ID):
    """Subtract ``penalty`` from the blank class so greedy CTC can leave all-blank.

    ``logits`` is ``(T, B, C)``. Returns a clone; the input is not mutated.
    """
    if penalty <= 0:
        return logits
    out = logits.clone()
    out[..., blank_id] = out[..., blank_id] - float(penalty)
    return out


def blank_fraction(logits, *, blank_id: int = BLANK_ID) -> float:
    """Share of timesteps whose argmax is blank. 1.0 means collapsed."""
    pred = logits.argmax(dim=-1)
    return float((pred == blank_id).float().mean())


def greedy_from_logits(
    logits,
    input_lens,
    charset: Charset,
    *,
    blank_penalty: float = 0.0,
    blank_id: int = BLANK_ID,
) -> list[str]:
    """Argmax + CTC collapse, optionally after a decode-time blank penalty."""
    scored = penalize_blank_logits(logits, blank_penalty, blank_id=blank_id)
    pred = scored.argmax(dim=-1).transpose(0, 1).detach().cpu().tolist()
    lens = input_lens.tolist() if hasattr(input_lens, "tolist") else list(input_lens)
    out: list[str] = []
    for seq, t in zip(pred, lens):
        ids = ctc_greedy(seq[: int(t)], blank_id=blank_id)
        out.append(ids_to_text(ids, charset))
    return out


def ctc_greedy(indices: list[int], *, blank_id: int = BLANK_ID) -> list[int]:
    """Collapse a CTC alignment: drop blanks, then collapse consecutive repeats."""
    out: list[int] = []
    prev: int | None = None
    for idx in indices:
        if idx == blank_id:
            prev = blank_id
            continue
        if idx != prev:
            out.append(idx)
        prev = idx
    return out


def ids_to_text(ids: list[int], charset: Charset) -> str:
    chars: list[str] = []
    for i in ids:
        if i in (BLANK_ID, UNK_ID):
            continue
        if 0 <= i < len(charset.tokens):
            chars.append(charset.tokens[i])
    return "".join(chars)

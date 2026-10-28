"""A large character n-gram LM from classical Arabic text (OpenITI), for the CTC beam search of rasam_lm_decode.py (2026-10-28).

    # build (Mac CPU, a few minutes; about 1-2 GB of RAM at the peak):
    python scripts/big_char_lm.py build --charset RUN/charset.json --out /Volumes/PortableSSD/models/rasam-replication/lm/openiti_o8.npz

Text: AbderrahmanSkiredj1/MLM_classical_arabic_openiti (1 M passages, about 113 M characters of OpenITI classical Arabic). NFC, characters
outside the recogniser's charset removed (brackets, Latin, digits in other scripts), whitespace collapsed, passages under 10 characters
dropped. Each passage is reversed (the CRNN emits visual order, like rasam_lm_decode.CharLM) and padded with BOS / EOS.

Same model as rasam_lm_decode.CharLM (Witten-Bell interpolation, order 8), but stored as sorted int64 keys per order (7 bits per
character) and looked up with numpy searchsorted, so 100 M characters fit in memory. MixLM interpolates it linearly with the small
in-domain LM: p = lam * p_in + (1 - lam) * p_big.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

BOS, EOS = "\x02", "\x03"
BITS = 7


class BigCharLM:
    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.order = int(z["order"])
        self.ids = {c: i for i, c in enumerate(json.loads(str(z["symbols"])))}
        self.v = int(z["v"])
        self.ng = [(z[f"ng{m}_k"], z[f"ng{m}_c"]) for m in range(1, self.order + 1)]   # m-gram keys / counts
        self.cx = [(z[f"cx{k}_k"], z[f"cx{k}_n"], z[f"cx{k}_u"]) for k in range(self.order)]  # context of length k: keys, total, distinct
        self.cache: dict = {}

    @staticmethod
    def _find(keys: np.ndarray, key: int) -> int:
        i = int(np.searchsorted(keys, key))
        return i if i < len(keys) and int(keys[i]) == key else -1

    def logp(self, c: str, hist: str) -> float:
        h = (BOS * (self.order - 1) + hist)[-(self.order - 1):]
        ck = (c, h)
        r = self.cache.get(ck)
        if r is not None:
            return r
        ci = self.ids.get(c)
        p = 1.0 / self.v
        if ci is not None:
            hid = [self.ids.get(x, -1) for x in h]
            for k in range(0, len(h) + 1):  # lowest order first
                ctx = hid[len(h) - k:]
                if -1 in ctx:
                    break
                key = 0
                for x in ctx:
                    key = (key << BITS) | x
                ck_keys, ck_n, ck_u = self.cx[k]
                j = self._find(ck_keys, key)
                if j < 0:
                    break  # a longer context containing this one cannot exist either
                n, u = int(ck_n[j]), int(ck_u[j])
                ng_keys, ng_c = self.ng[k]
                g = self._find(ng_keys, (key << BITS) | ci)
                p = ((int(ng_c[g]) if g >= 0 else 0) + u * p) / (n + u)
        r = math.log(max(p, 1e-12))
        self.cache[ck] = r
        return r


class MixLM:
    """Linear interpolation of two LMs with the CharLM interface (logp(c, hist), natural log)."""

    def __init__(self, small, big: BigCharLM, lam: float) -> None:
        self.small, self.big, self.lam = small, big, lam
        self.cache: dict = {}

    def logp(self, c: str, hist: str) -> float:
        k = (c, hist[-8:])
        r = self.cache.get(k)
        if r is None:
            r = math.log(self.lam * math.exp(self.small.logp(c, hist)) + (1 - self.lam) * math.exp(self.big.logp(c, hist)))
            self.cache[k] = r
        return r


def _count(seqs: np.ndarray, valid: np.ndarray, m: int, chunk: int = 15_000_000):
    """Unique m-gram keys ending at the valid positions, with counts (chunked to bound memory)."""
    parts_k, parts_c = [], []
    pos = np.flatnonzero(valid)
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        key = np.zeros(len(p), dtype=np.int64)
        for j in range(m - 1, -1, -1):
            key = (key << BITS) | seqs[p - j].astype(np.int64)
        k, c = np.unique(key, return_counts=True)
        parts_k.append(k); parts_c.append(c)
    k = np.concatenate(parts_k); c = np.concatenate(parts_c)
    order = np.argsort(k, kind="stable")
    k, c = k[order], c[order]
    uk, idx = np.unique(k, return_index=True)
    return uk, np.add.reduceat(c, idx).astype(np.int64)


def build(a) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    chars = json.loads(Path(a.charset).read_text(encoding="utf-8"))["chars"]
    symbols = ["<pad>"] + chars + [BOS, EOS]
    assert len(symbols) < 2 ** BITS
    ids = {c: i for i, c in enumerate(symbols)}
    keep = set(chars) | {" "}
    p = hf_hub_download("AbderrahmanSkiredj1/MLM_classical_arabic_openiti", "data/train-00000-of-00001-f202266784bed2d5.parquet", repo_type="dataset")
    texts = pq.read_table(p).column("text").to_pylist()
    pad = [ids[BOS]] * (a.order - 1)
    buf, n_txt, n_chr = [], 0, 0
    for t in texts:
        t = " ".join("".join(ch for ch in unicodedata.normalize("NFC", t or "") if ch in keep).split())
        if len(t) < 10:
            continue
        buf.extend(pad); buf.extend(ids[ch] for ch in t[::-1]); buf.append(ids[EOS])
        n_txt += 1; n_chr += len(t)
        if a.max_chars and n_chr >= a.max_chars:
            break
    seqs = np.asarray(buf, dtype=np.uint8)
    del buf, texts
    valid = seqs != ids[BOS]
    print(f"{n_txt} passages, {n_chr} characters", flush=True)
    out = {"order": np.array(a.order), "symbols": np.array(json.dumps(symbols)), "v": np.array(len(set(seqs[valid].tolist())) + 1)}
    for m in range(1, a.order + 1):
        k, c = _count(seqs, valid, m)
        out[f"ng{m}_k"], out[f"ng{m}_c"] = k, c.astype(np.int32)
        ctx = k >> BITS
        ck, idx, u = np.unique(ctx, return_index=True, return_counts=True)
        out[f"cx{m - 1}_k"], out[f"cx{m - 1}_n"], out[f"cx{m - 1}_u"] = ck, np.add.reduceat(c, idx).astype(np.int64), u.astype(np.int32)
        print(f"order {m}: {len(k):,} distinct n-grams, {len(ck):,} contexts", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--charset", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--order", type=int, default=8)
    b.add_argument("--max-chars", type=int, default=0)
    a = ap.parse_args()
    build(a)

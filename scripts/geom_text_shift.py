"""Text-side distance between the CRNN's training text and each evaluation set (R1.6 / R1.7 in the CRNN setting).

    .geomvenv/bin/python scripts/geom_text_shift.py OUT.json

Training text = every Muharaf train label plus the RASAM-1 and RASAM-2 labels (what the aligner trained on).
For Muharaf test, NAKBA test, NAKBA blind and KHATT it reports:
    char_oov      share of characters outside the training character set (they cannot be read)
    word_oov      share of word tokens whose type never occurs in training
    xent_bits     cross-entropy of a character 5-gram model trained on the training text (interpolated, add-k), bits per char
    rare_shape_10 / _50   share of letters whose (letter, positional form) occurs fewer than 10 / 50 times in training
    ttr_2000      type-token ratio on the first 2,000 word tokens (lexical diversity)
Labels are cleaned the same way as for the CRNN (harakat, tatweel and bidi marks removed).
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import HF, SOURCES  # noqa: E402

STRIP = re.compile("[ً-ْٰـ​-‏‪-‮⁦-⁩﻿؜]")
RIGHT = set("اأإآٱدذرزوؤة")  # join only to the previous letter
NONJ = set("ء")
LETTER = re.compile(r"[ء-ي]")


def clean(t):
    return " ".join(STRIP.sub("", unicodedata.normalize("NFC", t or "")).split())


def texts(pattern, col):
    out = []
    for p in sorted(glob.glob(str(HF / pattern))):
        mm = pa.memory_map(p)
        try:
            t = pa.ipc.open_stream(mm).read_all()
        except pa.ArrowInvalid:
            t = pa.ipc.open_file(pa.memory_map(p)).read_all()
        c = col or next(x for x in ("text", "transcription", "answer", "label") if x in t.column_names)
        out += [clean(x) for x in t.column(c).to_pylist()]
    return [x for x in out if x]


def forms(line):
    """(letter, form) for every Arabic letter; form from the Unicode joining types."""
    out = []
    for i, ch in enumerate(line):
        if not LETTER.match(ch):
            continue
        prev = line[i - 1] if i else " "
        nxt = line[i + 1] if i + 1 < len(line) else " "
        jp = bool(LETTER.match(prev)) and prev not in RIGHT and prev not in NONJ and ch not in NONJ
        jn = ch not in RIGHT and ch not in NONJ and bool(LETTER.match(nxt)) and nxt not in NONJ
        out.append((ch, {(False, False): "iso", (False, True): "ini", (True, True): "med", (True, False): "fin"}[(jp, jn)]))
    return out


class CharLM:
    def __init__(self, lines, n=5, k=0.01):
        self.n, self.k = n, k
        self.c = [Counter() for _ in range(n + 1)]
        self.vocab = set()
        for l in lines:
            s = "\x02" * (n - 1) + l + "\x03"
            self.vocab |= set(s)
            for i in range(n - 1, len(s)):
                for m in range(1, n + 1):
                    self.c[m][s[i - m + 1 : i + 1]] += 1
        self.V = len(self.vocab) + 1
        self.tot = sum(self.c[1].values())

    def prob(self, ctx, ch):
        p = (self.c[1][ch] + self.k) / (self.tot + self.k * self.V)
        for m in range(2, self.n + 1):  # simple interpolation, more weight on longer contexts that were seen
            h = ctx[-(m - 1):]
            den = self.c[m - 1][h] if m > 1 else self.tot
            if den == 0:
                break
            lam = den / (den + 5.0)
            p = lam * (self.c[m][h + ch] + self.k * p * self.V) / (den + self.k * self.V) + (1 - lam) * p
        return p

    def xent(self, lines):
        bits = n = 0
        for l in lines:
            s = "\x02" * (self.n - 1) + l + "\x03"
            for i in range(self.n - 1, len(s)):
                bits -= math.log2(self.prob(s[i - self.n + 1 : i], s[i]))
                n += 1
        return bits / max(n, 1)


def main():
    train = texts("aamijar___muharaf-public/default/*/*/muharaf-public-train-*.arrow", "text")
    train += texts("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "transcription") + texts("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "transcription")
    chars = Counter(ch for l in train for ch in l)
    words = Counter(w for l in train for w in l.split())
    shapes = Counter(f for l in train for f in forms(l))
    lm = CharLM(train)
    res = {"train_lines": len(train), "sets": {}}
    for tag in ["muharaf", "nakba", "nakba_blind", "khatt"]:
        pat, col = SOURCES[tag][0], SOURCES[tag][1]
        L = texts(pat, col)
        nch = sum(len(l) for l in L)
        toks = [w for l in L for w in l.split()]
        fs = [f for l in L for f in forms(l)]
        r = {"lines": len(L),
             "char_oov": sum(1 for l in L for ch in l if ch not in chars) / nch,
             "word_oov": sum(1 for w in toks if w not in words) / len(toks),
             "xent_bits": lm.xent(L),
             "rare_shape_10": sum(1 for f in fs if shapes[f] < 10) / len(fs),
             "rare_shape_50": sum(1 for f in fs if shapes[f] < 50) / len(fs),
             "ttr_2000": len(set(toks[:2000])) / min(2000, len(toks)),
             "mean_word_len": sum(len(w) for w in toks) / len(toks),
             "top_oov_chars": Counter(ch for l in L for ch in l if ch not in chars).most_common(8)}
        res["sets"][tag] = r
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

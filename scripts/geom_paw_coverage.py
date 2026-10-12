"""PAW inventory and coverage (queue item "PAW-inventory coverage curve"), plus unseen-PAW rates against the CRNN's training text.

    .geomvenv/bin/python scripts/geom_paw_coverage.py OUT.json

A word splits into PAWs (parts of Arabic words) after every letter that does not join to the next one
(ا أ إ آ ٱ د ذ ر ز و ؤ ة ء). The rasm version maps dotted letters onto their dotless skeleton (an approximation that ignores
the positional forms of ن and ي). For each set: distinct PAWs, the number of most frequent PAW types that cover 95 % and 99 %
of PAW tokens, and the share of PAW tokens never seen in the training text.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geom_data import SOURCES  # noqa: E402
from geom_text_shift import texts  # noqa: E402

BREAK = set("اأإآٱدذرزوؤةء")
RASM = str.maketrans({**{c: "ٮ" for c in "بتثنيىئ"}, **{c: "ح" for c in "جحخ"}, "ذ": "د", "ز": "ر", "ش": "س", "ض": "ص", "ظ": "ط", "غ": "ع",
                      **{c: "ڡ" for c in "فق"}, "ة": "ه", "ؤ": "و", **{c: "ا" for c in "أإآٱ"}})


def paws(word):
    out, cur = [], ""
    for ch in word:
        cur += ch
        if ch in BREAK:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def stats(lines, train_inv, rasm=False):
    toks = [p.translate(RASM) if rasm else p for l in lines for w in l.split() for p in paws(w)]
    c = Counter(toks)
    total = len(toks)
    cum, n95, n99 = 0, None, None
    for k, (_, v) in enumerate(c.most_common(), 1):
        cum += v
        if n95 is None and cum >= 0.95 * total:
            n95 = k
        if n99 is None and cum >= 0.99 * total:
            n99 = k
            break
    unseen = sum(v for p, v in c.items() if p not in train_inv) / total if train_inv is not None else None
    return {"paw_tokens": total, "distinct": len(c), "types_for_95pct": n95, "types_for_99pct": n99, "unseen_token_share": unseen, "paws_per_word": total / max(1, sum(len(l.split()) for l in lines))}


def main():
    train = texts("aamijar___muharaf-public/default/*/*/muharaf-public-train-*.arrow", "text")
    train += texts("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "transcription") + texts("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "transcription")
    inv = {p for l in train for w in l.split() for p in paws(w)}
    inv_r = {p.translate(RASM) for p in inv}
    res = {"train": {"dotted": stats(train, None), "rasm": stats(train, None, True)}}
    for tag in ["muharaf", "nakba", "nakba_blind", "khatt"]:
        L = texts(SOURCES[tag][0], SOURCES[tag][1])
        res[tag] = {"dotted": stats(L, inv), "rasm": stats(L, inv_r, True)}
    for k, v in res.items():
        print(k, "dotted", v["dotted"], "\n   rasm", v["rasm"])
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

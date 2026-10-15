"""The iʿjām question (theme T2/T6, 2026-10-15) on our training texts: after folding orthographic variants (hamza/alef, ya/maqsura, ta marbuta),
how many words become ambiguous when the dots are removed, and how much does one word of context resolve?
Held-out estimate: counts from 90 % of lines, entropy measured on the other 10 %."""
import math
import pickle
import random
import sys
from collections import Counter, defaultdict

from pathlib import Path
W = str(Path(__file__).resolve().parent.parent)
import json
RES = {}
sys.path.insert(0, W + "/scripts")
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from analyze_letters import words_view  # noqa: E402
from arabic_ocr_bench.characterize.errors import _orthographic, to_rasm  # noqa: E402

for name, f in (("Muharaf train", "aligner_data_standard.pkl"), ("NAKBA train", "aligner_data_nakba.pkl")):
    lines = [r["label"] for r in pickle.load(open(f"{W}/results/panel/{f}", "rb"))["rows"] if r["role"] == "train"]
    random.Random(0).shuffle(lines)
    cut = int(0.9 * len(lines))
    tr, te = lines[:cut], lines[cut:]
    norm = lambda ln: [_orthographic(w) for w in words_view(ln)]  # noqa: E731
    T = [norm(ln) for ln in tr]
    cw = Counter(w for ws in T for w in ws)
    R = {w: to_rasm(w) for w in cw}
    by_r = defaultdict(Counter)
    for w, n in cw.items():
        by_r[R[w]][w] += n
    toks = sum(cw.values())
    amb = sum(n for w, n in cw.items() if len(by_r[R[w]]) > 1) / toks
    big, ctx = Counter(), Counter()
    for ws in T:
        for p, w in zip(["<s>"] + ws, ws):
            big[(p, R[w], w)] += 1
            ctx[(p, R[w])] += 1
    # held-out: -log2 P(word | rasm) with add-0.1 smoothing over the rasm's candidates; with context, back off to unigram
    h1 = h2 = n = oov = 0
    acc1 = acc2 = 0
    for ln in te:
        ws = norm(ln)
        for p, w in zip(["<s>"] + ws, ws):
            r = to_rasm(w)
            cands = by_r.get(r)
            if not cands or w not in cands:
                oov += 1
                continue
            tot = sum(cands.values())
            p1 = (cands[w] + 0.1) / (tot + 0.1 * len(cands))
            c = ctx.get((p, r), 0)
            p2 = (big.get((p, r, w), 0) + 0.5 * p1 * 2) / (c + 1) if c else p1
            h1 -= math.log2(p1); h2 -= math.log2(p2); n += 1
            acc1 += max(cands, key=cands.get) == w
            best2 = max(cands, key=lambda x: (big.get((p, r, x), 0), cands[x]))
            acc2 += best2 == w
    RES[name] = {"train_tokens": toks, "ambiguous_token_share": amb, "heldout_known_tokens": n, "heldout_unseen_share": oov / (n + oov), "H_word_given_skeleton": h1 / n, "H_word_given_skeleton_prev": h2 / n, "restore_most_frequent": acc1 / n, "restore_with_prev": acc2 / n, "examples": [[w for w, _ in c.most_common(4)] for r_, c in sorted(((r, c) for r, c in by_r.items() if len(c) > 1), key=lambda t: -sum(t[1].values()) + max(t[1].values()))[:10]]}
    print(f"{name}: {toks} train tokens; tokens whose dotless form is shared by another (folded) word: {100 * amb:.1f}%")
    print(f"   held-out ({n} known tokens, {100 * oov / (n + oov):.1f}% unseen word or unseen skeleton): H(word | skeleton) {h1 / n:.3f} bits/word, "
          f"with the previous word {h2 / n:.3f}; restoring dots by the most frequent word: {100 * acc1 / n:.1f}% right, with the previous word {100 * acc2 / n:.1f}%")
    top = sorted(((r, c) for r, c in by_r.items() if len(c) > 1), key=lambda t: -sum(t[1].values()) + max(t[1].values()))[:6]
    for r, c in top:
        print("      ", " / ".join(f"{w}:{k}" for w, k in c.most_common(4)))

Path(W, "results/analysis/B/ijam.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

"""Is NAKBA's alif-hamza error a spelling convention? Hamza use in training against evaluation labels, and CER with hamza folded.

    .geomvenv/bin/python scripts/geom_hamza.py results/geometry/base_evaldata.json results/geometry/transfer_eval.pkl OUT.json

1. Share of alifs written with hamza (أ إ آ against bare ا) in the CRNN's training text and in each evaluation set's labels.
2. For every ground-truth أ / إ / آ in NAKBA and KHATT: what the CRNN wrote at that place (aligned with difflib).
3. CER as scored, and CER after folding أ إ آ ٱ to ا on both sides (the standard orthographic fold).
"""
import difflib
import json
import pickle
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from geom_text_shift import texts  # noqa: E402
from xai_train_aligner import edit_distance  # noqa: E402

HAMZA = "أإآ"
FOLD = str.maketrans({c: "ا" for c in "أإآٱ"})
per = json.load(open(sys.argv[1], encoding="utf-8"))["per_line"]
rows = pickle.load(open(sys.argv[2], "rb"))["rows"]
gt = {(r["dataset"], r["id"]): r["label"] for r in rows if "__" not in r["dataset"]}


def hamza_share(lines):
    c = Counter(ch for l in lines for ch in l if ch in HAMZA + "ا")
    return sum(c[h] for h in HAMZA) / max(1, sum(c.values()))


train = texts("aamijar___muharaf-public/default/*/*/muharaf-public-train-*.arrow", "text")
rasam = texts("calfa-ai___rasam-1/*/*/*/rasam-1-test.arrow", "transcription") + texts("calfa-ai___rasam-2/*/*/*/rasam-2-test.arrow", "transcription")
out = {"hamza_share_of_alifs": {"muharaf_train": hamza_share(train), "rasam": hamza_share(rasam)}, "sets": {}}
for tag in ("nakba_lines", "kitab_khatt", "nakba_blind"):
    lines = [(i, v) for i, v in per.get(tag, {}).items() if (tag, i) in gt]
    labels = [gt[(tag, i)] for i, _ in lines]
    out["hamza_share_of_alifs"][tag] = hamza_share(labels)
    read = Counter()
    e = ef = n = 0
    for (i, (err, ln, hyp)), lab in zip(lines, labels):
        n += len(lab)
        e += edit_distance(list(hyp), list(lab))
        ef += edit_distance(list(hyp.translate(FOLD)), list(lab.translate(FOLD)))
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, lab, hyp, autojunk=False).get_opcodes():
            for k in range(i1, i2):
                if lab[k] in HAMZA:
                    if op == "equal":
                        read["same"] += 1
                    elif op == "replace" and (i2 - i1) == (j2 - j1):
                        read[hyp[j1 + k - i1]] += 1
                    else:
                        read["other edit"] += 1
    out["sets"][tag] = {"lines": len(lines), "cer": e / n, "cer_hamza_folded": ef / n, "gt_hamza_read_as": dict(read.most_common(6))}
    print(tag, out["sets"][tag])
print(out["hamza_share_of_alifs"])
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)

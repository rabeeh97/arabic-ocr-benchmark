"""Agreement between labelers on the lines they both labelled (the calibration lines, and any others)."""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

from ..metrics import unicode_levenshtein
from ..normalize import strip_harakat
from . import labels


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return None if pe == 1 else (po - pe) / (1 - pe)


def _cer(a: str, b: str) -> float:
    return unicode_levenshtein(a, b) / max(len(a), len(b), 1)


def compare(labels_root, tranche_name: str, labelers: list[str] | None = None, top: int = 10) -> dict:
    ids = labelers or labels.list_labelers(labels_root, tranche_name)
    data = {lab: labels.load_labels(labels_root, lab, tranche_name) for lab in ids}
    out = {"labelers": ids, "pairs": []}
    for a, b in itertools.combinations(ids, 2):
        common = sorted(set(data[a]) & set(data[b]))
        va = [data[a][i]["verdict"] for i in common]
        vb = [data[b][i]["verdict"] for i in common]
        both_text = [i for i in common if data[a][i]["text_verified"] and data[b][i]["text_verified"]]
        per_ds = defaultdict(list)
        diffs = []
        for i in both_text:
            ta, tb = data[a][i]["text_verified"], data[b][i]["text_verified"]
            c_full, c_nod = _cer(ta, tb), _cer(strip_harakat(ta), strip_harakat(tb))
            per_ds[data[a][i]["dataset"]].append((ta == tb, c_full, c_nod))
            if ta != tb:
                diffs.append({"line_id": i, "cer": round(c_full, 3), a: ta, b: tb,
                              "verdicts": [data[a][i]["verdict"], data[b][i]["verdict"]]})
        diffs.sort(key=lambda d: -d["cer"])
        rows = {ds: {"n": len(v), "exact_text": round(sum(x[0] for x in v) / len(v), 3),
                     "cer_full": round(sum(x[1] for x in v) / len(v), 4),
                     "cer_no_diacritics": round(sum(x[2] for x in v) / len(v), 4)} for ds, v in per_ds.items()}
        out["pairs"].append({
            "a": a, "b": b, "common_lines": len(common),
            "verdict_agreement": round(sum(x == y for x, y in zip(va, vb)) / len(common), 3) if common else None,
            "verdict_kappa": None if cohen_kappa(va, vb) is None else round(cohen_kappa(va, vb), 3),
            "texts_compared": len(both_text),
            "exact_text": round(sum(data[a][i]["text_verified"] == data[b][i]["text_verified"] for i in both_text) / len(both_text), 3) if both_text else None,
            "per_dataset": rows, "biggest_disagreements": diffs[:top],
        })
    return out


def to_markdown(rep: dict) -> str:
    if not rep["pairs"]:
        return f"Labelers found: {rep['labelers']}. Need at least two with overlapping lines."
    L = []
    for p in rep["pairs"]:
        L += [f"## {p['a']} vs {p['b']}", f"- lines both labelled: {p['common_lines']}",
              f"- verdict agreement: {p['verdict_agreement']} (Cohen's kappa {p['verdict_kappa']})",
              f"- texts compared: {p['texts_compared']}, identical: {p['exact_text']}", "",
              "| dataset | n | identical | CER (with marks) | CER (no marks) |", "|---|---:|---:|---:|---:|"]
        for ds, r in sorted(p["per_dataset"].items()):
            L.append(f"| {ds} | {r['n']} | {r['exact_text']} | {r['cer_full']} | {r['cer_no_diacritics']} |")
        L += ["", "Biggest disagreements:"]
        for d in p["biggest_disagreements"]:
            L.append(f"- `{d['line_id']}` CER {d['cer']}: {d[p['a']]!r} vs {d[p['b']]!r} (verdicts {d['verdicts']})")
        L.append("")
    return "\n".join(L)

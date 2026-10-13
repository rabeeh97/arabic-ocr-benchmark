"""The per-line store: every reading of every line by every system, scored at every rung (plan 2026-10-13, §3.2).

    .geomvenv/bin/python scripts/store_build.py [--sources configs/analysis/sources.json] [--out results/store] [--no-vlm]

Writes OUT/samples.parquet (one row per line: GT, ids, ink extent, CTC filter flag, page / manuscript fields),
OUT/lines.parquet (one row per run × transform × line × evaluation pack), OUT/runs.json and OUT/reconcile.json.

Readings come from:
  crnn_per_line   geom_transfer_eval.py files: {per_line: {tag: {id: [edits, label_len, reading]}}}, tag = dataset[__transform]
  crnn_tta        xai_geom_tta.py files: tags ds, ds__tta, ds__inkcrop, ds__inkcrop__tta
  e8              xai_e8_spike.py per_line.json: {ds: {id: {gt, crnn, e8}}}
  crnn_panel      crnn_eval_panel.py files (fp32, one line at a time, with confidence)
  vlm             every predictions*.jsonl under the main checkout, via characterize/runs.py (standard prompt, latest
                  success per sample, smoke runs skipped, renames applied), one run per model name
Ids are mapped to the store key "{dataset}/{split}/{index:06d}" through the panel's legacy ids; GT always comes from the
panel (the HF row), never from the result file. Edits: errors.ladder_edits at the seven ladder rungs against the raw
GT, plus crnn_native = Levenshtein against the GT cleaned as for CRNN training (harakat, tatweel, bidi marks stripped).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, "/Volumes/PortableSSD/repos/arabic_ocr/src")
from arabic_ocr_bench.characterize import errors  # noqa: E402
from arabic_ocr_bench.characterize import runs as vruns  # noqa: E402
from geom_inkcrop_export import clean  # noqa: E402

RUNGS = list(errors.RUNGS) + ["crnn_native"]
LINE_SETS = {"kitab_khatt", "nakba_lines", "nakba_blind", "muharaf_lines", "rasam1", "rasam2"}
API = ("gemini", "gpt-", "claude", "pixtral", "mistral", "glm", "deepseek", "openrouter", "nvidia-")


def vlm_family(model: str) -> str:
    m = model.lower()
    if "paddle" in m or "trocr" in m or "qari" in m:
        return "ocr-engine"
    if m.startswith(API):
        return "api"
    if "-ft" in m or "visft" in m or m.startswith(("suite-v1", "nakba-", "baseer")) or "nakba" in m:
        return "vlm-ft"
    return "vlm-base"


def reversed_suspect(gt_clean: str, hyp: str, native: int) -> bool:
    """D22: the reading matches the GT much better backwards (a stored-order bug, plan §1.5 #5)."""
    return len(hyp) >= 8 and native > 0.3 * max(1, len(gt_clean)) and Levenshtein.distance(gt_clean, hyp[::-1]) < 0.5 * native


class Store:
    def __init__(self, samples: dict, legacy: dict):
        self.samples, self.legacy = samples, legacy
        self.rows: list[dict] = []
        self.runs: dict[str, dict] = {}
        self.rec: dict[str, dict] = {}

    aligner_train: set[str] = set()

    def in_train(self, run: str, key: str) -> str:
        """yes / no for CRNN runs (the aligner recipe trained on most Muharaf and RASAM test lines, CORRECTIONS C4; the
        standard split on none), unknown for everything else."""
        if run.startswith("crnn/std") :
            return "no"
        if run.startswith(("crnn/", "crnn+llm/")):
            return "yes" if key in self.aligner_train else "no"
        return "unknown"

    def key(self, ds: str, sid: str) -> str | None:
        if sid in self.samples:
            return sid
        return self.legacy.get(sid)

    def add(self, run, family, model, transform, key, hyp, pack, numerics, source, stored=None, conf=None):
        s = self.samples[key]
        gt, gtc = s["gt_raw"], s["gt_clean"]
        hyp = hyp or ""
        lad = errors.ladder_edits(gt, hyp)
        native = Levenshtein.distance(gtc, hyp)
        row = {"run_id": run, "family": family, "model": model, "transform": transform, "sample_key": key, "dataset": s["dataset"],
               "pack": pack, "numerics": numerics, "source": source, "hyp": hyp, "conf": conf,
               "stored_edits": None if stored is None else int(stored[0]), "stored_ref_len": None if stored is None else int(stored[1]),
               "reversed_suspect": reversed_suspect(gtc, hyp, native), "in_train": self.in_train(run, key)}
        for name, (e, n) in zip(errors.RUNGS, lad):
            row[f"e_{name}"], row[f"n_{name}"] = e, n
        row["e_crnn_native"], row["n_crnn_native"] = native, len(gtc)
        self.rows.append(row)
        self.runs.setdefault(run, {"family": family, "model": model, "sources": set()})["sources"].add(source)
        return row

    def reconcile(self, source: str, tag: str, rows: list[dict], unmapped: int):
        r = {"lines": len(rows), "unmapped_ids": unmapped, "reversed_suspect": sum(x["reversed_suspect"] for x in rows)}
        st = [x for x in rows if x["stored_edits"] is not None]
        if st:
            r["stored_cer"] = sum(x["stored_edits"] for x in st) / max(1, sum(x["stored_ref_len"] for x in st))
            r["native_cer"] = sum(x["e_crnn_native"] for x in st) / max(1, sum(x["n_crnn_native"] for x in st))
            r["native_equals_stored"] = sum(x["e_crnn_native"] == x["stored_edits"] for x in st) / len(st)
            r["ref_len_equals"] = sum(x["n_crnn_native"] == x["stored_ref_len"] for x in st) / len(st)
        if rows:
            r["no_diacritics_cer"] = sum(x["e_no_diacritics"] for x in rows) / max(1, sum(x["n_no_diacritics"] for x in rows))
        self.rec[f"{source}::{tag}"] = r


def split_tag(tag: str) -> tuple[str, str]:
    """'kitab_khatt__inkcrop__tta' -> ('kitab_khatt', 'inkcrop+tta_conf'); no suffix -> 'none'."""
    ds, *parts = tag.split("__")
    return ds, "+".join("tta_conf" if p == "tta" else p for p in parts) or "none"


def load_samples(panel: Path):
    blob = pickle.load(open(panel, "rb"))
    samples, legacy = {}, {}
    for r in blob["rows"]:
        if r["dataset"].endswith("__inkcrop"):
            continue
        ds, split, idx = r["id"].split("/")
        samples[r["id"]] = {"sample_key": r["id"], "dataset": ds, "split": split, "index": int(idx), "legacy_id": r["legacy_id"],
                            "gt_raw": r["label_raw"], "gt_clean": clean(r["label_raw"]), "ink_extent": r["ink_extent"], "ctc_ok": bool(r["ctc_ok"]),
                            "meta": json.dumps(r["meta"], ensure_ascii=False), "img_w": int(r["img"].shape[1])}
        legacy[r["legacy_id"]] = r["id"]
        if ds == "muharaf_lines":  # the transfer pack wrote muharaf_lines/NNNNNN for test rows
            legacy[f"{ds}/{idx}"] = r["id"]
    return samples, legacy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=W / "configs/analysis/sources.json")
    ap.add_argument("--out", type=Path, default=W / "results/store")
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    S = json.loads(a.sources.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    samples, legacy = load_samples(W / S["panel"])
    st = Store(samples, legacy)
    st.aligner_train = set(json.loads((W / S["crnn_aligner_train_ids"]).read_text()))
    print("samples", len(samples), Counter(s["dataset"] for s in samples.values()), flush=True)

    def ingest_pl(src, run, pack, numerics, per_line, conf=None):
        for tag, d in per_line.items():
            ds, tr = split_tag(tag)
            rows, unm = [], 0
            for sid, v in d.items():
                k = st.key(ds, sid)
                if k is None:
                    unm += 1
                    continue
                c = None if conf is None else conf.get(tag, {}).get(sid)
                rows.append(st.add(run, "crnn", run.split("/")[1], tr, k, v[2], pack, numerics, src, stored=v[:2], conf=c))
            st.reconcile(src, tag, rows, unm)

    for s in S["crnn_per_line"] + S["crnn_tta"]:
        d = json.loads((W / s["file"]).read_text(encoding="utf-8"))
        ingest_pl(s["file"], s["run"], s["pack"], s["numerics"], d["per_line"])
        print("ingested", s["file"], flush=True)
    for f in sorted(glob.glob(str(W / S["crnn_panel_glob"]))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        run = "crnn/" + d["run"].replace("aligner_run_", "").replace("aligner_run", "base") + ("" if d["ckpt"] == "model_final.pt" else "-best")
        ingest_pl(str(Path(f).relative_to(W)), run, d.get("panel", "panel"), d["numerics"], d["per_line"], d.get("conf"))
        print("ingested", f, flush=True)
    for s in S["e8"]:
        d = json.loads((W / s["file"]).read_text(encoding="utf-8"))
        for ds, m in d.items():
            for which, run in (("crnn", s["crnn_run"]), ("e8", s["llm_run"])):
                if run is None:
                    continue
                rows, unm = [], 0
                for sid, v in m.items():
                    k = st.key(ds, sid)
                    if k is None:
                        unm += 1
                        continue
                    rows.append(st.add(run, "crnn" if which == "crnn" else "crnn+llm", run.split("/")[1], "none", k, v[which], s["pack"], s["numerics"], s["file"]))
                st.reconcile(s["file"], f"{ds}::{which}", rows, unm)
        print("ingested", s["file"], flush=True)
    if not a.no_vlm:
        canon = vruns.load_canonical(S["vlm_root"])
        for (model, ds), c in sorted(canon.items()):
            if ds not in LINE_SETS:
                continue
            fam, run = vlm_family(model), f"vlm/{model}"
            rows, unm = [], 0
            for sid, rd in c["readings"].items():
                k = st.key(ds, sid)
                if k is None:
                    unm += 1
                    continue
                rows.append(st.add(run, fam, model, "none", k, "" if rd.error else rd.pred, "bench", "vlm", "vlm:" + c["prompt_version"]))
            st.reconcile("vlm", f"{model}::{ds}", rows, unm)
        print("ingested VLM pairs", sum(1 for (m, d) in canon if d in LINE_SETS), flush=True)

    pq.write_table(pa.Table.from_pylist(list(samples.values())), a.out / "samples.parquet")
    pq.write_table(pa.Table.from_pylist(st.rows), a.out / "lines.parquet")
    runs = {k: {**v, "sources": sorted(v["sources"])} for k, v in st.runs.items()}
    (a.out / "runs.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
    (a.out / "reconcile.json").write_text(json.dumps(st.rec, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = {k: v for k, v in st.rec.items() if v.get("unmapped_ids") or v.get("reversed_suspect", 0) > 0.2 * max(1, v["lines"]) or v.get("native_equals_stored", 1) < 0.95}
    print(f"lines {len(st.rows)}  runs {len(runs)}  flagged sources {len(bad)}")
    for k, v in list(bad.items())[:40]:
        print("  FLAG", k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()})


if __name__ == "__main__":
    main()

"""Scan any line dataset in configs/datasets.yaml with the data-QA checks.

  PYTHONPATH=src python scripts/dataqa_scan.py --dataset rasam1 --split test \
      [--limit N] [--reads results/rasam/clean/read_f0.json ...] [--id-prefix RASAM-1]

Writes results/dataqa/<dataset>_<split>.json (flags per line, counts) and
results/dataqa/<dataset>_<split>.html (up to --per-check flagged lines per
check: image, GT, reading, reason), to check the flags by eye.

--reads takes one or more JSON files {id: {"hyp": ..., "ms"?: ..., "page"?: ...}}
or {id: "reading"} from any recogniser. With page fields, `shifted` runs too.
--id-prefix renames sample ids '<dataset>/<split>/<idx>' to '<prefix>/<split>/<idx>'
so they match the reads' ids.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from pathlib import Path

import yaml

from arabic_ocr_bench import dataqa
from arabic_ocr_bench.datasets.registry import build_dataset


class Renamed:
    def __init__(self, ds, prefix, keep):
        self.ds, self.prefix, self.keep = ds, prefix, keep
        self.images = {}

    def __iter__(self):
        for s in self.ds:
            if self.prefix:
                p = s.id.split("/")
                s.id = "/".join([self.prefix] + p[1:])
            self.images[s.id] = s.image if self.keep(s.id) else None
            yield s


class HFParquet:
    """Offline reader for a cached HF image-to-text dataset (no `datasets` needed):
    parquet shards of <split> in the hub cache, in shard order, so the row index
    matches the HF dataset's."""

    GT_KEYS = ("text", "answer", "label", "ground_truth", "transcription", "gt", "target")

    def __init__(self, repo, split, limit=None, hub="/Volumes/PortableSSD/huggingface/hub"):
        import glob
        import os
        snap = Path(hub) / f"datasets--{repo.replace('/', '--')}" / "snapshots"
        files = sorted(f for f in glob.glob(str(snap / "*" / "**" / f"{split}-*.parquet"), recursive=True)
                       if not os.path.basename(f).startswith("._"))
        # one snapshot only (the newest) to avoid double rows
        if files:
            newest = max({Path(f).relative_to(snap).parts[0] for f in files},
                         key=lambda s: os.path.getmtime(snap / s))
            files = [f for f in files if Path(f).relative_to(snap).parts[0] == newest]
        if not files:
            raise SystemExit(f"no cached parquet for {repo} split {split} under {snap}")
        self.files, self.split, self.limit = files, split, limit
        self.name = repo.split("/")[-1]

    def __iter__(self):
        import pyarrow.parquet as pq
        from types import SimpleNamespace

        from PIL import Image
        i = 0
        for f in self.files:
            t = pq.read_table(f)
            cols = t.column_names
            gk = next(k for k in self.GT_KEYS if k in cols)
            for row in t.to_pylist():
                if self.limit is not None and i >= self.limit:
                    return
                im = row["image"]
                img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
                yield SimpleNamespace(id=f"{self.name}/{self.split}/{i:06d}", image=img,
                                      ground_truth=row[gk] or "", meta={})
                i += 1


def load_reads(paths):
    out = {}
    for p in paths or []:
        d = json.load(open(p))
        for k, v in d.items():
            out[k] = v if isinstance(v, dict) else {"hyp": v}
    return out


def thumb(img, h=56, max_w=900):
    if img is None:
        return ""
    w0, h0 = img.size
    s = h / max(h0, 1)
    w = int(w0 * s)
    if w > max_w:
        s = max_w / w0
    im = img.convert("RGB").resize((max(int(w0 * s), 1), max(int(h0 * s), 1)))
    if h0 > 3 * w0:  # vertical strips: show them at a fixed height instead
        im = img.convert("RGB").resize((max(int(w0 * 160 / h0), 1), 160))
    b = io.BytesIO()
    im.save(b, "WEBP", quality=70)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def gallery(name, res, images, gts, reads, per_check, seed=0):
    rng = random.Random(seed)
    by = {}
    for k, fs in res["flags"].items():
        for f in fs:
            by.setdefault(f["check"], []).append((f["score"], k, f))
    parts = [f"<h1>Data QA · {html.escape(name)}</h1><p>{res['n']} lines scanned, {res['n_read']} with a reading; "
             f"{res['any']} flagged by at least one check.</p><table class=c><tr><th>check</th><th>tier</th>"
             f"<th>lines</th><th>%</th><th>rule</th></tr>"]
    for c in dataqa.CHECKS.values():
        n = res["counts"].get(c.name, 0)
        parts.append(f"<tr><td><a href='#{c.name}'>{c.name}</a></td><td>{c.tier}</td><td>{n}</td>"
                     f"<td>{100 * n / max(res['n'], 1):.1f}</td><td>{html.escape(c.rule)}</td></tr>")
    parts.append("</table>")
    for c in dataqa.CHECKS.values():
        rows = by.get(c.name, [])
        if not rows:
            continue
        rows.sort(key=lambda r: -r[0])
        pick = rows[:per_check // 2]
        rest = rows[per_check // 2:]
        pick += rng.sample(rest, min(len(rest), per_check - len(pick)))
        parts.append(f"<h2 id='{c.name}'>{c.name} · {len(rows)} lines</h2><p class=r>{html.escape(c.rule)}"
                     f" <b>Fix:</b> {html.escape(c.fix)}</p><p class=r>Top {per_check // 2} by score, then random.</p>")
        for sc, k, f in pick:
            other = ", ".join(x["check"] for x in res["flags"][k] if x["check"] != c.name)
            hyp = reads.get(k, {}).get("hyp")
            parts.append(
                f"<div class=s><div class=m><code>{html.escape(k)}</code> · {html.escape(f['detail'])}"
                f"{' · also: ' + html.escape(other) if other else ''}</div>"
                f"<img src='{thumb(images.get(k))}'><div class=t dir=rtl><span class=l>GT</span> "
                f"{html.escape(gts.get(k, ''))}</div>"
                + (f"<div class=t dir=rtl><span class=l>read</span> {html.escape(hyp)}</div>" if hyp is not None else "")
                + "</div>")
    css = ("body{font:14px system-ui;margin:16px;max-width:1100px;background:#fff;color:#111}"
           "table.c{border-collapse:collapse;font-size:12px}table.c td,table.c th{border:1px solid #ddd;"
           "padding:3px 6px;vertical-align:top}.s{border-top:1px solid #eee;padding:8px 0}"
           ".m{font-size:12px;color:#555}img{max-width:100%;border:1px solid #ccc;margin:4px 0}"
           ".t{font-size:18px}.l{font:11px system-ui;color:#888;direction:ltr;unicode-bidi:isolate;"
           "display:inline-block;width:36px}.r{font-size:12px;color:#444}")
    return f"<!doctype html><meta charset=utf-8><title>Data QA {html.escape(name)}</title><style>{css}</style>" + "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="name in configs/datasets.yaml (uses its adapter)")
    ap.add_argument("--hf", help="org/name of a cached HF line dataset, read offline from parquet")
    ap.add_argument("--split")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reads", nargs="*")
    ap.add_argument("--id-prefix")
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", default="results/dataqa")
    ap.add_argument("--per-check", type=int, default=12)
    a = ap.parse_args()
    if a.hf:
        ds = HFParquet(a.hf, a.split or "test", a.limit)
        a.dataset = ds.name
        cfg = {"split": ds.split}
    else:
        cfg = dict(yaml.safe_load(open(a.config))[a.dataset])
        if a.split:
            cfg["split"] = a.split
        ds = build_dataset(a.dataset, cfg, limit=a.limit)
    reads = load_reads(a.reads)
    page_of = None
    if any("page" in v for v in reads.values()):
        def page_of(sid, r):
            return f"{r.get('ms', '')}|{r.get('page', '')}" if r.get("page") else None
    gts = {}

    class G:  # capture GTs while scanning
        def __init__(self, it):
            self.it = it

        def __iter__(self):
            for s in self.it:
                gts[s.id] = s.ground_truth
                yield s

    wrapped = Renamed(ds, a.id_prefix, keep=lambda sid: True)
    res = dataqa.scan(G(wrapped), reads=reads or None, page_of=page_of)
    split = cfg.get("split", "all")
    name = f"{a.dataset}_{split}" + (f"_n{a.limit}" if a.limit else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res["dataset"], res["split"], res["definitions"] = a.dataset, split, {
        k: {"tier": c.tier, "side": c.side, "rule": c.rule, "fix": c.fix} for k, c in dataqa.CHECKS.items()}
    (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    imgs = {k: wrapped.images.get(k) for k in res["flags"]}
    (out / f"{name}.html").write_text(gallery(name, res, imgs, gts, reads, a.per_check))
    print(json.dumps({"dataset": name, "n": res["n"], "n_read": res["n_read"], "any": res["any"],
                      "counts": res["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

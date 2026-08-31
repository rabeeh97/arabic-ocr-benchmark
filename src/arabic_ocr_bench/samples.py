"""Pick a few illuminating (model, dataset) sample transcriptions per run and
export them — image + ground truth + prediction + CER — as a self-contained
`samples.json` that the leaderboard page embeds.

Selection per (model, dataset):
- **anchor**  — a fixed set (first N lines of the split) so the same lines can be
  compared across every model
- **best / worst** — the k lowest / highest `cer_light`
- **typical** — the line closest to that model's median `cer_light`
- **perfect** — one line at `cer_light == 0`, if any
- **longest** — the line with the longest ground truth (a stress case)
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import statistics
from collections.abc import Callable
from pathlib import Path

from PIL import Image

ANCHOR_N = 12


def anchor_ids(dataset: str, n: int = ANCHOR_N) -> list[str]:
    return [f"{dataset}/test/{i:06d}" for i in range(n)]


def _row_index(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("/", 1)[-1])
    except ValueError:
        return -1


def select_sample_ids(
    scored: list[dict],
    anchors: list[str],
    gt_lengths: dict[str, int],
    k: int = 3,
) -> dict[str, list[str]]:
    """scored: scores.jsonl rows for ONE (model, dataset).
    Returns {sample_id: [categories]} in a stable, readable order."""
    present = {r["sample_id"] for r in scored}
    by_cer = sorted(scored, key=lambda r: (float(r.get("cer_light", 1.0)), r["sample_id"]))
    cats: dict[str, list[str]] = {}

    def tag(sid: str, cat: str) -> None:
        cats.setdefault(sid, [])
        if cat not in cats[sid]:
            cats[sid].append(cat)

    for a in anchors:
        if a in present:
            tag(a, "anchor")
    for r in by_cer[:k]:
        tag(r["sample_id"], "best")
    for r in by_cer[-k:] if len(by_cer) > k else []:
        tag(r["sample_id"], "worst")

    if by_cer:
        med = statistics.median(float(r.get("cer_light", 1.0)) for r in by_cer)
        typical = min(by_cer, key=lambda r: abs(float(r.get("cer_light", 1.0)) - med))
        tag(typical["sample_id"], "typical")

    for r in by_cer:
        if float(r.get("cer_light", 1.0)) == 0.0:
            tag(r["sample_id"], "perfect")
            break

    if gt_lengths:
        longest = max(present, key=lambda s: gt_lengths.get(s, 0))
        tag(longest, "longest")

    # order: anchors first (by id), then the rest by cer
    ordered = [a for a in anchors if a in cats]
    ordered += [r["sample_id"] for r in by_cer if r["sample_id"] in cats
                and r["sample_id"] not in ordered]
    return {sid: cats[sid] for sid in ordered}


def image_data_uri(img: Image.Image, max_w: int = 900, quality: int = 75) -> str:
    im = img.convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, max(1, h)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def export_run_samples(
    run_dir: str | Path,
    get_image: Callable[[int], Image.Image],
    *,
    k: int = 3,
    max_w: int = 900,
) -> Path:
    run_dir = Path(run_dir)
    preds = {(r["model"], r["sample_id"]): r for r in _read_jsonl(run_dir / "predictions.jsonl")}
    scores = _read_jsonl(run_dir / "scores.jsonl")

    pairs: dict[tuple[str, str], list[dict]] = {}
    for r in scores:
        pairs.setdefault((r["model"], r["dataset"]), []).append(r)

    from tqdm.auto import tqdm

    out_samples: list[dict] = []
    img_cache: dict[int, str] = {}
    selection = {
        (model, dataset): select_sample_ids(
            rows, anchor_ids(dataset),
            {r["sample_id"]: len(preds.get((model, r["sample_id"]), {}).get("ground_truth", ""))
             for r in rows},
            k=k,
        )
        for (model, dataset), rows in pairs.items()
    }
    n_total = sum(len(p) for p in selection.values())
    bar = tqdm(total=n_total, desc="sample export", unit="sample")
    for (model, dataset), rows in pairs.items():
        picked = selection[(model, dataset)]
        score_by_id = {r["sample_id"]: r for r in rows}
        for sid, categories in picked.items():
            bar.update(1)
            idx = _row_index(sid)
            if idx not in img_cache:
                try:
                    img_cache[idx] = image_data_uri(get_image(idx), max_w=max_w)
                except Exception as e:  # noqa: BLE001
                    img_cache[idx] = ""
                    print(f"samples: image {idx} failed: {e}")
            p = preds.get((model, sid), {})
            s = score_by_id.get(sid, {})
            out_samples.append({
                "sample_id": sid,
                "row": idx,
                "model": model,
                "dataset": dataset,
                "categories": categories,
                "image": img_cache[idx],
                "ground_truth": p.get("ground_truth", ""),
                "prediction": "" if p.get("error") else p.get("cleaned_text", ""),
                "error": p.get("error"),
                "cer_light": round(float(s.get("cer_light", 0.0)), 4),
                "cer_raw": round(float(s.get("cer_raw", 0.0)), 4),
                "wer_light": round(float(s.get("wer_light", 0.0)), 4),
            })
    bar.close()

    out = run_dir / "samples.json"
    out.write_text(json.dumps(
        {
            "run_id": run_dir.name,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "samples": out_samples,
        },
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    print(f"[samples] {len(out_samples)} entries, {len(img_cache)} images -> {out.name} "
          f"({out.stat().st_size // 1024} KB)", flush=True)
    return out

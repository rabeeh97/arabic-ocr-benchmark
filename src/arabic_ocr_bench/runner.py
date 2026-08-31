"""Execute a benchmark run: (sample x model) -> predictions.jsonl.

Crash-safe (rows are flushed as they complete), resumable (a persistent cache
keyed by sample+model+prompt is consulted first), and bounded-concurrent.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from .datasets.registry import build_dataset, load_dataset_configs
from .models.mock_client import build_mock_prompt
from .models.registry import build_model, is_mock, load_model_configs
from .prompting import get_prompt

_TRANSIENT = ("429", "rate", "timeout", "500", "502", "503", "504", "overloaded")

# Bump when generation behaviour changes (decoding params, max_new_tokens,
# image caps, output cleaning, refusal handling) so old cached predictions miss.
CACHE_REV = "3"


def _cache_key(sample_id: str, model: str, prompt_version: str) -> str:
    return hashlib.sha1(
        f"{sample_id}|{model}|{prompt_version}|r{CACHE_REV}".encode()
    ).hexdigest()


def _cost_usd(cfg: dict, in_tok: int, out_tok: int) -> float:
    pricing = cfg.get("pricing") or {}
    return (
        in_tok / 1e6 * pricing.get("input_usd_per_1m", 0.0)
        + out_tok / 1e6 * pricing.get("output_usd_per_1m", 0.0)
    )


def _is_transient(err: str | None) -> bool:
    return bool(err) and any(t in err.lower() for t in _TRANSIENT)


def run(run_cfg: dict, *, results_root: str | Path | None = None) -> Path:
    models_cfg = load_model_configs()
    datasets_cfg = load_dataset_configs()

    prompt_version = run_cfg.get("prompt_version", "line-v1")
    base_prompt = get_prompt(prompt_version)
    limit = run_cfg.get("limit")
    concurrency = int(run_cfg.get("concurrency", 4))
    max_retries = int(run_cfg.get("max_retries", 3))
    use_cache = bool(run_cfg.get("cache", True))
    show_progress = bool(run_cfg.get("progress", True))

    root = Path(results_root or run_cfg.get("results_dir", "results"))
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.snapshot.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    pred_path = run_dir / "predictions.jsonl"

    clients = {name: build_model(name, models_cfg[name]) for name in run_cfg["models"]}

    def _one(model_name: str, ds_name: str, sample) -> dict:
        cfg = models_cfg[model_name]
        key = _cache_key(sample.id, model_name, prompt_version)
        cache_file = cache_dir / f"{key}.json"
        if use_cache and cache_file.exists():
            row = json.loads(cache_file.read_text())
            row["cached"] = True
            return row

        prompt = base_prompt
        if is_mock(cfg):
            prompt = build_mock_prompt(base_prompt, sample.ground_truth, cfg.get("mode", "empty"))

        client = clients[model_name]
        attempts = 0
        while True:
            attempts += 1
            t0 = time.perf_counter()
            pred = client.transcribe(sample.image, prompt)
            if pred.latency_ms == 0.0:
                pred.latency_ms = (time.perf_counter() - t0) * 1000
            if not (_is_transient(pred.error) and attempts <= max_retries):
                break
            time.sleep(min(2 ** attempts, 30))

        row = {
            "sample_id": sample.id,
            "model": model_name,
            "dataset": ds_name,
            "track": sample.track,
            "prompt_version": prompt_version,
            "ground_truth": sample.ground_truth,
            "raw_text": pred.raw_text,
            "cleaned_text": pred.cleaned_text,
            "latency_ms": round(pred.latency_ms, 2),
            "input_tokens": pred.input_tokens,
            "output_tokens": pred.output_tokens,
            "cost_usd": round(_cost_usd(cfg, pred.input_tokens, pred.output_tokens), 8),
            "error": pred.error,
            "attempts": attempts,
            "cached": False,
        }
        if use_cache and pred.error is None:
            cache_file.write_text(json.dumps(row, ensure_ascii=False))
        return row

    def _chunks(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    n_models = len(run_cfg["models"])

    # Model-major: finish every job for one model, then free it before loading
    # the next (critical for a local-model sweep on a single GPU). Samples are
    # STREAMED from a fresh adapter per model and processed in small chunks, so
    # we never hold the whole decoded dataset in RAM.
    with pred_path.open("w", encoding="utf-8") as fh:
        for mi, model_name in enumerate(run_cfg["models"], 1):
            for ds_name in run_cfg["datasets"]:
                adapter = build_dataset(ds_name, datasets_cfg[ds_name], limit=limit)
                try:
                    total = len(adapter)
                except (TypeError, AttributeError):
                    total = None
                bar = tqdm(
                    total=total, unit="line", disable=not show_progress,
                    desc=f"[{mi}/{n_models}] {model_name} · {ds_name}",
                )
                seen = {"cached": 0, "err": 0}

                def _emit(row, bar=bar, seen=seen):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    seen["cached"] += bool(row.get("cached"))
                    seen["err"] += bool(row.get("error"))
                    bar.update(1)
                    bar.set_postfix(cached=seen["cached"], err=seen["err"], refresh=False)

                if concurrency <= 1:
                    for sample in adapter:
                        _emit(_one(model_name, ds_name, sample))
                else:
                    for chunk in _chunks(adapter, max(4, concurrency * 4)):
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            rows = list(pool.map(
                                lambda s, m=model_name, d=ds_name: _one(m, d, s), chunk
                            ))
                        for row in rows:
                            _emit(row)
                bar.close()
            close = getattr(clients[model_name], "close", None)
            if callable(close):
                close()

    return run_dir

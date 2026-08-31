"""Run a benchmark sweep unattended: every model, a partial pass then a full
pass, checkpointing to a manifest after each so an interrupted run resumes
instead of repeating work.

`run_sweep` does no git / Colab work itself:
- `on_progress(str)` — logging, best-effort.
- `on_done(entry, run_dir)` — persist this phase's results (push to the repo /
  Drive). **It MUST raise if persistence fails.** A phase is only recorded in
  the manifest as "done" after `on_done` returns without raising; on failure the
  manifest entry is rolled back and the phase re-runs on the next call.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path

from .normalize import PROFILES
from .report import score_run
from .runner import run

Entry = dict


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            data.setdefault("entries", [])
            return data
        except json.JSONDecodeError:
            pass
    return {"entries": []}


def _save_manifest(path: str | Path, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(p)


def _done_keys(manifest: dict) -> set[tuple[str, str, str]]:
    return {(e["model"], e["dataset"], e["phase"]) for e in manifest["entries"]}


def _reconcile(manifest: dict, results_dir: Path, log) -> dict:
    """Drop entries whose run directory isn't here — a 'done' marker is only
    trustworthy if its results are actually present (self-heals a manifest that
    was pushed while its run dirs were not)."""
    kept, dropped = [], []
    for e in manifest["entries"]:
        if (results_dir / e.get("run_id", "")).is_dir():
            kept.append(e)
        else:
            dropped.append(e)
    if dropped:
        for e in dropped:
            log(f"reconcile: dropping {e['model']}/{e['phase']} "
                f"(run dir {e.get('run_id')} missing) — will re-run")
        manifest["entries"] = kept
    return manifest


def _summary_row(run_dir: Path, model: str, dataset: str) -> dict:
    with (run_dir / "summary.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model and r["dataset"] == dataset:
                return r
    return {}


def run_sweep(
    models: list[str],
    dataset: str,
    partial_limit: int | None,
    run_full: bool,
    *,
    manifest_path: str | Path,
    results_dir: str | Path,
    prompt_version: str = "line-v2",
    profiles: tuple[str, ...] = PROFILES,
    concurrency: int = 1,
    cache: bool = True,
    on_progress: Callable[[str], None] | None = None,
    on_done: Callable[[Entry, Path], None] | None = None,
) -> list[Entry]:
    log = on_progress or (lambda _msg: None)
    manifest = _reconcile(load_manifest(manifest_path), Path(results_dir), log)
    _save_manifest(manifest_path, manifest)
    done = _done_keys(manifest)

    phases: list[tuple[str, int | None]] = [("partial", partial_limit)]
    if run_full:
        phases.append(("full", None))

    total = len(models) * len(phases)
    new_entries: list[Entry] = []
    step = 0
    for model in models:
        for phase, limit in phases:
            step += 1
            key = (model, dataset, phase)
            if key in done:
                log(f"[{step}/{total}] skip {model} / {phase} (already done)")
                continue
            log(f"[{step}/{total}] start {model} / {phase} (limit={limit})")
            try:
                run_dir = run(
                    {
                        "models": [model],
                        "datasets": [dataset],
                        "limit": limit,
                        "prompt_version": prompt_version,
                        "normalization_profiles": list(profiles),
                        "concurrency": concurrency,
                        "cache": cache,
                        "results_dir": str(results_dir),
                    },
                    results_root=results_dir,
                )
                score_run(run_dir, profiles=profiles)
                row = _summary_row(run_dir, model, dataset)
                entry: Entry = {
                    "model": model,
                    "dataset": dataset,
                    "phase": phase,
                    "limit": limit,
                    "run_id": run_dir.name,
                    "n": int(row.get("n", 0) or 0),
                    "errors": int(row.get("errors", 0) or 0),
                    "cer_micro_raw": float(row.get("cer_micro_raw", 0) or 0),
                    "cer_micro_light": float(row.get("cer_micro_light", 0) or 0),
                    "wer_micro_light": float(row.get("wer_micro_light", 0) or 0),
                    "done_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                }
            except Exception as e:  # noqa: BLE001 - keep the sweep going
                log(f"FAILED {model} / {phase}: {type(e).__name__}: {e}")
                continue

            # Publish first. Only record the phase as done if the results were
            # actually persisted (on_done pushes to the repo / Drive and RAISES
            # on failure). On failure we roll the manifest back so it re-runs.
            manifest["entries"].append(entry)
            _save_manifest(manifest_path, manifest)
            if on_done is not None:
                try:
                    on_done(entry, run_dir)
                except Exception as e:  # noqa: BLE001
                    manifest["entries"].pop()
                    _save_manifest(manifest_path, manifest)
                    log(
                        f"NOT PERSISTED {model} / {phase}: {type(e).__name__}: {e} "
                        f"— will retry next run"
                    )
                    continue
            done.add(key)
            new_entries.append(entry)
            log(
                f"done {model} / {phase}: CER-light "
                f"{entry['cer_micro_light'] * 100:.2f}% (n={entry['n']}, "
                f"errors={entry['errors']})"
            )
    return new_entries

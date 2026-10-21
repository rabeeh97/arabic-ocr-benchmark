"""Generate scripts/README.md: scripts grouped by component, one docstring line each."""
import ast
from pathlib import Path

ROOT = Path("scripts")
GROUPS = [
    ("Run & orchestrate (benchmarks, 3090 box, suites, queues)",
     ("run_", "queue_", "finetune_", "handoff_", "stage_", "fix_runbook", "eval_ain", "list_paid", "cost_report", "token_cap", "_detach", "_envfile", "_experiment_bases", "backfill_", "slim_ledger", "sweep_audit")),
    ("Experiments (single-question studies)", ("experiment_", "seed_", "vlm_", "pair_diff", "calib_", "rescore_")),
    ("Analysis (letters, dots, PAWs, text, errors)", ("analyze_", "decompose_", "eval_stats", "sample_char", "profile_", "dots_", "dec_", "shape_", "writer_", "line_end", "_feasibility")),
    ("Geometry (ink extent, transfer, augmentation)", ("geom_", "check_geom", "image_layer", "khatt_full", "inkcrop")),
    ("Explainability (xai_*)", ("xai_", "build_xai")),
    ("Audits & datasets", ("audit_", "fetch_", "prep_", "export_", "extract_", "store_", "regen_", "detectors_", "run_detectors", "crnn_")),
    ("Reports, galleries & site builders", ("build_", "make_leaderboard", "_leaderboard", "change_gallery", "research_map", "check_master", "pickle_")),
]


def doc_line(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, UnicodeDecodeError):
        doc = None
    return doc.strip().splitlines()[0].strip() if doc else ""


def group_of(name: str) -> int:
    for i, (_, prefixes) in enumerate(GROUPS):
        if name.startswith(prefixes):
            return i
    return len(GROUPS)


files = sorted(p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith("._") and p.suffix in {".py", ".bat", ".sh"})
buckets: dict[int, list[Path]] = {}
for p in files:
    buckets.setdefault(group_of(p.name), []).append(p)

out = [
    "# scripts/ index",
    "",
    "Command-line tools, grouped by component. Reusable logic lives in",
    "`src/arabic_ocr_bench/`; scripts are thin drivers around it. Run from the repo root:",
    "`PYTHONPATH=src .venv/bin/python scripts/<name>.py`.",
    "",
    "Generated from each script's module docstring — regenerate after adding scripts",
    "(`python scripts/_gen_index.py`). Sub-folders: `box/` (3090 machine queues/backups), `lexicon/` (word-list build).",
    "",
]
names = [g[0] for g in GROUPS] + ["Other"]
for i in sorted(buckets):
    out += [f"## {names[i]}", "", "| script | what it does |", "|---|---|"]
    for p in buckets[i]:
        out.append(f"| `{p.name}` | {doc_line(p).replace('|', '/')} |")
    out.append("")
(ROOT / "README.md").write_text("\n".join(out), encoding="utf-8")
print({names[i]: len(v) for i, v in buckets.items()})

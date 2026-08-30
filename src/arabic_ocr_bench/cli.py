"""`bench run` / `bench report` — thin wrappers over runner.run and report.score_run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .report import score_run
from .runner import run as run_benchmark


def _load_run_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute a benchmark run")
    p_run.add_argument("--config", default="configs/run.yaml")
    p_run.add_argument("--report", action="store_true", help="score immediately after")

    p_rep = sub.add_parser("report", help="score an existing run directory")
    p_rep.add_argument("run_dir")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = _load_run_cfg(args.config)
        run_dir = run_benchmark(cfg)
        print(f"run -> {run_dir}")
        if args.report:
            profiles = tuple(cfg.get("normalization_profiles") or ())
            md = score_run(run_dir, profiles=profiles) if profiles else score_run(run_dir)
            print(f"summary -> {md}")
            print(Path(md).read_text())
        return 0

    if args.cmd == "report":
        md = score_run(args.run_dir)
        print(f"summary -> {md}")
        print(Path(md).read_text())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

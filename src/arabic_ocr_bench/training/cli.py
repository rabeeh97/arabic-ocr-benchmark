"""`bench-train` — assemble data, check contamination, run SFT, merge, register.

    bench-train assemble    --config configs/training/smoke.yaml [--limit N]
    bench-train check-contam --config configs/training/smoke.yaml [--limit N]
    bench-train sft         --config configs/training/smoke.yaml
    bench-train merge       --adapter results/training/smoke --out results/training/smoke-merged
    bench-train register    --name smoke-ft --path results/training/smoke-merged --arch qwen2_vl

`assemble` and `check-contam` run in the core environment; `sft` and `merge`
need the `train` + `local` extras (torch/transformers/peft/trl).
"""

from __future__ import annotations

import argparse
import sys

from .config import load_training_config


def _cmd_assemble(args) -> int:
    from .data import assemble_dataset

    cfg = load_training_config(args.config)
    if args.limit is not None:
        for s in cfg.data.sources:
            s.limit = args.limit
    bundle = assemble_dataset(cfg.data, seed=cfg.train.seed)
    by_src: dict[str, int] = {}
    for row in bundle["train"] + bundle["val"]:
        by_src[row["source"]] = by_src.get(row["source"], 0) + 1
    print(f"experiment: {cfg.experiment}  method: {cfg.method}")
    for name, n in by_src.items():
        print(f"  {name}: {n} lines")
    print(f"train: {len(bundle['train'])}  val: {len(bundle['val'])}")
    return 0


def _cmd_check_contam(args) -> int:
    from .contamination import check_datasets

    cfg = load_training_config(args.config)
    train_specs = [
        {**s.model_dump(exclude_none=True), "name": s.name} for s in cfg.data.sources
    ]
    report = check_datasets(
        train_specs,
        cfg.eval_datasets,
        out_path=args.out,
        limit=args.limit,
    )
    for p in report["pairs"]:
        flag = "FLAG" if (p["n_text_collisions"] or p["n_near_dup_images"]) else "ok"
        print(
            f"[{flag}] {p['train_source']} vs {p['test_dataset']}: "
            f"{p['n_text_collisions']} text, {p['n_near_dup_images']} image"
        )
    print(f"report -> {args.out}  ({report['n_flagged']} flagged pair(s))")
    return 1 if report["n_flagged"] else 0


def _cmd_sft(args) -> int:
    cfg = load_training_config(args.config)
    if cfg.method == "trocr":
        from .sft_trocr import train
    else:
        from .sft_vlm import train
    out = train(cfg)
    print(f"trained -> {out}")
    return 0


def _cmd_merge(args) -> int:
    from .merge import merge_adapter

    out = merge_adapter(args.adapter, args.out, base_hf_id=args.base, arch=args.arch)
    print(f"merged -> {out}")
    return 0


def _cmd_register(args) -> int:
    from .registry import build_model_entry, register_model

    entry = build_model_entry(
        args.name,
        args.path,
        arch=args.arch,
        adapter_id=args.adapter_id,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        card=args.card,
    )
    register_model(args.name, entry, path=args.models)
    print(f"registered {args.name} -> {args.models}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench-train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("assemble", help="dry-run: report assembled data counts")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=_cmd_assemble)

    p = sub.add_parser("check-contam", help="scan training sources vs eval test sets")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/contamination_report.json")
    p.set_defaults(fn=_cmd_check_contam)

    p = sub.add_parser("sft", help="run supervised fine-tuning")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=_cmd_sft)

    p = sub.add_parser("merge", help="merge a LoRA adapter into base weights")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--arch", default=None)
    p.set_defaults(fn=_cmd_merge)

    p = sub.add_parser("register", help="add a trained model to models.yaml")
    p.add_argument("--name", required=True)
    p.add_argument("--path", required=True, help="merged dir, or base hf_id with --adapter-id")
    p.add_argument("--arch", required=True)
    p.add_argument("--adapter-id", dest="adapter_id", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                   help="score this model 4-bit-quantised (matches the 4-bit base rows)")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None,
                   help="generation cap for this row (default 160; raise for page-scale GT)")
    p.add_argument("--card", default=None)
    p.add_argument("--models", default="configs/models.yaml")
    p.set_defaults(fn=_cmd_register)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

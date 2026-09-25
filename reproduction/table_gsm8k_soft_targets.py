"""Run the paired Qwen GSM8K soft-target intervention."""

import json

from reproduction.suite_common import (
    Cell, QWEN_MODELS, common_parser, python_command, run_cells, seeds, summarize_cells,
)


def _audit(cells: list[Cell]) -> None:
    paired = {}
    for cell in cells:
        payload = json.loads(cell.output.read_text())
        paired.setdefault(cell.dimensions["seed"], {})[cell.dimensions["method"]] = payload
    for seed, methods in paired.items():
        if set(methods) != {"hard_replay", "cagd"}:
            raise ValueError(f"seed {seed}: incomplete method pair")
        hard, soft = methods["hard_replay"], methods["cagd"]
        for path in (
            ("metadata", "canonical_stage0_json_sha256"),
            ("metadata", "canonical_stage0_checkpoint"),
            ("metadata", "replay_sha256"),
            ("metadata", "anchor_manifest"),
            ("metadata", "settings"),
            ("data_sha256",),
        ):
            left, right = hard, soft
            for key in path:
                left, right = left[key], right[key]
            if left != right:
                raise ValueError(f"seed {seed}: unpaired {'/'.join(path)}")


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    if args.trainable == "last_block":
        parser.error("this frozen intervention requires --trainable all (or reported)")
    model = QWEN_MODELS["qwen3_0.6b"]
    stage0_cells = []
    for seed in seeds(args):
        output = args.run_root / "stage0" / f"s{seed}" / "stage0.json"
        checkpoint = output.parent / "model"
        stage0_cells.append(Cell(
            f"stage0-s{seed}",
            python_command(
                "ar_gsm8k_soft_targets", "--mode", "stage0", "--model", model["path"],
                "--trainable", "all", "--seed", seed, "--output", output,
                "--stage0-checkpoint", checkpoint, "--formal",
            ), output, {"stage": "gsm8k", "seed": seed},
        ))
    run_cells(stage0_cells, args)

    cells = []
    for seed in seeds(args):
        stage0_json = args.run_root / "stage0" / f"s{seed}" / "stage0.json"
        checkpoint = stage0_json.parent / "model"
        for method in ("hard_replay", "cagd"):
            output = args.run_root / "cells" / method / f"s{seed}.json"
            cells.append(Cell(
                f"{method}-s{seed}",
                python_command(
                    "ar_gsm8k_soft_targets", "--mode", "branch", "--method", method,
                    "--model", model["path"], "--trainable", "all", "--seed", seed,
                    "--output", output, "--stage0-json", stage0_json,
                    "--stage0-checkpoint", checkpoint, "--formal",
                ), output, {"model": model["display"], "method": method, "seed": seed},
            ))
    run_cells(cells, args)
    if not args.dry_run:
        _audit(cells)
        summarize_cells("table_gsm8k_soft_targets", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

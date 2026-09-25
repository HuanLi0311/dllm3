"""Run one matched scale extension of the GSM8K soft-target intervention."""

import json

from reproduction.suite_common import (
    Cell, QWEN_MODELS, SMDM_MODELS, common_parser, python_command, run_cells,
    seeds, summarize_cells,
)


MODELS = {**QWEN_MODELS, **SMDM_MODELS}
SUPPORTED = ("qwen3_1.7b", "smdm_219m", "smdm_1.14b")


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
            ("initial_gsm8k_benchmark",),
        ):
            left, right = hard, soft
            for key in path:
                left, right = left[key], right[key]
            if left != right:
                raise ValueError(f"seed {seed}: unpaired {'/'.join(path)}")


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model-id", choices=SUPPORTED, required=True)
    args = parser.parse_args()
    if args.trainable == "last_block":
        parser.error("this frozen intervention requires --trainable all (or reported)")
    model = MODELS[args.model_id]
    is_smdm = args.model_id.startswith("smdm")
    module = "smdm_gsm8k_soft_targets" if is_smdm else "ar_gsm8k_soft_targets_17b"

    def command(mode, seed, output, checkpoint, method=None, stage0=None):
        common = ["--mode", mode, "--trainable", "all", "--seed", seed,
                  "--output", output, "--stage0-checkpoint", checkpoint, "--formal"]
        common += (["--model-id", args.model_id, "--model-path", model["path"]]
                   if is_smdm else ["--model", model["path"]])
        if method is not None:
            common += ["--method", method, "--stage0-json", stage0]
        return python_command(module, *common)

    stage0_cells = []
    for seed in seeds(args):
        output = args.run_root / "stage0" / f"s{seed}" / "stage0.json"
        checkpoint = output.parent / ("model.safetensors" if is_smdm else "model")
        stage0_cells.append(Cell(
            f"stage0-s{seed}", command("stage0", seed, output, checkpoint), output,
            {"stage": "gsm8k", "seed": seed},
        ))
    run_cells(stage0_cells, args)

    cells = []
    for seed in seeds(args):
        stage0_json = args.run_root / "stage0" / f"s{seed}" / "stage0.json"
        checkpoint = stage0_json.parent / ("model.safetensors" if is_smdm else "model")
        for method in ("hard_replay", "cagd"):
            output = args.run_root / "cells" / method / f"s{seed}.json"
            cells.append(Cell(
                f"{method}-s{seed}",
                command("branch", seed, output, checkpoint, method, stage0_json),
                output,
                {"model": model["display"], "method": method, "seed": seed},
            ))
    run_cells(cells, args)
    if not args.dry_run:
        _audit(cells)
        summarize_cells(
            "table_gsm8k_soft_targets_scale", cells, args.run_root / "summary.json", args.resume
        )


if __name__ == "__main__":
    main()

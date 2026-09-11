"""Run all five checkpoint pairs and metrics required by the GSM8K table."""

from reproduction.suite_common import (
    Cell, QWEN_MODELS, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds,
    split_words, summarize_cells,
)


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--smdm-models", default="smdm_219m smdm_1.14b")
    parser.add_argument("--qwen-models", default="qwen3_0.6b qwen3_1.7b qwen3_4b")
    args = parser.parse_args()
    cells = []

    # SMDM branches share the exact post-GSM8K checkpoint within each model/seed.
    stage0_cells = []
    for model_id in split_words(args.smdm_models):
        model = SMDM_MODELS[model_id]
        for seed in seeds(args):
            stage0_json = args.run_root / "stage0" / model_id / f"s{seed}" / "stage0.json"
            stage0_checkpoint = stage0_json.with_suffix(".safetensors")
            stage0_cells.append(Cell(
                f"{model_id}-stage0-s{seed}",
                python_command(
                    "smdm_gsm8k", "--mode", "stage0", "--model-id", model_id, "--model-path", model["path"],
                    "--trainable", scope(args, "smdm"), "--seed", seed, "--output", stage0_json,
                    "--stage0-checkpoint", stage0_checkpoint,
                ), stage0_json, {"model": model["display"], "stage": "gsm8k", "seed": seed},
            ))
    run_cells(stage0_cells, args)

    for model_id in split_words(args.smdm_models):
        model = SMDM_MODELS[model_id]
        for seed in seeds(args):
            stage0_json = args.run_root / "stage0" / model_id / f"s{seed}" / "stage0.json"
            stage0_checkpoint = stage0_json.with_suffix(".safetensors")
            for method in ("seq", "cagd"):
                output = args.run_root / "cells" / model_id / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{model_id}-{method}-s{seed}",
                    python_command(
                        "smdm_gsm8k", "--mode", "branch", "--model-id", model_id, "--model-path", model["path"],
                        "--trainable", scope(args, "smdm"), "--method", method, "--seed", seed, "--output", output,
                        "--stage0-json", stage0_json, "--stage0-checkpoint", stage0_checkpoint,
                    ), output, {"model": model["display"], "backend": "smdm", "method": method, "seed": seed},
                ))
    for model_id in split_words(args.qwen_models):
        model = QWEN_MODELS[model_id]
        for seed in seeds(args):
            for method in ("seq", "cagd"):
                output = args.run_root / "cells" / model_id / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{model_id}-{method}-s{seed}",
                    python_command(
                        "gsm8k_scale", "--model-id", model_id, "--model-path", model["path"], "--method", method,
                        "--trainable", scope(args, "ar"), "--seed", seed, "--output", output,
                    ), output, {"model": model["display"], "backend": "ar", "method": method, "seed": seed},
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("table_gsm8k_behavior", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

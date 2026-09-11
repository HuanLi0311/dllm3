"""Run every model, method, seed, and metric required by the natural-task table."""

from reproduction.suite_common import (
    Cell, QWEN_MODELS, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, split_words,
    summarize_cells,
)


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--smdm-models", default="smdm_219m")
    parser.add_argument("--qwen-models", default="qwen3_0.6b")
    args = parser.parse_args()
    cells = []
    for model_id in split_words(args.smdm_models):
        model = SMDM_MODELS[model_id]
        for seed in seeds(args):
            for method in ("seq", "hard_replay", "cagd"):
                output = args.run_root / "cells" / model_id / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{model_id}-{method}-s{seed}",
                    python_command(
                        "smdm_natural", "--method", method, "--checkpoint", model["path"], "--model", model["config"],
                        "--trainable", scope(args, "smdm"), "--seed", seed, "--output", output,
                    ), output, {"model": model["display"], "backend": "smdm", "method": method, "seed": seed},
                ))
    for model_id in split_words(args.qwen_models):
        model = QWEN_MODELS[model_id]
        for seed in seeds(args):
            for method in ("seq", "hard_replay", "cagd"):
                output = args.run_root / "cells" / model_id / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{model_id}-{method}-s{seed}",
                    python_command(
                        "ar_natural", "--method", method, "--model", model["path"],
                        "--trainable", scope(args, "ar"), "--seed", seed, "--output", output,
                    ), output, {"model": model["display"], "backend": "ar", "method": method, "seed": seed},
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("table_cagd_natural", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

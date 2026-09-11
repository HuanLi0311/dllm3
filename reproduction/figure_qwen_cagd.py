"""Run all AR checkpoint cells and metrics required by the Qwen scale figure."""

from reproduction.suite_common import (
    Cell, QWEN_MODELS, common_parser, python_command, run_cells, scope, seeds, split_words,
    summarize_cells, write_model_inventory,
)


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--models", default="qwen3_0.6b qwen3_1.7b qwen3_4b")
    args = parser.parse_args()
    cells = []
    for model_id in split_words(args.models):
        model = QWEN_MODELS[model_id]
        inventory = args.run_root / "inventory" / f"{model_id}.json"
        if not args.dry_run:
            write_model_inventory(model["path"], inventory)
        for seed in seeds(args):
            for method in ("seq", "gd"):
                output = args.run_root / "cells" / model_id / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{model_id}-{method}-s{seed}",
                    python_command(
                        "ar_factual", "--run-kind", "paper", "--model", model["path"], "--model-label", model_id,
                        "--model-inventory", inventory, "--method", method, "--trainable", scope(args, "ar"),
                        "--seed", seed, "--output", output,
                    ), output, {"model": model["display"], "backend": "ar", "method": method, "seed": seed},
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("figure_qwen_cagd", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

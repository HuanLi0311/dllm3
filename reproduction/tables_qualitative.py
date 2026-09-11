"""Run both methods once and emit every output required by the qualitative tables."""

from reproduction.suite_common import Cell, QWEN_MODELS, common_parser, python_command, run_cells, scope, seeds, summarize_cells


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model", choices=QWEN_MODELS, default="qwen3_0.6b")
    args = parser.parse_args()
    selected_seeds = seeds(args)
    if len(selected_seeds) != 1:
        raise ValueError("qualitative tables require exactly one seed")
    seed = selected_seeds[0]
    model = QWEN_MODELS[args.model]
    cells = []
    for method in ("seq", "cagd"):
        output = args.run_root / "cells" / method / f"s{seed}.json"
        cells.append(Cell(
            f"{args.model}-{method}-s{seed}",
            python_command(
                "ar_qualitative", "--method", method, "--model", model["path"],
                "--model-display-name", model["display"], "--trainable", scope(args, "ar"),
                "--seed", seed, "--output", output,
            ), output, {"model": model["display"], "backend": "ar", "method": method, "seed": seed},
        ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("tables_qualitative", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

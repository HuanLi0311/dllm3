"""Run every anchor budget and order required by the sensitivity figure."""

from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, summarize_cells


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model", choices=SMDM_MODELS, default="smdm_219m")
    parser.add_argument("--budgets", default="4 8 16 32 64 120")
    args = parser.parse_args()
    model, trainable = SMDM_MODELS[args.model], scope(args, "smdm")
    budget_values = [int(item) for item in args.budgets.replace(",", " ").split()]
    if any(value < 1 for value in budget_values):
        raise ValueError("anchor budgets must be positive")
    cells = []
    for seed in seeds(args):
        for order in ("forward", "reverse"):
            for budget in budget_values:
                output = args.run_root / "cells" / order / f"m{budget}" / f"s{seed}.json"
                cells.append(Cell(
                    f"{order}-m{budget}-s{seed}",
                    python_command(
                        "smdm_factual", "--method", "cagd", "--checkpoint", model["path"], "--model", model["config"],
                        "--trainable", trainable, "--order", order, "--group-start", 8, "--tasks", 4,
                        "--group-count", 4, "--eval-mc-samples", 32, "--replay-per-task", budget,
                        "--seed", seed, "--generation-seed", seed, "--output", output,
                    ), output, {"model": model["display"], "order": order, "method": "cagd", "budget": budget, "seed": seed},
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("figure_anchor_budget", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

"""Run the complete trajectory matrix required by the loss-dynamics figure."""

from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, summarize_cells


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model", choices=SMDM_MODELS, default="smdm_219m")
    parser.add_argument("--interval", type=int, default=100)
    args = parser.parse_args()
    model, trainable = SMDM_MODELS[args.model], scope(args, "smdm")
    cells = []
    for seed in seeds(args):
        for order in ("forward", "reverse"):
            for method in ("seq", "cagd"):
                output = args.run_root / "cells" / order / method / f"s{seed}.json"
                cells.append(Cell(
                    f"{order}-{method}-s{seed}",
                    python_command(
                        "smdm_factual", "--method", method, "--checkpoint", model["path"], "--model", model["config"],
                        "--trainable", trainable, "--order", order, "--group-start", 8, "--tasks", 4,
                        "--group-count", 4, "--eval-mc-samples", 32, "--record-step-loss",
                        "--record-eval-loss-every", args.interval, "--seed", seed,
                        "--generation-seed", seed, "--output", output,
                    ), output, {"model": model["display"], "order": order, "method": method, "seed": seed},
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells(
            "figure_loss_dynamics", cells, args.run_root / "summary.json", args.resume, include=("stages",)
        )


if __name__ == "__main__":
    main()

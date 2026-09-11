"""Run every cell required by the main controlled-stream table."""

from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, summarize_cells


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model", choices=SMDM_MODELS, default="smdm_219m")
    args = parser.parse_args()
    model = SMDM_MODELS[args.model]
    trainable = scope(args, "smdm")
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
                        "--group-count", 4, "--eval-mc-samples", 32, "--seed", seed,
                        "--generation-seed", seed, "--output", output,
                    ), output, {"model": model["display"], "order": order, "method": method, "seed": seed},
                ))
        output = args.run_root / "cells" / "order_free" / "joint" / f"s{seed}.json"
        cells.append(Cell(
            f"joint-s{seed}",
            python_command(
                "smdm_factual", "--method", "joint", "--checkpoint", model["path"], "--model", model["config"],
                "--trainable", trainable, "--order", "forward", "--group-start", 8, "--tasks", 4,
                "--group-count", 4, "--eval-mc-samples", 32, "--seed", seed,
                "--generation-seed", seed, "--output", output,
            ), output, {"model": model["display"], "order": "order_free", "method": "joint", "seed": seed},
        ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("table_cagd_main", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

"""Run every cell required by the matched two-task component table."""

from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, summarize_cells


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--model", choices=SMDM_MODELS, default="smdm_219m")
    args = parser.parse_args()
    model, trainable = SMDM_MODELS[args.model], scope(args, "smdm")
    cells = []
    for seed in seeds(args):
        for method in ("hard_replay", "real_replay", "cagd"):
            output = args.run_root / "cells" / method / f"s{seed}.json"
            cells.append(Cell(
                f"{method}-s{seed}",
                python_command(
                    "smdm_factual", "--method", method, "--checkpoint", model["path"], "--model", model["config"],
                    "--trainable", trainable, "--order", "forward", "--group-start", 4, "--tasks", 2,
                    "--group-count", 2, "--eval-mc-samples", 32, "--seed", seed,
                    "--generation-seed", seed, "--output", output,
                ), output, {"model": model["display"], "order": "forward", "method": method, "seed": seed},
            ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("table_cagd_components", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

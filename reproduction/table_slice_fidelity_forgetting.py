"""Run the frozen slice-fidelity/forgetting grid and summarize it."""

from reproduction.slice_fidelity_forgetting import SEEDS, SLICES, summarize
from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, seeds


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    model = SMDM_MODELS["smdm_219m"]
    requested_seeds = seeds(args)
    if any(seed not in SEEDS for seed in requested_seeds):
        raise ValueError(f"seeds must be drawn from {SEEDS}")
    cells = []
    for seed in requested_seeds:
        for slice_name in SLICES:
            output = args.run_root / "cells" / slice_name / f"s{seed}.json"
            cells.append(Cell(
                f"{slice_name}-s{seed}",
                python_command(
                    "slice_fidelity_forgetting", "--slice", slice_name, "--seed", seed,
                    "--checkpoint", model["path"], "--output", output,
                ),
                output,
                {"slice": slice_name, "seed": seed},
            ))
    run_cells(cells, args)
    if not args.dry_run and set(requested_seeds) == set(SEEDS):
        summarize(args.run_root, args.run_root / "summary.json")


if __name__ == "__main__":
    main()

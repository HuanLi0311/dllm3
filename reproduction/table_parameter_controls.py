"""Run every SMDM cell and metric required by the parameter-control table."""

from reproduction.suite_common import Cell, SMDM_MODELS, common_parser, python_command, run_cells, scope, seeds, split_words, summarize_cells


FAMILIES = {"smdm_219m": ("r23", (1.0, 1_000_000.0)), "smdm_1.14b": ("r24", (1.0,))}


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument("--models", default="smdm_219m smdm_1.14b")
    args = parser.parse_args()
    cells = []
    for model_id in split_words(args.models):
        model = SMDM_MODELS[model_id]
        family, clips = FAMILIES[model_id]
        for seed in seeds(args):
            for clip in clips:
                for method in ("gd", "rank1_gd", "diag_gd"):
                    clip_label = "1" if clip == 1 else "1000000"
                    output = args.run_root / "cells" / model_id / f"clip{clip_label}" / method / f"s{seed}.json"
                    cells.append(Cell(
                        f"{model_id}-clip{clip_label}-{method}-s{seed}",
                        python_command(
                            "smdm_parameter_controls", "--family", family, "--checkpoint", model["path"],
                            "--trainable", scope(args, "smdm"), "--method", method, "--b-clip", clip,
                            "--seed", seed, "--output", output,
                        ), output, {
                            "model": model["display"], "backend": "smdm", "clip": clip,
                            "method": method, "seed": seed,
                        },
                    ))
    if "smdm_219m" in split_words(args.models):
        model = SMDM_MODELS["smdm_219m"]
        for seed in seeds(args):
            for method, coefficient in (("rank1", 100_000), ("diagonal", 1_000)):
                output = args.run_root / "cells" / "smdm_219m" / "four_task_ewc" / method / f"s{seed}.json"
                cells.append(Cell(
                    f"smdm_219m-four-task-{method}-s{seed}",
                    python_command(
                        "smdm_factual", "--method", method, "--checkpoint", model["path"],
                        "--model", model["config"], "--trainable", scope(args, "smdm"),
                        "--order", "forward", "--group-start", 8, "--tasks", 4, "--group-count", 4,
                        "--eval-mc-samples", 32, "--ewc-lambda", coefficient,
                        "--seed", seed, "--generation-seed", seed, "--output", output,
                    ), output, {
                        "study": "four_task_ewc", "model": model["display"], "backend": "smdm",
                        "order": "forward", "method": method, "seed": seed,
                    },
                ))
    run_cells(cells, args)
    if not args.dry_run:
        summarize_cells("table_parameter_controls", cells, args.run_root / "summary.json", args.resume)


if __name__ == "__main__":
    main()

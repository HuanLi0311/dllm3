#!/usr/bin/env python3
"""Plot Table 7 held-out loss throughout the complete training matrix."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDERS = ("forward", "reverse")
METHODS = {"seq": ("Sequential", "#607384"), "cagd": ("CAGD", "#ED028C")}
SEEDS = (3407, 3408, 3409)


def _load(root: Path) -> dict:
    runs = {}
    for order in ORDERS:
        for filename, (label, _) in METHODS.items():
            method_runs = []
            for seed in SEEDS:
                path = root / order / f"s{seed}" / f"{filename}.json"
                result = json.loads(path.read_text())
                metadata = result["metadata"]
                assert result["status"] == "ok"
                assert metadata["protocol"] == "r16_native_mask_v1"
                assert metadata["order"] == order and metadata["seed"] == seed
                assert metadata["method"] == ("seq" if filename == "seq" else "gd")
                assert metadata["records_step_loss"] is True
                assert metadata["eval_loss_recording_interval"] == 100
                assert len(result["stages"]) == 4
                for stage_index, stage in enumerate(result["stages"]):
                    history = stage["training"]["step_loss"]
                    assert [row["step"] for row in history] == list(range(1, 1001))
                    for row in history:
                        assert all(math.isfinite(row[key]) for key in ("current", "distill", "total"))
                        assert math.isclose(
                            row["total"], row["current"] + row["distill"],
                            rel_tol=2e-5, abs_tol=2e-5,
                        )
                    eval_history = stage["training"]["eval_loss"]
                    assert [row["step"] for row in eval_history] == list(range(100, 1001, 100))
                    task = stage["task"]
                    assert math.isclose(
                        eval_history[-1]["losses"][task],
                        result["summary"]["losses_when_learned"][stage_index],
                        rel_tol=1e-10, abs_tol=1e-10,
                    )
                assert math.isclose(
                    result["stages"][-1]["training"]["eval_loss"][-1]["losses"][result["stages"][0]["task"]],
                    result["summary"]["final_task_losses"][0],
                    rel_tol=1e-10, abs_tol=1e-10,
                )
                method_runs.append(result)
            runs[(order, label)] = method_runs
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output stem without suffix")
    args = parser.parse_args()
    runs = _load(args.root)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 4, figsize=(7.25, 3.45), sharex=True)
    for row, order in enumerate(ORDERS):
        for stage_index, axis in enumerate(axes[row]):
            task_name = runs[(order, "Sequential")][0]["stages"][stage_index]["task"]
            for _, (label, color) in METHODS.items():
                curves = []
                for run in runs[(order, label)]:
                    curves.append([
                        checkpoint["losses"][task_name]
                        for later_stage in run["stages"][stage_index:]
                        for checkpoint in later_stage["training"]["eval_loss"]
                    ])
                curves = np.asarray(curves)
                x = np.asarray([
                    1000 * later_index + checkpoint["step"]
                    for later_index in range(stage_index, 4)
                    for checkpoint in runs[(order, label)][0]["stages"][later_index]["training"]["eval_loss"]
                ])
                mean = curves.mean(axis=0)
                sem = curves.std(axis=0, ddof=1) / math.sqrt(len(SEEDS))
                axis.plot(x, mean, color=color, linewidth=1.35, marker="o", markersize=1.8, label=label)
                axis.fill_between(
                    x, np.maximum(mean - sem, 1e-8), mean + sem,
                    color=color, alpha=0.14, linewidth=0,
                )
            for boundary in (1000, 2000, 3000):
                axis.axvline(boundary, color="#C8D0D6", linewidth=0.55, linestyle=":", zorder=0)
            axis.set_yscale("log")
            axis.set_xlim(0, 4000)
            axis.set_xticks((1000, 2000, 3000, 4000), ("1k", "2k", "3k", "4k"))
            axis.grid(axis="y", which="both", color="#DCE3E8", linewidth=0.55)
            axis.spines["left"].set_color("#B9C3CA")
            axis.spines["bottom"].set_color("#B9C3CA")
            axis.set_title(f"Stage {stage_index + 1}: {task_name.replace('_', ' ')}", fontsize=7.1, pad=4)
            if row == 1:
                axis.set_xlabel("Global training step")
        axes[row, 0].set_ylabel(f"{order.title()} order\nHeld-out loss")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.14, wspace=0.30, hspace=0.42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(json.dumps({"status": "ok", "output": str(args.output), "runs": 12}))


if __name__ == "__main__":
    main()

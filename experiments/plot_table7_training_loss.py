#!/usr/bin/env python3
"""Plot fair per-step current-task losses for the complete Table 7 matrix."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDERS = ("forward", "reverse")
METHODS = {"seq": ("Sequential", "#607384"), "cagd": ("CAGD", "#ED028C")}
SEEDS = (3407, 3408, 3409)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    assert 1 <= window <= len(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


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
                assert len(result["stages"]) == 4
                for stage in result["stages"]:
                    history = stage["training"]["step_loss"]
                    assert [row["step"] for row in history] == list(range(1, 1001))
                    for row in history:
                        assert all(math.isfinite(row[key]) for key in ("current", "distill", "total"))
                        assert math.isclose(
                            row["total"], row["current"] + row["distill"],
                            rel_tol=2e-5, abs_tol=2e-5,
                        )
                method_runs.append(result)
            runs[(order, label)] = method_runs
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output stem without suffix")
    parser.add_argument("--smooth", type=int, default=20)
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
    x = np.arange(args.smooth, 1001)

    for row, order in enumerate(ORDERS):
        for stage_index, axis in enumerate(axes[row]):
            task_name = runs[(order, "Sequential")][0]["stages"][stage_index]["task"]
            for _, (label, color) in METHODS.items():
                curves = np.asarray([
                    _smooth(
                        np.asarray([point["current"] for point in run["stages"][stage_index]["training"]["step_loss"]]),
                        args.smooth,
                    )
                    for run in runs[(order, label)]
                ])
                mean = curves.mean(axis=0)
                sem = curves.std(axis=0, ddof=1) / math.sqrt(len(SEEDS))
                # ponytail: smoothing is only for legibility; exact raw losses remain in each JSON.
                axis.plot(x, np.maximum(mean, 1e-8), color=color, linewidth=1.35, label=label)
                axis.fill_between(
                    x, np.maximum(mean - sem, 1e-8), np.maximum(mean + sem, 1e-8),
                    color=color, alpha=0.14, linewidth=0,
                )
            axis.set_yscale("log")
            axis.set_xlim(args.smooth, 1000)
            axis.grid(axis="y", which="both", color="#DCE3E8", linewidth=0.55)
            axis.spines["left"].set_color("#B9C3CA")
            axis.spines["bottom"].set_color("#B9C3CA")
            axis.set_title(f"Stage {stage_index + 1}: {task_name.replace('_', ' ')}", fontsize=7.1, pad=4)
            if row == 1:
                axis.set_xlabel("Training step")
        axes[row, 0].set_ylabel(f"{order.title()} order\nCurrent-task loss")

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

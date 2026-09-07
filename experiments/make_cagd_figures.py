#!/usr/bin/env python3
"""Render the paper-specific CAGD figures with tight vector bounds."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


INK = "#243544"
MUTED = "#607384"
BLUE = "#4F91B2"
TEAL = "#4E9A8D"
ORANGE = "#DF8A62"
PINK = "#ED028C"
PALE_BLUE = "#EAF3F7"
PALE_TEAL = "#EAF5F2"
PALE_ORANGE = "#FBF0EA"


def _box(ax, x, y, width, height, face, edge, heading, lines):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + 0.024, y + height - 0.055, heading, color=edge, fontsize=9,
            fontweight="bold", va="top")
    ax.text(x + width / 2, y + height / 2 - 0.018, lines, color=INK,
            fontsize=11, va="center", ha="center", linespacing=1.35)


def _arrow(ax, start, end, label):
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": MUTED, "lw": 1.7,
        "shrinkA": 2, "shrinkB": 2, "mutation_scale": 13,
    })
    ax.text((start[0] + end[0]) / 2, start[1] + 0.055, label,
            color=MUTED, fontsize=8.5, ha="center", va="bottom")


def overview(output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(10.5, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.015, 0.25, 0.18, 0.50, PALE_BLUE, BLUE,
         "RETAIN", "Condition anchors\n" + r"$c_1,\ldots,c_m$" + "\n\nno old outputs")
    _box(ax, 0.245, 0.25, 0.19, 0.50, PALE_TEAL, TEAL,
         "RECONSTRUCT", "Frozen teacher\ncompletion " + r"$\widetilde y$" + "\nand relevant states")

    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (0.485, 0.16), 0.265, 0.68,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6, edgecolor=ORANGE, facecolor=PALE_ORANGE,
    ))
    ax.text(0.509, 0.785, "MATCH LOCAL DISTRIBUTIONS", color=ORANGE,
            fontsize=9, fontweight="bold", va="top")
    ax.text(0.6175, 0.625, "AR prefixes", color=INK, fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(0.6175, 0.535, "next-token KL", color=MUTED, fontsize=9.5, ha="center")
    ax.plot([0.515, 0.72], [0.47, 0.47], color="#D8C8BF", lw=1)
    ax.text(0.6175, 0.365, "Diffusion states", color=INK, fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(0.6175, 0.275, "kernel / denoiser KL", color=MUTED, fontsize=9.5, ha="center")
    ax.text(0.6175, 0.105, "SMDM: masked teacher completions", color=ORANGE,
            fontsize=8.2, ha="center")

    _box(ax, 0.80, 0.25, 0.185, 0.50, "#F7EFF5", PINK,
         "UPDATE", r"$\mathcal{L}_{\mathrm{new}}+\beta\mathcal{L}_{\mathrm{CAGD}}$" + "\n\npreserve old\nconditional behavior")

    _arrow(ax, (0.197, 0.50), (0.243, 0.50), "anchors")
    _arrow(ax, (0.437, 0.50), (0.483, 0.50), "states")
    _arrow(ax, (0.752, 0.50), (0.798, 0.50), "soft targets")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight",
                pad_inches=0.015, facecolor="white")
    plt.close(fig)


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "overview.pdf"
        overview(output)
        assert output.stat().st_size > 10_000
        assert output.with_suffix(".png").stat().st_size > 10_000
    print("self_check=ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("../assets/iclr_3/figures"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    overview(args.output_dir / "cagd_overview.pdf")
    print(f"wrote {args.output_dir / 'cagd_overview.pdf'}")


if __name__ == "__main__":
    main()

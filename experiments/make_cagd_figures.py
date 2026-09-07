#!/usr/bin/env python3
"""Render the paper-specific CAGD figures with tight vector bounds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
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
PALE_PINK = "#FBEAF4"


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
            fontsize=10, va="center", ha="center", linespacing=1.35)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": MUTED, "lw": 1.7,
        "shrinkA": 2, "shrinkB": 2, "mutation_scale": 13,
    })


def overview(output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(10.5, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.13, 0.87)
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
    ax.text(0.509, 0.785, "MATCH", color=ORANGE,
            fontsize=9, fontweight="bold", va="top")
    ax.text(0.6175, 0.625, "AR prefixes", color=INK, fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(0.6175, 0.535, "next-token KL", color=MUTED, fontsize=9.5, ha="center")
    ax.plot([0.515, 0.72], [0.47, 0.47], color="#D8C8BF", lw=1)
    ax.text(0.6175, 0.365, "Diffusion states", color=INK, fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(0.6175, 0.275, "kernel / denoiser KL", color=MUTED, fontsize=9.5, ha="center")
    ax.text(0.6175, 0.190, "SMDM: masked teacher completions", color=ORANGE,
            fontsize=8.2, ha="center")

    _box(ax, 0.80, 0.25, 0.185, 0.50, "#F7EFF5", PINK,
         "UPDATE", r"$\mathcal{L}_{\mathrm{new}}+\beta\mathcal{L}_{\mathrm{CAGD}}$" + "\n\npreserve old\nconditional behavior")

    _arrow(ax, (0.197, 0.50), (0.243, 0.50))
    _arrow(ax, (0.437, 0.50), (0.483, 0.50))
    _arrow(ax, (0.752, 0.50), (0.798, 0.50))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight",
                pad_inches=0.015, facecolor="white")
    plt.close(fig)


def _mean_sem(values):
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def _paired(left, right):
    return [a - b for a, b in zip(left["values"], right["values"])]


def result_summary(output: Path, factual: dict, fresh: dict,
                   component: dict, natural: dict, qwen: dict) -> None:
    """Render one compact evidence map from validated aggregate records."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 2.55),
        gridspec_kw={"width_ratios": [1.12, 0.90, 1.18]},
    )

    # A: CAGD versus sequential across controlled factual settings.
    labels, effects = [], []
    for name, summary, order in (
        ("SMDM main · F", factual, "forward"),
        ("SMDM main · R", factual, "reverse"),
        ("SMDM fresh · F", fresh, "forward"),
        ("SMDM fresh · R", fresh, "reverse"),
    ):
        values = _paired(
            summary["aggregate"][order]["gd"]["final_average_loss"],
            summary["aggregate"][order]["seq"]["final_average_loss"],
        )
        labels.append(name)
        effects.append(values)
    for scale, label in (("qwen3_0.6b", "AR · 0.6B"),
                         ("qwen3_1.7b", "AR · 1.7B"),
                         ("qwen3_4b", "AR · 4B")):
        labels.append(label)
        effects.append(qwen["paired_contrasts"][scale]["gd_minus_seq"]
                       ["final_average_loss"]["values"])
    ax = axes[0]
    y = list(range(len(labels)))
    means, sems = zip(*(_mean_sem(values) for values in effects))
    ax.axvspan(min(means) - 0.5, 0, color=PALE_TEAL, zorder=0)
    for yi, values in zip(y, effects):
        ax.scatter(values, [yi] * len(values), s=14, color=MUTED,
                   alpha=0.42, linewidths=0, zorder=2)
    ax.errorbar(means, y, xerr=sems, fmt="o", ms=5.2, lw=1.5,
                capsize=2.5, color=PINK, zorder=3)
    ax.axvline(0, color="#9AA8B2", lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("CAGD − Sequential\nfinal average loss")
    ax.set_title("A   Controlled retention", loc="left", fontweight="bold",
                 color=INK, pad=8)

    # B: causal soft-versus-hard intervention on identical generated support.
    contrast = component["paired"]["forward"]["cagd_minus_hard_replay"]
    endpoint_names = (
        ("Final average", "final_average_loss"),
        ("Past forgetting", "past_task_forgetting"),
        ("Final task", "final_task_loss"),
    )
    ax = axes[1]
    y = list(range(len(endpoint_names)))
    ax.axvspan(-0.14, 0, color=PALE_PINK, zorder=0)
    for yi, (_, key) in zip(y, endpoint_names):
        values = contrast[key]["values"]
        ax.scatter(values, [yi] * len(values), s=17, color=MUTED,
                   alpha=0.45, linewidths=0, zorder=2)
        ax.errorbar(contrast[key]["mean"], yi, xerr=contrast[key]["sem"],
                    fmt="o", ms=5.5, lw=1.6, capsize=2.5, color=PINK,
                    zorder=3)
    ax.axvline(0, color="#9AA8B2", lw=1)
    ax.set_yticks(y, [name for name, _ in endpoint_names])
    ax.invert_yaxis()
    ax.set_xlim(-0.14, 0.04)
    ax.set_xlabel("CAGD − hard replay")
    ax.set_title("B   Matched component", loc="left", fontweight="bold",
                 color=INK, pad=8)

    # C: same natural task stream on masked-diffusion and AR backends.
    rows = (
        ("SMDM vs Seq.", "smdm", "cagd_minus_seq"),
        ("SMDM vs Hard", "smdm", "cagd_minus_hard_replay"),
        ("AR vs Seq.", "qwen", "cagd_minus_seq"),
        ("AR vs Hard", "qwen", "cagd_minus_hard_replay"),
    )
    ax = axes[2]
    y = list(range(len(rows)))
    natural_means = []
    for yi, (_, backend, contrast_name) in zip(y, rows):
        record = natural["paired"][backend][contrast_name]
        for key, color, marker, offset in (
            ("final_average_loss", PINK, "o", -0.10),
            ("past_task_forgetting", BLUE, "s", 0.10),
        ):
            item = record[key]
            natural_means.append(item["mean"])
            ax.errorbar(item["mean"], yi + offset, xerr=item["sem"],
                        fmt=marker, ms=5.0, lw=1.4, capsize=2.3,
                        color=color, zorder=3)
    ax.axvspan(min(natural_means) - 0.2, 0, color=PALE_BLUE, zorder=0)
    ax.axvline(0, color="#9AA8B2", lw=1)
    ax.set_yticks(y, [name for name, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("CAGD − baseline")
    ax.set_title("C   Natural instructions", loc="left", fontweight="bold",
                 color=INK, pad=8)
    ax.plot([], [], "o", color=PINK, label="Final average")
    ax.plot([], [], "s", color=BLUE, label="Past forgetting")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left",
              handletextpad=0.4, borderpad=0.2)

    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B9C3CA")
        ax.grid(axis="x", color="#DCE3E8", lw=0.7, zorder=0)
        ax.tick_params(axis="y", length=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.86, bottom=0.22,
                        wspace=0.56)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight",
                pad_inches=0.015, facecolor="white")
    plt.close(fig)


def _fixture_summaries():
    metric = lambda values: {"mean": statistics.fmean(values), "sem": 0.02,
                             "values": values}
    aggregate = {
        order: {method: {"final_average_loss": metric(values)}
                for method, values in (("seq", [2.0, 2.1, 1.9]),
                                       ("gd", [1.0, 1.1, 0.9]))}
        for order in ("forward", "reverse")
    }
    component = {"paired": {"forward": {"cagd_minus_hard_replay": {
        key: metric([-0.08, -0.05, -0.06]) for key in
        ("final_average_loss", "past_task_forgetting", "final_task_loss")
    }}}}
    paired = {
        backend: {contrast: {
            "final_average_loss": metric([-0.2, -0.1, -0.3]),
            "past_task_forgetting": metric([-0.3, -0.2, -0.25]),
        } for contrast in ("cagd_minus_seq", "cagd_minus_hard_replay")}
        for backend in ("smdm", "qwen")
    }
    qwen = {"paired_contrasts": {
        scale: {"gd_minus_seq": {"final_average_loss": metric([-1.0, -1.1, -0.9])}}
        for scale in ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b")
    }}
    return {"aggregate": aggregate}, component, {"paired": paired}, qwen


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "overview.pdf"
        overview(output)
        assert output.stat().st_size > 10_000
        assert output.with_suffix(".png").stat().st_size > 10_000
        factual, component, natural, qwen = _fixture_summaries()
        result_output = Path(directory) / "results.pdf"
        result_summary(result_output, factual, factual, component, natural, qwen)
        assert result_output.stat().st_size > 10_000
        assert result_output.with_suffix(".png").stat().st_size > 10_000
    print("self_check=ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("../assets/iclr_3/figures"))
    parser.add_argument("--factual-summary", type=Path,
                        default=Path("runs/r16_native_mask/summary.json"))
    parser.add_argument("--fresh-summary", type=Path,
                        default=Path("runs/r16_native_mask/fresh_summary.json"))
    parser.add_argument("--component-summary", type=Path,
                        default=Path("runs/cagd_component/two_task_summary.json"))
    parser.add_argument("--natural-summary", type=Path,
                        default=Path("runs/cagd_natural/summary.json"))
    parser.add_argument("--qwen-summary", type=Path,
                        default=Path("runs/qwen_continual_scale/formal_summary_3scale.json"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    overview(args.output_dir / "cagd_overview.pdf")
    summary_paths = (args.factual_summary, args.fresh_summary,
                     args.component_summary, args.natural_summary,
                     args.qwen_summary)
    if all(path.is_file() for path in summary_paths):
        summaries = [json.loads(path.read_text()) for path in summary_paths]
        result_summary(args.output_dir / "cagd_results.pdf", *summaries)
        print(f"wrote {args.output_dir / 'cagd_results.pdf'}")
    else:
        missing = ", ".join(str(path) for path in summary_paths if not path.is_file())
        print(f"results_skipped_missing={missing}")
    print(f"wrote {args.output_dir / 'cagd_overview.pdf'}")


if __name__ == "__main__":
    main()

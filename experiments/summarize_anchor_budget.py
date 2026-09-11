#!/usr/bin/env python3
"""Audit the frozen anchor-budget sweep and render its appendix figure."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from matplotlib import font_manager

for font in Path('/usr/share/fonts/opentype/urw-base35').glob('NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(font)


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (4, 8, 16, 32, 64, 120)
ORDERS = ("forward", "reverse")
SEED = 3407
PROTOCOL_SHA256 = "50c84a21a24091908b33cf81a3acd53289e4d0f9f03f029e1c901bbc51118cf5"


def _path(order: str, budget: int) -> Path:
    if budget == 64:
        return ROOT / f"runs/cagd_component/main/{order}/s{SEED}/cagd.json"
    return ROOT / f"runs/cagd_anchor_budget/{order}/m{budget}_s{SEED}.json"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != "ok" or data.get("experiment") != "dllm_rank1_multitask":
        raise ValueError(f"invalid result: {path}")
    return data


def _comparable_metadata(metadata: dict) -> dict:
    ignored = {"protocol", "protocol_document", "protocol_document_sha256", "replay_per_task"}
    return {key: value for key, value in metadata.items() if key not in ignored}


def audit() -> list[dict]:
    protocol = ROOT / "report/cagd_anchor_budget_protocol.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("anchor-budget protocol changed after launch")

    rows = []
    for order in ORDERS:
        reference = _load(_path(order, 64))
        for budget in BUDGETS:
            path = _path(order, budget)
            run = _load(path)
            metadata = run["metadata"]
            if run["source_sha256"] != reference["source_sha256"]:
                raise ValueError(f"runner hash mismatch: {path}")
            if run["dependency_sha256"] != reference["dependency_sha256"]:
                raise ValueError(f"dependency hash mismatch: {path}")
            if _comparable_metadata(metadata) != _comparable_metadata(reference["metadata"]):
                raise ValueError(f"fixed metadata mismatch: {path}")
            if metadata["replay_per_task"] != budget:
                raise ValueError(f"anchor budget mismatch: {path}")
            for stage, record in enumerate(run["stages"]):
                if record["replay_examples"] != stage * budget:
                    raise ValueError(f"replay size mismatch: {path}, stage {stage + 1}")
                manifests = record["replay"]["per_task"]
                if len(manifests) != stage:
                    raise ValueError(f"replay task count mismatch: {path}, stage {stage + 1}")
                if stage and not record["replay"]["generated_rows_sha256"]:
                    raise ValueError(f"missing generated replay hash: {path}, stage {stage + 1}")
                for manifest in manifests:
                    counts = manifest["fact_counts"]
                    if manifest["count"] != budget or len(counts) != 4 or set(counts.values()) != {budget // 4}:
                        raise ValueError(f"unbalanced anchors: {path}, stage {stage + 1}")
            summary = run["summary"]
            values = {
                "final_average_loss": summary["final_average_loss"],
                "past_task_forgetting": summary["past_task_forgetting"],
                "final_task_loss": summary["final_task_losses"][-1],
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite endpoint: {path}")
            rows.append({"order": order, "anchors_per_old_task": budget, **values,
                         "source": str(path.relative_to(ROOT))})
    return rows


def render(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Nimbus Roman", "mathtext.fontset": "stix", "font.size": 7.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelcolor": "#243544", "xtick.color": "#607384",
        "ytick.color": "#607384",
    })
    metrics = (
        ("final_average_loss", "Final average loss", "a"),
        ("past_task_forgetting", "Past-task forgetting", "b"),
        ("final_task_loss", "Final-task loss", "c"),
    )
    colors = {"forward": "#4E9A8D", "reverse": "#ED028C"}
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.05), constrained_layout=True)
    for ax, (metric, label, panel) in zip(axes, metrics):
        for order in ORDERS:
            subset = [row for row in rows if row["order"] == order]
            ax.plot(BUDGETS, [row[metric] for row in subset], marker="o", ms=4.2,
                    lw=1.6, color=colors[order], label=order.capitalize())
        ax.axvline(64, color="#B8C2C9", lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        ax.set_xlabel("Anchors per old task")
        ax.set_ylabel(label)
        ax.set_title(f"({panel}) {label}", loc="left", fontsize=8.0, fontweight="bold")
        ax.grid(axis="y", color="#E2E8EC", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best", handlelength=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = audit()
    summary = ROOT / "runs/cagd_anchor_budget/summary.json"
    summary.write_text(json.dumps({
        "experiment": "cagd_anchor_budget_sensitivity",
        "protocol": "report/cagd_anchor_budget_protocol.md",
        "seed": SEED,
        "rows": rows,
    }, indent=2) + "\n")
    figure = ROOT.parent / "assets/iclr_3/figures/anchor_budget_sensitivity.pdf"
    render(rows, figure)
    print(json.dumps({"status": "ok", "summary": str(summary), "figure": str(figure)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit and summarize the paired GSM8K behavioral-retention runs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


METHODS = ("seq", "cagd")
SEEDS = (3407, 3408, 3409)


def _record_map(run: dict, stage: int) -> dict[int, dict]:
    records = run["stages"][stage]["benchmark"]["records"]
    return {row["source_index"]: row for row in records}


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / len(values) ** 0.5,
        "values": values,
    }


def _bootstrap_interval(differences: list[list[int]], repetitions: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    seed_count = len(differences)
    example_count = len(differences[0])
    estimates = []
    for _ in range(repetitions):
        selected_seeds = [rng.randrange(seed_count) for _ in range(seed_count)]
        estimate = 0.0
        for seed_index in selected_seeds:
            values = differences[seed_index]
            estimate += sum(values[rng.randrange(example_count)] for _ in range(example_count)) / example_count
        estimates.append(estimate / seed_count)
    estimates.sort()
    return [
        estimates[int(0.025 * repetitions)],
        estimates[min(int(0.975 * repetitions), repetitions - 1)],
    ]


def summarize(root: Path, repetitions: int, bootstrap_seed: int) -> dict:
    runs = {}
    for method in METHODS:
        for seed in SEEDS:
            path = root / "formal" / method / f"s{seed}.json"
            run = json.loads(path.read_text())
            if run.get("status") != "ok" or run["metadata"].get("protocol") != "cagd_gsm8k_behavior_v1":
                raise ValueError(f"unaccepted run: {path}")
            if run["metadata"]["method"] != method or run["metadata"]["seed"] != seed:
                raise ValueError(f"metadata mismatch: {path}")
            if run["metadata"]["benchmark_count"] != 1319:
                raise ValueError(f"incomplete benchmark: {path}")
            runs[method, seed] = run

    per_seed = []
    paired_examples = []
    for seed in SEEDS:
        seq = runs["seq", seed]
        cagd = runs["cagd", seed]
        seq_learned = _record_map(seq, 0)
        cagd_learned = _record_map(cagd, 0)
        if seq_learned != cagd_learned:
            raise ValueError(f"stage-1 outputs differ within paired seed {seed}")
        seq_final = _record_map(seq, -1)
        cagd_final = _record_map(cagd, -1)
        if seq_final.keys() != cagd_final.keys() or seq_final.keys() != seq_learned.keys():
            raise ValueError(f"benchmark rows differ within paired seed {seed}")
        indices = sorted(seq_final)
        differences = [int(cagd_final[i]["correct"]) - int(seq_final[i]["correct"]) for i in indices]
        paired_examples.append(differences)
        per_seed.append({
            "seed": seed,
            "when_learned": seq["summary"]["gsm8k_exact_match_when_learned"],
            "sequential_final": seq["summary"]["gsm8k_exact_match_final"],
            "cagd_final": cagd["summary"]["gsm8k_exact_match_final"],
            "cagd_minus_sequential": sum(differences) / len(differences),
            "sequential_retention_change": seq["summary"]["gsm8k_retention_change"],
            "cagd_retention_change": cagd["summary"]["gsm8k_retention_change"],
            "sequential_final_task_loss": seq["summary"]["final_task_loss"],
            "cagd_final_task_loss": cagd["summary"]["final_task_loss"],
        })

    seed_differences = [row["cagd_minus_sequential"] for row in per_seed]
    interval = _bootstrap_interval(paired_examples, repetitions, bootstrap_seed)
    learned = _record_map(runs["seq", 3407], 0)
    seq_final = _record_map(runs["seq", 3407], -1)
    cagd_final = _record_map(runs["cagd", 3407], -1)
    gains = [i for i in sorted(learned)
             if learned[i]["correct"] and not seq_final[i]["correct"] and cagd_final[i]["correct"]]
    reversals = [i for i in sorted(learned)
                 if learned[i]["correct"] and seq_final[i]["correct"] and not cagd_final[i]["correct"]]
    examples = [{
        "source_index": index,
        "question": learned[index]["question"],
        "target": learned[index]["target"],
        "when_learned_output": learned[index]["output"],
        "sequential_final_output": seq_final[index]["output"],
        "cagd_final_output": cagd_final[index]["output"],
    } for index in gains[:3]]

    return {
        "schema_version": 1,
        "status": "ok",
        "run_count": len(runs),
        "benchmark_count": 1319,
        "per_seed": per_seed,
        "paired_final_exact_match": _mean_sem(seed_differences),
        "seed_by_example_bootstrap_95_interval": interval,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "all_seeds_favor_cagd": all(value > 0 for value in seed_differences),
        "interval_excludes_zero": interval[0] > 0 or interval[1] < 0,
        "qualitative_rule": {
            "seed": 3407,
            "gain_count": len(gains),
            "reverse_count": len(reversals),
            "selected_count": len(examples),
            "examples": examples,
        },
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# GSM8K behavioral retention results",
        "",
        "| Seed | After GSM8K | Sequential final | CAGD final | CAGD - Seq. |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['when_learned']:.3f} | "
            f"{row['sequential_final']:.3f} | {row['cagd_final']:.3f} | "
            f"{row['cagd_minus_sequential']:+.3f} |"
        )
    paired = summary["paired_final_exact_match"]
    interval = summary["seed_by_example_bootstrap_95_interval"]
    rule = summary["qualitative_rule"]
    lines += [
        "",
        f"Paired improvement: {paired['mean']:+.3f} +/- {paired['sem']:.3f} SEM; "
        f"two-way 95% bootstrap interval [{interval[0]:+.3f}, {interval[1]:+.3f}].",
        "",
        f"Predeclared qualitative rule: {rule['gain_count']} gains and "
        f"{rule['reverse_count']} reversals among seed-3407 questions correct when learned.",
    ]
    for index, example in enumerate(rule["examples"], 1):
        lines += [
            "",
            f"## Example {index} (test index {example['source_index']})",
            "",
            f"**Question:** {example['question']}",
            "",
            f"**Target:** {example['target']}",
            "",
            f"**Sequential final:** {example['sequential_final_output']}",
            "",
            f"**CAGD final:** {example['cagd_final_output']}",
        ]
    return "\n".join(lines) + "\n"


def _self_check() -> None:
    interval = _bootstrap_interval([[1, 1], [1, 1], [1, 1]], 100, 7)
    assert interval == [1.0, 1.0]
    stats = _mean_sem([0.1, 0.2, 0.3])
    assert abs(stats["mean"] - 0.2) < 1e-12
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/cagd_gsm8k_behavior"))
    parser.add_argument("--output", type=Path, default=Path("runs/cagd_gsm8k_behavior/summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("report/cagd_gsm8k_behavior_results.md"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    summary = summarize(args.root, args.bootstrap_repetitions, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.markdown.write_text(_markdown(summary))
    print(json.dumps({
        "status": "ok",
        "output": str(args.output),
        "markdown": str(args.markdown),
        "paired": summary["paired_final_exact_match"],
        "interval": summary["seed_by_example_bootstrap_95_interval"],
    }, indent=2))


if __name__ == "__main__":
    main()

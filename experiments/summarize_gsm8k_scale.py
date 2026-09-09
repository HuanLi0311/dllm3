#!/usr/bin/env python3
"""Fail-closed audit and summary for the five-model GSM8K matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import cagd_gsm8k_scale as runner
from experiments import qwen_cagd_gsm8k_behavior as old_runner


MODELS = (
    ("smdm_219m", "SMDM-219M"),
    ("smdm_1.14b", "SMDM-1.14B"),
    ("qwen3_0.6b", "Qwen3-0.6B"),
    ("qwen3_1.7b", "Qwen3-1.7B"),
    ("qwen3_4b", "Qwen3-4B"),
)
METHODS = ("seq", "cagd")
SEEDS = (3407, 3408, 3409)
OLD_SOURCE = "7cb398e18773b30c9271ccfd4047e0579eb223cf9bcc8328cda5fe256fcf81ee"
OLD_PROTOCOL = "6571143f3fb4966fd95e862d1cdaa2b474402e1679b59150d5fb479c9df9d71c"
OLD_INVENTORY = "768bd5491a8c872f18c9edb62a8c4ece300963443fb1fa8b7b7be1d3d1fc053b"


def _mean_sem(values: list) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
    }


def _path(root: Path, model_id: str, method: str, seed: int) -> Path:
    if model_id == "qwen3_0.6b":
        return root.parent / "cagd_gsm8k_behavior/formal" / method / f"s{seed}.json"
    return root / "formal" / model_id / method / f"s{seed}.json"


def _records(run: dict, stage: int) -> list:
    return run["stages"][stage]["benchmark"]["records"]


def _audit_benchmark(run: dict, stage: int, path: Path, expected_rows: dict) -> None:
    benchmark = run["stages"][stage]["benchmark"]
    records = benchmark["records"]
    if benchmark["count"] != 1319 or len(records) != 1319:
        raise ValueError(f"incomplete benchmark: {path}")
    if sorted(row["source_index"] for row in records) != list(range(1319)):
        raise ValueError(f"benchmark indices differ: {path}")
    for row in records:
        expected = expected_rows[row["source_index"]]
        for key in ("example_id", "question", "target"):
            if row.get(key) != expected[key]:
                raise ValueError(f"source {key} differs: {path}:{row['source_index']}")
        prediction, marked = old_runner._extract_answer(row["output"])
        if prediction != row["prediction"] or marked != row["has_delimiter"]:
            raise ValueError(f"parser recomputation differs: {path}:{row['source_index']}")
        if row["correct"] != (prediction == row["target"]):
            raise ValueError(f"correctness recomputation differs: {path}:{row['source_index']}")
    exact = sum(row["correct"] for row in records) / len(records)
    delimiter = sum(row["has_delimiter"] for row in records) / len(records)
    if abs(exact - benchmark["exact_match"]) > 1e-15:
        raise ValueError(f"exact match recomputation differs: {path}")
    if abs(delimiter - benchmark["delimiter_rate"]) > 1e-15:
        raise ValueError(f"format-rate recomputation differs: {path}")


def _expected_data_hashes() -> dict:
    return {str(path.relative_to(runner.ROOT)): digest for path, digest in runner.DATA_HASHES.items()}


def _audit_old(run: dict, method: str, seed: int, path: Path) -> None:
    metadata = run.get("metadata", {})
    expected_dependencies = {
        "experiments/qwen_cagd_natural.py": "4a8be25e92865127c46ca12c0359de1692f866692849581d23da4c92aba8feea",
        "experiments/qwen_continual_transfer.py": "a8a48d15340d01b2261f0eba8551e42fd138fba10a73359794eba251c57947c0",
    }
    if (
        run.get("status") != "ok"
        or run.get("experiment") != "qwen_cagd_gsm8k_behavior"
        or run.get("source_sha256") != OLD_SOURCE
        or run.get("protocol_sha256") != OLD_PROTOCOL
        or run.get("dependency_sha256") != expected_dependencies
        or run.get("data_sha256") != _expected_data_hashes()
        or run.get("model_inventory", {}).get("sha256") != OLD_INVENTORY
        or metadata.get("protocol") != "cagd_gsm8k_behavior_v1"
        or metadata.get("method") != method
        or metadata.get("seed") != seed
        or metadata.get("benchmark_count") != 1319
    ):
        raise ValueError(f"reused Qwen3-0.6B provenance differs: {path}")


def _audit_new(run: dict, model_id: str, method: str, seed: int, path: Path) -> None:
    spec = runner.MODEL_SPECS[model_id]
    metadata = run.get("metadata", {})
    expected_settings = dict(runner.COMMON_FORMAL_SETTINGS)
    expected_settings["generation_batch_size"] = 16 if spec["backend"] == "ar" else 8
    if spec["backend"] == "smdm":
        expected_settings.update(runner.SMDM_FORMAL_SETTINGS)
    if (
        run.get("status") != "ok"
        or run.get("schema_version") != 2
        or run.get("experiment") != "cagd_gsm8k_scale"
        or run.get("source_sha256") != runner._sha256(Path(runner.__file__))
        or run.get("protocol_sha256") != runner._sha256(runner.PROTOCOL)
        or run.get("dependency_sha256") != runner._dependency_hashes()
        or run.get("data_sha256") != _expected_data_hashes()
        or metadata.get("formal") is not True
        or metadata.get("protocol") != "cagd_gsm8k_scale_v1"
        or metadata.get("model_id") != model_id
        or metadata.get("backend") != spec["backend"]
        or metadata.get("method") != method
        or metadata.get("seed") != seed
        or metadata.get("benchmark_count") != 1319
        or metadata.get("settings") != expected_settings
        or metadata.get("runner_sha256") != run.get("source_sha256")
        or metadata.get("protocol_sha256") != run.get("protocol_sha256")
        or metadata.get("dependency_sha256") != run.get("dependency_sha256")
    ):
        raise ValueError(f"new-run provenance differs: {path}")
    if spec["backend"] == "ar":
        if run.get("model_inventory", {}).get("sha256") != spec["inventory_sha256"]:
            raise ValueError(f"Qwen inventory differs: {path}")
    elif (
        metadata.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or metadata.get("model_parameter_count") != spec["parameter_count"]
        or metadata.get("trainable_parameter_count") != spec["parameter_count"]
        or run.get("tokenizer_sha256") != runner.TOKENIZER_SHA256
        or metadata.get("completion_only_scoring") is not True
    ):
        raise ValueError(f"SMDM model provenance differs: {path}")


def _audit_stages(run: dict, path: Path) -> None:
    stages = run.get("stages", [])
    if (
        len(stages) != 3
        or [stage.get("stage") for stage in stages] != [0, 1, 2]
        or [stage.get("task") for stage in stages] != list(runner.TASKS)
        or run.get("metadata", {}).get("task_sequence") != list(runner.TASKS)
        or stages[0].get("benchmark") is None
        or stages[1].get("benchmark") is not None
        or stages[2].get("benchmark") is None
        or stages[2].get("current_metrics") is None
    ):
        raise ValueError(f"stage structure differs: {path}")


def _audit_summary(run: dict, path: Path) -> None:
    learned = run["stages"][0]["benchmark"]["exact_match"]
    final = run["stages"][-1]["benchmark"]["exact_match"]
    summary = run["summary"]
    expected = {
        "gsm8k_exact_match_when_learned": learned,
        "gsm8k_exact_match_final": final,
        "gsm8k_retention_change": final - learned,
        "final_task_loss": run["stages"][-1]["current_metrics"]["loss"],
    }
    for key, value in expected.items():
        if not math.isfinite(summary[key]) or abs(summary[key] - value) > 1e-12:
            raise ValueError(f"summary recomputation differs for {key}: {path}")


def summarize(root: Path) -> dict:
    source_hash = runner._sha256(Path(runner.__file__))
    protocol_hash = runner._sha256(runner.PROTOCOL)
    runs = {}
    cells = []
    _, raw_evaluation = old_runner._raw_gsm()
    expected_rows = {
        row["source_index"]: {
            "example_id": row["example_id"],
            "question": row["question"],
            "target": row["target"],
        }
        for row in raw_evaluation
    }
    if sorted(expected_rows) != list(range(1319)):
        raise ValueError("vendored GSM8K indices differ")
    for model_id, display_name in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                path = _path(root, model_id, method, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"missing formal cell: {path}")
                run = json.loads(path.read_text())
                if model_id == "qwen3_0.6b":
                    _audit_old(run, method, seed, path)
                else:
                    _audit_new(run, model_id, method, seed, path)
                _audit_stages(run, path)
                _audit_benchmark(run, 0, path, expected_rows)
                _audit_benchmark(run, -1, path, expected_rows)
                _audit_summary(run, path)
                runs[model_id, method, seed] = run
                cells.append({
                    "model_id": model_id,
                    "model": display_name,
                    "method": method,
                    "seed": seed,
                    "status": "validated",
                    "path": str(path),
                    "sha256": runner._sha256(path),
                })
    groups = {}
    paired = {}
    for model_id, display_name in MODELS:
        groups[model_id] = {"display_name": display_name, "methods": {}}
        for method in METHODS:
            model_runs = [runs[model_id, method, seed] for seed in SEEDS]
            metrics = {
                "after_gsm8k_exact_match": [run["summary"]["gsm8k_exact_match_when_learned"] for run in model_runs],
                "final_exact_match": [run["summary"]["gsm8k_exact_match_final"] for run in model_runs],
                "retention_change": [run["summary"]["gsm8k_retention_change"] for run in model_runs],
                "final_format_rate": [run["stages"][-1]["benchmark"]["delimiter_rate"] for run in model_runs],
                "final_task_loss": [run["summary"]["final_task_loss"] for run in model_runs],
            }
            groups[model_id]["methods"][method] = {
                key: _mean_sem(values) for key, values in metrics.items()
            }
        differences = [
            runs[model_id, "cagd", seed]["summary"]["gsm8k_exact_match_final"]
            - runs[model_id, "seq", seed]["summary"]["gsm8k_exact_match_final"]
            for seed in SEEDS
        ]
        paired[model_id] = {
            "cagd_minus_sequential_final_exact_match": _mean_sem(differences),
            "all_seeds_favor_cagd": all(value > 0 for value in differences),
        }
        for seed in SEEDS:
            seq = {row["source_index"]: row for row in _records(runs[model_id, "seq", seed], 0)}
            cagd = {row["source_index"]: row for row in _records(runs[model_id, "cagd", seed], 0)}
            if seq != cagd:
                raise ValueError(f"paired post-GSM8K outputs differ: {model_id}, seed {seed}")
    return {
        "schema_version": 1,
        "status": "ok",
        "validated_run_count": len(cells),
        "new_run_count": len(cells) - 6,
        "reused_run_count": 6,
        "benchmark_count": 1319,
        "runner_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "models": groups,
        "paired": paired,
        "cells": cells,
    }


def _value(metric: dict, percent: bool = False) -> str:
    scale = 100 if percent else 1
    return f"{scale * metric['mean']:.2f} +/- {scale * metric['sem']:.2f}"


def _markdown(summary: dict) -> str:
    lines = [
        "# Five-model GSM8K behavioral-retention results",
        "",
        "All values are mean +/- SEM over three paired seeds.",
        "",
        "| Model | Method | After GSM8K EM (%) | Final EM (%) | Retention change (pp) | Format (%) | Final-task loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, display_name in MODELS:
        for method in METHODS:
            metrics = summary["models"][model_id]["methods"][method]
            lines.append(
                f"| {display_name} | {'Sequential' if method == 'seq' else 'CAGD'} | "
                f"{_value(metrics['after_gsm8k_exact_match'], True)} | "
                f"{_value(metrics['final_exact_match'], True)} | "
                f"{_value(metrics['retention_change'], True)} | "
                f"{_value(metrics['final_format_rate'], True)} | "
                f"{_value(metrics['final_task_loss'])} |"
            )
    lines += ["", "## Paired final-EM differences", ""]
    for model_id, display_name in MODELS:
        item = summary["paired"][model_id]
        difference = item["cagd_minus_sequential_final_exact_match"]
        lines.append(
            f"- {display_name}: {_value(difference, True)} pp; "
            f"all seeds favor CAGD: {item['all_seeds_favor_cagd']}."
        )
    lines += [
        "",
        "## Audit",
        "",
        f"Validated {summary['validated_run_count']} runs: "
        f"{summary['new_run_count']} new and {summary['reused_run_count']} reused Qwen3-0.6B runs.",
        "",
    ]
    return "\n".join(lines)


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)


def _self_check() -> None:
    stats = _mean_sem([0.1, 0.2, 0.3])
    assert abs(stats["mean"] - 0.2) < 1e-12
    assert len(MODELS) == 5 and len(METHODS) == 2 and len(SEEDS) == 3
    expected = {
        index: {"example_id": f"e{index}", "question": f"q{index}", "target": str(index)}
        for index in range(1319)
    }
    records = [{
        "source_index": index,
        **expected[index],
        "output": f"work #### {index}",
        "prediction": str(index),
        "has_delimiter": True,
        "correct": True,
    } for index in range(1319)]
    mock = {"stages": [{"benchmark": {
        "count": 1319, "exact_match": 1.0, "delimiter_rate": 1.0, "records": records,
    }}]}
    _audit_benchmark(mock, 0, Path("synthetic.json"), expected)
    records[7]["target"] = "wrong"
    try:
        _audit_benchmark(mock, 0, Path("tampered.json"), expected)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered target was accepted")
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/cagd_gsm8k_scale"))
    parser.add_argument("--output", type=Path, default=Path("runs/cagd_gsm8k_scale/summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("report/cagd_gsm8k_scale_results.md"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.output.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite an existing summary or report")
    summary = summarize(args.root)
    _write_new(args.output, json.dumps(summary, indent=2) + "\n")
    _write_new(args.markdown, _markdown(summary))
    print(json.dumps({"status": "ok", "output": str(args.output), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()

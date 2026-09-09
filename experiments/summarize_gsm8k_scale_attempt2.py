#!/usr/bin/env python3
"""Audit and summarize all 30 endpoints in the paired-stage0 scale matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import audit_gsm8k_scale_attempt2 as pair_audit  # noqa: E402
from experiments import cagd_gsm8k_scale as qwen_runner  # noqa: E402
from experiments import smdm_cagd_gsm8k_paired as smdm_runner  # noqa: E402
from experiments import summarize_gsm8k_scale as v1  # noqa: E402


MODELS = v1.MODELS
METHODS = v1.METHODS
SEEDS = v1.SEEDS


def _endpoint_path(root: Path, model_id: str, method: str, seed: int) -> Path:
    if model_id == "qwen3_0.6b":
        return root.parents[1] / "cagd_gsm8k_behavior/formal" / method / f"s{seed}.json"
    return root / "formal" / model_id / method / f"s{seed}.json"


def _canonical_path(root: Path, model_id: str, seed: int) -> Path:
    return root / "stage0" / model_id / f"s{seed}" / "stage0.json"


def _settings(backend: str) -> dict:
    settings = dict(qwen_runner.COMMON_FORMAL_SETTINGS)
    settings["generation_batch_size"] = 8 if backend == "smdm" else 16
    if backend == "smdm":
        settings.update(qwen_runner.SMDM_FORMAL_SETTINGS)
    return settings


def _require_fields(observed: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"{label} {key} differs")


def _audit_canonical(canonical: dict, path: Path, model_id: str, seed: int,
                     expected_rows: list[dict]) -> dict:
    spec = qwen_runner.MODEL_SPECS[model_id]
    source_hash = qwen_runner._sha256(Path(smdm_runner.__file__))
    protocol_hash = qwen_runner._sha256(smdm_runner.PROTOCOL)
    dependencies = smdm_runner._dependency_hashes()
    _require_fields(canonical, {
        "schema_version": smdm_runner.SCHEMA_VERSION,
        "status": "ok",
        "experiment": "cagd_gsm8k_scale_attempt2_stage0",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": smdm_runner._data_hashes(),
        "tokenizer_sha256": qwen_runner.TOKENIZER_SHA256,
    }, f"canonical {path}")
    _require_fields(canonical.get("metadata", {}), {
        "formal": True,
        "protocol": "cagd_gsm8k_scale_paired_stage0_v2",
        "model_id": model_id,
        "model_display_name": spec["display_name"],
        "backend": "smdm",
        "seed": seed,
        "base_checkpoint": str(spec["path"].resolve()),
        "base_checkpoint_sha256": spec["checkpoint_sha256"],
        "model_parameter_count": spec["parameter_count"],
        "trainable": "all_parameters",
        "trainable_parameter_count": spec["parameter_count"],
        "settings": _settings("smdm"),
        "runner_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
    }, f"canonical metadata {path}")
    checkpoint = canonical.get("canonical_checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    expected_checkpoint_path = path.with_name("stage0.safetensors").resolve()
    if checkpoint_path.resolve() != expected_checkpoint_path:
        raise ValueError(f"canonical checkpoint path differs: {path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"canonical checkpoint missing: {checkpoint_path}")
    actual_checkpoint = smdm_runner._file_descriptor(checkpoint_path)
    if checkpoint != actual_checkpoint:
        raise ValueError(f"canonical checkpoint descriptor differs: {path}")
    smdm_runner._audit_stage_record(canonical.get("stage", {}), expected_rows, 1000)
    return actual_checkpoint


def _audit_smdm_endpoint(run: dict, path: Path, model_id: str, method: str,
                         seed: int, canonical_path: Path, checkpoint: dict) -> None:
    spec = qwen_runner.MODEL_SPECS[model_id]
    source_hash = qwen_runner._sha256(Path(smdm_runner.__file__))
    protocol_hash = qwen_runner._sha256(smdm_runner.PROTOCOL)
    dependencies = smdm_runner._dependency_hashes()
    _require_fields(run, {
        "schema_version": 2,
        "status": "ok",
        "experiment": "cagd_gsm8k_scale_attempt2",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": smdm_runner._data_hashes(),
        "tokenizer_sha256": qwen_runner.TOKENIZER_SHA256,
    }, f"SMDM endpoint {path}")
    _require_fields(run.get("metadata", {}), {
        "formal": True,
        "protocol": "cagd_gsm8k_scale_paired_stage0_v2",
        "model_id": model_id,
        "model_display_name": spec["display_name"],
        "backend": "smdm",
        "method": method,
        "seed": seed,
        "task_sequence": list(qwen_runner.TASKS),
        "base_checkpoint": str(spec["path"].resolve()),
        "base_checkpoint_sha256": spec["checkpoint_sha256"],
        "config_name": spec["config_name"],
        "model_parameter_count": spec["parameter_count"],
        "trainable": "all_parameters",
        "trainable_parameter_count": spec["parameter_count"],
        "benchmark_count": 1319,
        "generation_batch_size": 8,
        "benchmark_decoding": "greedy_masked_diffusion_equal_prompt_length_v1",
        "completion_only_scoring": True,
        "settings": _settings("smdm"),
        "runner_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "canonical_stage0_json": str(canonical_path.resolve()),
        "canonical_stage0_json_sha256": qwen_runner._sha256(canonical_path),
        "canonical_stage0_checkpoint": checkpoint,
    }, f"SMDM metadata {path}")


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
    }


def _audit_training_structure(run: dict, model_id: str, method: str, path: Path) -> None:
    stages = run["stages"]
    if [stage.get("training", {}).get("steps") for stage in stages] != [1000, 1000, 1000]:
        raise ValueError(f"training step counts differ: {path}")
    anchor_counts = [
        [entry.get("count") for entry in stage.get("anchor_manifest", [])]
        for stage in stages
    ]
    expected_anchors = [[], [], []] if method == "seq" else [[], [64], [64, 64]]
    if anchor_counts != expected_anchors:
        raise ValueError(f"anchor counts differ: {path}")
    if model_id.startswith("smdm"):
        expected_replay = [0, 0, 0] if method == "seq" else [0, 64, 128]
        if [stage.get("generated_replay_count") for stage in stages] != expected_replay:
            raise ValueError(f"SMDM replay counts differ: {path}")


def summarize(root: Path) -> dict:
    _, raw_evaluation = v1.old_runner._raw_gsm()
    expected_map = {
        row["source_index"]: {
            "example_id": row["example_id"],
            "question": row["question"],
            "target": row["target"],
        }
        for row in raw_evaluation
    }
    if sorted(expected_map) != list(range(1319)):
        raise ValueError("vendored GSM8K indices differ")
    raw_by_index = {row["source_index"]: row for row in raw_evaluation}
    expected_list = [raw_by_index[index] for index in range(1319)]
    runs = {}
    cells = []
    canonical_cells = []
    for model_id, display_name in MODELS:
        if model_id.startswith("smdm"):
            for seed in SEEDS:
                canonical_path = _canonical_path(root, model_id, seed)
                if not canonical_path.is_file():
                    raise FileNotFoundError(f"missing canonical stage0: {canonical_path}")
                canonical = json.loads(canonical_path.read_text())
                checkpoint = _audit_canonical(
                    canonical, canonical_path, model_id, seed, expected_list
                )
                seq_path = _endpoint_path(root, model_id, "seq", seed)
                cagd_path = _endpoint_path(root, model_id, "cagd", seed)
                paired = pair_audit.audit(canonical_path, seq_path, cagd_path)
                canonical_cells.append({
                    "model_id": model_id,
                    "model": display_name,
                    "seed": seed,
                    "status": "validated",
                    "path": str(canonical_path),
                    "sha256": qwen_runner._sha256(canonical_path),
                    "checkpoint": checkpoint,
                    "paired_audit": paired,
                })
        for method in METHODS:
            for seed in SEEDS:
                path = _endpoint_path(root, model_id, method, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"missing formal endpoint: {path}")
                run = json.loads(path.read_text())
                if model_id == "qwen3_0.6b":
                    v1._audit_old(run, method, seed, path)
                elif model_id.startswith("qwen"):
                    v1._audit_new(run, model_id, method, seed, path)
                else:
                    canonical_path = _canonical_path(root, model_id, seed)
                    canonical = json.loads(canonical_path.read_text())
                    checkpoint_path = Path(canonical["canonical_checkpoint"]["path"])
                    checkpoint = smdm_runner._file_descriptor(checkpoint_path)
                    _audit_smdm_endpoint(
                        run, path, model_id, method, seed, canonical_path, checkpoint
                    )
                v1._audit_stages(run, path)
                _audit_training_structure(run, model_id, method, path)
                v1._audit_benchmark(run, 0, path, expected_map)
                v1._audit_benchmark(run, -1, path, expected_map)
                v1._audit_summary(run, path)
                runs[model_id, method, seed] = run
                cells.append({
                    "model_id": model_id,
                    "model": display_name,
                    "method": method,
                    "seed": seed,
                    "status": "validated",
                    "path": str(path),
                    "sha256": qwen_runner._sha256(path),
                })
    groups = {}
    paired_results = {}
    for model_id, display_name in MODELS:
        groups[model_id] = {"display_name": display_name, "methods": {}}
        for method in METHODS:
            model_runs = [runs[model_id, method, seed] for seed in SEEDS]
            metrics = {
                "after_gsm8k_exact_match": [
                    run["summary"]["gsm8k_exact_match_when_learned"] for run in model_runs
                ],
                "final_exact_match": [
                    run["summary"]["gsm8k_exact_match_final"] for run in model_runs
                ],
                "retention_change": [
                    run["summary"]["gsm8k_retention_change"] for run in model_runs
                ],
                "final_format_rate": [
                    run["stages"][-1]["benchmark"]["delimiter_rate"] for run in model_runs
                ],
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
        paired_results[model_id] = {
            "cagd_minus_sequential_final_exact_match": _mean_sem(differences),
            "all_seeds_favor_cagd": all(value > 0 for value in differences),
        }
        for seed in SEEDS:
            seq_stage0 = runs[model_id, "seq", seed]["stages"][0]
            cagd_stage0 = runs[model_id, "cagd", seed]["stages"][0]
            if model_id.startswith("smdm"):
                if seq_stage0 != cagd_stage0:
                    raise ValueError(f"paired stage0 differs: {model_id}, seed {seed}")
            else:
                seq_records = {
                    row["source_index"]: row for row in seq_stage0["benchmark"]["records"]
                }
                cagd_records = {
                    row["source_index"]: row for row in cagd_stage0["benchmark"]["records"]
                }
                if seq_records != cagd_records:
                    raise ValueError(
                        f"paired post-GSM8K outputs differ: {model_id}, seed {seed}"
                    )
    return {
        "schema_version": 2,
        "status": "ok",
        "validated_run_count": len(cells),
        "new_run_count": len(cells) - 6,
        "reused_run_count": 6,
        "validated_canonical_count": len(canonical_cells),
        "benchmark_count": 1319,
        "qwen_runner_sha256": qwen_runner._sha256(Path(qwen_runner.__file__)),
        "qwen_protocol_sha256": qwen_runner._sha256(qwen_runner.PROTOCOL),
        "smdm_runner_sha256": qwen_runner._sha256(Path(smdm_runner.__file__)),
        "smdm_protocol_sha256": qwen_runner._sha256(smdm_runner.PROTOCOL),
        "models": groups,
        "paired": paired_results,
        "canonical_cells": canonical_cells,
        "cells": cells,
    }


def _value(metric: dict, percent: bool = False) -> str:
    scale = 100 if percent else 1
    return f"{scale * metric['mean']:.2f} +/- {scale * metric['sem']:.2f}"


def _markdown(summary: dict) -> str:
    lines = [
        "# Five-model GSM8K behavioral-retention results (paired stage 0)",
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
        lines.append(
            f"- {display_name}: "
            f"{_value(item['cagd_minus_sequential_final_exact_match'], True)} pp; "
            f"all seeds favor CAGD: {item['all_seeds_favor_cagd']}."
        )
    lines += [
        "", "## Audit", "",
        f"Validated {summary['validated_run_count']} endpoints and "
        f"{summary['validated_canonical_count']} immutable SMDM prefixes; "
        f"{summary['reused_run_count']} Qwen3-0.6B endpoints were reused.", "",
    ]
    return "\n".join(lines)


def _self_check() -> None:
    stats = _mean_sem([0.1, 0.2, 0.3])
    assert abs(stats["mean"] - 0.2) < 1e-12
    assert len(MODELS) == 5 and len(METHODS) == 2 and len(SEEDS) == 3
    observed = {"status": "ok", "formal": True}
    _require_fields(observed, observed, "synthetic")
    tampered = dict(observed, formal=False)
    try:
        _require_fields(tampered, observed, "tampered")
    except ValueError:
        pass
    else:
        raise AssertionError("tampered provenance was accepted")
    pair_audit._self_check()
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/cagd_gsm8k_scale/attempt2")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/cagd_gsm8k_scale/attempt2/summary.json"),
    )
    parser.add_argument(
        "--markdown", type=Path,
        default=Path("report/cagd_gsm8k_scale_attempt2_results.md"),
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.output.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite an existing summary or report")
    summary = summarize(args.root)
    v1._write_new(args.output, json.dumps(summary, indent=2) + "\n")
    v1._write_new(args.markdown, _markdown(summary))
    print(json.dumps({
        "status": "ok", "output": str(args.output), "markdown": str(args.markdown),
    }, indent=2))


if __name__ == "__main__":
    main()

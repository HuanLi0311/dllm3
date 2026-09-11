#!/usr/bin/env python3
"""Audit and summarize the paired Qwen3-0.6B order extension."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVERSE_ROOT = ROOT / "runs/qwen_order_reverse/reverse"
FORWARD_ROOT = ROOT / "release_evidence/public/runs/qwen_continual_scale/formal"
OUTPUT = ROOT / "runs/qwen_order_reverse/summary.json"
SEEDS = (3407, 3408, 3409)
METHODS = ("seq", "gd")
TASK = "p2d_20-23"
FORWARD_TASKS = ("d2p_8-11", "p2d_12-15", "d2p_16-19", TASK)
REVERSE_TASKS = tuple(reversed(FORWARD_TASKS))


def _load(path: Path) -> dict:
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with handle:
        return json.load(handle)


def _mean_sem(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite or empty endpoint")
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
    }


def _forward_path(method: str, seed: int) -> Path:
    return FORWARD_ROOT / f"qwen3_0.6b_{method}_s{seed}.json.gz"


def _reverse_path(method: str, seed: int) -> Path:
    return REVERSE_ROOT / method / f"s{seed}.json"


def _check_pair(forward: dict, reverse: dict, method: str, seed: int) -> None:
    fm, rm = forward["metadata"], reverse["metadata"]
    if fm["method"] != method or rm["method"] != method or fm["seed"] != seed or rm["seed"] != seed:
        raise ValueError(f"method/seed mismatch for {method} seed {seed}")
    if tuple(fm["task_sequence"]) != FORWARD_TASKS or tuple(rm["task_sequence"]) != REVERSE_TASKS:
        raise ValueError(f"task-order mismatch for {method} seed {seed}")
    if Path(fm["model"]).name != Path(rm["model"]).name or forward["data_sha256"] != reverse["data_sha256"]:
        raise ValueError(f"model/data mismatch for {method} seed {seed}")
    for key in (
        "model_label", "trainable", "trainable_parameter_count", "steps_per_task",
        "batch_size", "eval_batch_size", "replay_per_task", "max_new_tokens",
        "distill_weight", "distill_temperature", "ewc_lambda", "lr", "clip",
    ):
        if fm[key] != rm[key]:
            raise ValueError(f"{key} mismatch for {method} seed {seed}")
    if fm["task_sha256"] != rm["task_sha256"]:
        raise ValueError(f"encoded task mismatch for {method} seed {seed}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    cells = {}
    for method in METHODS:
        for seed in SEEDS:
            forward = _load(_forward_path(method, seed))
            reverse = _load(_reverse_path(method, seed))
            _check_pair(forward, reverse, method, seed)
            cells[(method, seed)] = (forward, reverse)

    aggregate = {}
    for method in METHODS:
        rows = [cells[(method, seed)] for seed in SEEDS]
        forward_fixed = [row[0]["stages"][-1]["metrics"][TASK]["loss"] for row in rows]
        reverse_fixed = [row[1]["stages"][-1]["metrics"][TASK]["loss"] for row in rows]
        reverse_learned = [row[1]["stages"][0]["metrics"][TASK]["loss"] for row in rows]
        aggregate[method] = {
            "forward_fixed_task_final_loss": _mean_sem(forward_fixed),
            "reverse_fixed_task_final_loss": _mean_sem(reverse_fixed),
            "reverse_minus_forward_fixed_task_loss": _mean_sem([
                reverse_value - forward_value
                for forward_value, reverse_value in zip(forward_fixed, reverse_fixed)
            ]),
            "reverse_fixed_task_loss_when_learned": _mean_sem(reverse_learned),
            "reverse_fixed_task_forgetting": _mean_sem([
                final - learned for final, learned in zip(reverse_fixed, reverse_learned)
            ]),
            "reverse_final_average_loss": _mean_sem([
                row[1]["summary"]["final_average_loss"] for row in rows
            ]),
            "reverse_past_task_forgetting": _mean_sem([
                row[1]["summary"]["past_task_forgetting"] for row in rows
            ]),
            "reverse_final_task_loss": _mean_sem([
                row[1]["summary"]["final_task_losses"][-1] for row in rows
            ]),
        }
    aggregate["cagd_minus_sequential"] = {
        key: _mean_sem([
            aggregate["gd"][key]["values"][index] - aggregate["seq"][key]["values"][index]
            for index in range(len(SEEDS))
        ])
        for key in (
            "reverse_fixed_task_final_loss", "reverse_fixed_task_forgetting",
            "reverse_final_average_loss", "reverse_past_task_forgetting",
            "reverse_final_task_loss",
        )
    }
    result = {
        "schema_version": 1,
        "status": "ok",
        "protocol": "qwen_order_reverse_v1",
        "model": "Qwen3-0.6B",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "forward_tasks": list(FORWARD_TASKS),
        "reverse_tasks": list(REVERSE_TASKS),
        "fixed_task": TASK,
        "aggregate": aggregate,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(OUTPUT), "cells": 6}))


if __name__ == "__main__":
    main()

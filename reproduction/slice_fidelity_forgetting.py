#!/usr/bin/env python3
"""Paired slice-level Fisher fidelity and continual-forgetting experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.smdm_backend import (  # noqa: E402
    answer_token_accuracy,
    flat_parameters,
    load_model,
    set_seed,
    trainable_parameters,
)
from reproduction import smdm_factual as factual  # noqa: E402
from reproduction import smdm_transfer as transfer  # noqa: E402


PROTOCOL = ROOT / "report/slice_fidelity_forgetting_protocol.md"
SEEDS = (3407, 3408, 3409)
SLICES = {
    f"layer{layer}_{module}": (
        f"transformer.h.{layer}.norm_{'1' if module == 'attention' else '2'}.weight",
        f"transformer.h.{layer}.{'attn' if module == 'attention' else 'mlp' }.",
    )
    for layer in (0, 8, 17)
    for module in ("attention", "mlp")
}
TRACE_CHUNK = 1_048_576


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_sha256(rows) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _slice_parameters(model, name: str) -> tuple[list[str], list[torch.nn.Parameter]]:
    exact, prefix = SLICES[name]
    selected = [(key, value) for key, value in model.named_parameters() if key == exact or key.startswith(prefix)]
    if not selected:
        raise ValueError(f"slice {name!r} selected no parameters")
    return [key for key, _ in selected], [value for _, value in selected]


def _embed_slice(
    local: torch.Tensor,
    all_parameters: list[torch.nn.Parameter],
    slice_parameters: list[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    selected = {id(parameter) for parameter in slice_parameters}
    result = torch.zeros(sum(parameter.numel() for parameter in all_parameters), device=device)
    global_offset = local_offset = 0
    for parameter in all_parameters:
        width = parameter.numel()
        if id(parameter) in selected:
            result[global_offset : global_offset + width].copy_(
                local[local_offset : local_offset + width], non_blocking=True
            )
            local_offset += width
        global_offset += width
    if local_offset != local.numel():
        raise AssertionError(f"embedded {local_offset} of {local.numel()} slice values")
    return result


def _collect_test_gradients(model, rows, parameters, pad_id, device, seed, args) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    vectors = torch.empty((len(rows), sum(parameter.numel() for parameter in parameters)))
    model.eval()
    for index, row in enumerate(rows):
        loss = transfer._sft_losses(
            model, [row], pad_id, device, generator, args.mask_min, args.mask_max
        )[0]
        vectors[index].copy_(transfer._flat_gradient(loss, parameters))
        model.zero_grad(set_to_none=True)
        print(f"heldout_gradient={index + 1}/{len(rows)}", flush=True)
    return vectors


def _geometry(fisher: dict, test: torch.Tensor, chunk_size: int = TRACE_CHUNK) -> dict:
    direction = fisher["direction"].double()
    diagonal = fisher["diagonal"].double()
    coefficient = float(fisher["coefficient"])
    count, width = test.shape
    gram = torch.zeros((count, count), dtype=torch.float64)
    test_diagonal = torch.zeros(width, dtype=torch.float64)
    test_projections = torch.zeros(count, dtype=torch.float64)
    for left in range(0, width, chunk_size):
        right = min(left + chunk_size, width)
        block = test[:, left:right].double()
        gram.addmm_(block, block.T)
        test_diagonal[left:right] = block.square().mean(dim=0)
        test_projections.addmv_(block, direction[left:right])
    test_projection_sq = float(test_projections.square().mean())
    fisher_norm_sq = (gram / count).square().sum()
    direction_norm_sq = direction.square().sum()
    rank1_error_sq = (
        fisher_norm_sq
        - 2 * coefficient * test_projection_sq
        + coefficient**2 * direction_norm_sq**2
    ).clamp_min(0)
    diagonal_error_sq = (
        fisher_norm_sq
        - 2 * torch.dot(test_diagonal, diagonal)
        + diagonal.square().sum()
    ).clamp_min(0)
    rank1_error = torch.sqrt(rank1_error_sq / fisher_norm_sq)
    diagonal_error = torch.sqrt(diagonal_error_sq / fisher_norm_sq)
    result = {
        "test_examples": count,
        "parameter_count": width,
        "test_fisher_frobenius": float(torch.sqrt(fisher_norm_sq)),
        "rank1_relative_frobenius_error": float(rank1_error),
        "diagonal_relative_frobenius_error": float(diagonal_error),
        "score_log_diagonal_over_rank1": float(torch.log(diagonal_error / rank1_error)),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError(f"non-finite geometry: {result}")
    return result


def _matched_lambdas(fisher: dict) -> dict:
    direction_norm_sq = float(fisher["direction"].double().square().sum())
    diagonal_trace = float(fisher["diagonal"].double().sum())
    coefficient = float(fisher["coefficient"])
    rank1_trace = coefficient * direction_norm_sq
    if not all(math.isfinite(value) and value > 0 for value in (rank1_trace, diagonal_trace)):
        raise ValueError("Fisher slice has a non-positive trace")
    target = 1_000.0 * diagonal_trace
    lambdas = {"rank1": target / rank1_trace, "diagonal": 1_000.0}
    checks = {
        "rank1": lambdas["rank1"] * rank1_trace,
        "diagonal": lambdas["diagonal"] * diagonal_trace,
    }
    if not math.isclose(checks["rank1"], checks["diagonal"], rel_tol=1e-12):
        raise AssertionError(f"weighted traces differ: {checks}")
    return {
        "direction_norm_sq": direction_norm_sq,
        "rank1_trace": rank1_trace,
        "diagonal_trace": diagonal_trace,
        "weighted_trace_target": target,
        "lambdas": lambdas,
        "weighted_trace_checks": checks,
    }


def _metrics(model, task, pad_id, device, args) -> dict:
    return {
        "loss": transfer._evaluate_loss(
            model,
            task["eval_loss"],
            pad_id,
            device,
            args,
            args.seed + 81_001 + 100 * task["task_index"],
        ),
        "answer_token_accuracy": answer_token_accuracy(
            model, task["eval_loss"], device, args.eval_batch_size, pad_id
        ),
    }


def _update_fidelity(delta, fisher, test) -> dict:
    true = float((test @ delta).double().square().mean())
    rank1 = float(fisher["coefficient"]) * float(torch.dot(fisher["direction"], delta).square())
    diagonal = float(torch.dot(fisher["diagonal"].double(), delta.double().square()))
    denominator = max(abs(true), 1e-30)
    rank1_error = abs(rank1 - true) / denominator
    diagonal_error = abs(diagonal - true) / denominator
    return {
        "heldout_energy": true,
        "rank1_predicted_energy": rank1,
        "diagonal_predicted_energy": diagonal,
        "rank1_relative_error": rank1_error,
        "diagonal_relative_error": diagonal_error,
        "score_log_diagonal_over_rank1": math.log(max(diagonal_error, 1e-30) / max(rank1_error, 1e-30)),
        "slice_displacement_l2": float(delta.double().norm()),
    }


def _assert_finite(value, path="result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def run(args) -> dict:
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    if args.seed not in SEEDS or args.slice not in SLICES:
        raise ValueError("cell is outside the frozen grid")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("set PYTHONNOUSERSITE=1")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = factual._load_tasks(args, tokenizer)
    if [task["name"] for task in tasks] != ["d2p_8-11", "p2d_12-15"]:
        raise ValueError("unexpected task sequence")
    calibration_prompts = {row["prompt"] for row in tasks[0]["fisher_raw"]}
    test_prompts = {row["prompt"] for row in tasks[0]["eval_raw"]}
    if calibration_prompts & test_prompts:
        raise ValueError("calibration and test prompts overlap")

    source_hashes = {
        "runner": _sha256(Path(__file__)),
        "protocol": _sha256(PROTOCOL),
        "smdm_factual": _sha256(ROOT / "reproduction/smdm_factual.py"),
        "smdm_transfer": _sha256(ROOT / "reproduction/smdm_transfer.py"),
        "smdm_backend": _sha256(ROOT / "reproduction/smdm_backend.py"),
    }
    input_hashes = {
        "checkpoint": _sha256(args.checkpoint),
        "task_a_train": _records_sha256(tasks[0]["train_raw"]),
        "task_a_fisher": _records_sha256(tasks[0]["fisher_raw"]),
        "task_a_test": _records_sha256(tasks[0]["eval_raw"]),
        "task_b_train": _records_sha256(tasks[1]["train_raw"]),
    }

    model = load_model(args, device)
    all_parameters = trainable_parameters(model, "all")
    parameter_count = sum(parameter.numel() for parameter in all_parameters)
    if parameter_count != 219_050_496:
        raise ValueError(f"unexpected parameter count: {parameter_count}")
    slice_names, slice_parameters = _slice_parameters(model, args.slice)
    print(f"slice={args.slice} slice_parameters={sum(p.numel() for p in slice_parameters):,}", flush=True)

    task_a_training = factual._train_stage(
        model, tasks[0]["train"], all_parameters, pad_id, device, args, args.seed + 1000
    )
    task_a_metrics = _metrics(model, tasks[0], pad_id, device, args)
    fisher, fisher_stats = transfer._estimate_mean_and_diagonal_fisher(
        model, tasks[0]["fisher"], slice_parameters, pad_id, device, args
    )
    test_gradients = _collect_test_gradients(
        model, tasks[0]["eval_loss"], slice_parameters, pad_id, device, args.seed + 4101, args
    )
    geometry = _geometry(fisher, test_gradients)
    matching = _matched_lambdas(fisher)
    slice_anchor = flat_parameters(slice_parameters).detach().float().cpu().clone()

    prompts, replay_manifest = factual._replay_prompts(tasks, 1, 64)
    replay = transfer._generate_replay(model, tokenizer, prompts, device, args)
    expected = {str(fact): 16 for fact in range(8, 12)}
    if len(replay) != 64 or replay_manifest[0]["fact_counts"] != expected:
        raise AssertionError("replay is not balanced 16/16/16/16")
    replay_hash = factual._records_sha256(replay)
    teacher = load_model(args, device)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    reference = flat_parameters(all_parameters).detach().clone()

    methods = {}
    for method in ("rank1", "diagonal"):
        model.load_state_dict(teacher.state_dict())
        model.zero_grad(set_to_none=True)
        set_seed(args.seed + 9000)
        local = fisher["direction"] if method == "rank1" else fisher["diagonal"]
        represented = _embed_slice(local, all_parameters, slice_parameters, device)
        constraint = {"kind": method, "reference": reference, method: represented}
        if method == "rank1":
            constraint["direction"] = constraint.pop("rank1")
            constraint["coefficient"] = float(fisher["coefficient"])
        args.ewc_lambda = matching["lambdas"][method]
        training = factual._train_stage(
            model,
            tasks[1]["train"],
            all_parameters,
            pad_id,
            device,
            args,
            args.seed + 2000,
            teacher=teacher,
            replay_rows=replay,
            replay_objective="soft",
            constraints=[constraint],
        )
        final_metrics = {task["name"]: _metrics(model, task, pad_id, device, args) for task in tasks}
        delta = flat_parameters(slice_parameters).detach().float().cpu() - slice_anchor
        summary = {
            "past_task_forgetting": final_metrics[tasks[0]["name"]]["loss"] - task_a_metrics["loss"],
            "final_average_loss": sum(final_metrics[task["name"]]["loss"] for task in tasks) / 2,
            "final_task_loss": final_metrics[tasks[1]["name"]]["loss"],
            "final_average_answer_token_accuracy": sum(
                final_metrics[task["name"]]["answer_token_accuracy"] for task in tasks
            ) / 2,
        }
        methods[method] = {
            "ewc_lambda": args.ewc_lambda,
            "training": training,
            "final_metrics": final_metrics,
            "update_direction_fidelity": _update_fidelity(delta, fisher, test_gradients),
            "summary": summary,
        }
        del represented, constraint, delta
        if device.type == "cuda":
            torch.cuda.empty_cache()

    forgetting_preference = (
        methods["diagonal"]["summary"]["past_task_forgetting"]
        - methods["rank1"]["summary"]["past_task_forgetting"]
    )
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "slice_fidelity_forgetting",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": os.uname().nodename,
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "protocol": {
            "seed": args.seed,
            "slice": args.slice,
            "slice_parameter_names": slice_names,
            "slice_parameter_count": sum(parameter.numel() for parameter in slice_parameters),
            "all_parameter_count": parameter_count,
            "task_sequence": [task["name"] for task in tasks],
            "steps_per_task": args.steps_per_task,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "clip": args.clip,
            "calibration_examples": len(tasks[0]["fisher"]),
            "test_examples": len(tasks[0]["eval_loss"]),
            "replay_examples": len(replay),
            "replay_hash_shared_by_methods": replay_hash,
        },
        "task_a": {"training": task_a_training, "metrics": task_a_metrics},
        "fisher": fisher_stats,
        "heldout_geometry": geometry,
        "trace_matching": matching,
        "replay": {"manifest": replay_manifest, "generated_rows_sha256": replay_hash},
        "methods": methods,
        "summary": {
            "fidelity_score": geometry["score_log_diagonal_over_rank1"],
            "forgetting_preference": forgetting_preference,
            "final_loss_preference": (
                methods["diagonal"]["summary"]["final_average_loss"]
                - methods["rank1"]["summary"]["final_average_loss"]
            ),
            "winner_agrees": (geometry["score_log_diagonal_over_rank1"] > 0) == (forgetting_preference > 0),
        },
    }
    _assert_finite(result)
    current_hashes = {
        "runner": _sha256(Path(__file__)),
        "protocol": _sha256(PROTOCOL),
        "smdm_factual": _sha256(ROOT / "reproduction/smdm_factual.py"),
        "smdm_transfer": _sha256(ROOT / "reproduction/smdm_transfer.py"),
        "smdm_backend": _sha256(ROOT / "reproduction/smdm_backend.py"),
    }
    if current_hashes != source_hashes:
        raise RuntimeError("source changed while the cell was running")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        rank = (left + right - 1) / 2 + 1
        for index in order[left:right]:
            ranks[index] = rank
        left = right
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else float("nan")


def summarize(run_root: Path, output: Path, permutations: int = 100_000) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing existing summary: {output}")
    rows = []
    for seed in SEEDS:
        for slice_name in SLICES:
            path = run_root / "cells" / slice_name / f"s{seed}.json"
            payload = json.loads(path.read_text())
            if payload.get("status") != "ok":
                raise ValueError(f"invalid cell: {path}")
            rows.append({"seed": seed, "slice": slice_name, **payload["summary"], "source": str(path.resolve())})
    fidelity = [float(row["fidelity_score"]) for row in rows]
    forgetting = [float(row["forgetting_preference"]) for row in rows]
    observed = _correlation(_ranks(fidelity), _ranks(forgetting))
    rng = random.Random(20260926)
    extreme = 0
    groups = [[index for index, row in enumerate(rows) if row["seed"] == seed] for seed in SEEDS]
    for _ in range(permutations):
        shuffled = forgetting.copy()
        for indices in groups:
            values = [shuffled[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                shuffled[index] = value
        extreme += abs(_correlation(_ranks(fidelity), _ranks(shuffled))) >= abs(observed)
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "slice_fidelity_forgetting",
        "rows": rows,
        "summary": {
            "cells": len(rows),
            "spearman_fidelity_vs_forgetting_preference": observed,
            "pearson_fidelity_vs_forgetting_preference": _correlation(fidelity, forgetting),
            "within_seed_permutation_draws": permutations,
            "within_seed_permutation_p_two_sided": (extreme + 1) / (permutations + 1),
            "winner_agreement_count": sum(bool(row["winner_agrees"]) for row in rows),
            "winner_agreement_fraction": statistics.fmean(bool(row["winner_agrees"]) for row in rows),
            "per_seed_spearman": {
                str(seed): _correlation(
                    _ranks([row["fidelity_score"] for row in rows if row["seed"] == seed]),
                    _ranks([row["forgetting_preference"] for row in rows if row["seed"] == seed]),
                )
                for seed in SEEDS
            },
        },
    }
    _assert_finite(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _self_check() -> None:
    generator = torch.Generator().manual_seed(7)
    calibration = torch.randn(7, 5, generator=generator)
    test = torch.randn(9, 5, generator=generator)
    mean = calibration.mean(dim=0)
    direction = mean / mean.norm()
    coefficient = (calibration @ direction).square().mean()
    diagonal = calibration.square().mean(dim=0)
    actual = _geometry(
        {"direction": direction, "coefficient": coefficient, "diagonal": diagonal}, test, chunk_size=2
    )
    ftest = test.double().T @ test.double() / len(test)
    rank1 = coefficient.double() * torch.outer(direction.double(), direction.double())
    diag = torch.diag(diagonal.double())
    expected_rank1 = torch.linalg.matrix_norm(ftest - rank1) / torch.linalg.matrix_norm(ftest)
    expected_diag = torch.linalg.matrix_norm(ftest - diag) / torch.linalg.matrix_norm(ftest)
    assert math.isclose(actual["rank1_relative_frobenius_error"], float(expected_rank1), rel_tol=1e-10)
    assert math.isclose(actual["diagonal_relative_frobenius_error"], float(expected_diag), rel_tol=1e-10)
    matched = _matched_lambdas({"direction": direction, "coefficient": coefficient, "diagonal": diagonal})
    assert math.isclose(matched["weighted_trace_checks"]["rank1"], matched["weighted_trace_checks"]["diagonal"])
    assert _correlation(_ranks([3, 1, 2]), _ranks([30, 10, 20])) == 1.0
    print(json.dumps({"self_check": "ok", "slices": list(SLICES)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", choices=tuple(SLICES), default="layer0_attention")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3407)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "third_party/SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    args.model = 170
    args.start_direction = "d2p"
    args.order = "forward"
    args.group_start = 8
    args.tasks = 2
    args.group_count = 4
    args.fisher_per_fact = 10
    args.max_length = 128
    args.batch_size = 4
    args.eval_batch_size = 4
    args.eval_mc_samples = 32
    args.steps_per_task = 1000
    args.lr = 5e-5
    args.clip = 1.0
    args.mask_min = 1e-3
    args.mask_max = 1.0
    args.replay_per_task = 64
    args.replay_steps = 32
    args.replay_length = 52
    args.replay_cfg = 0.8
    args.replay_temperature = 0.0
    args.distill_weight = 1.0
    args.distill_temperature = 1.0
    args.ewc_lambda = 0.0
    args.generation_seed = args.seed
    args.generation_batch_size = 1
    args.reverse_steps = 32
    args.reverse_length = 52
    args.reverse_cfg = 0.8
    args.reverse_temperature = 0.0
    args.show_predictions = False
    args.record_step_loss = False
    args.record_eval_loss_every = 0
    args.final_protocol = False
    args.fresh_protocol = False
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

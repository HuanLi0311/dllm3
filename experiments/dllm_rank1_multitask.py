#!/usr/bin/env python3
"""DLLM continual learning with soft, hard-generated, and real replay."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_benchmark import encode_benchmark_rows  # noqa: E402
from continual_mdm import (  # noqa: E402
    answer_token_accuracy,
    flat_parameters,
    load_model,
    set_seed,
    trainable_parameters,
)
from continual_reverse import fact_rows, measure  # noqa: E402
from dllm_rank1_transfer import (  # noqa: E402
    _diagonal_penalty,
    _distillation_losses,
    _estimate_mean_and_diagonal_fisher,
    _evaluate_loss,
    _generate_replay,
    _overlap_metrics,
    _projection,
    _self_check as _transfer_self_check,
    _sft_losses,
    _split_calibration,
)


FINAL_TASKS = (("d2p", 8), ("p2d", 12), ("d2p", 16), ("p2d", 20))
FRESH_TASKS = (("d2p", 24), ("p2d", 26), ("d2p", 28))
FINAL_FORWARD_METHODS = {"seq", "gd", "rank1", "diagonal", "rank1_gd", "diag_gd", "joint"}
FINAL_REVERSE_METHODS = {"seq", "gd", "rank1_gd", "diag_gd"}
FRESH_METHODS = {"seq", "gd", "rank1_gd", "diag_gd"}
VALIDATION_SUMMARY_SHA256 = "b621509724d73059362bd095d9e67a45cdc690da68d84d0b0a37b2e33dc610e4"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(member.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _records_sha256(rows) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fact_counts(rows) -> dict[str, int]:
    counts = {}
    for row in rows:
        key = str(row["fact_id"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dependency_hashes() -> dict[str, str]:
    names = (
        "continual_benchmark.py",
        "continual_mdm.py",
        "continual_reverse.py",
        "experiments/dllm_rank1_transfer.py",
    )
    return {name: _sha256(ROOT / name) for name in names}


def _sequence_specs(
    start_direction: str,
    group_start: int,
    tasks: int,
    group_count: int,
    order: str = "forward",
) -> list[dict]:
    other = "p2d" if start_direction == "d2p" else "d2p"
    specs = [
        {
            "task_index": index,
            "direction": start_direction if index % 2 == 0 else other,
            "group_start": group_start + index * group_count,
            "group_count": group_count,
        }
        for index in range(tasks)
    ]
    # ponytail: exact reversal is enough to isolate order; no general task-graph parser.
    return specs if order == "forward" else list(reversed(specs))


def _load_tasks(args, tokenizer) -> list[dict]:
    tasks = []
    for spec in _sequence_specs(
        args.start_direction, args.group_start, args.tasks, args.group_count, args.order
    ):
        train_raw = fact_rows(
            args.reverse_dir, spec["direction"], "train", spec["group_start"], spec["group_count"]
        )
        _, fisher_raw = _split_calibration(train_raw, args.fisher_per_fact)
        eval_raw = fact_rows(
            args.reverse_dir, spec["direction"], "test", spec["group_start"], spec["group_count"]
        )
        tasks.append({
            **spec,
            "name": f"{spec['direction']}_{spec['group_start']}-{spec['group_start'] + spec['group_count'] - 1}",
            "train_raw": train_raw,
            "fisher_raw": fisher_raw,
            "eval_raw": eval_raw,
            "train": encode_benchmark_rows(train_raw, tokenizer, args.max_length),
            "fisher": encode_benchmark_rows(fisher_raw, tokenizer, args.max_length),
            "eval_loss": encode_benchmark_rows(
                [{"prompt": row["prompt"], "answer": row["target"]} for row in eval_raw],
                tokenizer,
                args.max_length,
            ),
        })
    return tasks


def _measure_task(model, tokenizer, task, pad_id, device, args) -> dict:
    generation, items = measure(model, tokenizer, task["eval_raw"], args, device)
    generation.update(_overlap_metrics(items))
    return {
        "loss": _evaluate_loss(
            model, task["eval_loss"], pad_id, device, args,
            args.seed + 81_001 + 100 * task["task_index"],
        ),
        "answer_token_accuracy": answer_token_accuracy(
            model, task["eval_loss"], device, args.eval_batch_size, pad_id
        ),
        "generation": generation,
        "generation_items": items,
    }


def _train_stage(
    model,
    rows,
    parameters,
    pad_id,
    device,
    args,
    seed,
    teacher=None,
    replay_rows=None,
    replay_objective=None,
    constraints=None,
) -> dict:
    started = time.monotonic()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    current_rng = random.Random(seed)
    replay_rng = random.Random(seed + 303)
    current_generator = torch.Generator(device=device).manual_seed(seed + 101)
    replay_generator = torch.Generator(device=device).manual_seed(seed + 202)
    totals = {
        "current": 0.0,
        "distill": 0.0,
        "hard_replay": 0.0,
        "penalty": 0.0,
        "total": 0.0,
    }
    penalty_max = 0.0
    gradient_norm_max = 0.0
    clipped_steps = 0
    constraints = constraints or []
    model.train()
    for step in range(args.steps_per_task):
        batch = [rows[current_rng.randrange(len(rows))] for _ in range(args.batch_size)]
        current = _sft_losses(
            model, batch, pad_id, device, current_generator, args.mask_min, args.mask_max
        ).mean()
        distill = torch.zeros((), device=device)
        hard_replay = torch.zeros((), device=device)
        if replay_objective == "soft":
            if teacher is None or not replay_rows:
                raise ValueError("soft replay requires a teacher and replay rows")
            replay_batch = [replay_rows[replay_rng.randrange(len(replay_rows))] for _ in range(args.batch_size)]
            distill = _distillation_losses(
                model, teacher, replay_batch, pad_id, device, replay_generator,
                args.mask_min, args.mask_max, args.distill_temperature,
            ).mean()
        elif replay_objective == "hard":
            if not replay_rows:
                raise ValueError("hard replay requires replay rows")
            replay_batch = [replay_rows[replay_rng.randrange(len(replay_rows))] for _ in range(args.batch_size)]
            hard_replay = _sft_losses(
                model, replay_batch, pad_id, device, replay_generator,
                args.mask_min, args.mask_max,
            ).mean()
        elif replay_objective is not None:
            raise ValueError(f"unknown replay objective: {replay_objective}")
        penalty = torch.zeros((), device=device)
        for constraint in constraints:
            if constraint["kind"] == "rank1":
                projected = _projection(parameters, constraint["reference"], constraint["direction"])
                penalty = penalty + 0.5 * constraint["coefficient"] * projected.square()
            else:
                penalty = penalty + _diagonal_penalty(
                    parameters, constraint["reference"], constraint["diagonal"]
                )
        total = current + args.distill_weight * (distill + hard_replay) + args.ewc_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        clipped_steps += gradient_norm > args.clip
        optimizer.step()
        values = {
            "current": float(current.detach().cpu()),
            "distill": float(distill.detach().cpu()),
            "hard_replay": float(hard_replay.detach().cpu()),
            "penalty": float(penalty.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        for key, value in values.items():
            totals[key] += value
        penalty_max = max(penalty_max, values["penalty"])
        gradient_norm_max = max(gradient_norm_max, gradient_norm)
        if step == 0 or step + 1 == args.steps_per_task or (step + 1) % 100 == 0:
            print(
                f"stage_step={step + 1}/{args.steps_per_task} current={values['current']:.5f} "
                f"distill={values['distill']:.5f} hard_replay={values['hard_replay']:.5f} "
                f"penalty={values['penalty']:.6g}",
                flush=True,
            )
    return {
        "steps": args.steps_per_task,
        "wall_time_seconds": time.monotonic() - started,
        "penalty_max": penalty_max,
        "gradient_norm_max": gradient_norm_max,
        "clip_fraction": clipped_steps / args.steps_per_task,
        **{f"{key}_mean": value / args.steps_per_task for key, value in totals.items()},
        "ewc_loss_mean": args.ewc_lambda * totals["penalty"] / args.steps_per_task,
        "distill_loss_weighted_mean": args.distill_weight * totals["distill"] / args.steps_per_task,
        "hard_replay_loss_weighted_mean": (
            args.distill_weight * totals["hard_replay"] / args.steps_per_task
        ),
    }


def _replay_prompts(tasks: list[dict], seen: int, per_task: int) -> tuple[list[str], list[dict]]:
    prompts = []
    manifest = []
    for task in tasks[:seen]:
        pools = {}
        for row in task["train_raw"]:
            pools.setdefault(int(row["fact_id"]), []).append(row["prompt"])
        fact_ids = sorted(pools)
        quotient, remainder = divmod(per_task, len(fact_ids))
        selected_rows = []
        per_fact_prompt_sha256 = {}
        for index, fact_id in enumerate(fact_ids):
            count = quotient + (index < remainder)
            pool = pools[fact_id]
            selected = [pool[item % len(pool)] for item in range(count)]
            selected_rows.extend({"fact_id": fact_id, "prompt": prompt} for prompt in selected)
            per_fact_prompt_sha256[str(fact_id)] = _records_sha256(selected)
        selected_prompts = [row["prompt"] for row in selected_rows]
        prompts.extend(selected_prompts)
        manifest.append({
            "task": task["name"],
            "count": len(selected_rows),
            "fact_counts": _fact_counts(selected_rows),
            "prompt_sha256": _records_sha256(selected_prompts),
            "selection_sha256": _records_sha256(selected_rows),
            "per_fact_prompt_sha256": per_fact_prompt_sha256,
        })
    return prompts, manifest


def _real_replay_rows(
    tasks: list[dict], seen: int, per_task: int, tokenizer, max_length: int
) -> tuple[list[dict], list[dict]]:
    rows = []
    manifest = []
    for task in tasks[:seen]:
        pools = {}
        for row in task["train_raw"]:
            pools.setdefault(int(row["fact_id"]), []).append(row)
        fact_ids = sorted(pools)
        quotient, remainder = divmod(per_task, len(fact_ids))
        selected = []
        for index, fact_id in enumerate(fact_ids):
            count = quotient + (index < remainder)
            pool = pools[fact_id]
            selected.extend(pool[item % len(pool)] for item in range(count))
        encoded = encode_benchmark_rows(selected, tokenizer, max_length)
        if len(encoded) != len(selected):
            raise ValueError("real replay rows were truncated by max_length")
        rows.extend(encoded)
        manifest.append({
            "task": task["name"],
            "count": len(selected),
            "fact_counts": _fact_counts(selected),
            "prompt_sha256": _records_sha256([row["prompt"] for row in selected]),
            "selection_sha256": _records_sha256([
                {"fact_id": row["fact_id"], "prompt": row["prompt"]} for row in selected
            ]),
            "answer_sha256": _records_sha256([row["answer"] for row in selected]),
        })
    return rows, manifest


def _summary(stages: list[dict], tasks: list[dict]) -> dict:
    final = stages[-1]["metrics"]
    losses = [final[task["name"]]["loss"] for task in tasks]
    learned = [stages[index]["metrics"][task["name"]]["loss"] for index, task in enumerate(tasks)]
    forgetting = [after - before for after, before in zip(losses, learned)]
    return {
        "final_average_loss": sum(losses) / len(losses),
        "past_task_forgetting": sum(forgetting[:-1]) / max(len(forgetting) - 1, 1),
        "final_average_answer_token_accuracy": sum(
            final[task["name"]]["answer_token_accuracy"] for task in tasks
        ) / len(tasks),
        "final_average_target_containment": sum(
            final[task["name"]]["generation"]["accuracy"] for task in tasks
        ) / len(tasks),
        "final_average_exact_match": sum(
            final[task["name"]]["generation"]["strict_accuracy"] for task in tasks
        ) / len(tasks),
        "final_average_rouge_l_f1": sum(
            final[task["name"]]["generation"]["rouge_l_f1"] for task in tasks
        ) / len(tasks),
        "final_task_losses": losses,
        "losses_when_learned": learned,
        "task_forgetting": forgetting,
    }


def _metadata(args, tasks) -> dict:
    locked = args.final_protocol or args.fresh_protocol
    metadata = {
        "protocol": (
            "r16_native_mask_v1" if args.final_protocol
            else "r16_fresh_facts_v1" if args.fresh_protocol
            else "development"
        ),
        "mask_sampling": "independent_bernoulli_allow_empty_v1",
        "minibatch_sampling": "separate_current_replay_rng_v1",
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_dir": str(args.reverse_dir.resolve()),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "model": args.model,
        "order": args.order,
        "sequence": [
            {key: task[key] for key in ("task_index", "name", "direction", "group_start", "group_count")}
            for task in tasks
        ],
        "task_artifacts": [
            {
                "task": task["name"],
                "train_count": len(task["train_raw"]),
                "train_fact_counts": _fact_counts(task["train_raw"]),
                "train_sha256": _records_sha256(task["train_raw"]),
                "fisher_count": len(task["fisher_raw"]),
                "fisher_fact_counts": _fact_counts(task["fisher_raw"]),
                "fisher_sha256": _records_sha256(task["fisher_raw"]),
                "eval_count": len(task["eval_raw"]),
                "eval_fact_counts": _fact_counts(task["eval_raw"]),
                "eval_sha256": _records_sha256(task["eval_raw"]),
            }
            for task in tasks
        ],
        "fisher_source": "task_training_examples",
        "fisher_per_fact": args.fisher_per_fact,
        "trainable": args.trainable,
        "steps_per_task": args.steps_per_task,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_mc_samples": args.eval_mc_samples,
        "clip": args.clip,
        "lr": args.lr,
        "mask_min": args.mask_min,
        "mask_max": args.mask_max,
        "replay_per_task": args.replay_per_task,
        "replay_sampling": "balanced_by_fact_in_source_order",
        "replay_source": (
            "teacher_generated" if args.method in ("gd", "cagd", "hard_replay", "rank1_gd", "diag_gd")
            else "stored_real" if args.method == "real_replay"
            else "none"
        ),
        "replay_objective": (
            "teacher_kl" if args.method in ("gd", "cagd", "rank1_gd", "diag_gd")
            else "hard_cross_entropy" if args.method in ("hard_replay", "real_replay")
            else "none"
        ),
        "distill_weight": args.distill_weight,
        "ewc_lambda": args.ewc_lambda,
        "seed": args.seed,
        "generation_seed": args.generation_seed,
    }
    if locked:
        validation = json.loads(args.validation_summary.read_text())
        metadata.update({
            "validation_summary": str(args.validation_summary.resolve()),
            "validation_summary_sha256": _sha256(args.validation_summary),
            "selected_lambdas": {
                method: validation["selected"][method]["ewc_lambda"]
                for method in ("rank1_gd", "diag_gd")
            },
        })
    return metadata


def _validate_locked_protocol(args, tasks, selected_lambdas=None) -> None:
    if args.final_protocol == args.fresh_protocol:
        raise ValueError("select exactly one locked protocol")
    task_spec = FINAL_TASKS if args.final_protocol else FRESH_TASKS
    expected = list(task_spec if args.order == "forward" else reversed(task_spec))
    actual = [(task["direction"], task["group_start"]) for task in tasks]
    errors = []
    if actual != expected:
        errors.append(f"task sequence {actual!r} != {expected!r}")
    expected_values = {
        "start_direction": "d2p",
        "group_start": 8 if args.final_protocol else 24,
        "tasks": 4 if args.final_protocol else 3,
        "group_count": 4 if args.final_protocol else 2,
        "fisher_per_fact": 10, "steps_per_task": 1000, "batch_size": 4,
        "eval_batch_size": 4, "eval_mc_samples": 32, "replay_per_task": 64,
        "replay_steps": 32, "replay_length": 52, "replay_cfg": 0.8,
        "replay_temperature": 0.0, "reverse_steps": 32, "reverse_length": 52,
        "reverse_cfg": 0.8, "reverse_temperature": 0.0,
        "generation_batch_size": 1, "max_length": 128, "model": 170,
        "mask_min": 1e-3, "mask_max": 1.0, "trainable": "all",
        "distill_weight": 1.0, "distill_temperature": 1.0,
        "lr": 5e-5, "clip": 1.0,
    }
    for name, expected_value in expected_values.items():
        if getattr(args, name) != expected_value:
            errors.append(f"{name}={getattr(args, name)!r} != {expected_value!r}")
    allowed = (
        FINAL_FORWARD_METHODS if args.final_protocol and args.order == "forward"
        else FINAL_REVERSE_METHODS if args.final_protocol
        else FRESH_METHODS
    )
    if args.method not in allowed:
        errors.append(f"method {args.method!r} is not allowed for {args.order}")
    if args.seed not in (3407, 3408, 3409):
        errors.append(f"unexpected seed {args.seed}")
    if args.generation_seed != args.seed:
        errors.append(f"generation_seed={args.generation_seed} != seed={args.seed}")
    if selected_lambdas is None:
        if _sha256(args.validation_summary) != VALIDATION_SUMMARY_SHA256:
            raise ValueError("validation summary does not match the frozen R16 artifact")
        validation = json.loads(args.validation_summary.read_text())
        if validation.get("protocol") != "r16_native_mask_validation_v1":
            raise ValueError("invalid validation summary protocol")
        selected_lambdas = {
            method: validation["selected"][method]["ewc_lambda"]
            for method in ("rank1_gd", "diag_gd")
        }
        if any(value not in (1e3, 1e4, 1e5, 1e6, 1e7) for value in selected_lambdas.values()):
            raise ValueError("selected lambda is outside the audited validation grid")
    expected_lambda = (
        selected_lambdas["rank1_gd"] if args.method in ("rank1", "rank1_gd")
        else selected_lambdas["diag_gd"] if args.method in ("diagonal", "diag_gd")
        else 0.0
    )
    if args.ewc_lambda != expected_lambda:
        errors.append(f"ewc_lambda={args.ewc_lambda!r} != selected {expected_lambda!r}")
    for task in tasks:
        facts = set(range(task["group_start"], task["group_start"] + args.group_count))
        train_counts = {str(fact): 30 for fact in facts}
        eval_counts = {str(fact): 10 for fact in facts}
        if len(task["train_raw"]) != 30 * args.group_count or _fact_counts(task["train_raw"]) != train_counts:
            errors.append(f"{task['name']} training rows/facts mismatch")
        if len(task["fisher_raw"]) != 10 * args.group_count or _fact_counts(task["fisher_raw"]) != eval_counts:
            errors.append(f"{task['name']} Fisher rows/facts mismatch")
        if len(task["eval_raw"]) != 10 * args.group_count or _fact_counts(task["eval_raw"]) != eval_counts:
            errors.append(f"{task['name']} evaluation rows/facts mismatch")
    if args.replay_per_task % args.group_count:
        errors.append("replay_per_task must divide evenly across task facts")
    if errors:
        raise ValueError("locked protocol mismatch: " + "; ".join(errors))


def run(args) -> dict:
    started = time.monotonic()
    set_seed(args.seed)
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = _load_tasks(args, tokenizer)
    if args.final_protocol or args.fresh_protocol:
        _validate_locked_protocol(args, tasks)
    source_sha256 = _sha256(Path(__file__))
    dependency_sha256 = _dependency_hashes()
    metadata = _metadata(args, tasks)
    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"method={args.method} trainable_parameters={sum(p.numel() for p in parameters):,}", flush=True)

    if args.method == "joint":
        joint_rows = [row for task in tasks for row in task["train"]]
        original_steps = args.steps_per_task
        args.steps_per_task *= len(tasks)
        training = _train_stage(
            model, joint_rows, parameters, pad_id, device, args, args.seed + 1
        )
        args.steps_per_task = original_steps
        metrics = {
            task["name"]: _measure_task(model, tokenizer, task, pad_id, device, args)
            for task in tasks
        }
        losses = [metrics[task["name"]]["loss"] for task in tasks]
        result = {
            "schema_version": 1,
            "status": "ok",
            "experiment": "dllm_rank1_multitask",
            "created_utc": _utc_now(),
            "wall_time_seconds": time.monotonic() - started,
            "host": __import__("os").uname().nodename,
            "source_sha256": source_sha256,
            "dependency_sha256": dependency_sha256,
            "metadata": metadata,
            "joint_training": training,
            "final_metrics": metrics,
            "summary": {
                "final_average_loss": sum(losses) / len(losses),
                "final_average_answer_token_accuracy": sum(
                    metrics[task["name"]]["answer_token_accuracy"] for task in tasks
                ) / len(tasks),
                "final_average_target_containment": sum(
                    metrics[task["name"]]["generation"]["accuracy"] for task in tasks
                ) / len(tasks),
                "final_average_exact_match": sum(
                    metrics[task["name"]]["generation"]["strict_accuracy"] for task in tasks
                ) / len(tasks),
                "final_average_rouge_l_f1": sum(
                    metrics[task["name"]]["generation"]["rouge_l_f1"] for task in tasks
                ) / len(tasks),
                "final_task_losses": losses,
            },
        }
    else:
        uses_soft_replay = args.method in ("gd", "cagd", "rank1_gd", "diag_gd")
        uses_generated_replay = uses_soft_replay or args.method == "hard_replay"
        uses_real_replay = args.method == "real_replay"
        ewc_kind = "rank1" if args.method in ("rank1", "rank1_gd") else (
            "diagonal" if args.method in ("diagonal", "diag_gd") else None
        )
        constraints = []
        stages = []
        for stage, task in enumerate(tasks):
            print(f"stage={stage + 1}/{len(tasks)} task={task['name']}", flush=True)
            teacher = replay = None
            replay_manifest = []
            if stage and uses_generated_replay:
                teacher = load_model(args, device)
                teacher.load_state_dict(model.state_dict())
                teacher.eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                prompts, replay_manifest = _replay_prompts(tasks, stage, args.replay_per_task)
                replay = _generate_replay(model, tokenizer, prompts, device, args)
            elif stage and uses_real_replay:
                replay, replay_manifest = _real_replay_rows(
                    tasks, stage, args.replay_per_task, tokenizer, args.max_length
                )
            training = _train_stage(
                model, task["train"], parameters, pad_id, device, args,
                args.seed + 1000 * (stage + 1),
                teacher=teacher if uses_soft_replay else None,
                replay_rows=replay,
                replay_objective=("soft" if uses_soft_replay else "hard" if replay is not None else None),
                constraints=constraints,
            )
            if teacher is not None:
                del teacher
                torch.cuda.empty_cache()
            metrics = {
                seen_task["name"]: _measure_task(
                    model, tokenizer, seen_task, pad_id, device, args
                )
                for seen_task in tasks[: stage + 1]
            }
            fisher_stats = None
            if stage + 1 < len(tasks) and ewc_kind:
                fisher, fisher_stats = _estimate_mean_and_diagonal_fisher(
                    model, task["fisher"], parameters, pad_id, device, args
                )
                fisher_stats["source"] = "task_training_examples"
                fisher_stats["task"] = task["name"]
                constraint = {
                    "kind": ewc_kind,
                    "reference": flat_parameters(parameters).detach().clone(),
                }
                if ewc_kind == "rank1":
                    constraint.update({
                        "direction": fisher["direction"].to(device),
                        "coefficient": fisher["coefficient"],
                    })
                else:
                    constraint["diagonal"] = fisher["diagonal"].to(device)
                constraints.append(constraint)
                del fisher
            stages.append({
                "stage": stage,
                "task": task["name"],
                "training": training,
                "replay_examples": len(replay) if replay is not None else 0,
                "replay": {
                    "per_task": replay_manifest,
                    "generated_rows_sha256": _records_sha256(replay) if replay is not None else None,
                },
                "fisher": fisher_stats,
                "metrics": metrics,
            })
        result = {
            "schema_version": 1,
            "status": "ok",
            "experiment": "dllm_rank1_multitask",
            "created_utc": _utc_now(),
            "wall_time_seconds": time.monotonic() - started,
            "host": __import__("os").uname().nodename,
            "source_sha256": source_sha256,
            "dependency_sha256": dependency_sha256,
            "metadata": metadata,
            "stages": stages,
            "summary": _summary(stages, tasks),
        }
    if (
        _sha256(Path(__file__)) != source_sha256
        or _dependency_hashes() != dependency_sha256
        or _metadata(args, tasks) != metadata
    ):
        raise RuntimeError("experiment provenance changed while the run was active")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    _transfer_self_check()
    specs = _sequence_specs("d2p", 8, 4, 4)
    assert [spec["direction"] for spec in specs] == ["d2p", "p2d", "d2p", "p2d"]
    assert [spec["group_start"] for spec in specs] == [8, 12, 16, 20]
    reverse = _sequence_specs("d2p", 8, 4, 4, "reverse")
    assert [(spec["direction"], spec["group_start"]) for spec in reverse] == [
        ("p2d", 20), ("d2p", 16), ("p2d", 12), ("d2p", 8)
    ]
    stages = []
    for i in range(4):
        metrics = {}
        for j in range(i + 1):
            metrics[f"t{j}"] = {
                "loss": float(i + j + 1),
                "answer_token_accuracy": 0.5,
                "generation": {
                    "accuracy": 0.5, "strict_accuracy": 0.25,
                    "rouge_l_f1": 0.5, "lcs_recall": 0.5,
                },
            }
        stages.append({"metrics": metrics})
    tasks = [{"name": f"t{i}"} for i in range(4)]
    summary = _summary(stages, tasks)
    assert summary["final_task_losses"] == [4.0, 5.0, 6.0, 7.0]
    assert summary["losses_when_learned"] == [1.0, 3.0, 5.0, 7.0]
    assert math.isclose(summary["past_task_forgetting"], 2.0)
    protocol_args = argparse.Namespace(
        start_direction="d2p", order="forward", group_start=8, tasks=4, group_count=4,
        fisher_per_fact=10, steps_per_task=1000, batch_size=4, eval_batch_size=4,
        eval_mc_samples=32, replay_per_task=64, replay_steps=32, replay_length=52,
        replay_cfg=0.8, replay_temperature=0.0, reverse_steps=32, reverse_length=52,
        reverse_cfg=0.8, reverse_temperature=0.0, generation_batch_size=1,
        max_length=128, model=170, mask_min=1e-3, mask_max=1.0, trainable="all",
        distill_weight=1.0, distill_temperature=1.0, ewc_lambda=0.0, lr=5e-5,
        clip=1.0, method="gd", seed=3407, generation_seed=3407,
        final_protocol=True, fresh_protocol=False,
    )
    mock_tasks = []
    for spec in specs:
        facts = range(spec["group_start"], spec["group_start"] + 4)
        mock_tasks.append({
            **spec,
            "name": f"{spec['direction']}_{spec['group_start']}-{spec['group_start'] + 3}",
            "train_raw": [
                {"fact_id": fact, "prompt": f"fact-{fact}-prompt-{index}"}
                for fact in facts for index in range(30)
            ],
            "fisher_raw": [{"fact_id": fact} for fact in facts for _ in range(10)],
            "eval_raw": [{"fact_id": fact} for fact in facts for _ in range(10)],
        })
    selected = {"rank1_gd": 1e5, "diag_gd": 1e3}
    _validate_locked_protocol(protocol_args, mock_tasks, selected)
    replay_prompts, replay_manifest = _replay_prompts(mock_tasks, 2, 64)
    assert len(replay_prompts) == 128
    assert [item["fact_counts"] for item in replay_manifest] == [
        {str(fact): 16 for fact in range(8, 12)},
        {str(fact): 16 for fact in range(12, 16)},
    ]
    assert all(len(item["selection_sha256"]) == 64 for item in replay_manifest)
    class _Tokenizer:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            del add_special_tokens
            return {"input_ids": list(range(1, len(text.split()) + 2))}

    for task in mock_tasks:
        for row in task["train_raw"]:
            row["answer"] = "answer"
    real_rows, real_manifest = _real_replay_rows(mock_tasks, 2, 64, _Tokenizer(), 128)
    assert len(real_rows) == 128
    assert [item["fact_counts"] for item in real_manifest] == [
        {str(fact): 16 for fact in range(8, 12)},
        {str(fact): 16 for fact in range(12, 16)},
    ]
    protocol_args.group_count = 2
    try:
        _validate_locked_protocol(protocol_args, mock_tasks, selected)
    except ValueError:
        pass
    else:
        raise AssertionError("locked protocol validation must reject drift")
    fresh_specs = _sequence_specs("d2p", 24, 3, 2)
    fresh_tasks = []
    for spec in fresh_specs:
        facts = range(spec["group_start"], spec["group_start"] + 2)
        fresh_tasks.append({
            **spec,
            "name": f"{spec['direction']}_{spec['group_start']}-{spec['group_start'] + 1}",
            "train_raw": [{"fact_id": fact} for fact in facts for _ in range(30)],
            "fisher_raw": [{"fact_id": fact} for fact in facts for _ in range(10)],
            "eval_raw": [{"fact_id": fact} for fact in facts for _ in range(10)],
        })
    protocol_args.final_protocol = False
    protocol_args.fresh_protocol = True
    protocol_args.group_start = 24
    protocol_args.tasks = 3
    protocol_args.group_count = 2
    protocol_args.method = "rank1_gd"
    protocol_args.ewc_lambda = selected["rank1_gd"]
    _validate_locked_protocol(protocol_args, fresh_tasks, selected)
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(
            "seq", "gd", "cagd", "hard_replay", "real_replay", "rank1",
            "diagonal", "rank1_gd", "diag_gd", "joint",
        ),
        default="cagd",
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--start-direction", choices=("d2p", "p2d"), default="d2p")
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--group-start", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument("--fisher-per-fact", type=int, default=10)
    parser.add_argument("--trainable", choices=("all", "last_block", "last_mlp"), default="all")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-mc-samples", type=int, default=4)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--replay-length", type=int, default=52)
    parser.add_argument("--replay-cfg", type=float, default=0.8)
    parser.add_argument("--replay-temperature", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--reverse-length", type=int, default=52)
    parser.add_argument("--reverse-cfg", type=float, default=0.8)
    parser.add_argument("--reverse-temperature", type=float, default=0.0)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--show-predictions", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--final-protocol", action="store_true")
    parser.add_argument("--fresh-protocol", action="store_true")
    parser.add_argument(
        "--validation-summary", type=Path,
        default=ROOT / "runs/r16_native_mask/validation_summary.json",
    )
    args = parser.parse_args()
    if args.self_check:
        return args
    if args.output is None:
        parser.error("--output is required")
    if args.generation_seed is None:
        args.generation_seed = args.seed
    if args.final_protocol and args.fresh_protocol:
        parser.error("--final-protocol and --fresh-protocol are mutually exclusive")
    if args.tasks < 2 or args.group_count < 1 or args.fisher_per_fact < 1:
        parser.error("tasks >= 2, group_count >= 1, and fisher_per_fact >= 1 are required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

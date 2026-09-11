#!/usr/bin/env python3
"""Qwen3 continual-learning scale extension with GD and Fisher EWC."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
PROTOCOLS = {
    "qwen3_0.6b": ROOT / "report/qwen_continual_scale_protocol.md",
    "qwen3_1.7b": ROOT / "report/qwen_continual_scale_1.7b_protocol.md",
    "qwen3_4b": ROOT / "report/qwen_continual_scale_protocol.md",
}
PROTOCOL_TAGS = {
    "qwen3_0.6b": "qwen_continual_scale_v1",
    "qwen3_1.7b": "qwen_continual_scale_1.7b_addendum_v1",
    "qwen3_4b": "qwen_continual_scale_v1",
}
DEFAULT_REVERSE = ROOT / "SMDM/data/reverse_experiments/june_version_7921032488"
SEEDS = (3407, 3408, 3409)
METHODS = ("seq", "gd", "rank1_gd", "diag_gd")
SPECS = {
    "validation": (("d2p", 0, 2), ("p2d", 2, 2)),
    "formal": (("d2p", 8, 4), ("p2d", 12, 4), ("d2p", 16, 4), ("p2d", 20, 4)),
    "paper": (("d2p", 8, 4), ("p2d", 12, 4), ("d2p", 16, 4), ("p2d", 20, 4)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fact_rows(directory: Path, direction: str, split: str, start: int, count: int) -> list[dict]:
    path = directory / f"{direction}_prompts_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    group_size = 30 if split == "train" else 10
    selected = rows[start * group_size : (start + count) * group_size]
    if len(selected) != count * group_size:
        raise ValueError(f"missing fact groups {start}:{start + count} in {path}")
    answer_key = "answer" if split == "train" else "target"
    return [
        {
            "prompt": row["prompt"],
            answer_key: row["completion"],
            "fact_id": start + index // group_size,
            "template_id": index % group_size,
        }
        for index, row in enumerate(selected)
    ]


def _encode(row: dict, answer_key: str, tokenizer, max_length: int) -> dict | None:
    prefix = ([] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id])
    prefix += tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(row[answer_key], add_special_tokens=False)
    if not answer or len(prefix) + len(answer) + 1 > max_length:
        return None
    return {
        "ids": prefix + answer + [tokenizer.eos_token_id],
        "answer_start": len(prefix),
        "fact_id": row["fact_id"],
        "template_id": row["template_id"],
        "prompt": row["prompt"],
        "prompt_ids": prefix,
    }


def _load_tasks(args, tokenizer) -> list[dict]:
    tasks = []
    for task_index, (direction, start, count) in enumerate(SPECS[args.run_kind]):
        train_raw = _fact_rows(args.reverse_dir, direction, "train", start, count)
        eval_raw = _fact_rows(args.reverse_dir, direction, "test", start, count)
        train = [_encode(row, "answer", tokenizer, args.max_length) for row in train_raw]
        evaluate = [_encode(row, "target", tokenizer, args.max_length) for row in eval_raw]
        if any(row is None for row in train + evaluate):
            raise ValueError("a locked task row is empty or exceeds max_length")
        fisher = [row for row in train if row["template_id"] >= 30 - args.fisher_per_fact]
        name = f"{direction}_{start}-{start + count - 1}"
        tasks.append({
            "index": task_index,
            "name": name,
            "direction": direction,
            "group_start": start,
            "group_count": count,
            "train": train,
            "fisher": fisher,
            "eval": evaluate,
        })
    return tasks


def _batch(rows, pad_id, device):
    import torch

    width = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    target = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["ids"])
        ids[index, :length] = torch.tensor(row["ids"], device=device)
        attention[index, :length] = 1
        target[index, row["answer_start"] : length] = True
    return ids, attention, target


def _answer_losses(model, rows, pad_id, device):
    import torch.nn.functional as F

    ids, attention, target = _batch(rows, pad_id, device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    labels = ids[:, 1:]
    mask = target[:, 1:]
    token_losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    return (token_losses * mask).sum(1) / mask.sum(1).clamp_min(1)


def _distillation_loss(student, teacher, rows, pad_id, device, temperature):
    import torch
    import torch.nn.functional as F

    ids, attention, target = _batch(rows, pad_id, device)
    mask = target[:, 1:]
    with torch.no_grad():
        teacher_logits = teacher(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        probabilities = F.softmax(teacher_logits[mask].float() / temperature, dim=-1)
    student_logits = student(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
    log_probabilities = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    return F.kl_div(log_probabilities, probabilities, reduction="batchmean") * temperature**2


def _select_parameters(model, mode="last_block"):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = f"model.layers.{model.config.num_hidden_layers - 1}."
    if mode == "all":
        selected = list(model.named_parameters())
    elif mode == "last_block":
        selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)]
    else:
        raise ValueError(f"unknown trainable scope: {mode}")
    if not selected:
        raise ValueError(f"no parameters under {prefix}")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return [name for name, _ in selected], [parameter for _, parameter in selected]


def _fisher(model, rows, parameters, pad_id, device) -> tuple[dict, dict]:
    import torch

    model.eval()
    mean = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    diagonal = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    for index, row in enumerate(rows, 1):
        loss = _answer_losses(model, [row], pad_id, device).mean()
        gradients = torch.autograd.grad(loss, parameters)
        for total, square, gradient in zip(mean, diagonal, gradients):
            value = gradient.detach().float()
            total.add_(value)
            square.addcmul_(value, value)
        if index % 10 == 0:
            print(f"fisher_pass=1 example={index}/{len(rows)}", flush=True)
    for value in mean:
        value.div_(len(rows))
    for value in diagonal:
        value.div_(len(rows))
    mean_norm_sq = sum(float(torch.sum(value.square())) for value in mean)
    projection_sq = 0.0
    for index, row in enumerate(rows, 1):
        loss = _answer_losses(model, [row], pad_id, device).mean()
        gradients = torch.autograd.grad(loss, parameters)
        projection = sum(torch.sum(gradient.detach().float() * direction) for gradient, direction in zip(gradients, mean))
        projection_sq += float(projection.square())
        if index % 10 == 0:
            print(f"fisher_pass=2 example={index}/{len(rows)}", flush=True)
    coefficient = (projection_sq / len(rows)) / max(mean_norm_sq**2, 1e-30)
    trace = sum(float(torch.sum(value)) for value in diagonal)
    return (
        {"mean": mean, "diagonal": diagonal, "coefficient": coefficient},
        {
            "sample_count": len(rows),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trace": trace,
            "mean_norm_sq": mean_norm_sq,
            "rank1_coefficient": coefficient,
            "rank1_trace_fraction": coefficient * mean_norm_sq / max(trace, 1e-30),
        },
    )


def _constraint(kind, fisher, parameters):
    import torch

    if kind == "rank1":
        direction = fisher["mean"]
        anchor = sum(torch.sum(parameter.detach().float() * value) for parameter, value in zip(parameters, direction)).detach()
        return {"kind": kind, "direction": direction, "anchor": anchor, "coefficient": fisher["coefficient"]}
    return {
        "kind": kind,
        "diagonal": fisher["diagonal"],
        "reference": [parameter.detach().clone() for parameter in parameters],
    }


def _penalty(parameters, constraints):
    import torch

    if not constraints:
        return torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    result = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    for constraint in constraints:
        if constraint["kind"] == "rank1":
            projection = sum(
                torch.sum(parameter.float() * direction)
                for parameter, direction in zip(parameters, constraint["direction"])
            ) - constraint["anchor"]
            result = result + 0.5 * constraint["coefficient"] * projection.square()
        else:
            result = result + 0.5 * sum(
                torch.sum(diagonal * (parameter.float() - reference.float()).square())
                for parameter, reference, diagonal in zip(
                    parameters, constraint["reference"], constraint["diagonal"]
                )
            )
    return result


def _balanced_prompts(tasks, seen, per_task):
    selected = []
    manifest = []
    for task in tasks[:seen]:
        pools = {}
        for row in task["train"]:
            pools.setdefault(row["fact_id"], []).append(row)
        quotient, remainder = divmod(per_task, len(pools))
        task_rows = []
        for index, fact_id in enumerate(sorted(pools)):
            count = quotient + int(index < remainder)
            task_rows.extend(pools[fact_id][:count])
        selected.extend(task_rows)
        manifest.append({
            "task": task["name"],
            "count": len(task_rows),
            "fact_counts": {
                str(fact_id): sum(row["fact_id"] == fact_id for row in task_rows)
                for fact_id in sorted(pools)
            },
            "prompt_sha256": _stable_hash([row["prompt_ids"] for row in task_rows]),
        })
    return selected, manifest


def _generate_replay(teacher, prompt_rows, pad_id, device, batch_size, max_new_tokens):
    import torch

    teacher.eval()
    replay = []
    for start in range(0, len(prompt_rows), batch_size):
        rows = prompt_rows[start : start + batch_size]
        width = max(len(row["prompt_ids"]) for row in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            length = len(row["prompt_ids"])
            ids[index, width - length :] = torch.tensor(row["prompt_ids"], device=device)
            attention[index, width - length :] = 1
        with torch.no_grad():
            outputs = teacher.generate(
                input_ids=ids,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
            )
        generated = outputs[:, width:].cpu().tolist()
        for source, answer in zip(rows, generated):
            if pad_id in answer:
                answer = answer[: answer.index(pad_id) + 1]
            if not answer:
                answer = [pad_id]
            replay.append({
                "ids": source["prompt_ids"] + answer,
                "answer_start": len(source["prompt_ids"]),
                "fact_id": source["fact_id"],
                "template_id": source["template_id"],
            })
        print(f"replay_generated={min(start + batch_size, len(prompt_rows))}/{len(prompt_rows)}", flush=True)
    return replay


@__import__("contextlib").contextmanager
def _evaluation_mode(model):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


def _evaluate(model, rows, pad_id, device, batch_size):
    import torch

    losses, correct, total = [], 0, 0
    with _evaluation_mode(model), torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            ids, attention, target = _batch(batch, pad_id, device)
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
            labels = ids[:, 1:]
            mask = target[:, 1:]
            import torch.nn.functional as F

            token_losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
            losses.extend(((token_losses * mask).sum(1) / mask.sum(1)).cpu().tolist())
            predictions = logits.argmax(-1)
            correct += int(((predictions == labels) & mask).sum())
            total += int(mask.sum())
    return {"loss": sum(losses) / len(losses), "answer_token_accuracy": correct / total}


def _train_stage(model, teacher, current_rows, replay_rows, parameters, constraints, pad_id, device, args, stage):
    import torch

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    current_rng = random.Random(args.seed + 10_000 * (stage + 1))
    replay_rng = random.Random(args.seed + 10_000 * (stage + 1) + 303)
    totals = {"current": 0.0, "distill": 0.0, "penalty": 0.0, "total": 0.0}
    clipped = 0
    started = time.monotonic()
    model.train()
    for step in range(args.steps_per_task):
        current = [current_rows[current_rng.randrange(len(current_rows))] for _ in range(args.batch_size)]
        current_loss = _answer_losses(model, current, pad_id, device).mean()
        distill = torch.zeros((), device=device)
        if teacher is not None:
            replay = [replay_rows[replay_rng.randrange(len(replay_rows))] for _ in range(args.batch_size)]
            distill = _distillation_loss(model, teacher, replay, pad_id, device, args.distill_temperature)
        penalty = _penalty(parameters, constraints)
        total = current_loss + args.distill_weight * distill + args.ewc_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        clipped += norm > args.clip
        optimizer.step()
        values = {
            "current": float(current_loss.detach()),
            "distill": float(distill.detach()),
            "penalty": float(penalty.detach()),
            "total": float(total.detach()),
        }
        for key in totals:
            totals[key] += values[key]
        if step == 0 or step + 1 == args.steps_per_task or (step + 1) % 100 == 0:
            print(
                f"stage={stage + 1} step={step + 1}/{args.steps_per_task} "
                f"current={values['current']:.5f} distill={values['distill']:.5f} "
                f"penalty={values['penalty']:.6g}",
                flush=True,
            )
    del optimizer
    return {
        "wall_time_seconds": time.monotonic() - started,
        "clip_fraction": clipped / args.steps_per_task,
        **{f"{key}_mean": value / args.steps_per_task for key, value in totals.items()},
    }


def _summarize(stages, tasks):
    final = stages[-1]["metrics"]
    final_losses = [final[task["name"]]["loss"] for task in tasks]
    learned_losses = [stages[index]["metrics"][task["name"]]["loss"] for index, task in enumerate(tasks)]
    forgetting = [final_loss - learned for final_loss, learned in zip(final_losses, learned_losses)]
    return {
        "final_average_loss": sum(final_losses) / len(final_losses),
        "past_task_forgetting": sum(forgetting[:-1]) / (len(forgetting) - 1),
        "final_average_answer_token_accuracy": sum(
            final[task["name"]]["answer_token_accuracy"] for task in tasks
        ) / len(tasks),
        "final_task_losses": final_losses,
        "losses_when_learned": learned_losses,
        "task_forgetting": forgetting,
    }


def _validate_args(args):
    errors = []
    protocol = PROTOCOLS[args.model_label]
    expected_tasks = 2 if args.run_kind == "validation" else 4
    for name, actual, expected in (
        ("steps_per_task", args.steps_per_task, 1000),
        ("batch_size", args.batch_size, 4),
        ("eval_batch_size", args.eval_batch_size, 4),
        ("fisher_per_fact", args.fisher_per_fact, 10),
        ("replay_per_task", args.replay_per_task, 64),
        ("max_new_tokens", args.max_new_tokens, 32),
        ("max_length", args.max_length, 128),
    ):
        if not args.dev and actual != expected:
            errors.append(f"{name}={actual} != {expected}")
    if args.method in ("seq", "gd") and args.ewc_lambda != 0:
        errors.append("seq/gd require lambda=0")
    if args.method in ("rank1_gd", "diag_gd") and args.ewc_lambda <= 0:
        errors.append("EWC methods require a positive lambda")
    if args.seed not in SEEDS:
        errors.append(f"seed must be one of {SEEDS}")
    if args.run_kind == "formal":
        if args.selection is None:
            errors.append("formal runs require --selection")
        else:
            selection = json.loads(args.selection.read_text())
            if selection.get("status") != "ok":
                errors.append(f"selection status is {selection.get('status')!r}")
            if selection.get("audit", {}).get("runner_sha256") != _sha256(Path(__file__)):
                errors.append("selection was produced by a different runner")
            if selection.get("audit", {}).get("protocol_sha256") != _sha256(protocol):
                errors.append("selection was produced under a different protocol")
            inventory_sha = _sha256(args.model_inventory)
            expected_inventory = selection.get("audit", {}).get("model_inventory_sha256", {}).get(args.model_label)
            if inventory_sha != expected_inventory:
                errors.append("model inventory differs from validation")
            expected = selection["selected"].get(args.model_label, {}).get(args.method, 0.0)
            if args.ewc_lambda != expected:
                errors.append(f"lambda={args.ewc_lambda} != selected {expected}")
    if len(SPECS[args.run_kind]) != expected_tasks:
        errors.append("internal task specification mismatch")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_args(args)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("a CUDA device is required")
    source_sha = _sha256(Path(__file__))
    protocol = PROTOCOLS[args.model_label]
    protocol_sha = _sha256(protocol)
    inventory = json.loads(args.model_inventory.read_text())
    if Path(inventory["model_dir"]).resolve() != args.model.resolve():
        raise ValueError("model inventory path does not match --model")
    _set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    pad_id = int(tokenizer.eos_token_id)
    tasks = _load_tasks(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    names, parameters = _select_parameters(model, args.trainable)
    print(
        f"model={args.model_label} method={args.method} seed={args.seed} "
        f"trainable={sum(parameter.numel() for parameter in parameters):,}",
        flush=True,
    )
    uses_gd = args.method != "seq"
    ewc_kind = "rank1" if args.method == "rank1_gd" else "diagonal" if args.method == "diag_gd" else None
    constraints, stages = [], []
    for stage, task in enumerate(tasks):
        teacher = None
        replay_rows = []
        replay_manifest = []
        if stage and uses_gd:
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            prompts, replay_manifest = _balanced_prompts(tasks, stage, args.replay_per_task)
            replay_rows = _generate_replay(
                teacher, prompts, pad_id, device, args.generation_batch_size, args.max_new_tokens
            )
        training = _train_stage(
            model, teacher, task["train"], replay_rows, parameters, constraints,
            pad_id, device, args, stage,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        metrics = {
            seen["name"]: _evaluate(model, seen["eval"], pad_id, device, args.eval_batch_size)
            for seen in tasks[: stage + 1]
        }
        fisher_stats = None
        if ewc_kind and stage + 1 < len(tasks):
            fisher, fisher_stats = _fisher(model, task["fisher"], parameters, pad_id, device)
            constraints.append(_constraint(ewc_kind, fisher, parameters))
            del fisher
        stages.append({
            "stage": stage,
            "task": task["name"],
            "training": training,
            "metrics": metrics,
            "replay_count": len(replay_rows),
            "replay_manifest": replay_manifest,
            "replay_sha256": _stable_hash(replay_rows) if replay_rows else None,
            "fisher": fisher_stats,
        })
    metadata = {
        "protocol": PROTOCOL_TAGS[args.model_label],
        "run_kind": args.run_kind,
        "model_label": args.model_label,
        "model": str(args.model.resolve()),
        "model_inventory": str(args.model_inventory.resolve()),
        "model_inventory_sha256": _sha256(args.model_inventory),
        "method": args.method,
        "seed": args.seed,
        "task_sequence": [task["name"] for task in tasks],
        "task_sha256": {
            task["name"]: {
                "train": _stable_hash(task["train"]),
                "fisher": _stable_hash(task["fisher"]),
                "eval": _stable_hash(task["eval"]),
            }
            for task in tasks
        },
        "trainable": args.trainable,
        "trainable_names": names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "steps_per_task": args.steps_per_task,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "fisher_per_fact": args.fisher_per_fact,
        "replay_per_task": args.replay_per_task,
        "max_new_tokens": args.max_new_tokens,
        "generation": "greedy",
        "distillation": "teacher KL on teacher-generated answer tokens",
        "distill_weight": args.distill_weight,
        "distill_temperature": args.distill_temperature,
        "ewc_lambda": args.ewc_lambda,
        "lr": args.lr,
        "clip": args.clip,
        "selection": str(args.selection.resolve()) if args.selection else None,
        "selection_sha256": _sha256(args.selection) if args.selection else None,
    }
    result = {
        "schema_version": 1,
        "status": "ok",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "source_sha256": source_sha,
        "protocol_sha256": protocol_sha,
        "data_sha256": {
            name: _sha256(args.reverse_dir / name)
            for name in ("d2p_prompts_train.jsonl", "d2p_prompts_test.jsonl", "p2d_prompts_train.jsonl", "p2d_prompts_test.jsonl")
        },
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "metadata": metadata,
        "stages": stages,
        "summary": _summarize(stages, tasks),
    }
    if _sha256(Path(__file__)) != source_sha or _sha256(protocol) != protocol_sha:
        raise RuntimeError("source or protocol changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check():
    import torch

    assert set(PROTOCOLS) == {"qwen3_0.6b", "qwen3_1.7b", "qwen3_4b"}

    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([-1.0]))
    parameters = [first, second]
    rank1 = {
        "kind": "rank1",
        "direction": [torch.tensor([2.0, -1.0]), torch.tensor([3.0])],
        "anchor": torch.tensor(-3.0),
        "coefficient": 0.25,
    }
    flat = torch.cat([first, second])
    direction = torch.tensor([2.0, -1.0, 3.0])
    expected = 0.5 * 0.25 * (torch.dot(flat, direction) + 3.0).square()
    assert torch.allclose(_penalty(parameters, [rank1]), expected)
    diagonal = {
        "kind": "diagonal",
        "diagonal": [torch.tensor([1.0, 2.0]), torch.tensor([4.0])],
        "reference": [torch.tensor([0.0, 1.0]), torch.tensor([1.0])],
    }
    ref = torch.tensor([0.0, 1.0, 1.0])
    precision = torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(_penalty(parameters, [diagonal]), 0.5 * torch.sum(precision * (flat - ref).square()))
    stages = [
        {"metrics": {"a": {"loss": 1.0}}},
        {"metrics": {"a": {"loss": 2.0}, "b": {"loss": 3.0}}},
        {"metrics": {"a": {"loss": 4.0}, "b": {"loss": 5.0}, "c": {"loss": 6.0, "answer_token_accuracy": 0.5}}},
    ]
    for stage in stages:
        for value in stage["metrics"].values():
            value.setdefault("answer_token_accuracy", 0.5)
    tasks = [{"name": name} for name in "abc"]
    summary = _summarize(stages, tasks)
    assert summary["final_average_loss"] == 5.0
    assert summary["past_task_forgetting"] == 2.5
    print(json.dumps({"self_check": "ok"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--run-kind", choices=tuple(SPECS), default="validation")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-label", choices=tuple(PROTOCOLS))
    parser.add_argument("--model-inventory", type=Path)
    parser.add_argument("--method", choices=METHODS, default="seq")
    parser.add_argument("--trainable", choices=("last_block", "all"), default="last_block")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reverse-dir", type=Path, default=DEFAULT_REVERSE)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--fisher-per-fact", type=int, default=10)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    for name in ("model", "model_label", "model_inventory", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    run(args)


if __name__ == "__main__":
    main()

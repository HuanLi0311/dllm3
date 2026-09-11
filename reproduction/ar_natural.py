#!/usr/bin/env python3
"""Autoregressive CAGD study on the fixed natural-language task stream."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.ar_factual import (  # noqa: E402
    _answer_losses,
    _distillation_loss,
    _evaluate,
    _generate_replay,
    _select_parameters,
    _set_seed,
    _stable_hash,
)


TASKS = ("closed_qa", "summarization", "creative_writing")
METHODS = ("seq", "cagd", "hard_replay")
SEEDS = (3407, 3408, 3409)
PROTOCOL = ROOT / "report/cagd_natural_protocol.md"
PROTOCOL_TAG = "cagd_natural_qwen_v1"
DEFAULT_MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_inventory(path: Path) -> dict:
    files = []
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        target = member.resolve()
        files.append({
            "path": member.relative_to(path).as_posix(),
            "blob": target.name,
            "bytes": target.stat().st_size,
        })
    return {"snapshot": path.name, "files": files, "sha256": _stable_hash(files)}


def _encode(row: dict, tokenizer, max_length: int, template_id: int) -> dict:
    prompt_ids = ([] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id])
    prompt_ids += tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    if not answer_ids or len(ids) > max_length:
        raise ValueError(f"{row['example_id']}: row violates the frozen token limit")
    return {
        "ids": ids,
        "answer_start": len(prompt_ids),
        "fact_id": row["task"],
        "template_id": template_id,
        "prompt": row["prompt"],
        "prompt_ids": prompt_ids,
        "source_index": row["source_index"],
    }


def _read_tasks(path: Path, tokenizer, max_length: int, order: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = TASKS if order == "forward" else tuple(reversed(TASKS))
    tasks = []
    for name in names:
        train_raw = [row for row in rows if row["task"] == name and row["split"] == "train"]
        eval_raw = [row for row in rows if row["task"] == name and row["split"] == "test"]
        tasks.append({
            "name": name,
            "train_raw": train_raw,
            "eval_raw": eval_raw,
            "train": [_encode(row, tokenizer, max_length, index) for index, row in enumerate(train_raw)],
            "eval": [_encode(row, tokenizer, max_length, index) for index, row in enumerate(eval_raw)],
        })
    return tasks


def _anchors(tasks: list[dict], seen: int, per_task: int) -> tuple[list[dict], list[dict]]:
    selected, manifest = [], []
    for task in tasks[:seen]:
        rows = task["train"][:per_task]
        if len(rows) != per_task:
            raise ValueError(f"{task['name']}: insufficient anchors")
        selected.extend(rows)
        manifest.append({
            "task": task["name"],
            "count": len(rows),
            "prompt_sha256": _stable_hash([row["prompt_ids"] for row in rows]),
            "source_indices": [row["source_index"] for row in rows],
        })
    return selected, manifest


def _train(model, teacher, current_rows, replay_rows, parameters, pad_id, device, args, stage) -> dict:
    import torch

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    current_rng = random.Random(args.seed + 10_000 * (stage + 1))
    replay_rng = random.Random(args.seed + 10_000 * (stage + 1) + 303)
    totals = {"current": 0.0, "soft_replay": 0.0, "hard_replay": 0.0, "total": 0.0}
    clipped = 0
    started = time.monotonic()
    model.train()
    for step in range(args.steps_per_task):
        current = [current_rows[current_rng.randrange(len(current_rows))] for _ in range(args.batch_size)]
        current_loss = _answer_losses(model, current, pad_id, device).mean()
        soft = hard = torch.zeros((), device=device)
        if replay_rows:
            replay = [replay_rows[replay_rng.randrange(len(replay_rows))] for _ in range(args.batch_size)]
            if teacher is not None:
                soft = _distillation_loss(
                    model, teacher, replay, pad_id, device, args.distill_temperature
                )
            else:
                hard = _answer_losses(model, replay, pad_id, device).mean()
        total = current_loss + args.distill_weight * (soft + hard)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        clipped += norm > args.clip
        optimizer.step()
        values = {
            "current": float(current_loss.detach()),
            "soft_replay": float(soft.detach()),
            "hard_replay": float(hard.detach()),
            "total": float(total.detach()),
        }
        for key, value in values.items():
            totals[key] += value
        if step == 0 or step + 1 == args.steps_per_task or (step + 1) % 100 == 0:
            print(
                f"stage={stage + 1} step={step + 1}/{args.steps_per_task} "
                f"current={values['current']:.5f} soft={values['soft_replay']:.5f} "
                f"hard={values['hard_replay']:.5f}",
                flush=True,
            )
    del optimizer
    return {
        "steps": args.steps_per_task,
        "wall_time_seconds": time.monotonic() - started,
        "clip_fraction": clipped / args.steps_per_task,
        **{f"{key}_mean": value / args.steps_per_task for key, value in totals.items()},
    }


def _summarize(stages: list[dict], tasks: list[dict]) -> dict:
    final = stages[-1]["metrics"]
    losses = [final[task["name"]]["loss"] for task in tasks]
    learned = [stages[index]["metrics"][task["name"]]["loss"] for index, task in enumerate(tasks)]
    forgetting = [after - before for after, before in zip(losses, learned)]
    return {
        "final_average_loss": sum(losses) / len(losses),
        "past_task_forgetting": sum(forgetting[:-1]) / (len(forgetting) - 1),
        "final_average_answer_token_accuracy": sum(
            final[task["name"]]["answer_token_accuracy"] for task in tasks
        ) / len(tasks),
        "final_task_losses": losses,
        "losses_when_learned": learned,
        "task_forgetting": forgetting,
    }


def _validate(args, tasks: list[dict]) -> None:
    manifest = json.loads(args.manifest.read_text())
    errors = []
    if args.order != "forward":
        errors.append("the formal natural stream uses the predeclared forward order")
    expected_manifest = {
        "dataset_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "selection_seed": 20260907,
        "task_order": list(TASKS),
        "train_per_task": 120,
        "test_per_task": 40,
        "prompt_disjoint": True,
        "selected_rows_sha256": "5083b050bdc3255b2e77c4d51af9fff012d536cdf101a391104a8ab25a3adc88",
    }
    for key, wanted in expected_manifest.items():
        if manifest.get(key) != wanted:
            errors.append(f"manifest {key} differs")
    locked = {
        "steps_per_task": 1000,
        "batch_size": 2,
        "eval_batch_size": 2,
        "replay_per_task": 64,
        "max_new_tokens": 96,
        "max_length": 256,
        "distill_weight": 1.0,
        "distill_temperature": 1.0,
        "lr": 5e-5,
        "clip": 1.0,
    }
    for key, wanted in locked.items():
        if getattr(args, key) != wanted:
            errors.append(f"{key}={getattr(args, key)!r}, expected {wanted!r}")
    if args.method not in METHODS or args.seed not in SEEDS:
        errors.append("method or seed is outside the formal matrix")
    if args.model.resolve() != DEFAULT_MODEL.resolve():
        errors.append("model snapshot differs")
    if [task["name"] for task in tasks] != list(TASKS if args.order == "forward" else reversed(TASKS)):
        errors.append("task order differs")
    if any(len(task["train_raw"]) != 120 or len(task["eval_raw"]) != 40 for task in tasks):
        errors.append("task counts differ")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def run(args) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the natural AR study requires CUDA")
    _set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = _read_tasks(args.data, tokenizer, args.max_length, args.order)
    if args.formal:
        _validate(args, tasks)
    source_sha256 = _sha256(Path(__file__))
    dependency_sha256 = _sha256(ROOT / "reproduction/ar_factual.py")
    protocol_sha256 = _sha256(PROTOCOL) if args.formal else None
    data_sha256 = _sha256(args.data)
    manifest_sha256 = _sha256(args.manifest)
    inventory = _model_inventory(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    trainable_names, parameters = _select_parameters(model)
    print(f"method={args.method} order={args.order} trainable={sum(p.numel() for p in parameters):,}", flush=True)
    stages = []
    for stage, task in enumerate(tasks):
        teacher = None
        replay_rows = []
        anchor_manifest = []
        if stage and args.method != "seq":
            if args.method == "cagd":
                teacher = copy.deepcopy(model).eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
            anchors, anchor_manifest = _anchors(tasks, stage, args.replay_per_task)
            replay_rows = _generate_replay(
                teacher if teacher is not None else model,
                anchors, pad_id, device, args.generation_batch_size, args.max_new_tokens,
            )
        training = _train(
            model, teacher if args.method == "cagd" else None, task["train"], replay_rows,
            parameters, pad_id, device, args, stage,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        metrics = {
            seen["name"]: _evaluate(model, seen["eval"], pad_id, device, args.eval_batch_size)
            for seen in tasks[: stage + 1]
        }
        stages.append({
            "stage": stage,
            "task": task["name"],
            "training": training,
            "anchor_manifest": anchor_manifest,
            "generated_rows_sha256": _stable_hash(replay_rows) if replay_rows else None,
            "metrics": metrics,
        })
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "qwen_cagd_natural",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_sha256,
        "dependency_sha256": dependency_sha256,
        "protocol_sha256": protocol_sha256,
        "data_sha256": data_sha256,
        "manifest_sha256": manifest_sha256,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "model_inventory": inventory,
        "metadata": {
            "protocol": PROTOCOL_TAG if args.formal else "development",
            "method": args.method,
            "order": args.order,
            "seed": args.seed,
            "task_sequence": [task["name"] for task in tasks],
            "model": str(args.model.resolve()),
            "trainable": "last_transformer_block",
            "trainable_names": trainable_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "steps_per_task": args.steps_per_task,
            "batch_size": args.batch_size,
            "replay_per_task": args.replay_per_task,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "distill_weight": args.distill_weight,
        },
        "stages": stages,
        "summary": _summarize(stages, tasks),
    }
    if (
        _sha256(Path(__file__)) != source_sha256
        or _sha256(ROOT / "reproduction/ar_factual.py") != dependency_sha256
        or _sha256(args.data) != data_sha256
        or _sha256(args.manifest) != manifest_sha256
        or _model_inventory(args.model) != inventory
        or (args.formal and _sha256(PROTOCOL) != protocol_sha256)
    ):
        raise RuntimeError("source, protocol, model, or data changed during execution")
    if not all(math.isfinite(value) for value in result["summary"]["final_task_losses"]):
        raise RuntimeError("non-finite endpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    tasks = [{"name": name} for name in "abc"]
    stages = [
        {"metrics": {"a": {"loss": 1.0, "answer_token_accuracy": 0.5}}},
        {"metrics": {
            "a": {"loss": 2.0, "answer_token_accuracy": 0.5},
            "b": {"loss": 3.0, "answer_token_accuracy": 0.5},
        }},
        {"metrics": {
            "a": {"loss": 4.0, "answer_token_accuracy": 0.5},
            "b": {"loss": 5.0, "answer_token_accuracy": 0.5},
            "c": {"loss": 6.0, "answer_token_accuracy": 0.5},
        }},
    ]
    assert _summarize(stages, tasks)["past_task_forgetting"] == 2.5
    mock = [{"name": "a", "train": [{"prompt_ids": [i], "source_index": i} for i in range(70)]}]
    anchors, manifest = _anchors(mock, 1, 64)
    assert len(anchors) == manifest[0]["count"] == 64
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/data/dolly_natural_stream.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/data/dolly_natural_stream_manifest.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

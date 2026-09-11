#!/usr/bin/env python3
"""Continual CAGD study on the fixed natural-language task stream."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reproduction.continual_benchmark import encode_benchmark_rows  # noqa: E402
from reproduction.smdm_backend import answer_token_accuracy, load_model, set_seed, trainable_parameters  # noqa: E402
from reproduction.smdm_factual import _records_sha256, _train_stage  # noqa: E402
from reproduction.smdm_transfer import _evaluate_loss, _generate_replay  # noqa: E402


TASKS = ("closed_qa", "summarization", "creative_writing")
METHODS = ("seq", "cagd", "hard_replay")
SEEDS = (3407, 3408, 3409)
PROTOCOL = ROOT / "report/cagd_natural_protocol.md"
PROTOCOL_TAG = "cagd_natural_smdm_v1"


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
        digest.update(bytes.fromhex(_sha256(member)))
    return digest.hexdigest()


def _dependency_hashes() -> dict[str, str]:
    names = (
        "continual_benchmark.py",
        "continual_mdm.py",
        "continual_reverse.py",
        "experiments/dllm_rank1_multitask.py",
        "experiments/dllm_rank1_transfer.py",
    )
    return {name: _sha256(ROOT / name) for name in names}


def _read_tasks(path: Path, tokenizer, max_length: int, order: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = TASKS if order == "forward" else tuple(reversed(TASKS))
    tasks = []
    for task_index, name in enumerate(names):
        train_raw = [row for row in rows if row["task"] == name and row["split"] == "train"]
        eval_raw = [row for row in rows if row["task"] == name and row["split"] == "test"]
        train = encode_benchmark_rows(train_raw, tokenizer, max_length)
        evaluate = encode_benchmark_rows(eval_raw, tokenizer, max_length)
        if len(train) != len(train_raw) or len(evaluate) != len(eval_raw):
            raise ValueError(f"{name}: encoded rows differ from the frozen data")
        tasks.append({
            "task_index": task_index,
            "name": name,
            "train_raw": train_raw,
            "eval_raw": eval_raw,
            "train": train,
            "eval": evaluate,
        })
    return tasks


def _anchor_prompts(tasks: list[dict], seen: int, per_task: int) -> tuple[list[str], list[dict]]:
    prompts, manifest = [], []
    for task in tasks[:seen]:
        selected = task["train_raw"][:per_task]
        if len(selected) != per_task:
            raise ValueError(f"{task['name']}: insufficient anchors")
        values = [row["prompt"] for row in selected]
        prompts.extend(values)
        manifest.append({
            "task": task["name"],
            "count": len(values),
            "prompt_sha256": _records_sha256(values),
            "source_indices": [row["source_index"] for row in selected],
        })
    return prompts, manifest


def _measure(model, task, pad_id, device, args) -> dict:
    return {
        "loss": _evaluate_loss(
            model, task["eval"], pad_id, device, args,
            args.seed + 81_001 + 100 * task["task_index"],
        ),
        "answer_token_accuracy": answer_token_accuracy(
            model, task["eval"], device, args.eval_batch_size, pad_id
        ),
    }


def _summarize(stages: list[dict], tasks: list[dict]) -> dict:
    final = stages[-1]["metrics"]
    final_losses = [final[task["name"]]["loss"] for task in tasks]
    learned_losses = [stages[index]["metrics"][task["name"]]["loss"] for index, task in enumerate(tasks)]
    forgetting = [after - before for after, before in zip(final_losses, learned_losses)]
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


def _validate(args, tasks: list[dict]) -> None:
    if args.method not in METHODS or args.seed not in SEEDS:
        raise ValueError("formal method or seed is outside the protocol")
    manifest = json.loads(args.manifest.read_text())
    errors = []
    if args.order != "forward":
        errors.append("the formal natural stream uses the predeclared forward order")
    expected = {
        "dataset_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "selection_seed": 20260907,
        "task_order": list(TASKS),
        "train_per_task": 120,
        "test_per_task": 40,
        "max_length": 256,
        "max_prompt_tokens": 160,
        "prompt_disjoint": True,
        "selected_rows_sha256": "5083b050bdc3255b2e77c4d51af9fff012d536cdf101a391104a8ab25a3adc88",
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            errors.append(f"manifest {key} differs")
    locked = {
        "steps_per_task": 1000,
        "batch_size": 2,
        "eval_batch_size": 2,
        "eval_mc_samples": 16,
        "max_length": 256,
        "replay_per_task": 64,
        "replay_steps": 32,
        "replay_length": 256,
        "replay_cfg": 0.8,
        "replay_temperature": 0.0,
        "distill_weight": 1.0,
        "distill_temperature": 1.0,
        "ewc_lambda": 0.0,
        "mask_min": 1e-3,
        "mask_max": 1.0,
        "lr": 5e-5,
        "clip": 1.0,
        "model": 170,
        "trainable": "all",
    }
    for key, wanted in locked.items():
        if getattr(args, key) != wanted:
            errors.append(f"{key}={getattr(args, key)!r}, expected {wanted!r}")
    if [task["name"] for task in tasks] != list(TASKS if args.order == "forward" else reversed(TASKS)):
        errors.append("task order differs")
    if any(len(task["train_raw"]) != 120 or len(task["eval_raw"]) != 40 for task in tasks):
        errors.append("task counts differ")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def run(args) -> dict:
    from transformers import AutoTokenizer

    started = time.monotonic()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the natural SMDM study requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = _read_tasks(args.data, tokenizer, args.max_length, args.order)
    if args.formal:
        _validate(args, tasks)
    source_sha256 = _sha256(Path(__file__))
    dependencies = _dependency_hashes()
    protocol_sha256 = _sha256(PROTOCOL) if args.formal else None
    data_sha256 = _sha256(args.data)
    manifest_sha256 = _sha256(args.manifest)
    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"method={args.method} order={args.order} trainable={sum(p.numel() for p in parameters):,}", flush=True)
    stages = []
    for stage, task in enumerate(tasks):
        print(f"stage={stage + 1}/{len(tasks)} task={task['name']}", flush=True)
        teacher = replay = None
        anchor_manifest = []
        if stage and args.method != "seq":
            if args.method == "cagd":
                teacher = load_model(args, device)
                teacher.load_state_dict(model.state_dict())
                teacher.eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
            prompts, anchor_manifest = _anchor_prompts(tasks, stage, args.replay_per_task)
            replay = _generate_replay(teacher if teacher is not None else model, tokenizer, prompts, device, args)
        training = _train_stage(
            model, task["train"], parameters, pad_id, device, args,
            args.seed + 1000 * (stage + 1),
            teacher=teacher,
            replay_rows=replay,
            replay_objective=("soft" if teacher is not None else "hard" if replay else None),
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        metrics = {seen["name"]: _measure(model, seen, pad_id, device, args) for seen in tasks[: stage + 1]}
        stages.append({
            "stage": stage,
            "task": task["name"],
            "training": training,
            "anchor_manifest": anchor_manifest,
            "generated_rows_sha256": _records_sha256(replay) if replay else None,
            "metrics": metrics,
        })
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "smdm_cagd_natural",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_sha256,
        "dependency_sha256": dependencies,
        "protocol_sha256": protocol_sha256,
        "data_sha256": data_sha256,
        "manifest_sha256": manifest_sha256,
        "metadata": {
            "protocol": PROTOCOL_TAG if args.formal else "development",
            "method": args.method,
            "order": args.order,
            "seed": args.seed,
            "task_sequence": [task["name"] for task in tasks],
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "tokenizer_sha256": _tree_sha256(args.tokenizer),
            "steps_per_task": args.steps_per_task,
            "batch_size": args.batch_size,
            "eval_mc_samples": args.eval_mc_samples,
            "replay_per_task": args.replay_per_task,
            "max_length": args.max_length,
            "replay_length": args.replay_length,
            "distill_weight": args.distill_weight,
        },
        "stages": stages,
        "summary": _summarize(stages, tasks),
    }
    if (
        _sha256(Path(__file__)) != source_sha256
        or _dependency_hashes() != dependencies
        or _sha256(args.data) != data_sha256
        or _sha256(args.manifest) != manifest_sha256
        or (args.formal and _sha256(PROTOCOL) != protocol_sha256)
    ):
        raise RuntimeError("source, protocol, or data changed during execution")
    if not all(math.isfinite(value) for value in result["summary"]["final_task_losses"]):
        raise RuntimeError("non-finite endpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    mock = [
        {"name": "a"}, {"name": "b"}, {"name": "c"},
    ]
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
    summary = _summarize(stages, mock)
    assert summary["final_average_loss"] == 5.0
    assert summary["past_task_forgetting"] == 2.5
    tasks = [{"name": "a", "train_raw": [{"prompt": str(i), "source_index": i} for i in range(70)]}]
    prompts, manifest = _anchor_prompts(tasks, 1, 64)
    assert len(prompts) == manifest[0]["count"] == 64
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/data/dolly_natural_stream.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/data/dolly_natural_stream_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--trainable", choices=("all", "last_block", "last_mlp"), default="all")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-mc-samples", type=int, default=16)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--replay-length", type=int, default=256)
    parser.add_argument("--replay-cfg", type=float, default=0.8)
    parser.add_argument("--replay-temperature", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
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

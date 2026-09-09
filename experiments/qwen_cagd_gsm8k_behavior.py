#!/usr/bin/env python3
"""Behavioral GSM8K retention after continual Qwen language adaptation."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qwen_cagd_natural import (  # noqa: E402
    DEFAULT_MODEL,
    _anchors,
    _encode,
    _model_inventory,
    _sha256,
    _train,
)
from experiments.qwen_continual_transfer import (  # noqa: E402
    _evaluate,
    _evaluation_mode,
    _generate_replay,
    _select_parameters,
    _set_seed,
)


METHODS = ("seq", "cagd")
SEEDS = (3407, 3408, 3409)
TASKS = ("gsm8k", "summarization", "creative_writing")
PROTOCOL = ROOT / "report/cagd_gsm8k_behavior_protocol.md"
GSM_TRAIN = ROOT / "SMDM/data/gsm8k/train_no_aug.txt"
GSM_TEST = ROOT / "SMDM/data/gsm8k/test.jsonl"
DOLLY = ROOT / "runs/data/dolly_natural_stream.jsonl"
DEPENDENCIES = (
    ROOT / "experiments/qwen_cagd_natural.py",
    ROOT / "experiments/qwen_continual_transfer.py",
)
LOCKED_HASHES = {
    GSM_TRAIN: "52ebf7c73927f7434abbb2f7b705a82fb3dbdd4695438b7654de78b701c23b36",
    GSM_TEST: "8530a3775b96385370842171f226d83de7c5b27d779be54ef0411d255a939818",
    DOLLY: "a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0",
}
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")


def _canonical_number(value: str) -> str | None:
    value = value.replace(",", "").strip()
    try:
        number = Fraction(value) if "/" in value else Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"


def _extract_answer(text: str) -> tuple[str | None, bool]:
    marked = re.findall(r"####\s*([^\n]+)", text)
    candidates = NUMBER.findall(marked[-1] if marked else text)
    return (_canonical_number(candidates[-1]) if candidates else None, bool(marked))


def _gsm_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _raw_gsm() -> tuple[list[dict], list[dict]]:
    train = []
    for index, line in enumerate(GSM_TRAIN.read_text().split("\n")):
        if not line.strip():
            continue
        question, answer = line.split("||", 1)
        train.append({
            "task": "gsm8k", "split": "train", "source_index": index,
            "example_id": f"gsm8k:train:{index}",
            "prompt": _gsm_prompt(question), "answer": " " + answer.strip(),
            "question": question.strip(),
            "target": _extract_answer(answer)[0],
        })
    evaluate = []
    for index, line in enumerate(GSM_TEST.read_text().split("\n")):
        if not line.strip():
            continue
        source = json.loads(line)
        evaluate.append({
            "task": "gsm8k", "split": "test", "source_index": index,
            "example_id": f"gsm8k:test:{index}",
            "prompt": _gsm_prompt(source["question"]),
            "answer": " " + source["answer"].strip(),
            "question": source["question"].strip(),
            "target": _canonical_number(str(source["target"])),
        })
    return train, evaluate


def _encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    encoded = []
    for index, row in enumerate(rows):
        item = _encode(row, tokenizer, max_length, index)
        item.update({
            "example_id": row["example_id"],
            "question": row.get("question"),
            "target": row.get("target"),
        })
        encoded.append(item)
    return encoded


def _tasks(tokenizer, max_length: int) -> list[dict]:
    gsm_train, gsm_eval = _raw_gsm()
    dolly = [json.loads(line) for line in DOLLY.read_text().splitlines() if line.strip()]
    tasks = [{
        "name": "gsm8k",
        "train": _encode_rows(gsm_train, tokenizer, max_length),
        "eval": _encode_rows(gsm_eval, tokenizer, max_length),
    }]
    for name in TASKS[1:]:
        train = [row for row in dolly if row["task"] == name and row["split"] == "train"]
        evaluate = [row for row in dolly if row["task"] == name and row["split"] == "test"]
        tasks.append({
            "name": name,
            "train": _encode_rows(train, tokenizer, max_length),
            "eval": _encode_rows(evaluate, tokenizer, max_length),
        })
    return tasks


def _benchmark(model, rows, tokenizer, pad_id, device, batch_size, max_new_tokens) -> dict:
    import torch

    records = []
    started = time.monotonic()
    with _evaluation_mode(model), torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for index, row in enumerate(batch):
                length = len(row["prompt_ids"])
                ids[index, width - length:] = torch.tensor(row["prompt_ids"], device=device)
                attention[index, width - length:] = 1
            outputs = model.generate(
                input_ids=ids,
                attention_mask=attention,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=pad_id,
                use_cache=True,
            )
            for row, generated in zip(batch, outputs[:, width:].cpu().tolist()):
                if pad_id in generated:
                    generated = generated[:generated.index(pad_id)]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                prediction, marked = _extract_answer(text)
                records.append({
                    "example_id": row["example_id"],
                    "source_index": row["source_index"],
                    "question": row["question"],
                    "target": row["target"],
                    "prediction": prediction,
                    "has_delimiter": marked,
                    "correct": prediction == row["target"],
                    "output": text,
                })
            print(f"benchmark_generated={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return {
        "count": len(records),
        "exact_match": sum(row["correct"] for row in records) / len(records),
        "delimiter_rate": sum(row["has_delimiter"] for row in records) / len(records),
        "wall_time_seconds": time.monotonic() - started,
        "records": records,
    }


def _validate(args) -> None:
    locked = {
        "steps_per_task": 1000,
        "batch_size": 2,
        "eval_batch_size": 4,
        "generation_batch_size": 16,
        "replay_per_task": 64,
        "replay_max_new_tokens": 128,
        "benchmark_max_new_tokens": 256,
        "max_length": 576,
        "distill_weight": 1.0,
        "distill_temperature": 1.0,
        "lr": 5e-5,
        "clip": 1.0,
        "benchmark_limit": 0,
    }
    errors = [f"{key}={getattr(args, key)!r}, expected {value!r}"
              for key, value in locked.items() if getattr(args, key) != value]
    if args.method not in METHODS or args.seed not in SEEDS:
        errors.append("method or seed is outside the formal matrix")
    if args.model.resolve() != DEFAULT_MODEL.resolve():
        errors.append("model snapshot differs")
    for path, wanted in LOCKED_HASHES.items():
        if _sha256(path) != wanted:
            errors.append(f"data hash differs: {path}")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def run(args) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires CUDA")
    if args.formal:
        _validate(args)
    started = time.monotonic()
    _set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = _tasks(tokenizer, args.max_length)
    benchmark_rows = tasks[0]["eval"][:args.benchmark_limit or None]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    trainable_names, parameters = _select_parameters(model)
    source_hash = _sha256(Path(__file__))
    protocol_hash = _sha256(PROTOCOL)
    dependency_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in DEPENDENCIES}
    inventory = _model_inventory(args.model)
    stages = []

    for stage, task in enumerate(tasks):
        teacher = None
        replay_rows = []
        anchor_manifest = []
        if stage and args.method == "cagd":
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            anchors, anchor_manifest = _anchors(tasks, stage, args.replay_per_task)
            replay_rows = _generate_replay(
                teacher, anchors, pad_id, device, args.generation_batch_size,
                args.replay_max_new_tokens,
            )
        training = _train(
            model, teacher, task["train"], replay_rows, parameters, pad_id,
            device, args, stage,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        current_metrics = _evaluate(
            model, task["eval"], pad_id, device, args.eval_batch_size
        )
        benchmark = None
        if stage in (0, len(tasks) - 1):
            benchmark = _benchmark(
                model, benchmark_rows, tokenizer, pad_id, device,
                args.generation_batch_size, args.benchmark_max_new_tokens,
            )
        stages.append({
            "stage": stage,
            "task": task["name"],
            "training": training,
            "anchor_manifest": anchor_manifest,
            "current_metrics": current_metrics,
            "benchmark": benchmark,
        })

    learned = stages[0]["benchmark"]["exact_match"]
    final = stages[-1]["benchmark"]["exact_match"]
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "qwen_cagd_gsm8k_behavior",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependency_hashes,
        "data_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in LOCKED_HASHES},
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model_inventory": inventory,
        "metadata": {
            "protocol": "cagd_gsm8k_behavior_v1" if args.formal else "development",
            "method": args.method,
            "seed": args.seed,
            "task_sequence": list(TASKS),
            "trainable": "last_transformer_block",
            "trainable_names": trainable_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "benchmark_count": len(benchmark_rows),
        },
        "stages": stages,
        "summary": {
            "gsm8k_exact_match_when_learned": learned,
            "gsm8k_exact_match_final": final,
            "gsm8k_retention_change": final - learned,
            "final_task_loss": stages[-1]["current_metrics"]["loss"],
            "final_task_answer_token_accuracy": stages[-1]["current_metrics"]["answer_token_accuracy"],
        },
    }
    if (
        _sha256(Path(__file__)) != source_hash
        or _sha256(PROTOCOL) != protocol_hash
        or {str(path.relative_to(ROOT)): _sha256(path) for path in DEPENDENCIES} != dependency_hashes
        or _model_inventory(args.model) != inventory
    ):
        raise RuntimeError("source, protocol, dependency, or model changed during execution")
    if args.formal:
        _validate(args)
    if not all(math.isfinite(value) for value in result["summary"].values()):
        raise RuntimeError("non-finite endpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    assert _extract_answer("work\n#### 1,234")[0] == "1234"
    assert _extract_answer("the last value is -2.50")[0] == "-5/2"
    assert _extract_answer("no numeric answer") == (None, False)
    train, evaluate = _raw_gsm()
    assert len(train) == 5250 and len(evaluate) == 1319
    assert all(row["target"] is not None for row in train + evaluate)
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--replay-max-new-tokens", type=int, default=128)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=256)
    parser.add_argument("--benchmark-limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=576)
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

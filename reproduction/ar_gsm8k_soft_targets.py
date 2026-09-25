#!/usr/bin/env python3
"""Matched Qwen GSM8K soft-target versus hard-replay intervention."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.ar_factual import _evaluate, _generate_replay, _select_parameters, _set_seed, _stable_hash  # noqa: E402
from reproduction.ar_gsm8k import (  # noqa: E402
    DOLLY,
    GSM_TEST,
    GSM_TRAIN,
    LOCKED_HASHES,
    _benchmark,
    _extract_answer,
    _model_inventory,
    _sha256,
    _tasks,
)
from reproduction.ar_natural import DEFAULT_MODEL, _anchors, _train  # noqa: E402


METHODS = ("hard_replay", "cagd")
SEEDS = (3407, 3408, 3409)
PROTOCOL = ROOT / "report/qwen_gsm8k_soft_targets_protocol.md"
DEPENDENCIES = (
    ROOT / "reproduction/ar_factual.py",
    ROOT / "reproduction/ar_gsm8k.py",
    ROOT / "reproduction/ar_natural.py",
)
MODEL_INVENTORY_SHA256 = "768bd5491a8c872f18c9edb62a8c4ece300963443fb1fa8b7b7be1d3d1fc053b"
SETTINGS = {
    "steps_per_task": 1000,
    "batch_size": 2,
    "eval_batch_size": 4,
    "generation_batch_size": 16,
    "replay_per_task": 64,
    "replay_max_new_tokens": 128,
    "benchmark_max_new_tokens": 256,
    "benchmark_limit": 0,
    "max_length": 576,
    "distill_weight": 1.0,
    "distill_temperature": 1.0,
    "lr": 5e-5,
    "clip": 1.0,
}


def _dependencies() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in DEPENDENCIES}


def _tree_descriptor(path: Path) -> dict[str, object]:
    files = [{
        "path": member.relative_to(path).as_posix(),
        "bytes": member.stat().st_size,
        "sha256": _sha256(member),
    } for member in sorted(item for item in path.rglob("*") if item.is_file())]
    return {"path": str(path.resolve()), "bytes": sum(row["bytes"] for row in files),
            "sha256": _stable_hash(files), "files": files}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _save_checkpoint(model, path: Path) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _tree_descriptor(path)


def _lock(path: Path) -> Path:
    lock = Path(str(path) + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
    except FileExistsError as error:
        raise FileExistsError(f"output is already locked: {lock}") from error
    return lock


def _validate(args) -> None:
    errors = []
    if args.seed not in SEEDS:
        errors.append("seed is outside the frozen matrix")
    if args.mode == "branch" and args.method not in METHODS:
        errors.append("branch method is outside the frozen matrix")
    if args.trainable != "all":
        errors.append("the frozen intervention is full-parameter")
    if args.model.resolve() != DEFAULT_MODEL.resolve():
        errors.append("model snapshot differs")
    elif _model_inventory(args.model)["sha256"] != MODEL_INVENTORY_SHA256:
        errors.append("model inventory differs")
    for key, wanted in SETTINGS.items():
        if getattr(args, key) != wanted:
            errors.append(f"{key}={getattr(args, key)!r}, expected {wanted!r}")
    for path, wanted in LOCKED_HASHES.items():
        if not path.is_file() or _sha256(path) != wanted:
            errors.append(f"data hash differs: {path}")
    if not PROTOCOL.is_file() or not PROTOCOL.read_text().startswith(
        "# Qwen GSM8K soft-target intervention\n\nStatus: frozen\n"
    ):
        errors.append("protocol is not frozen")
    if args.mode == "branch" and (args.stage0_json is None or args.stage0_checkpoint is None):
        errors.append("branch requires --stage0-json and --stage0-checkpoint")
    if args.mode == "stage0" and args.stage0_checkpoint is None:
        errors.append("stage0 requires --stage0-checkpoint")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def _load_context(args):
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires CUDA")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    tasks = _tasks(tokenizer, args.max_length)[:2]
    if [task["name"] for task in tasks] != ["gsm8k", "summarization"]:
        raise RuntimeError("task stream differs")
    return device, tokenizer, tasks, int(tokenizer.eos_token_id)


def _audit_benchmark(benchmark: dict, expected: list[dict]) -> None:
    records = benchmark.get("records")
    if benchmark.get("count") != len(expected) or not isinstance(records, list):
        raise ValueError("benchmark cardinality differs")
    for record, row in zip(records, expected):
        if any(record.get(key) != row.get(key) for key in ("example_id", "source_index", "question", "target")):
            raise ValueError("benchmark identity differs")
        prediction, marked = _extract_answer(record.get("output", ""))
        if (record.get("prediction"), record.get("has_delimiter"), record.get("correct")) != (
            prediction, marked, prediction == row["target"]
        ):
            raise ValueError("benchmark derived fields differ")


def _audit_stage0(args, tasks: list[dict], source_hash: str, protocol_hash: str,
                  dependencies: dict) -> tuple[dict, dict]:
    payload = json.loads(args.stage0_json.read_text())
    expected = {
        "status": "ok",
        "experiment": "qwen_gsm8k_soft_targets_stage0",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in LOCKED_HASHES},
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            raise ValueError(f"stage0 {key} differs")
    metadata = payload.get("metadata", {})
    for key, wanted in {
        "seed": args.seed,
        "trainable": "all",
        "settings": SETTINGS,
        "model_inventory_sha256": MODEL_INVENTORY_SHA256,
    }.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"stage0 metadata {key} differs")
    checkpoint = _tree_descriptor(args.stage0_checkpoint)
    if payload.get("checkpoint") != checkpoint:
        raise ValueError("stage0 checkpoint differs")
    _audit_benchmark(payload.get("benchmark", {}), tasks[0]["eval"][:args.benchmark_limit or None])
    replay = payload.get("replay_rows")
    anchors = tasks[0]["train"][:args.replay_per_task]
    if not isinstance(replay, list) or len(replay) != len(anchors):
        raise ValueError("stage0 replay count differs")
    if payload.get("replay_sha256") != _stable_hash(replay):
        raise ValueError("stage0 replay digest differs")
    for generated, anchor in zip(replay, anchors):
        if generated["ids"][:generated["answer_start"]] != anchor["prompt_ids"]:
            raise ValueError("stage0 replay prompt differs")
    return payload, checkpoint


def _conditional_retention(initial: dict, final: dict) -> tuple[int, float]:
    initial_records, final_records = initial["records"], final["records"]
    if [row["example_id"] for row in initial_records] != [row["example_id"] for row in final_records]:
        raise ValueError("initial and final GSM8K records are not aligned")
    initially_correct = [row["correct"] for row in initial_records]
    count = sum(initially_correct)
    if not count:
        raise ValueError("no initially correct GSM8K examples")
    retained = sum(was_correct and row["correct"] for was_correct, row in zip(initially_correct, final_records))
    return count, retained / count


def _stage0(args, source_hash: str, protocol_hash: str, dependencies: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    _set_seed(args.seed)
    device, tokenizer, tasks, pad_id = _load_context(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    names, parameters = _select_parameters(model, "all")
    training = _train(model, None, tasks[0]["train"], [], parameters, pad_id, device, args, 0)
    benchmark = _benchmark(
        model, tasks[0]["eval"][:args.benchmark_limit or None], tokenizer, pad_id, device,
        args.generation_batch_size, args.benchmark_max_new_tokens,
    )
    anchors, anchor_manifest = _anchors(tasks, 1, args.replay_per_task)
    replay = _generate_replay(
        model, anchors, pad_id, device, args.generation_batch_size, args.replay_max_new_tokens
    )
    _audit_benchmark(benchmark, tasks[0]["eval"][:args.benchmark_limit or None])
    checkpoint = _save_checkpoint(model, args.stage0_checkpoint)
    payload = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "qwen_gsm8k_soft_targets_stage0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in LOCKED_HASHES},
        "metadata": {
            "protocol": "qwen_gsm8k_soft_targets_v1",
            "seed": args.seed,
            "model": str(args.model.resolve()),
            "model_inventory_sha256": MODEL_INVENTORY_SHA256,
            "trainable": "all",
            "trainable_names": names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "settings": SETTINGS,
        },
        "training": training,
        "benchmark": benchmark,
        "anchor_manifest": anchor_manifest,
        "replay_rows": replay,
        "replay_sha256": _stable_hash(replay),
        "checkpoint": checkpoint,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _atomic_json(args.output, payload)
    return payload


def _branch(args, source_hash: str, protocol_hash: str, dependencies: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    _set_seed(args.seed)
    device, tokenizer, tasks, pad_id = _load_context(args)
    canonical, checkpoint = _audit_stage0(args, tasks, source_hash, protocol_hash, dependencies)
    model = AutoModelForCausalLM.from_pretrained(
        args.stage0_checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    names, parameters = _select_parameters(model, "all")
    teacher = None
    if args.method == "cagd":
        teacher = AutoModelForCausalLM.from_pretrained(
            args.stage0_checkpoint, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        teacher.config.use_cache = False
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(device)
    training = _train(
        model, teacher, tasks[1]["train"], canonical["replay_rows"], parameters,
        pad_id, device, args, 1,
    )
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    current = _evaluate(model, tasks[1]["eval"], pad_id, device, args.eval_batch_size)
    final = _benchmark(
        model, tasks[0]["eval"][:args.benchmark_limit or None], tokenizer, pad_id, device,
        args.generation_batch_size, args.benchmark_max_new_tokens,
    )
    _audit_benchmark(final, tasks[0]["eval"][:args.benchmark_limit or None])
    initial_count, conditional = _conditional_retention(canonical["benchmark"], final)
    branch_wall = time.monotonic() - started
    summary = {
        "gsm8k_exact_match_when_learned": canonical["benchmark"]["exact_match"],
        "gsm8k_exact_match_final": final["exact_match"],
        "gsm8k_retention_change": final["exact_match"] - canonical["benchmark"]["exact_match"],
        "gsm8k_initially_correct_count": initial_count,
        "gsm8k_initially_correct_retention": conditional,
        "format_compliance_final": final["delimiter_rate"],
        "new_task_loss": current["loss"],
        "new_task_answer_token_accuracy": current["answer_token_accuracy"],
        "adaptation_wall_time_seconds": training["wall_time_seconds"],
        "branch_wall_time_seconds": branch_wall,
        "stage0_wall_time_seconds": canonical["wall_time_seconds"],
        "amortized_wall_time_seconds": branch_wall + canonical["wall_time_seconds"] / len(METHODS),
    }
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "qwen_gsm8k_soft_targets",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": branch_wall,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": canonical["data_sha256"],
        "metadata": {
            "protocol": "qwen_gsm8k_soft_targets_v1",
            "method": args.method,
            "seed": args.seed,
            "task_sequence": ["gsm8k", "summarization"],
            "model": str(args.model.resolve()),
            "model_inventory_sha256": MODEL_INVENTORY_SHA256,
            "trainable": "all",
            "trainable_names": names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "settings": SETTINGS,
            "canonical_stage0_json": str(args.stage0_json.resolve()),
            "canonical_stage0_json_sha256": _sha256(args.stage0_json),
            "canonical_stage0_checkpoint": checkpoint,
            "replay_sha256": canonical["replay_sha256"],
            "anchor_manifest": canonical["anchor_manifest"],
        },
        "training": training,
        "new_task_metrics": current,
        "initial_gsm8k_benchmark": canonical["benchmark"],
        "final_gsm8k_benchmark": final,
        "summary": summary,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise RuntimeError("non-finite endpoint")
    _atomic_json(args.output, result)
    return result


def run(args) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock(args.output)
    try:
        _validate(args)
        source_hash, protocol_hash, dependencies = _sha256(Path(__file__)), _sha256(PROTOCOL), _dependencies()
        result = (_stage0 if args.mode == "stage0" else _branch)(
            args, source_hash, protocol_hash, dependencies
        )
        if (_sha256(Path(__file__)), _sha256(PROTOCOL), _dependencies()) != (
            source_hash, protocol_hash, dependencies
        ):
            raise RuntimeError("runner, protocol, or dependency changed during execution")
        print(json.dumps({"status": "ok", "mode": args.mode, "output": str(args.output),
                          "summary": result.get("summary")}, indent=2))
        return result
    finally:
        lock.unlink(missing_ok=True)


def _self_check() -> None:
    initial = {"records": [
        {"example_id": "a", "correct": True},
        {"example_id": "b", "correct": True},
        {"example_id": "c", "correct": False},
    ]}
    final = {"records": [
        {"example_id": "a", "correct": True},
        {"example_id": "b", "correct": False},
        {"example_id": "c", "correct": True},
    ]}
    assert _conditional_retention(initial, final) == (2, 0.5)
    train, evaluate = __import__("reproduction.ar_gsm8k", fromlist=["_raw_gsm"])._raw_gsm()
    assert len(train) == 5250 and len(evaluate) == 1319
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage0", "branch"))
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage0-json", type=Path)
    parser.add_argument("--stage0-checkpoint", type=Path)
    parser.add_argument("--trainable", choices=("last_block", "all"), default="all")
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
    if args.self_check:
        return args
    if args.mode is None or args.output is None:
        parser.error("--mode and --output are required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

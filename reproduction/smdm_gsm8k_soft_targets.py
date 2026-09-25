#!/usr/bin/env python3
"""Matched SMDM GSM8K soft-target versus hard-replay intervention."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction import gsm8k_scale as scale  # noqa: E402
from reproduction import smdm_gsm8k as paired  # noqa: E402


METHODS = ("hard_replay", "cagd")
MODELS = ("smdm_219m", "smdm_1.14b")
SEEDS = (3407, 3408, 3409)
PROTOCOL = ROOT / "report/smdm_gsm8k_soft_targets_protocol.md"
DEPENDENCIES = (*scale.DEPENDENCIES, Path(paired.__file__).resolve(), Path(scale.__file__).resolve())


def _stable(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dependencies() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): scale._sha256(path) for path in DEPENDENCIES}


def _data_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): scale._sha256(path) for path in scale.DATA_HASHES}


def _settings(args) -> dict[str, object]:
    return scale._actual_settings(args, "smdm")


def _assert_sources(source_hash: str, protocol_hash: str, dependencies: dict) -> None:
    if (scale._sha256(Path(__file__)), scale._sha256(PROTOCOL), _dependencies()) != (
        source_hash, protocol_hash, dependencies
    ):
        raise RuntimeError("runner, protocol, or dependency changed during execution")


def _validate(args, spec: dict) -> None:
    errors = []
    if args.model_id not in MODELS or spec["backend"] != "smdm":
        errors.append("runner accepts only SMDM-219M and SMDM-1.14B")
    if args.seed not in SEEDS or args.method not in METHODS:
        errors.append("method or seed is outside the frozen matrix")
    if args.trainable != "all":
        errors.append("the frozen intervention is full-parameter")
    if args.mode == "branch" and (args.stage0_json is None or args.stage0_checkpoint is None):
        errors.append("branch requires --stage0-json and --stage0-checkpoint")
    if args.mode == "stage0" and args.stage0_checkpoint is None:
        errors.append("stage0 requires --stage0-checkpoint")
    if not PROTOCOL.is_file() or not PROTOCOL.read_text().startswith(
        "# SMDM GSM8K soft-target intervention\n\nStatus: frozen\n"
    ):
        errors.append("protocol is not frozen")
    if not spec["path"].is_file() or scale._sha256(spec["path"]) != spec["checkpoint_sha256"]:
        errors.append("base SMDM checkpoint differs")
    if scale._tree_sha256(scale.TOKENIZER) != scale.TOKENIZER_SHA256:
        errors.append("tokenizer differs")
    for path, wanted in scale.DATA_HASHES.items():
        if not path.is_file() or scale._sha256(path) != wanted:
            errors.append(f"data hash differs: {path}")
    if args.formal:
        locked = {**scale.COMMON_FORMAL_SETTINGS, **scale.SMDM_FORMAL_SETTINGS}
        locked["generation_batch_size"] = 8
        for key, wanted in locked.items():
            if getattr(args, key) != wanted:
                errors.append(f"{key}={getattr(args, key)!r}, expected {wanted!r}")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def _audit_benchmark(benchmark: dict, expected: list[dict]) -> None:
    records = benchmark.get("records")
    if benchmark.get("count") != len(expected) or not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("benchmark cardinality differs")
    correct = marked = 0
    for record, row in zip(records, expected):
        for key in ("example_id", "source_index", "question", "target"):
            if record.get(key) != row.get(key):
                raise ValueError(f"benchmark {key} differs")
        prediction, has_delimiter = scale.qwen_base._extract_answer(record.get("output", ""))
        is_correct = prediction == row["target"]
        if (record.get("prediction"), record.get("has_delimiter"), record.get("correct")) != (
            prediction, has_delimiter, is_correct
        ):
            raise ValueError("benchmark derived fields differ")
        correct += is_correct
        marked += has_delimiter
    count = len(expected)
    if count and (
        not math.isclose(benchmark.get("exact_match", -1), correct / count, abs_tol=1e-15)
        or not math.isclose(benchmark.get("delimiter_rate", -1), marked / count, abs_tol=1e-15)
    ):
        raise ValueError("benchmark aggregate differs")


def _conditional_retention(initial: dict, final: dict, formal: bool) -> tuple[int, float]:
    initial_rows, final_rows = initial["records"], final["records"]
    if [row["example_id"] for row in initial_rows] != [row["example_id"] for row in final_rows]:
        raise ValueError("initial and final GSM8K records are not aligned")
    selected = [row["correct"] for row in initial_rows]
    count = sum(selected)
    if not count:
        if formal:
            raise ValueError("no initially correct GSM8K examples")
        return 0, 0.0
    retained = sum(was_correct and row["correct"] for was_correct, row in zip(selected, final_rows))
    return count, retained / count


def _load_context(args, spec: dict):
    device, tokenizer, tasks, model, parameters = paired._load_context(args, spec)
    tasks = tasks[:2]
    if [task["name"] for task in tasks] != ["gsm8k", "summarization"]:
        raise RuntimeError("task stream differs")
    return device, tokenizer, tasks, model, parameters


def _audit_stage0(args, spec: dict, tasks: list[dict], source_hash: str,
                  protocol_hash: str, dependencies: dict) -> tuple[dict, dict]:
    if not args.stage0_json.is_file() or not args.stage0_checkpoint.is_file():
        raise FileNotFoundError("stage0 JSON or checkpoint is missing")
    payload = json.loads(args.stage0_json.read_text())
    for key, wanted in {
        "status": "ok",
        "experiment": "smdm_gsm8k_soft_targets_stage0",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": _data_hashes(),
        "tokenizer_sha256": scale._tree_sha256(scale.TOKENIZER),
    }.items():
        if payload.get(key) != wanted:
            raise ValueError(f"stage0 {key} differs")
    metadata = payload.get("metadata", {})
    for key, wanted in {
        "formal": bool(args.formal), "seed": args.seed, "model_id": args.model_id,
        "trainable": "all", "settings": _settings(args),
        "base_checkpoint_sha256": spec["checkpoint_sha256"],
    }.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"stage0 metadata {key} differs")
    checkpoint = paired._file_descriptor(args.stage0_checkpoint)
    if payload.get("checkpoint") != checkpoint:
        raise ValueError("stage0 checkpoint differs")
    expected = tasks[0]["eval"][:args.benchmark_limit or None]
    _audit_benchmark(payload.get("benchmark", {}), expected)
    replay = payload.get("replay_rows")
    anchors = tasks[0]["train"][:args.replay_per_task]
    if not isinstance(replay, list) or len(replay) != len(anchors) or payload.get("replay_sha256") != _stable(replay):
        raise ValueError("stage0 replay differs")
    for generated, anchor in zip(replay, anchors):
        if generated.get("ids", [])[:generated.get("answer_start", -1)] != anchor["prompt_ids"]:
            raise ValueError("stage0 replay prompt differs")
    return payload, checkpoint


def _stage0(args, spec: dict, source_hash: str, protocol_hash: str, dependencies: dict) -> dict:
    import torch
    import transformers
    from reproduction.smdm_backend import set_seed
    from reproduction.smdm_factual import _train_stage

    started = time.monotonic()
    set_seed(args.seed)
    device, tokenizer, tasks, model, parameters = _load_context(args, spec)
    torch.cuda.reset_peak_memory_stats(device)
    pad_id = int(tokenizer.eos_token_id)
    training = _train_stage(
        model, tasks[0]["train"], parameters, pad_id, device, args,
        args.seed + 1000, teacher=None, replay_rows=[], replay_objective=None,
    )
    torch.cuda.empty_cache()
    benchmark_rows = tasks[0]["eval"][:args.benchmark_limit or None]
    benchmark = scale._benchmark(model, benchmark_rows, tokenizer, device, args)
    anchors, anchor_manifest = scale._anchors(tasks, 1, args.replay_per_task)
    replay = scale._smdm_generate(
        model, anchors, tokenizer, device, args.generation_batch_size,
        args.replay_steps, args.replay_max_new_tokens, args.replay_cfg, False,
    )
    _audit_benchmark(benchmark, benchmark_rows)
    _assert_sources(source_hash, protocol_hash, dependencies)
    checkpoint = paired._atomic_checkpoint(args.stage0_checkpoint, model)
    payload = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "smdm_gsm8k_soft_targets_stage0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": _data_hashes(),
        "tokenizer_sha256": scale._tree_sha256(scale.TOKENIZER),
        "metadata": {
            "formal": bool(args.formal),
            "protocol": "smdm_gsm8k_soft_targets_v1" if args.formal else "development",
            "model_id": args.model_id,
            "model_display_name": spec["display_name"],
            "seed": args.seed,
            "base_checkpoint": str(spec["path"].resolve()),
            "base_checkpoint_sha256": spec["checkpoint_sha256"],
            "model_parameter_count": spec["parameter_count"],
            "trainable": "all",
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "settings": _settings(args),
        },
        "training": training,
        "benchmark": benchmark,
        "anchor_manifest": anchor_manifest,
        "replay_rows": replay,
        "replay_sha256": _stable(replay),
        "checkpoint": checkpoint,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _assert_sources(source_hash, protocol_hash, dependencies)
    paired._atomic_json(args.output, payload)
    return payload


def _branch(args, spec: dict, source_hash: str, protocol_hash: str, dependencies: dict) -> dict:
    import torch
    import transformers
    from safetensors.torch import load_file
    from reproduction.smdm_backend import answer_token_accuracy, load_model, set_seed
    from reproduction.smdm_factual import _train_stage
    from reproduction.smdm_transfer import _evaluate_loss

    started = time.monotonic()
    set_seed(args.seed)
    device, tokenizer, tasks, model, parameters = _load_context(args, spec)
    canonical, checkpoint = _audit_stage0(args, spec, tasks, source_hash, protocol_hash, dependencies)
    state = load_file(str(args.stage0_checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    teacher = None
    if args.method == "cagd":
        teacher = load_model(args, device)
        teacher.load_state_dict(model.state_dict(), strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(device)
    pad_id = int(tokenizer.eos_token_id)
    training = _train_stage(
        model, tasks[1]["train"], parameters, pad_id, device, args, args.seed + 2000,
        teacher=teacher, replay_rows=canonical["replay_rows"],
        replay_objective="soft" if teacher is not None else "hard",
    )
    if teacher is not None:
        del teacher
    torch.cuda.empty_cache()
    current = {
        "loss": _evaluate_loss(
            model, tasks[1]["eval"], pad_id, device, args,
            args.seed + 81_001 + 100 * tasks[1]["task_index"],
        ),
        "answer_token_accuracy": answer_token_accuracy(
            model, tasks[1]["eval"], device, args.eval_batch_size, pad_id
        ),
    }
    benchmark_rows = tasks[0]["eval"][:args.benchmark_limit or None]
    final = scale._benchmark(model, benchmark_rows, tokenizer, device, args)
    _audit_benchmark(final, benchmark_rows)
    initial_count, conditional = _conditional_retention(canonical["benchmark"], final, args.formal)
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
        "experiment": "smdm_gsm8k_soft_targets",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": branch_wall,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": canonical["data_sha256"],
        "tokenizer_sha256": canonical["tokenizer_sha256"],
        "metadata": {
            "formal": bool(args.formal),
            "protocol": "smdm_gsm8k_soft_targets_v1" if args.formal else "development",
            "model_id": args.model_id,
            "model_display_name": spec["display_name"],
            "method": args.method,
            "seed": args.seed,
            "task_sequence": ["gsm8k", "summarization"],
            "base_checkpoint": str(spec["path"].resolve()),
            "base_checkpoint_sha256": spec["checkpoint_sha256"],
            "model_parameter_count": spec["parameter_count"],
            "trainable": "all",
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "settings": _settings(args),
            "canonical_stage0_json": str(args.stage0_json.resolve()),
            "canonical_stage0_json_sha256": scale._sha256(args.stage0_json),
            "canonical_stage0_checkpoint": checkpoint,
            "replay_sha256": canonical["replay_sha256"],
            "anchor_manifest": copy.deepcopy(canonical["anchor_manifest"]),
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
    _assert_sources(source_hash, protocol_hash, dependencies)
    paired._atomic_json(args.output, result)
    return result


def run(args) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    output_lock = paired._open_lock(args.output)
    checkpoint_lock = None
    try:
        if args.mode == "stage0":
            args.stage0_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_lock = paired._open_lock(args.stage0_checkpoint)
        spec = scale._model_spec(args)
        _validate(args, spec)
        source_hash = scale._sha256(Path(__file__))
        protocol_hash = scale._sha256(PROTOCOL)
        dependencies = _dependencies()
        result = (_stage0 if args.mode == "stage0" else _branch)(
            args, spec, source_hash, protocol_hash, dependencies
        )
        print(json.dumps({"status": "ok", "mode": args.mode, "output": str(args.output),
                          "summary": result.get("summary"), "resources": result.get("resources")}, indent=2))
        return result
    finally:
        if checkpoint_lock is not None:
            checkpoint_lock.unlink(missing_ok=True)
        output_lock.unlink(missing_ok=True)


def _self_check() -> None:
    initial = {"records": [{"example_id": "a", "correct": True}, {"example_id": "b", "correct": False}]}
    final = {"records": [{"example_id": "a", "correct": False}, {"example_id": "b", "correct": True}]}
    assert _conditional_retention(initial, final, True) == (1, 0.0)
    assert _stable({"b": 2, "a": 1}) == _stable({"a": 1, "b": 2})
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "result.json"
        paired._atomic_json(target, {"status": "ok"})
        try:
            paired._atomic_json(target, {"status": "bad"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("immutable output was overwritten")
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage0", "branch"))
    parser.add_argument("--model-id", choices=MODELS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--trainable", choices=("last_block", "all"), default="all")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage0-json", type=Path)
    parser.add_argument("--stage0-checkpoint", type=Path)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--replay-max-new-tokens", type=int, default=128)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=256)
    parser.add_argument("--benchmark-steps", type=int, default=256)
    parser.add_argument("--benchmark-limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--replay-cfg", type=float, default=0.8)
    parser.add_argument("--benchmark-cfg", type=float, default=0.1)
    parser.add_argument("--eval-mc-samples", type=int, default=16)
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
    parser.add_argument("--record-step-loss", action="store_true")
    parser.add_argument("--record-eval-loss-every", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return args
    if args.mode is None or args.model_id is None or args.output is None:
        parser.error("--mode, --model-id, and --output are required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

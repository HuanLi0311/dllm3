#!/usr/bin/env python3
"""Run SMDM cells with one immutable, shared stage-0 prefix per model/seed.

``stage0`` trains GSM8K once, evaluates it once, then writes a safetensors
checkpoint followed by a JSON manifest. ``branch`` verifies both artifacts,
loads that checkpoint, embeds the manifest's stage record verbatim, and runs
either Sequential or CAGD on the remaining two tasks.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
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

from reproduction import gsm8k_scale as base  # noqa: E402


PROTOCOL = ROOT / "report/cagd_gsm8k_scale_attempt2_protocol.md"
BASE_RUNNER = ROOT / "reproduction/gsm8k_scale.py"
SMDM_MODELS = ("smdm_219m", "smdm_1.14b")
SCHEMA_VERSION = 1


def _dependency_hashes() -> dict[str, str]:
    paths = (*base.DEPENDENCIES, BASE_RUNNER, base.PROTOCOL)
    return {str(path.relative_to(ROOT)): base._sha256(path) for path in paths}


def _data_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): base._sha256(path) for path in base.DATA_HASHES}


def _file_descriptor(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": base._sha256(path),
    }


def _assert_implementation(source_hash: str, protocol_hash: str, dependencies: dict) -> None:
    if (
        base._sha256(Path(__file__)) != source_hash
        or base._sha256(PROTOCOL) != protocol_hash
        or _dependency_hashes() != dependencies
    ):
        raise RuntimeError("runner, protocol, or dependency changed during execution")


def _protocol_status(path: Path) -> str | None:
    return next((line for line in path.read_text().splitlines() if line.startswith("Status:")), None)


def _validate(args: argparse.Namespace, spec: dict) -> None:
    errors = []
    if args.model_id not in SMDM_MODELS or spec["backend"] != "smdm":
        errors.append("attempt-2 paired runner accepts only the two SMDM models")
    if args.mode == "stage0" and args.method != "seq":
        errors.append("canonical stage0 must use --method seq")
    if args.mode == "branch" and args.stage0_json is None:
        errors.append("branch requires --stage0-json")
    if args.mode == "stage0" and args.stage0_json is not None:
        errors.append("stage0 writes --output; do not pass --stage0-json")
    if args.stage0_checkpoint is None:
        errors.append("--stage0-checkpoint is required")
    if not PROTOCOL.is_file():
        errors.append(f"missing attempt-2 protocol: {PROTOCOL}")
    elif args.formal and _protocol_status(PROTOCOL) != "Status: frozen":
        errors.append("attempt-2 protocol is not frozen")
    if not spec["path"].is_file() or base._sha256(spec["path"]) != spec["checkpoint_sha256"]:
        errors.append("base SMDM checkpoint is missing or has a different SHA-256")
    if base._tree_sha256(base.TOKENIZER) != base.TOKENIZER_SHA256:
        errors.append("SMDM tokenizer hash differs")
    for path, wanted in base.DATA_HASHES.items():
        if not path.is_file() or base._sha256(path) != wanted:
            errors.append(f"data hash differs: {path}")
    if args.formal:
        try:
            base._validate(args, spec)
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("attempt-2 protocol mismatch: " + "; ".join(errors))


def _open_lock(target: Path) -> Path:
    lock = Path(str(target) + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
    except FileExistsError as error:
        raise FileExistsError(f"target is already locked: {lock}") from error
    return lock


def _commit_exclusive(temporary: Path, target: Path) -> None:
    """Atomically publish ``temporary`` only when ``target`` is absent."""
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {target}") from error


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, model) -> dict[str, object]:
    """Write model state without replacing an existing artifact."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        save_file(state, str(temporary), metadata={"format": "pt"})
        del state
        _commit_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_descriptor(path)


def _audit_stage_record(stage: dict, expected_rows: list[dict], steps: int) -> None:
    if set(stage) != {
        "stage", "task", "training", "anchor_manifest",
        "generated_replay_count", "current_metrics", "benchmark",
    }:
        raise ValueError("canonical stage record has unexpected fields")
    if (
        stage["stage"] != 0
        or stage["task"] != "gsm8k"
        or stage["anchor_manifest"] != []
        or stage["generated_replay_count"] != 0
        or stage["current_metrics"] is not None
        or stage["training"].get("steps") != steps
    ):
        raise ValueError("canonical stage record structure differs")
    benchmark = stage["benchmark"]
    records = benchmark.get("records")
    if benchmark.get("count") != len(expected_rows) or not isinstance(records, list):
        raise ValueError("canonical benchmark cardinality differs")
    if len(records) != len(expected_rows):
        raise ValueError("canonical benchmark record count differs")
    correct = 0
    marked = 0
    for record, expected in zip(records, expected_rows):
        for key in ("example_id", "source_index", "question", "target"):
            if record.get(key) != expected.get(key):
                raise ValueError(f"canonical benchmark {key} differs")
        prediction, has_delimiter = base.qwen_base._extract_answer(record.get("output", ""))
        is_correct = prediction == expected["target"]
        if (
            record.get("prediction") != prediction
            or record.get("has_delimiter") is not has_delimiter
            or record.get("correct") is not is_correct
        ):
            raise ValueError("canonical benchmark derived fields differ")
        correct += is_correct
        marked += has_delimiter
    count = len(records)
    if not math.isclose(benchmark.get("exact_match", -1), correct / count, abs_tol=1e-15):
        raise ValueError("canonical benchmark exact match differs")
    if not math.isclose(benchmark.get("delimiter_rate", -1), marked / count, abs_tol=1e-15):
        raise ValueError("canonical benchmark delimiter rate differs")


def _audit_canonical(args, spec: dict, tasks: list[dict], source_hash: str,
                     protocol_hash: str, dependencies: dict) -> tuple[dict, dict]:
    if not args.stage0_json.is_file() or not args.stage0_checkpoint.is_file():
        raise FileNotFoundError("canonical stage0 JSON or checkpoint is missing")
    payload = json.loads(args.stage0_json.read_text())
    expected_settings = base._actual_settings(args, "smdm")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "experiment": "cagd_gsm8k_scale_attempt2_stage0",
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "data_sha256": _data_hashes(),
        "tokenizer_sha256": base._tree_sha256(base.TOKENIZER),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"canonical stage0 {key} differs")
    metadata = payload.get("metadata", {})
    checks = {
        "formal": bool(args.formal),
        "protocol": "cagd_gsm8k_scale_paired_stage0_v2" if args.formal else "development",
        "model_id": args.model_id,
        "model_display_name": spec["display_name"],
        "backend": "smdm",
        "seed": args.seed,
        "base_checkpoint": str(spec["path"].resolve()),
        "base_checkpoint_sha256": spec["checkpoint_sha256"],
        "model_parameter_count": spec["parameter_count"],
        "trainable": args.trainable,
        "trainable_parameter_count": spec["parameter_count"],
        "settings": expected_settings,
        "runner_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise ValueError(f"canonical metadata {key} differs")
    descriptor = _file_descriptor(args.stage0_checkpoint)
    if payload.get("canonical_checkpoint") != descriptor:
        raise ValueError("canonical checkpoint bytes, hash, or path differs")
    _audit_stage_record(
        payload.get("stage", {}), tasks[0]["eval"][:args.benchmark_limit or None],
        args.steps_per_task,
    )
    return payload, descriptor


def _load_context(args, spec: dict):
    import torch
    from transformers import AutoTokenizer
    from reproduction.smdm_backend import load_model, trainable_parameters

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SMDM attempt-2 experiments require CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        base.TOKENIZER, local_files_only=True, use_fast=True
    )
    tasks = base._raw_tasks(tokenizer, args.max_length)
    args.model = spec["config_size"]
    args.checkpoint = spec["path"]
    model = load_model(args, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec["parameter_count"]:
        raise RuntimeError(
            f"parameter count {parameter_count} differs from {spec['parameter_count']}"
        )
    return device, tokenizer, tasks, model, trainable_parameters(model, args.trainable)


def _stage0(args, spec: dict, source_hash: str, protocol_hash: str,
            dependencies: dict) -> dict:
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
    stage = {
        "stage": 0,
        "task": "gsm8k",
        "training": training,
        "anchor_manifest": [],
        "generated_replay_count": 0,
        "current_metrics": None,
        "benchmark": base._benchmark(
            model, tasks[0]["eval"][:args.benchmark_limit or None], tokenizer, device, args
        ),
    }
    _audit_stage_record(
        stage, tasks[0]["eval"][:args.benchmark_limit or None], args.steps_per_task
    )
    _assert_implementation(source_hash, protocol_hash, dependencies)
    checkpoint = _atomic_checkpoint(args.stage0_checkpoint, model)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "experiment": "cagd_gsm8k_scale_attempt2_stage0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "metadata": {
            "formal": bool(args.formal),
            "protocol": (
                "cagd_gsm8k_scale_paired_stage0_v2" if args.formal else "development"
            ),
            "model_id": args.model_id,
            "model_display_name": spec["display_name"],
            "backend": "smdm",
            "seed": args.seed,
            "base_checkpoint": str(spec["path"].resolve()),
            "base_checkpoint_sha256": spec["checkpoint_sha256"],
            "model_parameter_count": spec["parameter_count"],
            "trainable": args.trainable,
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "settings": base._actual_settings(args, "smdm"),
            "runner_sha256": source_hash,
            "protocol_sha256": protocol_hash,
            "dependency_sha256": dependencies,
        },
        "data_sha256": _data_hashes(),
        "tokenizer_sha256": base._tree_sha256(base.TOKENIZER),
        "canonical_checkpoint": checkpoint,
        "stage": stage,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _assert_implementation(source_hash, protocol_hash, dependencies)
    _atomic_json(args.output, payload)  # JSON deliberately lands last.
    return payload


def _branch(args, spec: dict, source_hash: str, protocol_hash: str,
            dependencies: dict) -> dict:
    import torch
    import transformers
    from safetensors.torch import load_file
    from reproduction.smdm_backend import answer_token_accuracy, load_model, set_seed
    from reproduction.smdm_factual import _train_stage
    from reproduction.smdm_transfer import _evaluate_loss

    started = time.monotonic()
    set_seed(args.seed)
    device, tokenizer, tasks, model, parameters = _load_context(args, spec)
    canonical, checkpoint = _audit_canonical(
        args, spec, tasks, source_hash, protocol_hash, dependencies
    )
    state = load_file(str(args.stage0_checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    torch.cuda.reset_peak_memory_stats(device)
    pad_id = int(tokenizer.eos_token_id)
    stages = [copy.deepcopy(canonical["stage"])]
    for stage_index, task in enumerate(tasks[1:], start=1):
        print(f"stage={stage_index + 1}/{len(tasks)} task={task['name']}", flush=True)
        teacher = None
        replay = []
        anchor_manifest = []
        if args.method == "cagd":
            teacher = load_model(args, device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            anchors, anchor_manifest = base._anchors(tasks, stage_index, args.replay_per_task)
            replay = base._smdm_generate(
                teacher, anchors, tokenizer, device, args.generation_batch_size,
                args.replay_steps, args.replay_max_new_tokens, args.replay_cfg, False,
            )
        training = _train_stage(
            model, task["train"], parameters, pad_id, device, args,
            args.seed + 1000 * (stage_index + 1), teacher=teacher, replay_rows=replay,
            replay_objective="soft" if teacher is not None else None,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        current_metrics = None
        if stage_index == len(tasks) - 1:
            current_metrics = {
                "loss": _evaluate_loss(
                    model, task["eval"], pad_id, device, args,
                    args.seed + 81_001 + 100 * task["task_index"],
                ),
                "answer_token_accuracy": answer_token_accuracy(
                    model, task["eval"], device, args.eval_batch_size, pad_id
                ),
            }
        benchmark = base._benchmark(
            model, tasks[0]["eval"][:args.benchmark_limit or None], tokenizer, device, args
        ) if stage_index == len(tasks) - 1 else None
        stages.append({
            "stage": stage_index,
            "task": task["name"],
            "training": training,
            "anchor_manifest": anchor_manifest,
            "generated_replay_count": len(replay),
            "current_metrics": current_metrics,
            "benchmark": benchmark,
        })
    learned = stages[0]["benchmark"]["exact_match"]
    final = stages[-1]["benchmark"]["exact_match"]
    result = {
        "schema_version": 2,
        "status": "ok",
        "experiment": "cagd_gsm8k_scale_attempt2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metadata": {
            "formal": bool(args.formal),
            "protocol": (
                "cagd_gsm8k_scale_paired_stage0_v2" if args.formal else "development"
            ),
            "model_id": args.model_id,
            "model_display_name": spec["display_name"],
            "backend": "smdm",
            "method": args.method,
            "seed": args.seed,
            "task_sequence": list(base.TASKS),
            "base_checkpoint": str(spec["path"].resolve()),
            "base_checkpoint_sha256": spec["checkpoint_sha256"],
            "config_name": spec["config_name"],
            "model_parameter_count": spec["parameter_count"],
            "trainable": args.trainable,
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "benchmark_count": len(tasks[0]["eval"][:args.benchmark_limit or None]),
            "generation_batch_size": args.generation_batch_size,
            "benchmark_decoding": "greedy_masked_diffusion_equal_prompt_length_v1",
            "completion_only_scoring": True,
            "settings": base._actual_settings(args, "smdm"),
            "runner_sha256": source_hash,
            "protocol_sha256": protocol_hash,
            "dependency_sha256": dependencies,
            "canonical_stage0_json": str(args.stage0_json.resolve()),
            "canonical_stage0_json_sha256": base._sha256(args.stage0_json),
            "canonical_stage0_checkpoint": checkpoint,
        },
        "data_sha256": _data_hashes(),
        "tokenizer_sha256": base._tree_sha256(base.TOKENIZER),
        "stages": stages,
        "summary": {
            "gsm8k_exact_match_when_learned": learned,
            "gsm8k_exact_match_final": final,
            "gsm8k_retention_change": final - learned,
            "final_task_loss": stages[-1]["current_metrics"]["loss"],
            "final_task_answer_token_accuracy": (
                stages[-1]["current_metrics"]["answer_token_accuracy"]
            ),
        },
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _assert_implementation(source_hash, protocol_hash, dependencies)
    if not all(math.isfinite(value) for value in result["summary"].values()):
        raise RuntimeError("non-finite endpoint")
    _atomic_json(args.output, result)
    return result


def run(args: argparse.Namespace) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.mode == "stage0"
        and args.stage0_checkpoint is not None
        and args.stage0_checkpoint.exists()
    ):
        raise FileExistsError(f"refusing to overwrite {args.stage0_checkpoint}")
    output_lock = _open_lock(args.output)
    checkpoint_lock = None
    try:
        if args.mode == "stage0":
            args.stage0_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_lock = _open_lock(args.stage0_checkpoint)
        spec = base._model_spec(args)
        _validate(args, spec)
        source_hash = base._sha256(Path(__file__))
        protocol_hash = base._sha256(PROTOCOL)
        dependencies = _dependency_hashes()
        result = (
            _stage0(args, spec, source_hash, protocol_hash, dependencies)
            if args.mode == "stage0"
            else _branch(args, spec, source_hash, protocol_hash, dependencies)
        )
        print(json.dumps({
            "status": "ok",
            "mode": args.mode,
            "output": str(args.output),
            "model_id": args.model_id,
            "method": args.method,
            "seed": args.seed,
            "summary": result.get("summary"),
            "resources": result.get("resources"),
        }, indent=2))
        return result
    finally:
        if checkpoint_lock is not None:
            checkpoint_lock.unlink(missing_ok=True)
        output_lock.unlink(missing_ok=True)


def _self_check() -> None:
    expected = [{
        "example_id": "gsm8k-test-0", "source_index": 0,
        "question": "1+1?", "target": "2",
    }]
    record = {
        **expected[0], "prediction": "2", "has_delimiter": True,
        "correct": True, "output": "work\n#### 2",
    }
    stage = {
        "stage": 0,
        "task": "gsm8k",
        "training": {"steps": 2},
        "anchor_manifest": [],
        "generated_replay_count": 0,
        "current_metrics": None,
        "benchmark": {
            "count": 1, "exact_match": 1.0, "delimiter_rate": 1.0,
            "wall_time_seconds": 0.0, "records": [record],
        },
    }
    _audit_stage_record(stage, expected, 2)
    broken = copy.deepcopy(stage)
    broken["benchmark"]["records"][0]["target"] = "3"
    try:
        _audit_stage_record(broken, expected, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered canonical target was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "sentinel"
        path.write_bytes(b"paired-stage0")
        descriptor = _file_descriptor(path)
        assert descriptor["bytes"] == 13
        assert descriptor["sha256"] == base._sha256(path)
        output = root / "result.json"
        _atomic_json(output, {"version": 1})
        try:
            _atomic_json(output, {"version": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive JSON commit overwrote an existing target")
        assert json.loads(output.read_text()) == {"version": 1}
        protocol = root / "protocol.md"
        protocol.write_text("Status: freeze candidate\ntext says Status: frozen later\n")
        assert _protocol_status(protocol) == "Status: freeze candidate"
        protocol.write_text("Status: frozen\n")
        assert _protocol_status(protocol) == "Status: frozen"
    assert copy.deepcopy(stage) == stage
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage0", "branch"), required=True)
    parser.add_argument("--model-id", choices=SMDM_MODELS, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--method", choices=base.METHODS, default="seq")
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

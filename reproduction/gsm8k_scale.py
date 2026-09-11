#!/usr/bin/env python3
"""Run missing checkpoints in the five-model GSM8K retention matrix.

The completed Qwen3-0.6B cell is deliberately rejected here: its six formal
runs are reused verbatim by ``summarize_gsm8k_scale.py``.  Qwen checkpoints
delegate to the audited AR runner.  SMDM checkpoints use answer-only masked
diffusion training and deterministic, prompt-length-safe diffusion decoding.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction import ar_gsm8k as qwen_base  # noqa: E402


METHODS = ("seq", "cagd")
SEEDS = (3407, 3408, 3409)
TASKS = ("gsm8k", "summarization", "creative_writing")
PROTOCOL = ROOT / "report/cagd_gsm8k_scale_protocol.md"
GSM_TRAIN = ROOT / "third_party/SMDM/data/gsm8k/train_no_aug.txt"
GSM_TEST = ROOT / "third_party/SMDM/data/gsm8k/test.jsonl"
DOLLY = ROOT / "runs/data/dolly_natural_stream.jsonl"
TOKENIZER = ROOT / "tokenizer"
DATA_HASHES = {
    GSM_TRAIN: "52ebf7c73927f7434abbb2f7b705a82fb3dbdd4695438b7654de78b701c23b36",
    GSM_TEST: "8530a3775b96385370842171f226d83de7c5b27d779be54ef0411d255a939818",
    DOLLY: "a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0",
}
QWEN_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
MODEL_SPECS = {
    "qwen3_0.6b": {
        "backend": "ar",
        "display_name": "Qwen3-0.6B",
        "path": QWEN_CACHE / "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "inventory_sha256": "768bd5491a8c872f18c9edb62a8c4ece300963443fb1fa8b7b7be1d3d1fc053b",
    },
    "qwen3_1.7b": {
        "backend": "ar",
        "display_name": "Qwen3-1.7B",
        "path": QWEN_CACHE / "models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "inventory_sha256": "ec3da139e35f8bdd716b8f3858ebb9ba5042629b0e5bbd122a283ddfcea48ade",
    },
    "qwen3_4b": {
        "backend": "ar",
        "display_name": "Qwen3-4B",
        "path": QWEN_CACHE / "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "inventory_sha256": "62d9e88fca7b194225dbc4d7caafc7d64b96608c33fa68103a1cf39e98b187e8",
    },
    "smdm_219m": {
        "backend": "smdm",
        "display_name": "SMDM-219M",
        "config_name": "Diff_LLaMA_170M",
        "config_size": 170,
        "parameter_count": 219_050_496,
        "path": ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors",
        "checkpoint_sha256": "2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb",
    },
    "smdm_1.14b": {
        "backend": "smdm",
        "display_name": "SMDM-1.14B",
        "config_name": "Diff_LLaMA_1028M",
        "config_size": 1028,
        "parameter_count": 1_142_367_744,
        "path": ROOT.parent / "checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors",
        "checkpoint_sha256": "ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6",
    },
}
TOKENIZER_SHA256 = "f2bb6b928f472c0a69142cf12ab7b6d74b7e43050f561eded292ac33a2edf559"
COMMON_FORMAL_SETTINGS = {
    "steps_per_task": 1000,
    "batch_size": 2,
    "eval_batch_size": 4,
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
SMDM_FORMAL_SETTINGS = {
    "replay_steps": 32,
    "benchmark_steps": 256,
    "mask_min": 1e-3,
    "mask_max": 1.0,
    "replay_cfg": 0.8,
    "benchmark_cfg": 0.1,
    "eval_mc_samples": 16,
    "ewc_lambda": 0.0,
}
DEPENDENCIES = (
    ROOT / "reproduction/continual_benchmark.py",
    ROOT / "reproduction/smdm_backend.py",
    ROOT / "reproduction/smdm_factual.py",
    ROOT / "reproduction/smdm_transfer.py",
    ROOT / "reproduction/ar_gsm8k.py",
    ROOT / "reproduction/ar_natural.py",
    ROOT / "reproduction/ar_factual.py",
)


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


def _safetensor_parameter_count(path: Path) -> int:
    from safetensors import safe_open

    count = 0
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                count += math.prod(handle.get_slice(name).get_shape())
    if not count:
        raise ValueError(f"no safetensor parameters found in {path}")
    return count


def _dependency_hashes() -> dict:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in DEPENDENCIES}


def _model_spec(args) -> dict:
    spec = dict(MODEL_SPECS[args.model_id])
    spec["path"] = args.model_path or spec["path"]
    return spec


def _validate(args, spec: dict) -> None:
    errors = []
    if args.model_id not in MODEL_SPECS:
        errors.append("unknown model")
    if args.method not in METHODS or args.seed not in SEEDS:
        errors.append("method or seed is outside the formal matrix")
    locked = dict(COMMON_FORMAL_SETTINGS)
    if spec["backend"] == "smdm":
        locked.update(SMDM_FORMAL_SETTINGS)
    for key, wanted in locked.items():
        if getattr(args, key) != wanted:
            errors.append(f"{key}={getattr(args, key)!r}, expected {wanted!r}")
    expected_generation_batch = 16 if spec["backend"] == "ar" else 8
    if args.generation_batch_size != expected_generation_batch:
        errors.append(
            f"generation_batch_size={args.generation_batch_size!r}, "
            f"expected {expected_generation_batch!r} for {spec['backend']}"
        )
    for path, wanted in DATA_HASHES.items():
        if not path.is_file() or _sha256(path) != wanted:
            errors.append(f"data hash differs: {path}")
    if not spec["path"].exists():
        errors.append(f"missing model: {spec['path']}")
    elif spec["backend"] == "ar":
        if qwen_base._model_inventory(spec["path"])["sha256"] != spec["inventory_sha256"]:
            errors.append("Qwen snapshot inventory differs")
    else:
        if _sha256(spec["path"]) != spec["checkpoint_sha256"]:
            errors.append("SMDM checkpoint hash differs")
        if _tree_sha256(TOKENIZER) != TOKENIZER_SHA256:
            errors.append("SMDM tokenizer hash differs")
    if not PROTOCOL.is_file() or "Status: frozen" not in PROTOCOL.read_text():
        errors.append("protocol is not frozen")
    if errors:
        raise ValueError("protocol mismatch: " + "; ".join(errors))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)


def _formal_provenance(args, spec, source_hash, protocol_hash, dependencies) -> dict:
    return {
        "formal": bool(args.formal),
        "protocol": "cagd_gsm8k_scale_v1" if args.formal else "development",
        "model_id": args.model_id,
        "model_display_name": spec["display_name"],
        "backend": spec["backend"],
        "runner_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "dependency_sha256": dependencies,
    }


def _actual_settings(args, backend: str) -> dict:
    keys = list(COMMON_FORMAL_SETTINGS) + ["generation_batch_size"]
    if backend == "smdm":
        keys += list(SMDM_FORMAL_SETTINGS)
    return {key: getattr(args, key) for key in keys}


def _run_qwen(args, spec: dict, source_hash: str, protocol_hash: str, dependencies: dict) -> dict:
    import torch

    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.base.json")
    base_args = copy.copy(args)
    base_args.model = spec["path"]
    base_args.output = temporary
    base_args.formal = False
    try:
        result = qwen_base.run(base_args)
        if result["metadata"]["benchmark_count"] != (args.benchmark_limit or 1319):
            raise RuntimeError("Qwen benchmark count differs")
        result["schema_version"] = 2
        result["experiment"] = "cagd_gsm8k_scale"
        result["source_sha256"] = source_hash
        result["protocol_sha256"] = protocol_hash
        result["dependency_sha256"] = dependencies
        result["metadata"].update(_formal_provenance(
            args, spec, source_hash, protocol_hash, dependencies
        ))
        result["metadata"]["trainable"] = args.trainable
        result["metadata"]["generation_batch_size"] = args.generation_batch_size
        result["metadata"]["model_inventory_sha256"] = result["model_inventory"]["sha256"]
        result["metadata"]["model_parameter_count"] = _safetensor_parameter_count(spec["path"])
        result["metadata"]["settings"] = _actual_settings(args, spec["backend"])
        result["resources"] = {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)),
        }
        if args.formal:
            _validate(args, spec)
        if _sha256(Path(__file__)) != source_hash or _sha256(PROTOCOL) != protocol_hash:
            raise RuntimeError("scale runner or protocol changed during execution")
        _atomic_json(args.output, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _raw_tasks(tokenizer, max_length: int) -> list[dict]:
    from reproduction.continual_benchmark import encode_benchmark_rows

    gsm_train, gsm_eval = qwen_base._raw_gsm()
    dolly = [json.loads(line) for line in DOLLY.read_text().splitlines() if line.strip()]
    raw_tasks = [("gsm8k", gsm_train, gsm_eval)]
    for name in TASKS[1:]:
        raw_tasks.append((
            name,
            [row for row in dolly if row["task"] == name and row["split"] == "train"],
            [row for row in dolly if row["task"] == name and row["split"] == "test"],
        ))
    tasks = []
    for task_index, (name, train_raw, eval_raw) in enumerate(raw_tasks):
        expected_train, expected_eval = (5250, 1319) if name == "gsm8k" else (120, 40)
        if len(train_raw) != expected_train or len(eval_raw) != expected_eval:
            raise ValueError(
                f"{name}: raw counts {(len(train_raw), len(eval_raw))} differ from "
                f"{(expected_train, expected_eval)}"
            )

        def encode(rows):
            encoded = []
            for row in rows:
                single = encode_benchmark_rows([row], tokenizer, max_length)
                if len(single) != 1:
                    raise ValueError(f"{row['example_id']}: row was not encoded exactly once")
                item = single[0]
                prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
                item.update({
                    "prompt": row["prompt"],
                    "prompt_ids": prompt_ids,
                    "source_index": row["source_index"],
                    "example_id": row["example_id"],
                    "question": row.get("question"),
                    "target": row.get("target"),
                })
                encoded.append(item)
            if len(encoded) != len(rows):
                raise ValueError(f"{name}: encoded row count differs")
            return encoded
        tasks.append({
            "task_index": task_index,
            "name": name,
            "train_raw": train_raw,
            "eval_raw": eval_raw,
            "train": encode(train_raw),
            "eval": encode(eval_raw),
        })
    return tasks


def _anchors(tasks: list, seen: int, per_task: int) -> tuple:
    anchors, manifest = [], []
    for task in tasks[:seen]:
        selected = task["train"][:per_task]
        if len(selected) != per_task:
            raise ValueError(f"{task['name']}: insufficient anchors")
        anchors.extend(selected)
        manifest.append({
            "task": task["name"],
            "count": len(selected),
            "source_indices": [row["source_index"] for row in selected],
            "prompt_sha256": hashlib.sha256(json.dumps(
                [row["prompt_ids"] for row in selected], separators=(",", ":")
            ).encode()).hexdigest(),
        })
    return anchors, manifest


def _diffuse_equal_prompt_length(model, prompt_ids: list, device, steps: int,
                                 max_new_tokens: int, cfg: float):
    """Greedy masked-diffusion decoding with per-example transfer counts."""
    import torch
    from reproduction.continual_benchmark import MASK_ID

    if not prompt_ids or len({len(row) for row in prompt_ids}) != 1:
        raise ValueError("generation batch must have one exact prompt length")
    prompt_length = len(prompt_ids[0])
    context_length = prompt_length + max_new_tokens
    x = torch.full(
        (len(prompt_ids), context_length), MASK_ID, dtype=torch.long, device=device
    )
    for index, row in enumerate(prompt_ids):
        x[index, :prompt_length] = torch.tensor(row, dtype=torch.long, device=device)
    timesteps = torch.linspace(1, 1e-5, steps + 1, device=device)
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            mask = x == MASK_ID
            if cfg > 0:
                unconditional = x.clone()
                unconditional[:, :prompt_length] = MASK_ID
                logits = model(torch.cat((x, unconditional), dim=0))
                logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + (cfg + 1) * (logits - uncond_logits)
            else:
                logits = model(x)
            fraction = float(1 - timesteps[step + 1] / timesteps[step])
            for row_index in range(len(prompt_ids)):
                positions = mask[row_index].nonzero(as_tuple=False).flatten()
                if not len(positions):
                    continue
                selected_logits = logits[row_index, positions]
                predictions = selected_logits.argmax(dim=-1)
                transfer = len(positions) if step + 1 == steps else int(len(positions) * fraction)
                if transfer:
                    if transfer == len(positions):
                        chosen = torch.arange(len(positions), device=device)
                    else:
                        confidence = selected_logits.float().log_softmax(dim=-1).amax(dim=-1)
                        chosen = confidence.topk(transfer).indices
                    x[row_index, positions[chosen]] = predictions[chosen]
                    if transfer != len(positions):
                        del confidence
                del positions, selected_logits, predictions
            if cfg > 0:
                del unconditional, uncond_logits
            del logits, mask
    return x


def _smdm_generate(model, rows: list, tokenizer, device, batch_size: int,
                   steps: int, max_new_tokens: int, cfg: float, records: bool) -> list:
    grouped = defaultdict(list)
    for output_index, row in enumerate(rows):
        grouped[len(row["prompt_ids"])].append((output_index, row))
    outputs = [None] * len(rows)
    completed = 0
    for prompt_length in sorted(grouped):
        group = grouped[prompt_length]
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            generated = _diffuse_equal_prompt_length(
                model, [row["prompt_ids"] for _, row in batch], device,
                steps, max_new_tokens, cfg,
            ).cpu().tolist()
            for (output_index, row), token_ids in zip(batch, generated):
                completion, text = _decode_completion(token_ids, prompt_length, tokenizer)
                if records:
                    prediction, marked = qwen_base._extract_answer(text)
                    outputs[output_index] = {
                        "example_id": row["example_id"],
                        "source_index": row["source_index"],
                        "question": row["question"],
                        "target": row["target"],
                        "prediction": prediction,
                        "has_delimiter": marked,
                        "correct": prediction == row["target"],
                        "output": text,
                    }
                else:
                    outputs[output_index] = {
                        "ids": row["prompt_ids"] + completion,
                        "answer_start": prompt_length,
                        "answer_end": prompt_length + len(completion),
                        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    }
                    if not completion:
                        raise RuntimeError("teacher generated an empty replay completion")
            completed += len(batch)
            print(f"generated={completed}/{len(rows)} prompt_tokens={prompt_length}", flush=True)
    return outputs


def _decode_completion(token_ids: list, prompt_length: int, tokenizer) -> tuple:
    completion = token_ids[prompt_length:]
    if tokenizer.eos_token_id in completion:
        completion = completion[:completion.index(tokenizer.eos_token_id)]
    return completion, tokenizer.decode(completion, skip_special_tokens=True).strip()


def _benchmark(model, rows, tokenizer, device, args) -> dict:
    started = time.monotonic()
    records = _smdm_generate(
        model, rows, tokenizer, device, args.generation_batch_size,
        args.benchmark_steps, args.benchmark_max_new_tokens, args.benchmark_cfg, True,
    )
    return {
        "count": len(records),
        "exact_match": sum(row["correct"] for row in records) / len(records),
        "delimiter_rate": sum(row["has_delimiter"] for row in records) / len(records),
        "wall_time_seconds": time.monotonic() - started,
        "records": records,
    }


def _run_smdm(args, spec: dict, source_hash: str, protocol_hash: str,
              dependencies: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer
    from reproduction.smdm_backend import answer_token_accuracy, load_model, set_seed, trainable_parameters
    from reproduction.smdm_factual import _train_stage
    from reproduction.smdm_transfer import _evaluate_loss

    started = time.monotonic()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SMDM scale experiments require CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    tasks = _raw_tasks(tokenizer, args.max_length)
    if args.formal:
        _validate(args, spec)
    benchmark_rows = tasks[0]["eval"][:args.benchmark_limit or None]
    args.model = spec["config_size"]
    args.checkpoint = spec["path"]
    model = load_model(args, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec["parameter_count"]:
        raise RuntimeError(f"parameter count {parameter_count} differs from {spec['parameter_count']}")
    parameters = trainable_parameters(model, args.trainable)
    pad_id = int(tokenizer.eos_token_id)
    stages = []
    for stage, task in enumerate(tasks):
        print(f"stage={stage + 1}/{len(tasks)} task={task['name']}", flush=True)
        teacher = None
        replay = []
        anchor_manifest = []
        if stage and args.method == "cagd":
            teacher = load_model(args, device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            anchors, anchor_manifest = _anchors(tasks, stage, args.replay_per_task)
            replay = _smdm_generate(
                teacher, anchors, tokenizer, device, args.generation_batch_size,
                args.replay_steps, args.replay_max_new_tokens, args.replay_cfg, False,
            )
        training = _train_stage(
            model, task["train"], parameters, pad_id, device, args,
            args.seed + 1000 * (stage + 1), teacher=teacher, replay_rows=replay,
            replay_objective="soft" if teacher is not None else None,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        current_metrics = None
        if stage == len(tasks) - 1:
            current_metrics = {
                "loss": _evaluate_loss(
                    model, task["eval"], pad_id, device, args,
                    args.seed + 81_001 + 100 * task["task_index"],
                ),
                "answer_token_accuracy": answer_token_accuracy(
                    model, task["eval"], device, args.eval_batch_size, pad_id
                ),
            }
        benchmark = _benchmark(model, benchmark_rows, tokenizer, device, args) \
            if stage in (0, len(tasks) - 1) else None
        stages.append({
            "stage": stage,
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
        "experiment": "cagd_gsm8k_scale",
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
            **_formal_provenance(args, spec, source_hash, protocol_hash, dependencies),
            "method": args.method,
            "seed": args.seed,
            "task_sequence": list(TASKS),
            "checkpoint": str(spec["path"].resolve()),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "config_name": spec["config_name"],
            "model_parameter_count": parameter_count,
            "trainable": args.trainable,
            "trainable_parameter_count": sum(p.numel() for p in parameters),
            "benchmark_count": len(benchmark_rows),
            "generation_batch_size": args.generation_batch_size,
            "benchmark_decoding": "greedy_masked_diffusion_equal_prompt_length_v1",
            "completion_only_scoring": True,
            "settings": _actual_settings(args, spec["backend"]),
        },
        "data_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in DATA_HASHES},
        "tokenizer_sha256": _tree_sha256(TOKENIZER),
        "stages": stages,
        "summary": {
            "gsm8k_exact_match_when_learned": learned,
            "gsm8k_exact_match_final": final,
            "gsm8k_retention_change": final - learned,
            "final_task_loss": stages[-1]["current_metrics"]["loss"],
            "final_task_answer_token_accuracy": stages[-1]["current_metrics"]["answer_token_accuracy"],
        },
        "resources": {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    if args.formal:
        _validate(args, spec)
    if (
        _sha256(Path(__file__)) != source_hash
        or _sha256(PROTOCOL) != protocol_hash
        or _dependency_hashes() != dependencies
    ):
        raise RuntimeError("runner, protocol, or dependency changed during execution")
    if not all(math.isfinite(value) for value in result["summary"].values()):
        raise RuntimeError("non-finite endpoint")
    _atomic_json(args.output, result)
    return result


def run(args) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = Path(str(args.output) + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
    except FileExistsError as error:
        raise FileExistsError(f"output is already locked: {lock}") from error
    try:
        spec = _model_spec(args)
        if args.formal:
            _validate(args, spec)
        source_hash = _sha256(Path(__file__))
        protocol_hash = _sha256(PROTOCOL)
        dependencies = _dependency_hashes()
        result = (
            _run_qwen(args, spec, source_hash, protocol_hash, dependencies)
            if spec["backend"] == "ar"
            else _run_smdm(args, spec, source_hash, protocol_hash, dependencies)
        )
        print(json.dumps({
            "status": "ok",
            "output": str(args.output),
            "model_id": args.model_id,
            "method": args.method,
            "seed": args.seed,
            "summary": result["summary"],
            "resources": result.get("resources"),
        }, indent=2))
        return result
    finally:
        lock.unlink(missing_ok=True)


def _self_check() -> None:
    import torch

    class ConstantModel:
        def eval(self):
            return self

        def __call__(self, ids):
            logits = torch.zeros((*ids.shape, 8))
            logits[..., 7] = 1
            return logits

    class TinyTokenizer:
        eos_token_id = 6

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            del skip_special_tokens
            return ",".join(map(str, ids))

    assert set(MODEL_SPECS) == {"qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "smdm_219m", "smdm_1.14b"}
    assert qwen_base._extract_answer("work\n#### 1,234") == ("1234", True)
    assert qwen_base._extract_answer("therefore -2.50") == ("-5/2", False)
    train, evaluate = qwen_base._raw_gsm()
    assert len(train) == 5250 and len(evaluate) == 1319
    assert COMMON_FORMAL_SETTINGS["benchmark_max_new_tokens"] == 256
    generated = _diffuse_equal_prompt_length(
        ConstantModel(), [[1, 2], [3, 4]], torch.device("cpu"), 2, 3, 0.1
    )
    assert generated.tolist() == [[1, 2, 7, 7, 7], [3, 4, 7, 7, 7]]
    completion, text = _decode_completion([1, 2, 7, 6, 5], 2, TinyTokenizer())
    assert completion == [7] and text == "7"
    rows = [
        {"prompt_ids": [1, 2], "prompt": "a", "source_index": 20,
         "example_id": "a", "question": "a", "target": "7"},
        {"prompt_ids": [3], "prompt": "b", "source_index": 10,
         "example_id": "b", "question": "b", "target": "7"},
        {"prompt_ids": [4, 5], "prompt": "c", "source_index": 30,
         "example_id": "c", "question": "c", "target": "7"},
    ]
    records = _smdm_generate(
        ConstantModel(), rows, TinyTokenizer(), torch.device("cpu"), 2, 1, 2, 0.0, True
    )
    assert [row["source_index"] for row in records] == [20, 10, 30]
    assert all(row["output"] == "7,7" for row in records)
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=tuple(MODEL_SPECS))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS, default="cagd")
    parser.add_argument("--trainable", choices=("last_block", "all"), default="last_block")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int)
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
    if args.self_check:
        return args
    if args.model_id is None:
        parser.error("--model-id is required")
    if args.generation_batch_size is None:
        args.generation_batch_size = 16 if args.model_id.startswith("qwen") else 8
    if args.output is None:
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

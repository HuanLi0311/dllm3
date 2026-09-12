#!/usr/bin/env python3
"""Corrected stored-direction trace controls for the R23/R24 DLLM grids."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.smdm_backend import flat_parameters, load_model, set_seed, trainable_parameters  # noqa: E402
from reproduction import smdm_factual as multitask  # noqa: E402
from reproduction import smdm_transfer as transfer  # noqa: E402


METHODS = ("gd", "rank1_gd", "diag_gd")
SEEDS = (3407, 3408, 3409)
CLIPS = (1.0, 1_000_000.0)
TRACE_CHUNK = 1_048_576
R16_ANCHORS = {
    3407: {"current_mean": 0.15928860665256275, "clip_fraction": 0.236,
           "rank1_coefficient": 0.0008613997596079841, "diagonal_trace": 0.002578950487077236},
    3408: {"current_mean": 0.14222952271252223, "clip_fraction": 0.239,
           "rank1_coefficient": 0.002652776763081589, "diagonal_trace": 0.005044732242822647},
    3409: {"current_mean": 0.1648787468732853, "clip_fraction": 0.253,
           "rank1_coefficient": 0.01223424401850455, "diagonal_trace": 0.019160481169819832},
}
FAMILIES = {
    "r23": {
        "run_dir": "r23_corrected_trace",
        "protocol": ROOT / "report/r23_corrected_trace_protocol.md",
        "experiment": "r23_corrected_trace_match",
        "role": "exploratory_predeclared_complete_grid",
        "model": 170,
        "parameter_count": 219_050_496,
        "checkpoint": ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors",
        "checkpoint_sha256": "2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb",
        "clips": CLIPS,
        "r16_anchors": True,
    },
    "r24": {
        "run_dir": "r24_scale1028_corrected_trace",
        "protocol": ROOT / "report/r24_scale1028_corrected_trace_protocol.md",
        "experiment": "r24_scale1028_corrected_trace_match",
        "role": "predeclared_exploratory_scale_extension",
        "model": 1028,
        "parameter_count": 1_142_367_744,
        "checkpoint": ROOT.parent / "checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors",
        "checkpoint_sha256": "ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6",
        "clips": (1.0,),
        "r16_anchors": False,
    },
}


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
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _dependencies(config: dict) -> dict[str, str]:
    paths = (
        Path(__file__), config["protocol"],
        ROOT / "reproduction/smdm_factual.py",
        ROOT / "reproduction/smdm_transfer.py",
        ROOT / "reproduction/continual_benchmark.py", ROOT / "reproduction/smdm_backend.py",
        ROOT / "reproduction/continual_reverse.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _inputs(args, tasks) -> dict:
    return {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "task_artifacts": [
            {
                "name": task["name"],
                "train": multitask._records_sha256(task["train_raw"]),
                "fisher": multitask._records_sha256(task["fisher_raw"]),
                "test": multitask._records_sha256(task["eval_raw"]),
            }
            for task in tasks
        ],
    }


def _runtime() -> dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "xformers": importlib.metadata.version("xformers"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _clip_name(value: float) -> str:
    return "1" if value == 1.0 else "1000000"


def _expected_output(config: dict, method: str, clip: float, seed: int) -> Path:
    name = f"{method}_clip{_clip_name(clip)}_s{seed}.json" if len(config["clips"]) > 1 else f"{method}_s{seed}.json"
    return ROOT / "runs" / config["run_dir"] / "formal" / name


def _direction_norm_sq(direction: torch.Tensor, chunk_size: int = TRACE_CHUNK) -> float:
    if direction.ndim != 1 or not direction.is_floating_point() or chunk_size < 1:
        raise ValueError("direction must be a flat floating tensor and chunk_size must be positive")
    flat = direction.detach().cpu()
    partials = []
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size].to(torch.float64)
        partials.append(float(torch.dot(chunk, chunk)))
    value = math.fsum(partials)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"stored direction has invalid squared norm: {value}")
    return value


def _tensor_sum_float64(vector: torch.Tensor, chunk_size: int = TRACE_CHUNK) -> float:
    if vector.ndim != 1 or not vector.is_floating_point() or chunk_size < 1:
        raise ValueError("vector must be a flat floating tensor and chunk_size must be positive")
    flat = vector.detach().cpu()
    partials = []
    for start in range(0, flat.numel(), chunk_size):
        partials.append(float(flat[start : start + chunk_size].sum(dtype=torch.float64)))
    value = math.fsum(partials)
    if not math.isfinite(value):
        raise ValueError(f"stored tensor has non-finite float64 sum: {value}")
    return value


def _matched_stiffness(alpha: float, diagonal_trace_exact: float, direction_norm_sq: float) -> dict:
    values = (alpha, diagonal_trace_exact, direction_norm_sq)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"invalid Fisher quantities for trace matching: {values}")
    rank1_trace = alpha * direction_norm_sq
    target = 1_000.0 * diagonal_trace_exact
    lambdas = {"rank1": target / rank1_trace, "diagonal": 1_000.0}
    checks = {
        "rank1": lambdas["rank1"] * alpha * direction_norm_sq,
        "diagonal": lambdas["diagonal"] * diagonal_trace_exact,
    }
    if any(not math.isclose(value, target, rel_tol=1e-12, abs_tol=0.0) for value in checks.values()):
        raise AssertionError(f"corrected weighted-trace matching failed: {checks}")
    return {
        "direction_norm_sq_float64_chunked": direction_norm_sq,
        "direction_norm_float64_chunked": math.sqrt(direction_norm_sq),
        "rank1_unweighted_trace": rank1_trace,
        "diagonal_unweighted_trace": diagonal_trace_exact,
        "weighted_trace_target": target,
        "matched_lambdas": lambdas,
        "weighted_trace_checks": checks,
    }


def _summary(stages, tasks) -> dict:
    learned_a = stages[0]["metrics"][tasks[0]["name"]]["loss"]
    final = stages[1]["metrics"]
    a_final = final[tasks[0]["name"]]["loss"]
    b_final = final[tasks[1]["name"]]["loss"]
    return {
        "final_average_loss": (a_final + b_final) / 2,
        "past_task_forgetting": a_final - learned_a,
        "final_task_loss": b_final,
        "final_average_answer_token_accuracy": (
            final[tasks[0]["name"]]["answer_token_accuracy"]
            + final[tasks[1]["name"]]["answer_token_accuracy"]
        ) / 2,
        "task_a_loss_when_learned": learned_a,
        "task_a_final_loss": a_final,
    }


def _assert_finite(value, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _contract_guard(family: str, config: dict, dependencies: dict, inputs: dict) -> tuple[Path, str]:
    path = ROOT / "runs" / config["run_dir"] / "contract.json"
    contract = json.loads(path.read_text())
    expected_grid = {
        "methods": list(METHODS), "b_clip": list(config["clips"]),
        "seeds": list(SEEDS), "cell_count": len(METHODS) * len(config["clips"]) * len(SEEDS),
    }
    if contract.get("status") != "frozen_before_pilot" or contract.get("family") != family:
        raise ValueError(f"unexpected contract status: {path}")
    if contract.get("grid") != expected_grid:
        raise ValueError(f"unexpected contract grid: {path}")
    for relative, digest in dependencies.items():
        if contract["sha256"].get(relative) != digest:
            raise ValueError(f"dependency outside frozen contract: {relative}")
    if contract["sha256"].get("checkpoint") != inputs["checkpoint_sha256"]:
        raise ValueError("checkpoint outside frozen contract")
    if contract.get("inputs") != inputs:
        raise ValueError("tokenizer/data/task artifacts are outside the frozen contract")
    runtime = _runtime()
    if any(runtime.get(key) != value for key, value in contract.get("environment", {}).items()):
        raise ValueError("runtime environment is outside the frozen contract")
    return path, _sha256(path)


def _r16_anchor_checks(seed: int, training: dict, fisher_stats: dict) -> tuple[dict, dict]:
    anchor = R16_ANCHORS[seed]
    checks = {
        "training_current_mean_abs_difference": abs(training["current_mean"] - anchor["current_mean"]),
        "training_clip_fraction_abs_difference": abs(training["clip_fraction"] - anchor["clip_fraction"]),
        "rank1_coefficient_relative_difference": abs(fisher_stats["rank1_coefficient"] - anchor["rank1_coefficient"]) / anchor["rank1_coefficient"],
        "diagonal_trace_relative_difference": abs(fisher_stats["diagonal_trace"] - anchor["diagonal_trace"]) / anchor["diagonal_trace"],
    }
    if (
        checks["training_current_mean_abs_difference"] > 1e-9
        or checks["training_clip_fraction_abs_difference"] > 0
        or checks["rank1_coefficient_relative_difference"] > 1e-6
        or checks["diagonal_trace_relative_difference"] > 1e-6
    ):
        raise AssertionError(f"Task-A state does not reproduce R16: {checks}")
    return anchor, checks


def run(args) -> dict:
    started = time.monotonic()
    config = FAMILIES[args.family]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.method not in METHODS or args.seed not in SEEDS or args.b_clip not in config["clips"]:
        raise ValueError(f"run is outside the frozen {args.family} grid")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("frozen environment requires PYTHONNOUSERSITE=1")
    if _sha256(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match the frozen family")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = multitask._load_tasks(args, tokenizer)
    if [task["name"] for task in tasks] != ["d2p_8-11", "p2d_12-15"]:
        raise ValueError("unexpected task sequence")
    dependencies = _dependencies(config)
    inputs = _inputs(args, tasks)

    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if args.trainable == "all" and parameter_count != config["parameter_count"]:
        raise ValueError(f"unexpected full parameter count: {parameter_count}")

    args.clip = 1.0
    task_a_training = multitask._train_stage(
        model, tasks[0]["train"], parameters, pad_id, device, args, args.seed + 1000
    )
    task_a_metrics = multitask._measure_task(model, tokenizer, tasks[0], pad_id, device, args)
    fisher, fisher_stats = transfer._estimate_mean_and_diagonal_fisher(
        model, tasks[0]["fisher"], parameters, pad_id, device, args
    )
    if (
        fisher_stats["calibration_examples"] != 40
        or fisher_stats["parameter_count"] != parameter_count
        or fisher_stats["repeat_loss_max_abs_difference"] != 0.0
    ):
        raise AssertionError(f"unexpected Fisher audit: {fisher_stats}")
    fisher_stats["source"] = "task_a_training_examples"
    fisher_stats["task"] = tasks[0]["name"]
    anchor = anchor_checks = None
    if config["r16_anchors"] and args.trainable == "all":
        anchor, anchor_checks = _r16_anchor_checks(args.seed, task_a_training, fisher_stats)

    direction = fisher["direction"]
    diagonal = fisher["diagonal"]
    if (
        direction.numel() != parameter_count or diagonal.numel() != parameter_count
        or direction.dtype != torch.float32 or diagonal.dtype != torch.float32
        or direction.device.type != "cpu" or diagonal.device.type != "cpu"
    ):
        raise AssertionError("Fisher tensors do not match the frozen float32 CPU representation")
    direction_norm_sq = _direction_norm_sq(direction)
    diagonal_trace_exact = _tensor_sum_float64(diagonal)
    stiffness = _matched_stiffness(
        float(fisher["coefficient"]), diagonal_trace_exact, direction_norm_sq
    )
    stiffness.update({
        "stored_tensor_dtype": "torch.float32",
        "stored_tensor_device": "cpu",
        "stored_tensor_numel": parameter_count,
        "float64_reduction_chunk_elements": TRACE_CHUNK,
        "float64_reduction_chunks": math.ceil(parameter_count / TRACE_CHUNK),
        "diagonal_trace_float32_reported": float(fisher_stats["diagonal_trace"]),
        "diagonal_trace_float64_chunked": diagonal_trace_exact,
        "diagonal_trace_reported_relative_difference": abs(
            diagonal_trace_exact - float(fisher_stats["diagonal_trace"])
        ) / diagonal_trace_exact,
    })
    prompts, replay_manifest = multitask._replay_prompts(tasks, 1, 64)
    expected_counts = {str(fact): 16 for fact in range(8, 12)}
    if len(replay_manifest) != 1 or replay_manifest[0]["fact_counts"] != expected_counts:
        raise AssertionError("replay prompts are not balanced 16/16/16/16")
    replay = transfer._generate_replay(model, tokenizer, prompts, device, args)
    if len(replay) != 64:
        raise AssertionError(f"unexpected replay size: {len(replay)}")

    teacher = load_model(args, device)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    constraint = None
    reference = None
    effective_lambda = 0.0
    if args.method != "gd":
        reference = flat_parameters(parameters).detach().clone()
        if args.method == "rank1_gd":
            effective_lambda = stiffness["matched_lambdas"]["rank1"]
            constraint = {
                "kind": "rank1", "reference": reference,
                "direction": fisher["direction"].to(device),
                "coefficient": float(fisher["coefficient"]),
            }
        else:
            effective_lambda = stiffness["matched_lambdas"]["diagonal"]
            constraint = {
                "kind": "diagonal", "reference": reference,
                "diagonal": fisher["diagonal"].to(device),
            }
    del fisher, direction, diagonal
    args.clip = args.b_clip
    args.ewc_lambda = effective_lambda
    task_b_training = multitask._train_stage(
        model, tasks[1]["train"], parameters, pad_id, device, args,
        args.seed + 2000, teacher=teacher, replay_rows=replay,
        constraints=[] if constraint is None else [constraint],
    )
    del teacher, reference, constraint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_metrics = {
        task["name"]: multitask._measure_task(model, tokenizer, task, pad_id, device, args)
        for task in tasks
    }
    stages = [
        {"task": tasks[0]["name"], "training": task_a_training,
         "metrics": {tasks[0]["name"]: task_a_metrics}, "fisher": fisher_stats},
        {"task": tasks[1]["name"], "training": task_b_training, "metrics": final_metrics},
    ]
    stiffness["effective_lambda"] = effective_lambda
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": config["experiment"],
        "family": args.family,
        "role": config["role"],
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": os.uname().nodename,
        "runtime": _runtime(),
        "dependencies": dependencies,
        "contract": {"kind": "reproduction_v2", "dependency_sha256": dependencies},
        "inputs": inputs,
        "protocol": {
            "method": args.method, "seed": args.seed,
            "model": config["model"], "parameter_count": parameter_count,
            "task_sequence": [task["name"] for task in tasks],
            "trainable": args.trainable,
            "full_parameter_training": args.trainable == "all",
            "objective": "r16_answer_only_independent_bernoulli_allow_empty_importance_weighted",
            "steps_per_task": 1000, "batch_size": 4, "learning_rate": 5e-5,
            "task_a_clip": 1.0, "task_b_clip": args.b_clip,
            "fisher_examples": 40,
            "matching": "lambda_rank1_times_alpha_times_stored_direction_norm_sq_equals_lambda_diagonal_times_trace_diagonal",
            "stored_trace_reduction": f"float64_cpu_chunks_{TRACE_CHUNK}",
            "diagonal_anchor_lambda": 1_000.0,
            "replay_fact_counts": replay_manifest[0]["fact_counts"],
        },
        "training_state": {
            "fisher_sampling_stage": "immediately_after_task_a_training",
            "task_a_training": task_a_training,
            "task_a_metrics": task_a_metrics,
        },
        "task_a_anchor": anchor,
        "task_a_anchor_checks": anchor_checks,
        "fisher": fisher_stats,
        "stiffness_match": stiffness,
        "replay": {
            "examples": len(replay), "manifest": replay_manifest,
            "generated_rows_sha256": multitask._records_sha256(replay),
        },
        "stages": stages,
        "summary": _summary(stages, tasks),
    }
    _assert_finite(result)
    if _dependencies(config) != dependencies:
        raise RuntimeError("source/protocol provenance changed during run")
    if _inputs(args, tasks) != inputs:
        raise RuntimeError("input provenance changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check(args: argparse.Namespace) -> None:
    assert args.record_step_loss is False and args.record_eval_loss_every == 0
    direction = torch.tensor([3.0, 4.0], dtype=torch.float32)
    direct_norm_sq = float(direction.double().square().sum())
    chunked_norm_sq = _direction_norm_sq(direction, chunk_size=1)
    assert chunked_norm_sq == direct_norm_sq == 25.0
    diagonal = torch.tensor([0.5, 0.125, 1.5, 0.25, 2.0, 0.0625, 0.75], dtype=torch.float32)
    direct_diagonal_trace = float(diagonal.double().sum())
    chunked_diagonal_trace = _tensor_sum_float64(diagonal, chunk_size=2)
    assert chunked_diagonal_trace == direct_diagonal_trace
    match = _matched_stiffness(2.5, chunked_diagonal_trace, chunked_norm_sq)
    rank1 = match["matched_lambdas"]["rank1"] * 2.5 * torch.outer(direction.double(), direction.double())
    rank1_trace = float(torch.trace(rank1))
    diagonal_trace = match["matched_lambdas"]["diagonal"] * direct_diagonal_trace
    assert math.isclose(rank1_trace, diagonal_trace, rel_tol=1e-12)
    assert math.isclose(match["weighted_trace_checks"]["rank1"], match["weighted_trace_target"], rel_tol=1e-12)
    assert multitask._sft_losses is transfer._sft_losses
    mock = [{"name": "a", "train_raw": [
        {"fact_id": fact, "prompt": f"p-{fact}-{index}"}
        for fact in range(8, 12) for index in range(30)
    ]}]
    prompts, manifest = multitask._replay_prompts(mock, 1, 64)
    assert len(prompts) == 64 and manifest[0]["fact_counts"] == {str(fact): 16 for fact in range(8, 12)}
    _assert_finite({"nested": [0.0, 1.0]})
    try:
        _assert_finite({"nested": [float("nan")]})
    except ValueError:
        pass
    else:
        raise AssertionError("recursive finite check accepted NaN")
    assert _expected_output(FAMILIES["r23"], "gd", 1e6, 3409).name == "gd_clip1000000_s3409.json"
    assert _expected_output(FAMILIES["r24"], "gd", 1.0, 3409).name == "gd_s3409.json"
    print(json.dumps({"self_check": "ok", "nonunit_direction_norm_sq": chunked_norm_sq}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=False, default="r23")
    parser.add_argument("--method", choices=METHODS, default="gd")
    parser.add_argument("--trainable", choices=("last_block", "all"), default="all")
    parser.add_argument("--b-clip", type=float, choices=CLIPS, default=1.0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3407)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "third_party/SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    config = FAMILIES[args.family]
    if args.checkpoint is None:
        args.checkpoint = config["checkpoint"]
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    args.model = config["model"]
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
        _self_check(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

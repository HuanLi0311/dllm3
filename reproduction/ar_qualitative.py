#!/usr/bin/env python3
"""Reproduce one Qwen natural-task pair and retain fixed qualitative outputs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction import ar_natural as base  # noqa: E402
from reproduction.ar_factual import (  # noqa: E402
    _evaluate,
    _evaluation_mode,
    _generate_replay,
    _select_parameters,
    _set_seed,
)


SEED = 3407
METHODS = ("seq", "cagd")
SELECTED_IDS = {
    "summarization": (
        "summarization:test:0",
        "summarization:test:2",
        "summarization:test:33",
    ),
    "creative_writing": (
        "creative_writing:test:0",
        "creative_writing:test:6",
        "creative_writing:test:13",
    ),
}
PROTOCOL = ROOT / "report/cagd_qualitative_examples_protocol.md"
DATA = ROOT / "runs/data/dolly_natural_stream.jsonl"
MANIFEST = ROOT / "runs/data/dolly_natural_stream_manifest.json"
BASE_RUNNER = ROOT / "reproduction/ar_natural.py"
DEPENDENCY = ROOT / "reproduction/ar_factual.py"
ARCHIVE_ROOT = ROOT / "runs/cagd_natural/formal/qwen/s3407"
LOCKED_HASHES = {
    BASE_RUNNER: "4a8be25e92865127c46ca12c0359de1692f866692849581d23da4c92aba8feea",
    DEPENDENCY: "a8a48d15340d01b2261f0eba8551e42fd138fba10a73359794eba251c57947c0",
    DATA: "a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0",
    MANIFEST: "ce47eb566d6115e95f31a5379768c3d3c42b005ad3b4a9705fee42c867a437fa",
}
SETTINGS = {
    "steps_per_task": 1000,
    "batch_size": 2,
    "eval_batch_size": 2,
    "generation_batch_size": 4,
    "replay_per_task": 64,
    "max_new_tokens": 96,
    "max_length": 256,
    "distill_weight": 1.0,
    "distill_temperature": 1.0,
    "lr": 5e-5,
    "clip": 1.0,
}


def _fixed_args(method: str, device: str, output: Path) -> Namespace:
    return Namespace(
        data=DATA,
        manifest=MANIFEST,
        model=base.DEFAULT_MODEL,
        output=output,
        method=method,
        order="forward",
        seed=SEED,
        device=device,
        formal=True,
        self_check=False,
        **SETTINGS,
    )


def _generate_examples(model, task: dict, tokenizer, pad_id: int, device) -> list[dict]:
    import torch

    raw = {row["example_id"]: row for row in task["eval_raw"]}
    encoded = {row["example_id"]: item for row, item in zip(task["eval_raw"], task["eval"])}
    records = []
    with _evaluation_mode(model), torch.no_grad():
        for example_id in SELECTED_IDS[task["name"]]:
            row, item = raw[example_id], encoded[example_id]
            prompt = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=SETTINGS["max_new_tokens"],
                pad_token_id=pad_id,
                eos_token_id=pad_id,
                use_cache=True,
            )[0, prompt.shape[1]:].cpu().tolist()
            if pad_id in generated:
                generated = generated[:generated.index(pad_id)]
            records.append({
                "example_id": example_id,
                "source_index": row["source_index"],
                "prompt": row["prompt"],
                "reference": row["answer"].strip(),
                "output": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            })
    return records


def _metric_difference(observed: list[dict], archived: list[dict]) -> float:
    differences = []
    for actual_stage, archived_stage in zip(observed, archived):
        if actual_stage["stage"] != archived_stage["stage"]:
            raise ValueError("stage indices differ from the archived formal run")
        for task, metrics in actual_stage["metrics"].items():
            expected = archived_stage["metrics"][task]
            differences.extend(abs(metrics[key] - expected[key]) for key in metrics)
    return max(differences, default=0.0)


def run(args: Namespace) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("qualitative reproduction requires CUDA")
    if not PROTOCOL.read_text().startswith("# Qwen qualitative-output protocol\n\nStatus: frozen"):
        raise ValueError("qualitative protocol is not frozen")
    for path, digest in LOCKED_HASHES.items():
        if base._sha256(path) != digest:
            raise ValueError(f"locked dependency differs: {path}")

    archive_path = ARCHIVE_ROOT / f"{args.method}.json"
    archive = json.loads(archive_path.read_text())
    if archive.get("status") != "ok" or archive["metadata"].get("seed") != SEED:
        raise ValueError("archived comparison endpoint is invalid")

    started = time.monotonic()
    _set_seed(SEED)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = base._read_tasks(DATA, tokenizer, SETTINGS["max_length"], "forward")
    base._validate(args, tasks)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    trainable_names, parameters = _select_parameters(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    stages = []

    for stage, task in enumerate(tasks):
        teacher, replay_rows, anchor_manifest = None, [], []
        if stage and args.method == "cagd":
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            anchors, anchor_manifest = base._anchors(tasks, stage, SETTINGS["replay_per_task"])
            replay_rows = _generate_replay(
                teacher, anchors, pad_id, device,
                SETTINGS["generation_batch_size"], SETTINGS["max_new_tokens"],
            )
        training = base._train(
            model, teacher, task["train"], replay_rows, parameters,
            pad_id, device, args, stage,
        )
        if teacher is not None:
            del teacher
            torch.cuda.empty_cache()
        metrics = {
            seen["name"]: _evaluate(model, seen["eval"], pad_id, device, SETTINGS["eval_batch_size"])
            for seen in tasks[:stage + 1]
        }
        stages.append({
            "stage": stage,
            "task": task["name"],
            "training": training,
            "anchor_manifest": anchor_manifest,
            "metrics": metrics,
        })

    max_metric_difference = _metric_difference(stages, archive["stages"])
    if max_metric_difference > 5e-6:
        raise RuntimeError(f"reproduction differs from archived endpoint by {max_metric_difference}")
    examples = {
        task["name"]: _generate_examples(model, task, tokenizer, pad_id, device)
        for task in tasks if task["name"] in SELECTED_IDS
    }
    tracked = (Path(__file__), PROTOCOL, *LOCKED_HASHES)
    hashes = {str(path.relative_to(ROOT)): base._sha256(path) for path in tracked}
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "qwen_cagd_qualitative",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": hashes,
        "upstream_artifact": str(archive_path.relative_to(ROOT)),
        "upstream_artifact_sha256": base._sha256(archive_path),
        "metadata": {
            "model": "Qwen3-0.6B",
            "model_parameter_count": total_parameters,
            "trainable": "last_transformer_block",
            "trainable_names": trainable_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
            "method": args.method,
            "seed": SEED,
            "task_sequence": list(base.TASKS),
            "selected_example_ids": SELECTED_IDS,
            "settings": SETTINGS,
            "max_archived_metric_difference": max_metric_difference,
        },
        "stages": stages,
        "examples": examples,
    }
    if any(not record["output"] for records in examples.values() for record in records):
        raise RuntimeError("empty qualitative output")
    if {str(path.relative_to(ROOT)): base._sha256(path) for path in tracked} != hashes:
        raise RuntimeError("source, protocol, or dependency changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "ok", "output": str(args.output),
                      "max_archived_metric_difference": max_metric_difference}, indent=2))
    return result


def _self_check() -> None:
    for path, digest in LOCKED_HASHES.items():
        assert base._sha256(path) == digest
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    observed = {(row["task"], row["example_id"]) for row in rows if row["split"] == "test"}
    selected = [(task, example_id) for task, ids in SELECTED_IDS.items() for example_id in ids]
    assert len(selected) == len(set(selected)) == 6
    assert all(item in observed for item in selected)
    print(json.dumps({"self_check": "ok", "selected_examples": len(selected)}))


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parsed = parser.parse_args()
    if parsed.self_check:
        return parsed
    if parsed.method is None or parsed.output is None:
        parser.error("--method and --output are required")
    return _fixed_args(parsed.method, parsed.device, parsed.output)


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()

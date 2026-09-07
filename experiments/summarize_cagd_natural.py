#!/usr/bin/env python3
"""Strictly audit and summarize the frozen natural-task CAGD matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


BACKENDS = ("smdm", "qwen")
METHODS = ("seq", "cagd", "hard_replay")
SEEDS = (3407, 3408, 3409)
TAGS = {"smdm": "cagd_natural_smdm_v1", "qwen": "cagd_natural_qwen_v1"}
RUNNER_SHA256 = {
    "smdm": "eb2e88ceb8de3aec433187c367b3a73fe534260a8f8504be3b483e87610994cc",
    "qwen": "a45839488e8f848069bdea0da5ecddb975305cd1e909ae268cce891377693dd3",
}
PROTOCOL_SHA256 = "299c9e15cf6ac8873a7922bb93b4c0711f88584f3b06feb1a19ab31d42466f89"
DATA_SHA256 = "a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0"
MANIFEST_SHA256 = "ce47eb566d6115e95f31a5379768c3d3c42b005ad3b4a9705fee42c867a437fa"
TASKS = ["closed_qa", "summarization", "creative_writing"]
ENDPOINTS = (
    "final_average_loss",
    "past_task_forgetting",
    "final_task_loss",
    "final_average_answer_token_accuracy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
    }


def _paths(root: Path) -> dict[tuple[str, int, str], Path]:
    return {
        (backend, seed, method): root / backend / f"s{seed}" / f"{method}.json"
        for backend in BACKENDS for seed in SEEDS for method in METHODS
    }


def _endpoints(result: dict) -> dict[str, float]:
    summary = result["summary"]
    return {
        "final_average_loss": float(summary["final_average_loss"]),
        "past_task_forgetting": float(summary["past_task_forgetting"]),
        "final_task_loss": float(summary["final_task_losses"][-1]),
        "final_average_answer_token_accuracy": float(summary["final_average_answer_token_accuracy"]),
    }


def summarize(root: Path) -> dict:
    expected = _paths(root)
    actual = set(root.rglob("*.json"))
    missing = sorted(str(path) for path in set(expected.values()) - actual)
    extra = sorted(str(path) for path in actual - set(expected.values()))
    if missing or extra:
        raise ValueError(f"matrix mismatch: missing={missing}, extra={extra}")

    results, input_hashes = {}, {}
    for key, path in expected.items():
        backend, seed, method = key
        result = json.loads(path.read_text())
        metadata = result.get("metadata", {})
        wanted = {
            "status": (result.get("status"), "ok"),
            "experiment": (result.get("experiment"), f"{backend}_cagd_natural"),
            "protocol": (metadata.get("protocol"), TAGS[backend]),
            "method": (metadata.get("method"), method),
            "seed": (metadata.get("seed"), seed),
            "order": (metadata.get("order"), "forward"),
            "tasks": (metadata.get("task_sequence"), TASKS),
            "runner": (result.get("source_sha256"), RUNNER_SHA256[backend]),
            "protocol_sha256": (result.get("protocol_sha256"), PROTOCOL_SHA256),
            "data_sha256": (result.get("data_sha256"), DATA_SHA256),
            "manifest_sha256": (result.get("manifest_sha256"), MANIFEST_SHA256),
        }
        errors = [f"{name}={actual!r}, expected {wanted!r}" for name, (actual, wanted) in wanted.items() if actual != wanted]
        if errors:
            raise ValueError(f"{path}: " + "; ".join(errors))
        endpoints = _endpoints(result)
        if not all(math.isfinite(value) for value in endpoints.values()):
            raise ValueError(f"{path}: non-finite endpoint")
        results[key] = {"result": result, "endpoints": endpoints}
        input_hashes[str(path.relative_to(root))] = _sha256(path)

    generated_match = {}
    for backend in BACKENDS:
        generated_match[backend] = {}
        for seed in SEEDS:
            cagd = results[(backend, seed, "cagd")]["result"]["stages"][1]["generated_rows_sha256"]
            hard = results[(backend, seed, "hard_replay")]["result"]["stages"][1]["generated_rows_sha256"]
            if cagd != hard:
                raise ValueError(f"{backend} seed {seed}: first-transition generated rows differ")
            generated_match[backend][str(seed)] = cagd

    aggregate, paired = {}, {}
    for backend in BACKENDS:
        aggregate[backend] = {
            method: {
                endpoint: _mean_sem([
                    results[(backend, seed, method)]["endpoints"][endpoint] for seed in SEEDS
                ])
                for endpoint in ENDPOINTS
            }
            for method in METHODS
        }
        paired[backend] = {}
        for baseline in ("seq", "hard_replay"):
            label = f"cagd_minus_{baseline}"
            paired[backend][label] = {}
            for endpoint in ENDPOINTS:
                values = [
                    results[(backend, seed, "cagd")]["endpoints"][endpoint]
                    - results[(backend, seed, baseline)]["endpoints"][endpoint]
                    for seed in SEEDS
                ]
                paired[backend][label][endpoint] = {
                    **_mean_sem(values),
                    "wins": sum(value > 0 for value in values) if endpoint.endswith("accuracy") else sum(value < 0 for value in values),
                }
    return {
        "schema_version": 1,
        "status": "ok",
        "protocol": "cagd_natural_v1",
        "run_count": len(results),
        "backends": list(BACKENDS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "aggregate": aggregate,
        "paired": paired,
        "audit": {
            "runner_sha256": RUNNER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "data_sha256": DATA_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "first_transition_generated_rows_match": generated_match,
            "input_sha256": input_hashes,
        },
    }


def _self_check() -> None:
    values = _mean_sem([1.0, 2.0, 3.0])
    assert values["mean"] == 2.0 and math.isclose(values["sem"], 1 / math.sqrt(3))
    paths = _paths(Path("formal"))
    assert len(paths) == 18
    assert paths[("qwen", 3409, "hard_replay")] == Path("formal/qwen/s3409/hard_replay.json")
    assert _endpoints({"summary": {
        "final_average_loss": 1,
        "past_task_forgetting": 2,
        "final_task_losses": [3, 4],
        "final_average_answer_token_accuracy": 0.5,
    }})["final_task_loss"] == 4.0
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.run_root is None or args.output is None:
        parser.error("--run-root and --output are required")
    result = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "run_count": result["run_count"]}))


if __name__ == "__main__":
    main()

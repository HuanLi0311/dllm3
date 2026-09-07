#!/usr/bin/env python3
"""Strictly summarize condition-anchored distillation component runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


METHODS = ("cagd", "hard_replay", "real_replay")
SEEDS = (3407, 3408, 3409)
ORDERS = {
    "two_task": ("forward",),
    "main": ("forward", "reverse"),
    "fresh": ("forward", "reverse"),
}
PROTOCOLS = {
    "two_task": "cagd_component_two_task_v1",
    "main": "cagd_component_main_v1",
    "fresh": "cagd_component_fresh_v1",
}
ENDPOINTS = ("final_average_loss", "past_task_forgetting", "final_task_loss")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _expected_paths(root: Path, protocol: str) -> dict[tuple[str, int, str], Path]:
    paths = {}
    for order in ORDERS[protocol]:
        for seed in SEEDS:
            for method in METHODS:
                relative = Path(f"s{seed}/{method}.json")
                if protocol != "two_task":
                    relative = Path(order) / relative
                paths[(order, seed, method)] = root / relative
    return paths


def _endpoints(result: dict) -> dict[str, float]:
    summary = result["summary"]
    return {
        "final_average_loss": float(summary["final_average_loss"]),
        "past_task_forgetting": float(summary["past_task_forgetting"]),
        "final_task_loss": float(summary["final_task_losses"][-1]),
    }


def summarize(root: Path, protocol: str) -> dict:
    expected = _expected_paths(root, protocol)
    actual = set(root.rglob("*.json"))
    missing = sorted(str(path) for path in set(expected.values()) - actual)
    extra = sorted(str(path) for path in actual - set(expected.values()))
    if missing or extra:
        raise ValueError(f"matrix mismatch: missing={missing}, extra={extra}")

    results = {}
    source_hashes = set()
    protocol_hashes = set()
    input_hashes = {}
    for key, path in expected.items():
        order, seed, method = key
        result = json.loads(path.read_text())
        metadata = result.get("metadata", {})
        if result.get("status") != "ok":
            raise ValueError(f"{path}: non-ok status")
        for name, value, wanted in (
            ("method", metadata.get("method"), method),
            ("seed", metadata.get("seed"), seed),
            ("order", metadata.get("order"), order),
            ("protocol", metadata.get("protocol"), PROTOCOLS[protocol]),
        ):
            if value != wanted:
                raise ValueError(f"{path}: {name}={value!r}, expected {wanted!r}")
        values = _endpoints(result)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{path}: non-finite endpoint")
        if metadata.get("ewc_lambda") != 0.0:
            raise ValueError(f"{path}: component study must not use EWC")
        source_hashes.add(result["source_sha256"])
        protocol_hashes.add(metadata["protocol_document_sha256"])
        input_hashes[str(path.relative_to(root))] = _sha256(path)
        results[key] = {"result": result, "endpoints": values}

    if len(source_hashes) != 1 or len(protocol_hashes) != 1:
        raise ValueError("mixed runner or protocol hashes")

    generated_match = {}
    if protocol == "two_task":
        for seed in SEEDS:
            cagd = results[("forward", seed, "cagd")]["result"]["stages"][1]
            hard = results[("forward", seed, "hard_replay")]["result"]["stages"][1]
            cagd_hash = cagd["replay"]["generated_rows_sha256"]
            hard_hash = hard["replay"]["generated_rows_sha256"]
            if cagd_hash != hard_hash:
                raise ValueError(f"seed {seed}: CAGD and hard replay rows differ")
            generated_match[str(seed)] = cagd_hash

    aggregate = {}
    paired = {}
    for order in ORDERS[protocol]:
        aggregate[order] = {}
        for method in METHODS:
            aggregate[order][method] = {
                endpoint: _mean_sem([
                    results[(order, seed, method)]["endpoints"][endpoint] for seed in SEEDS
                ])
                for endpoint in ENDPOINTS
            }
        paired[order] = {}
        for baseline in ("hard_replay", "real_replay"):
            label = f"cagd_minus_{baseline}"
            paired[order][label] = {}
            for endpoint in ENDPOINTS:
                values = [
                    results[(order, seed, "cagd")]["endpoints"][endpoint]
                    - results[(order, seed, baseline)]["endpoints"][endpoint]
                    for seed in SEEDS
                ]
                paired[order][label][endpoint] = {
                    **_mean_sem(values),
                    "wins": sum(value < 0 for value in values),
                }

    return {
        "schema_version": 1,
        "status": "ok",
        "protocol": PROTOCOLS[protocol],
        "run_count": len(results),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "aggregate": aggregate,
        "paired": paired,
        "audit": {
            "runner_sha256": next(iter(source_hashes)),
            "protocol_document_sha256": next(iter(protocol_hashes)),
            "two_task_generated_rows_match": generated_match,
            "input_sha256": input_hashes,
        },
    }


def _self_check() -> None:
    values = _mean_sem([1.0, 2.0, 3.0])
    assert values == {"mean": 2.0, "sem": 1 / math.sqrt(3), "values": [1.0, 2.0, 3.0]}
    paths = _expected_paths(Path("runs"), "two_task")
    assert len(paths) == 9
    assert paths[("forward", 3407, "cagd")] == Path("runs/s3407/cagd.json")
    assert _endpoints({"summary": {
        "final_average_loss": 1,
        "past_task_forgetting": 2,
        "final_task_losses": [3, 4],
    }}) == {"final_average_loss": 1.0, "past_task_forgetting": 2.0, "final_task_loss": 4.0}
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--protocol", choices=tuple(ORDERS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.run_root is None or args.protocol is None or args.output is None:
        parser.error("--run-root, --protocol, and --output are required")
    result = summarize(args.run_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "run_count": result["run_count"]}))


if __name__ == "__main__":
    main()

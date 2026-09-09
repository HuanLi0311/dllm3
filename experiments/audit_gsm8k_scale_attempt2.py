#!/usr/bin/env python3
"""Fail-closed audit for a paired SMDM attempt-2 canonical/branch triplet."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import cagd_gsm8k_scale as base  # noqa: E402


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _audit_benchmark(benchmark: dict, expected_rows: list[dict]) -> None:
    records = benchmark.get("records")
    if not isinstance(records, list) or len(records) != len(expected_rows):
        raise ValueError("benchmark record count differs")
    correct = 0
    marked = 0
    for record, expected in zip(records, expected_rows):
        for key in ("example_id", "source_index", "question", "target"):
            if record.get(key) != expected.get(key):
                raise ValueError(f"benchmark {key} differs from vendored GSM8K")
        prediction, has_delimiter = base.qwen_base._extract_answer(record.get("output", ""))
        is_correct = prediction == expected["target"]
        if (
            record.get("prediction") != prediction
            or record.get("has_delimiter") is not has_delimiter
            or record.get("correct") is not is_correct
        ):
            raise ValueError("benchmark parsed fields differ")
        correct += is_correct
        marked += has_delimiter
    count = len(records)
    checks = {
        "count": count,
        "exact_match": correct / count,
        "delimiter_rate": marked / count,
    }
    for key, value in checks.items():
        observed = benchmark.get(key)
        if observed != value and not (
            isinstance(observed, float) and math.isclose(observed, value, abs_tol=1e-15)
        ):
            raise ValueError(f"benchmark {key} differs")


def _audit_branch(result: dict, expected_rows: list[dict]) -> None:
    stages = result.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("branch must contain exactly three stages")
    if [(stage.get("stage"), stage.get("task")) for stage in stages] != [
        (0, "gsm8k"), (1, "summarization"), (2, "creative_writing")
    ]:
        raise ValueError("branch task order differs")
    if stages[0].get("benchmark") is None or stages[1].get("benchmark") is not None:
        raise ValueError("benchmark appears on the wrong stage")
    if stages[2].get("benchmark") is None or stages[2].get("current_metrics") is None:
        raise ValueError("final benchmark or current-task metrics are missing")
    _audit_benchmark(stages[0]["benchmark"], expected_rows)
    _audit_benchmark(stages[2]["benchmark"], expected_rows)
    learned = stages[0]["benchmark"]["exact_match"]
    final = stages[2]["benchmark"]["exact_match"]
    expected_summary = {
        "gsm8k_exact_match_when_learned": learned,
        "gsm8k_exact_match_final": final,
        "gsm8k_retention_change": final - learned,
        "final_task_loss": stages[2]["current_metrics"]["loss"],
        "final_task_answer_token_accuracy": (
            stages[2]["current_metrics"]["answer_token_accuracy"]
        ),
    }
    if result.get("summary") != expected_summary:
        raise ValueError("branch summary is not derivable from stages")


def _audit_reference(metadata: dict, method: str, canonical_path: Path,
                     canonical_json_sha256: str, checkpoint: dict) -> None:
    if metadata.get("method") != method:
        raise ValueError(f"{method} branch method differs")
    if metadata.get("canonical_stage0_json") != str(canonical_path.resolve()):
        raise ValueError(f"{method} canonical JSON path differs")
    if metadata.get("canonical_stage0_json_sha256") != canonical_json_sha256:
        raise ValueError(f"{method} canonical JSON SHA-256 differs")
    if metadata.get("canonical_stage0_checkpoint") != checkpoint:
        raise ValueError(f"{method} canonical checkpoint differs")


def audit(canonical_path: Path, seq_path: Path, cagd_path: Path) -> dict:
    canonical = _read(canonical_path)
    seq = _read(seq_path)
    cagd = _read(cagd_path)
    canonical_json_sha256 = base._sha256(canonical_path)
    checkpoint = canonical.get("canonical_checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("canonical checkpoint is missing")
    actual_checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": base._sha256(checkpoint_path),
    }
    if checkpoint != actual_checkpoint:
        raise ValueError("canonical checkpoint descriptor differs from file")
    _, gsm_eval = base.qwen_base._raw_gsm()
    count = canonical.get("stage", {}).get("benchmark", {}).get("count")
    if not isinstance(count, int) or count < 1 or count > len(gsm_eval):
        raise ValueError("invalid canonical benchmark count")
    expected_rows = gsm_eval[:count]
    base_stage = canonical.get("stage")
    for method, result in (("seq", seq), ("cagd", cagd)):
        metadata = result.get("metadata", {})
        _audit_reference(
            metadata, method, canonical_path, canonical_json_sha256, actual_checkpoint
        )
        if result.get("stages", [None])[0] != base_stage:
            raise ValueError(f"{method} did not embed canonical stage0 verbatim")
        _audit_branch(result, expected_rows)
    if seq["stages"][0] != cagd["stages"][0]:
        raise ValueError("paired stage0 objects differ")
    return {
        "status": "ok",
        "canonical_json_sha256": canonical_json_sha256,
        "canonical_checkpoint_sha256": actual_checkpoint["sha256"],
        "canonical_checkpoint_bytes": actual_checkpoint["bytes"],
        "stage0_equal": True,
        "benchmark_count": count,
        "seq_summary": seq["summary"],
        "cagd_summary": cagd["summary"],
    }


def _self_check() -> None:
    expected = [{
        "example_id": "gsm8k-test-0", "source_index": 0,
        "question": "1+1?", "target": "2",
    }]
    benchmark = {
        "count": 1,
        "exact_match": 1.0,
        "delimiter_rate": 1.0,
        "records": [{
            **expected[0], "prediction": "2", "has_delimiter": True,
            "correct": True, "output": "#### 2",
        }],
    }
    _audit_benchmark(benchmark, expected)
    broken = copy.deepcopy(benchmark)
    broken["records"][0]["prediction"] = "3"
    try:
        _audit_benchmark(broken, expected)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered parsed output was accepted")
    canonical_path = Path("/tmp/canonical.json")
    checkpoint = {"path": "/tmp/canonical.safetensors", "bytes": 7, "sha256": "abc"}
    reference = {
        "method": "seq",
        "canonical_stage0_json": str(canonical_path.resolve()),
        "canonical_stage0_json_sha256": "def",
        "canonical_stage0_checkpoint": checkpoint,
    }
    _audit_reference(reference, "seq", canonical_path, "def", checkpoint)
    tampered_reference = copy.deepcopy(reference)
    tampered_reference["canonical_stage0_json_sha256"] = "tampered"
    try:
        _audit_reference(tampered_reference, "seq", canonical_path, "def", checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered canonical JSON hash was accepted")
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--seq", type=Path)
    parser.add_argument("--cagd", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and None in (args.canonical, args.seq, args.cagd):
        parser.error("--canonical, --seq, and --cagd are required")
    return args


def main() -> None:
    args = parse_args()
    result = (
        _self_check() if args.self_check
        else print(json.dumps(audit(args.canonical, args.seq, args.cagd), indent=2))
    )
    del result


if __name__ == "__main__":
    main()

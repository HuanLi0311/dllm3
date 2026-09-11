#!/usr/bin/env python3
"""Run the exact-reverse Qwen3-0.6B factual order extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import qwen_continual_transfer as base  # noqa: E402


PROTOCOL = ROOT / "report/qwen_order_reverse_protocol.md"
DEFAULT_MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_INVENTORY = ROOT / "runs/qwen_full_parameter_diagnostic/model_inventory.json"
FORWARD_SPECS = tuple(base.SPECS["formal"])
REVERSE_SPECS = tuple(reversed(FORWARD_SPECS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_check() -> None:
    assert REVERSE_SPECS == (
        ("p2d", 20, 4),
        ("d2p", 16, 4),
        ("p2d", 12, 4),
        ("d2p", 8, 4),
    )
    assert REVERSE_SPECS[::-1] == FORWARD_SPECS
    print(json.dumps({"self_check": "ok", "tasks": REVERSE_SPECS}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--method", choices=("seq", "gd"))
    parser.add_argument("--seed", type=int, choices=base.SEEDS, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.method is None or args.output is None:
        parser.error("--method and --output are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    wrapper_sha = _sha256(Path(__file__))
    temporary = args.output.with_suffix(args.output.suffix + ".base.tmp")
    if temporary.exists():
        raise FileExistsError(f"remove stale temporary output first: {temporary}")

    # ponytail: reuse the established runner and change only the task sequence.
    base.SPECS["formal_reverse"] = REVERSE_SPECS
    base.PROTOCOLS["qwen3_0.6b"] = PROTOCOL
    base.PROTOCOL_TAGS["qwen3_0.6b"] = "qwen_order_reverse_v1"
    run_args = Namespace(
        dev=False,
        run_kind="formal_reverse",
        model=args.model,
        model_label="qwen3_0.6b",
        model_inventory=args.model_inventory,
        method=args.method,
        seed=args.seed,
        ewc_lambda=0.0,
        selection=None,
        reverse_dir=base.DEFAULT_REVERSE,
        steps_per_task=1000,
        batch_size=4,
        eval_batch_size=4,
        generation_batch_size=8,
        fisher_per_fact=10,
        replay_per_task=64,
        max_new_tokens=32,
        max_length=128,
        distill_weight=1.0,
        distill_temperature=1.0,
        lr=5e-5,
        clip=1.0,
        device=args.device,
        output=temporary,
    )
    result = base.run(run_args)
    result["order_extension_provenance"] = {
        "wrapper_sha256": wrapper_sha,
        "base_runner_sha256": _sha256(Path(base.__file__)),
        "order": "reverse",
        "fixed_task": "p2d_20-23",
    }
    if _sha256(Path(__file__)) != wrapper_sha:
        raise RuntimeError("wrapper changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.unlink()
    print(json.dumps({"status": "ok", "output": str(args.output)}))


if __name__ == "__main__":
    main()

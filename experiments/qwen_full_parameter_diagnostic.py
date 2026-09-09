#!/usr/bin/env python3
"""Run the locked Qwen factual stream while updating every model parameter."""

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


DEFAULT_MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_RUN_DIR = ROOT / "runs/qwen_full_parameter_diagnostic"
_SELECTED_COUNT = 0
_SELECTED_NAMES: list[str] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_parameters(model):
    global _SELECTED_COUNT, _SELECTED_NAMES
    selected = list(model.named_parameters())
    if not selected:
        raise ValueError("model has no parameters")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    _SELECTED_NAMES = [name for name, _ in selected]
    _SELECTED_COUNT = sum(parameter.numel() for _, parameter in selected)
    return _SELECTED_NAMES, [parameter for _, parameter in selected]


def _self_check() -> None:
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[0].weight.requires_grad_(False)
    names, parameters = _all_parameters(model)
    assert names == [name for name, _ in model.named_parameters()]
    assert len(parameters) == len(list(model.parameters()))
    assert all(parameter.requires_grad for parameter in parameters)
    print(json.dumps({"self_check": "ok", "parameter_tensors": len(parameters)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--method", choices=("seq", "gd"))
    parser.add_argument("--seed", type=int, choices=base.SEEDS, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_RUN_DIR / "model_inventory.json")
    parser.add_argument("--selection", type=Path, default=DEFAULT_RUN_DIR / "selection.json")
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

    # ponytail: reuse the locked experiment implementation; this wrapper changes
    # only its parameter selector and can be retired after the diagnostic.
    base._select_parameters = _all_parameters
    run_args = Namespace(
        dev=False,
        run_kind="formal",
        model=args.model,
        model_label="qwen3_0.6b",
        model_inventory=args.model_inventory,
        method=args.method,
        seed=args.seed,
        ewc_lambda=0.0,
        selection=args.selection,
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
    result["metadata"]["trainable"] = "all_parameters"
    result["diagnostic_provenance"] = {
        "wrapper_sha256": wrapper_sha,
        "base_runner_sha256": _sha256(Path(base.__file__)),
        "purpose": "single-seed full-parameter sensitivity check",
    }
    if result["metadata"]["trainable_parameter_count"] != _SELECTED_COUNT:
        raise RuntimeError("recorded trainable parameter count is not the full model count")
    if result["metadata"]["trainable_names"] != _SELECTED_NAMES:
        raise RuntimeError("recorded trainable parameter names are incomplete")
    if _sha256(Path(__file__)) != wrapper_sha:
        raise RuntimeError("wrapper changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.unlink()
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
if __name__ == "__main__":
    main()

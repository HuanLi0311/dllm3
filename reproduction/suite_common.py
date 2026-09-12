"""Shared launch utilities and model profiles for figure/table experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shlex
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("PAPER_PYTHON", "/home/JJ_Group/lih2511/.conda/envs/opr/bin/python"))
AR_PYTHON = Path(os.environ.get("AR_PYTHON", PYTHON))
SMDM_PYTHON = Path(os.environ.get(
    "SMDM_PYTHON",
    PYTHON if "PAPER_PYTHON" in os.environ else "/home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python",
))
SMDM_MODULES = {"smdm_factual", "smdm_natural", "smdm_gsm8k", "smdm_parameter_controls"}

SMDM_MODELS = {
    "smdm_219m": {
        "display": "SMDM-219M",
        "config": "170",
        "path": ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors",
    },
    "smdm_1.14b": {
        "display": "SMDM-1.14B",
        "config": "1028",
        "path": ROOT.parent / "checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors",
    },
}

QWEN_MODELS = {
    "qwen3_0.6b": {
        "display": "Qwen3-0.6B",
        "path": Path(
            "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
            "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
        ),
    },
    "qwen3_1.7b": {
        "display": "Qwen3-1.7B",
        "path": Path(
            "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/"
            "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        ),
    },
    "qwen3_4b": {
        "display": "Qwen3-4B",
        "path": Path(
            "/home/JJ_Group/lih2511/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
            "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        ),
    },
}

TRACE_MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)


@dataclass(frozen=True)
class Cell:
    name: str
    command: tuple[str, ...]
    output: Path
    dimensions: dict[str, object] = field(default_factory=dict)


def split_words(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default=os.environ.get("PAPER_SEEDS", "3407 3408 3409"))
    parser.add_argument("--gpus", default=os.environ.get("PAPER_GPUS", "0 1 2 3 4 5 6 7"))
    parser.add_argument(
        "--trainable",
        choices=("reported", "last_block", "all"),
        default=os.environ.get("TRAINABLE_SCOPE", "reported"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def seeds(args) -> list[int]:
    values = [int(item) for item in split_words(args.seeds)]
    if not values:
        raise ValueError("at least one seed is required")
    return values


def gpus(args) -> list[str]:
    values = split_words(args.gpus)
    if not values:
        raise ValueError("at least one GPU is required")
    return values


def scope(args, backend: str) -> str:
    if args.trainable != "reported":
        return args.trainable
    return "all" if backend == "smdm" else "last_block"


def python_command(module: str, *arguments: object) -> tuple[str, ...]:
    executable = SMDM_PYTHON if module in SMDM_MODULES else AR_PYTHON
    return (str(executable), "-m", f"reproduction.{module}", *(str(item) for item in arguments))


def valid_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status", "ok") == "ok"


def _run_cell(cell: Cell, gpu: str, resume: bool) -> str:
    if cell.output.exists():
        if resume and valid_result(cell.output):
            return f"reuse {cell.name}"
        raise FileExistsError(f"refusing existing output: {cell.output}")
    cell.output.parent.mkdir(parents=True, exist_ok=True)
    log = cell.output.parent / f"{cell.output.stem}.log"
    if log.exists():
        raise FileExistsError(f"refusing existing log: {log}")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false"})
    # ponytail: ten retries cover observed shared-filesystem import jitter; raise immediately otherwise.
    for attempt in range(1, 11):
        with log.open("x", encoding="utf-8") as handle:
            handle.write("command=" + shlex.join(cell.command) + "\n")
            handle.write(f"physical_gpu={gpu}\n")
            handle.flush()
            completed = subprocess.run(cell.command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if not completed.returncode:
            break
        failure = log.read_text(encoding="utf-8", errors="replace")
        retryable = any(marker in failure for marker in (
            "partially initialized module 'multiprocessing'",
            "No module named 'email.",
            "No module named 'json.",
            "No module named 'unittest.",
        ))
        if not retryable or attempt == 10:
            raise RuntimeError(f"{cell.name} failed with exit code {completed.returncode}; see {log}")
        archive_index = attempt
        archived = log.with_name(f"{log.stem}.failed_attempt{archive_index}.log")
        while archived.exists():
            archive_index += 1
            archived = log.with_name(f"{log.stem}.failed_attempt{archive_index}.log")
        log.rename(archived)
    if not valid_result(cell.output):
        raise RuntimeError(f"{cell.name} did not produce a valid result: {cell.output}")
    return f"done {cell.name}"


def run_cells(cells: list[Cell], args) -> None:
    if args.dry_run:
        for cell in cells:
            print(f"{cell.name}\t{cell.output}\t{shlex.join(cell.command)}")
        return
    devices = gpus(args)
    available = queue.Queue()
    for device in devices:
        available.put(device)

    def leased(cell: Cell) -> str:
        device = available.get()
        try:
            return _run_cell(cell, device, args.resume)
        finally:
            available.put(device)

    failures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(leased, cell): cell for cell in cells}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append(f"{cell.name}: {error}")
    if failures:
        raise RuntimeError("\n".join(failures))


def summarize_cells(
    experiment: str,
    cells: list[Cell],
    output: Path,
    resume: bool = False,
    include: tuple[str, ...] = (),
) -> None:
    if output.exists():
        if resume and valid_result(output):
            return
        raise FileExistsError(f"refusing existing summary: {output}")
    rows = []
    dimension_names = sorted({key for cell in cells for key in cell.dimensions})
    for cell in cells:
        payload = json.loads(cell.output.read_text())
        if payload.get("status", "ok") != "ok" or not isinstance(payload.get("summary"), dict):
            raise ValueError(f"invalid or incomplete cell: {cell.output}")
        metadata = payload.get("metadata") or payload.get("protocol", {})
        row = {
            **cell.dimensions,
            "cell": cell.name,
            "source": str(cell.output.resolve()),
            "trainable": metadata.get("trainable", metadata.get("trainable_scope")),
            "trainable_parameter_count": metadata.get(
                "trainable_parameter_count", metadata.get("trainable_parameters")
            ),
            **payload["summary"],
            **{key: payload[key] for key in include},
        }
        losses = row.get("final_task_losses")
        if isinstance(losses, list) and losses:
            row.setdefault("final_task_loss", losses[-1])
            if row.get("order") in ("forward", "order_free"):
                row["fixed_task_loss"] = losses[-1]
            elif row.get("order") == "reverse":
                row["fixed_task_loss"] = losses[0]
        rows.append(row)
    aggregates, differences = aggregate_rows(rows, dimension_names)
    result = {
        "schema_version": 2,
        "status": "ok",
        "experiment": experiment,
        "dimension_names": dimension_names,
        "rows": rows,
        "aggregates": aggregates,
        "paired_differences": differences,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


def _estimate(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _numeric_metrics(rows: list[dict], excluded: set[str]) -> dict[str, dict]:
    metrics = {}
    common = set.intersection(*(set(row) for row in rows)) - excluded
    for key in sorted(common):
        values = [row[key] for row in rows]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            metrics[key] = _estimate([float(value) for value in values])
        elif all(isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value) for value in values):
            widths = {len(value) for value in values}
            if len(widths) == 1:
                metrics[key] = [_estimate([float(value[index]) for value in values]) for index in range(len(values[0]))]
    return metrics


def aggregate_rows(rows: list[dict], dimension_names: list[str]) -> tuple[list[dict], list[dict]]:
    group_names = [name for name in dimension_names if name != "seed"]
    grouped = {}
    for row in rows:
        key = tuple((name, row.get(name)) for name in group_names)
        grouped.setdefault(key, []).append(row)
    excluded = set(dimension_names) | {"cell", "source", "trainable", "trainable_parameter_count"}
    aggregates = [
        {**dict(key), "metrics": _numeric_metrics(group, excluded)}
        for key, group in sorted(grouped.items(), key=lambda item: repr(item[0]))
    ]

    paired = []
    if "method" in dimension_names and "seed" in dimension_names:
        base_names = [name for name in dimension_names if name not in ("method", "seed")]
        bases = {}
        for row in rows:
            key = tuple((name, row.get(name)) for name in base_names)
            bases.setdefault(key, {}).setdefault(row["method"], {})[row["seed"]] = row
        for key, methods in sorted(bases.items(), key=lambda item: repr(item[0])):
            for left in sorted(methods):
                for right in sorted(methods):
                    if left == right:
                        continue
                    common_seeds = sorted(set(methods[left]) & set(methods[right]))
                    deltas = []
                    for seed in common_seeds:
                        left_row, right_row = methods[left][seed], methods[right][seed]
                        common_metrics = set(left_row) & set(right_row) - excluded
                        deltas.append({
                            metric: float(left_row[metric]) - float(right_row[metric])
                            for metric in common_metrics
                            if isinstance(left_row[metric], (int, float))
                            and not isinstance(left_row[metric], bool)
                            and isinstance(right_row[metric], (int, float))
                            and not isinstance(right_row[metric], bool)
                        })
                    if deltas:
                        paired.append({
                            **dict(key), "left_method": left, "right_method": right,
                            "metrics": _numeric_metrics(deltas, set()),
                        })
    return aggregates, paired


def self_check() -> None:
    forbidden = (
        'ROOT / "experiments', "from experiments", "import experiments",
        "from continual_", "import continual_", "from dllm_", "import dllm_", "OnPolicyReplay",
    )
    offenders = []
    for path in Path(__file__).parent.iterdir():
        if path == Path(__file__) or path.suffix not in (".py", ".sh"):
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.name}: {token}" for token in forbidden if token in text)
    assert not offenders, "legacy code dependency: " + ", ".join(offenders)
    assert SMDM_MODELS and QWEN_MODELS and all(path.is_absolute() for path in (PYTHON, AR_PYTHON, SMDM_PYTHON))
    mock = [
        {"model": "m", "method": "a", "seed": 1, "score": 2.0},
        {"model": "m", "method": "a", "seed": 2, "score": 4.0},
        {"model": "m", "method": "b", "seed": 1, "score": 1.0},
        {"model": "m", "method": "b", "seed": 2, "score": 3.0},
    ]
    aggregates, paired = aggregate_rows(mock, ["method", "model", "seed"])
    assert next(row for row in aggregates if row["method"] == "a")["metrics"]["score"]["mean"] == 3.0
    assert next(row for row in paired if row["left_method"] == "a")["metrics"]["score"]["mean"] == 1.0
    checks = {
        SMDM_PYTHON: (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT / 'third_party/SMDM')!r}); "
            "import torch, transformers, safetensors, lightning; from lit_gpt.config import Config"
        ),
        AR_PYTHON: "import torch, transformers, safetensors",
    }
    for executable, code in checks.items():
        completed = subprocess.run(
            [str(executable), "-c", code], capture_output=True, text=True, timeout=120
        )
        if completed.returncode:
            raise RuntimeError(f"environment check failed for {executable}: {completed.stderr.strip()}")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        marker, output = temporary / "marker", temporary / "result.json"
        (temporary / "result.failed_attempt1.log").write_text("preserved\n")
        code = (
            "import json,sys; from pathlib import Path; "
            f"marker=Path({str(marker)!r}); output=Path({str(output)!r}); "
            "first=not marker.exists(); marker.touch(); "
            "print(\"partially initialized module 'multiprocessing'\", file=sys.stderr) if first else None; "
            "output.write_text(json.dumps({'status':'ok','summary':{}})) if not first else None; "
            "sys.exit(1 if first else 0)"
        )
        assert _run_cell(Cell("retry", (sys.executable, "-c", code), output), "", False) == "done retry"
        assert (temporary / "result.failed_attempt1.log").read_text() == "preserved\n"
        assert (temporary / "result.failed_attempt2.log").is_file()
    print(json.dumps({"self_check": "ok", "independent_files": len(list(Path(__file__).parent.glob("*")))}))


def write_model_inventory(model: Path, output: Path) -> None:
    model = model.resolve()
    files = [item for item in sorted(model.rglob("*")) if item.is_file()]
    records = [
        {"path": item.relative_to(model).as_posix(), "bytes": item.stat().st_size, "target": item.resolve().name}
        for item in files
    ]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {"model_dir": str(model), "files": records, "inventory_sha256": digest}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise ValueError(f"model inventory changed: {output}")
        return
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    self_check()

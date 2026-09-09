#!/usr/bin/env python3
"""Reproduce the locked full-parameter TRACE comparison of CAGD and OPR-RU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/trace_opr"
RUNS = ROOT / "runs/trace_opr_cagd"
THIRD_PARTY = ROOT / "third_party/OnPolicyReplay"
MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
TASKS = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)
EPOCHS = (5, 3, 7, 5, 3, 5, 5, 7)
MAX_LENGTH = 2048
BUFFER_SIZE = 50
MICRO_BATCH = 2
GRADIENT_ACCUMULATION = 8


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocations(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def model_inventory(model: Path) -> dict:
    important = [model / "config.json", model / "tokenizer_config.json", model / "model.safetensors.index.json"]
    hashes = {}
    for path in important:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    shards = sorted(model.glob("model-*.safetensors"))
    return {
        "path": str(model),
        "snapshot": model.name,
        "files": hashes,
        "weight_shards": [{"name": path.name, "bytes": path.stat().st_size} for path in shards],
        "weight_bytes": sum(path.stat().st_size for path in shards),
    }


def apply_template(tokenizer, prompt: str, answer: str | None = None) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )


def encode_training_example(tokenizer, prompt: str, answer: str) -> dict | None:
    prompt_ids = apply_template(tokenizer, prompt)
    full_ids = apply_template(tokenizer, prompt, answer)
    if len(full_ids) > MAX_LENGTH or full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]}


def official_tools():
    tools_dir = str(THIRD_PARTY / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import eval as opr_eval  # type: ignore
    import generate_opr_ru as opr_ru  # type: ignore

    return opr_eval, opr_ru


def score_rows(task_id: int, gold: list[str], responses: list[str], prompts: list[str]) -> float:
    opr_eval, _ = official_tools()
    if task_id in (0, 1, 4):
        return opr_eval.eval_acc(gold, responses)
    if task_id == 2:
        return opr_eval.eval_rougel(gold, responses)
    if task_id == 3:
        return opr_eval.eval_code(gold, responses)
    if task_id in (5, 6):
        return opr_eval.eval_math(gold, responses)
    return opr_eval.eval_sari(gold, responses, prompts)


def score_one(task_id: int, gold: str, response: str, prompt: str) -> float:
    _, opr_ru = official_tools()
    if task_id in (0, 1, 4):
        return opr_ru.score_acc(gold, response)
    if task_id == 2:
        return opr_ru.score_rougel(gold, response)
    if task_id == 3:
        return opr_ru.score_code(gold, response)
    if task_id in (5, 6):
        return opr_ru.score_math(gold, response)
    return opr_ru.score_sari_batch([gold], [response], [prompt])[0]


def generation_length(task_id: int) -> int:
    return 1 if task_id in (0, 1) else 512


def load_eligible(tokenizer, task_id: int, split: str) -> list[dict]:
    rows = read_jsonl(DATA / TASKS[task_id] / f"{split}.jsonl")
    return [row for row in rows if len(apply_template(tokenizer, row["prompt"], row["answer"])) <= MAX_LENGTH]


def make_llm(checkpoint: Path, seed: int):
    from vllm import LLM

    return LLM(
        model=str(checkpoint),
        tensor_parallel_size=8,
        dtype="bfloat16",
        seed=seed,
        max_model_len=2560,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )


def chat(llm, messages: list[list[dict]], temperature: float, max_tokens: int | list[int], n: int = 1):
    from vllm import SamplingParams

    params = (
        [SamplingParams(temperature=temperature, max_tokens=length, n=n) for length in max_tokens]
        if isinstance(max_tokens, list)
        else SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    )
    return llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})


def evaluate_tasks(llm, tokenizer, task_ids: list[int]) -> dict:
    result = {}
    for task_id in task_ids:
        rows = load_eligible(tokenizer, task_id, "test")
        messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
        outputs = chat(llm, messages, 0.1, generation_length(task_id), n=8)
        repeat_scores = []
        for repeat in range(8):
            responses = [output.outputs[repeat].text for output in outputs]
            repeat_scores.append(
                score_rows(
                    task_id,
                    [row["answer"] for row in rows],
                    responses,
                    [row["prompt"] for row in rows],
                )
            )
        result[TASKS[task_id]] = {
            "mean": sum(repeat_scores) / len(repeat_scores),
            "repetitions": repeat_scores,
            "eligible_test_examples": len(rows),
        }
    return result


def make_opr_buffer(llm, tokenizer, stage: int) -> list[dict]:
    selected = []
    for task_id, keep in enumerate(allocations(BUFFER_SIZE, stage)):
        rows = load_eligible(tokenizer, task_id, "train")
        messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
        outputs = chat(llm, messages, 0.1, generation_length(task_id))
        candidates = []
        for index, (row, output) in enumerate(zip(rows, outputs)):
            response = output.outputs[0].text
            candidates.append(
                {
                    "prompt": row["prompt"],
                    "answer": response,
                    "score": score_one(task_id, row["answer"], response, row["prompt"]),
                    "source_task": TASKS[task_id],
                    "source_index": index,
                }
            )
        candidates.sort(key=lambda row: (-row["score"], row["source_index"]))
        selected.extend(candidates[:keep])
    assert len(selected) == BUFFER_SIZE
    return selected


def make_cagd_anchors(llm, tokenizer, stage: int, seed: int) -> list[dict]:
    selected = []
    for task_id, keep in enumerate(allocations(BUFFER_SIZE, stage)):
        rows = load_eligible(tokenizer, task_id, "train")
        indices = list(range(len(rows)))
        random.Random(seed * 1000 + task_id).shuffle(indices)
        for index in indices[:keep]:
            selected.append({"prompt": rows[index]["prompt"], "source_task": TASKS[task_id], "source_index": index})
    messages = [[{"role": "user", "content": row["prompt"]}] for row in selected]
    lengths = [max(1, min(512, MAX_LENGTH - len(apply_template(tokenizer, row["prompt"])))) for row in selected]
    outputs = chat(llm, messages, 0.0, lengths)
    anchors = []
    for row, output in zip(selected, outputs):
        answer = output.outputs[0].text
        if encode_training_example(tokenizer, row["prompt"], answer) is None:
            raise RuntimeError("teacher trajectory exceeded the locked 2,048-token training context")
        anchors.append({**row, "answer": answer})
    assert len(anchors) == BUFFER_SIZE
    return anchors


def stage_inference(args) -> None:
    if args.evaluation is None and args.next_support is None:
        raise ValueError("stage-inference needs --evaluation and/or --next-support")
    if args.next_support is not None and args.stage == len(TASKS) - 1:
        raise ValueError("the final stage has no next-task support")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = make_llm(args.checkpoint, args.seed)
    task_ids = list(range(args.stage + 1)) if args.stage == len(TASKS) - 1 else [args.stage]
    if args.evaluation is not None:
        evaluation = {
            "checkpoint": str(args.checkpoint),
            "stage": args.stage,
            "task_scores": evaluate_tasks(llm, tokenizer, task_ids),
        }
        write_json(args.evaluation, evaluation)
    if args.next_support is not None:
        next_stage = args.stage + 1
        rows = (
            make_opr_buffer(llm, tokenizer, next_stage)
            if args.method == "opr"
            else make_cagd_anchors(llm, tokenizer, next_stage, args.seed)
        )
        write_jsonl(args.next_support, rows)


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.rglob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError(f"Swift produced no checkpoint below {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def train_opr(args) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    datasets = [DATA / TASKS[args.stage] / "train.jsonl"]
    if args.support is not None:
        datasets.append(args.support)
    command = [
        str(Path(sys.executable).parent / "swift"),
        "sft",
        "--model",
        str(args.checkpoint),
        "--train_type",
        "full",
        "--dataset",
        *map(str, datasets),
        "--num_train_epochs",
        str(EPOCHS[args.stage]),
        "--per_device_train_batch_size",
        str(MICRO_BATCH),
        "--gradient_accumulation_steps",
        str(GRADIENT_ACCUMULATION),
        "--learning_rate",
        "1e-5",
        "--max_length",
        str(MAX_LENGTH),
        "--output_dir",
        str(args.output),
        "--warmup_ratio",
        "0",
        "--weight_decay",
        "0",
        "--lr_scheduler_type",
        "cosine",
        "--max_grad_norm",
        "1",
        "--save_strategy",
        "epoch",
        "--save_total_limit",
        "1",
        "--save_only_model",
        "true",
        "--use_liger_kernel",
        "true",
        "--gradient_checkpointing",
        "true",
        "--columns",
        '{"prompt":"query","answer":"response"}',
        "--deepspeed",
        "zero2",
        "--report_to",
        "none",
        "--seed",
        str(args.seed),
        "--data_seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    env.update({"NPROC_PER_NODE": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
    started = time.time()
    subprocess.run(command, check=True, env=env)
    checkpoint = latest_checkpoint(args.output)
    write_json(
        args.output / "stage_result.json",
        {
            "method": "shared" if args.stage == 0 else "opr-ru",
            "stage": args.stage,
            "checkpoint": str(checkpoint),
            "elapsed_seconds": time.time() - started,
            "command": command,
        },
    )


class PairedDataset:
    def __init__(self, current: list[dict], anchors: list[dict]):
        self.current = current
        self.anchors = anchors

    def __len__(self):
        return len(self.current)

    def __getitem__(self, index):
        return {"current": self.current[index], "anchor": self.anchors[index % len(self.anchors)]}


def paired_collator(tokenizer):
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad(rows: list[dict], prefix: str) -> dict:
        length = max(len(row["input_ids"]) for row in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            missing = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [pad_id] * missing)
            masks.append([1] * len(row["input_ids"]) + [0] * missing)
            labels.append(row["labels"] + [-100] * missing)
        return {
            f"{prefix}_input_ids": torch.tensor(ids, dtype=torch.long),
            f"{prefix}_attention_mask": torch.tensor(masks, dtype=torch.long),
            f"{prefix}_labels": torch.tensor(labels, dtype=torch.long),
        }

    def collate(features: list[dict]) -> dict:
        anchors = [row["anchor"] for row in features]
        result = {**pad([row["current"] for row in features], "current"), **pad(anchors, "anchor")}
        if "teacher_logits" in anchors[0]:
            result["anchor_teacher_logits"] = torch.cat([row["teacher_logits"] for row in anchors])
        return result

    return collate


def train_cagd(args) -> None:
    import torch
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from torch.nn import functional as F
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not args.smoke and args.support is None:
        raise ValueError("formal CAGD training requires --support")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    current = [
        encoded
        for row in read_jsonl(DATA / TASKS[args.stage] / "train.jsonl")
        if (encoded := encode_training_example(tokenizer, row["prompt"], row["answer"])) is not None
    ]
    support_rows = (
        read_jsonl(args.support)
        if args.support is not None
        else read_jsonl(DATA / TASKS[0] / "train.jsonl")[:BUFFER_SIZE]
    )
    anchors = [
        encoded
        for row in support_rows
        if (encoded := encode_training_example(tokenizer, row["prompt"], row["answer"])) is not None
    ]
    if len(anchors) != BUFFER_SIZE:
        raise RuntimeError(f"expected {BUFFER_SIZE} valid anchors, found {len(anchors)}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": local_rank},
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        for start in range(0, len(anchors), 4):
            batch = anchors[start : start + 4]
            length = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.tensor(
                [row["input_ids"] + [pad_id] * (length - len(row["input_ids"])) for row in batch],
                dtype=torch.long,
                device=local_rank,
            )
            attention_mask = torch.tensor(
                [[1] * len(row["input_ids"]) + [0] * (length - len(row["input_ids"])) for row in batch],
                dtype=torch.long,
                device=local_rank,
            )
            hidden = teacher.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            head = teacher.get_output_embeddings()
            for index, row in enumerate(batch):
                labels = torch.tensor(row["labels"][1:], dtype=torch.long, device=local_rank)
                answer_hidden = hidden[index, : len(row["input_ids"]) - 1][labels != -100]
                row["teacher_logits"] = F.linear(answer_hidden, head.weight, getattr(head, "bias", None)).cpu()
    teacher_cache_peak = torch.cuda.max_memory_allocated()
    del teacher
    torch.cuda.empty_cache()

    student = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="sdpa"
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class CAGDModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = student
            self.ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        @staticmethod
        def hidden(model, input_ids, attention_mask):
            return model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

        def forward(
            self,
            current_input_ids,
            current_attention_mask,
            current_labels,
            anchor_input_ids,
            anchor_attention_mask,
            anchor_labels,
            anchor_teacher_logits,
        ):
            current_hidden = self.hidden(self.student, current_input_ids, current_attention_mask)
            current_target = current_labels[:, 1:]
            current_mask = current_target != -100
            current_answer_hidden = current_hidden[:, :-1][current_mask]
            sft_loss = self.ce(
                self.student.get_output_embeddings().weight,
                current_answer_hidden,
                current_target[current_mask],
                getattr(self.student.get_output_embeddings(), "bias", None),
            )
            anchor_student = self.hidden(self.student, anchor_input_ids, anchor_attention_mask)
            anchor_target = anchor_labels[:, 1:]
            anchor_mask = anchor_target != -100
            answer_hidden = anchor_student[:, :-1][anchor_mask]
            head = self.student.get_output_embeddings()
            student_logits = F.linear(answer_hidden, head.weight, getattr(head, "bias", None)).float()
            teacher_log_probs = F.log_softmax(anchor_teacher_logits.float(), dim=-1)
            distill_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1), teacher_log_probs, reduction="batchmean", log_target=True
            )
            return {"loss": sft_loss + distill_loss, "sft_loss": sft_loss.detach(), "distill_loss": distill_loss.detach()}

    deepspeed_config = {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": MICRO_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "train_batch_size": 128,
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
    }
    if args.smoke:
        current = current[:128]
    training_args = TrainingArguments(
        output_dir=str(args.output / "trainer"),
        num_train_epochs=1 if args.smoke else EPOCHS[args.stage],
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        deepspeed=deepspeed_config,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=CAGDModel(),
        args=training_args,
        train_dataset=PairedDataset(current, anchors),
        data_collator=paired_collator(tokenizer),
    )
    started = time.time()
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    peak = torch.tensor(max(teacher_cache_peak, torch.cuda.max_memory_allocated()), device=trainer.accelerator.device)
    torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    if trainer.accelerator.is_main_process:
        output_model = args.output / "model"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        if not args.smoke:
            unwrapped.student.config.use_cache = True
            unwrapped.student.save_pretrained(output_model, safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(output_model)
        write_json(
            args.output / "stage_result.json",
            {
                "method": "cagd",
                "stage": args.stage,
                "checkpoint": None if args.smoke else str(output_model),
                "smoke": args.smoke,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_bytes": int(peak.item()),
                "eligible_current_examples": len(current),
                "anchor_examples": len(anchors),
                "trainable_parameters": sum(parameter.numel() for parameter in student.parameters()),
            },
        )


def summarize(args) -> None:
    method_root = args.run / args.method
    diagonal = []
    for stage in range(len(TASKS)):
        root = args.run / ("shared" if stage == 0 else args.method) / f"stage{stage}"
        evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        diagonal.append(evaluation["task_scores"][TASKS[stage]]["mean"])
    final_eval = json.loads((method_root / "stage7/evaluation.json").read_text(encoding="utf-8"))
    final = [final_eval["task_scores"][task]["mean"] for task in TASKS]
    summary = {
        "method": args.method,
        "diagonal": dict(zip(TASKS, diagonal)),
        "final": dict(zip(TASKS, final)),
        "ACC": sum(final) / len(final),
        "BWT": sum(final[index] - diagonal[index] for index in range(len(TASKS) - 1)) / (len(TASKS) - 1),
    }
    write_json(method_root / "summary.json", summary)


def self_check() -> None:
    assert allocations(50, 3) == [17, 17, 16]
    assert allocations(50, 7) == [8, 7, 7, 7, 7, 7, 7]
    toy = PairedDataset([{"x": i} for i in range(3)], [{"y": 1}, {"y": 2}])
    assert toy[2] == {"current": {"x": 2}, "anchor": {"y": 1}}
    assert generation_length(0) == 1 and generation_length(4) == 512
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        value = torch.tensor([dist.get_rank() + 1.0], device=torch.cuda.current_device())
        dist.all_reduce(value)
        assert value.item() == dist.get_world_size() * (dist.get_world_size() + 1) / 2
        dist.destroy_process_group()
    print(json.dumps({"self_check": "ok", "world_size": int(os.environ.get("WORLD_SIZE", "1"))}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--model", type=Path, default=MODEL)
    inventory.add_argument("--output", type=Path, required=True)

    inference = sub.add_parser("stage-inference")
    inference.add_argument("--method", choices=("opr", "cagd"), required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--stage", type=int, choices=range(8), required=True)
    inference.add_argument("--seed", type=int, default=3407)
    inference.add_argument("--evaluation", type=Path)
    inference.add_argument("--next-support", type=Path)

    opr = sub.add_parser("train-opr")
    opr.add_argument("--checkpoint", type=Path, required=True)
    opr.add_argument("--stage", type=int, choices=range(8), required=True)
    opr.add_argument("--seed", type=int, default=3407)
    opr.add_argument("--support", type=Path)
    opr.add_argument("--output", type=Path, required=True)

    cagd = sub.add_parser("train-cagd")
    cagd.add_argument("--checkpoint", type=Path, required=True)
    cagd.add_argument("--stage", type=int, choices=range(1, 8), required=True)
    cagd.add_argument("--seed", type=int, default=3407)
    cagd.add_argument("--support", type=Path)
    cagd.add_argument("--smoke", action="store_true")
    cagd.add_argument("--output", type=Path, required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--run", type=Path, required=True)
    summary.add_argument("--method", choices=("opr", "cagd"), required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "inventory":
        write_json(args.output, model_inventory(args.model))
    elif args.command == "stage-inference":
        stage_inference(args)
    elif args.command == "train-opr":
        train_opr(args)
    elif args.command == "train-cagd":
        train_cagd(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()

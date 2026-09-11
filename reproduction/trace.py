#!/usr/bin/env python3
"""Run the locked full-parameter TRACE continual-learning comparison."""

from __future__ import annotations

import argparse
import encodings.unicode_escape  # Preload before distributed workers compile the Jinja chat template.
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/trace_opr"
MODEL = Path(
    "/home/JJ_Group/lih2511/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
CANONICAL_TASKS = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)
CANONICAL_EPOCHS = (5, 3, 7, 5, 3, 5, 5, 7)
TASKS = CANONICAL_TASKS
EPOCHS = CANONICAL_EPOCHS
MAX_LENGTH = 2048
BUFFER_SIZE = 50


def select_trainable_parameters(model, mode: str):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "all":
        selected = list(model.named_parameters())
    elif mode == "last_block":
        prefix = f"model.layers.{model.config.num_hidden_layers - 1}."
        selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)]
    else:
        raise ValueError(f"unknown trainable scope: {mode}")
    if not selected:
        raise RuntimeError(f"no parameters selected for {mode}")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return selected
MICRO_BATCH = 4
GRADIENT_ACCUMULATION = 4
SDFT_MICRO_BATCH = 1
SDFT_GRADIENT_ACCUMULATION = 16
SDFT_EMA_RATE = 0.01
SDFT_KL_CHUNK = 16
SDFT_LOSS_TOKENS_TO_SKIP = 3


def configure_task_order(order: str) -> None:
    global TASKS, EPOCHS
    TASKS = CANONICAL_TASKS if order == "canonical" else CANONICAL_TASKS[::-1]
    EPOCHS = CANONICAL_EPOCHS if order == "canonical" else CANONICAL_EPOCHS[::-1]


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


def _code_text(text: str) -> str:
    text = text.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    for kind, value in re.findall(r"<(STR|NUM|CHAR)_LIT:(.*?)>", text, re.S):
        text = text.replace(f"<{kind}_LIT:{value}>", value)
    return text


def _math_answer(text: str) -> str | None:
    values = re.findall(r"(\-?[0-9\.\,]+)", text)
    return next((value for value in reversed(values) if value not in ("", ".")), None)


def _sari(gold: list[str], responses: list[str], prompts: list[str]) -> float:
    from evaluate import load

    sources = [prompt.split("Paragraph:\n", 1)[1].split("\n\nSimplification:", 1)[0] for prompt in prompts]
    return float(load("sari").compute(
        sources=sources, predictions=responses, references=[[answer] for answer in gold]
    )["sari"])


def score_rows(task_id: int, gold: list[str], responses: list[str], prompts: list[str]) -> float:
    task = TASKS[task_id]
    if task in ("C-STANCE", "FOMC", "ScienceQA"):
        return 100 * sum(bool(response) and answer[:1] == response[:1] for answer, response in zip(gold, responses)) / len(gold)
    if task == "MeetingBank":
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"])
        return 100 * sum(scorer.score(answer, response)["rougeL"].fmeasure for answer, response in zip(gold, responses)) / len(gold)
    if task == "Py150":
        from fuzzywuzzy import fuzz

        return sum(fuzz.ratio(_code_text(response), _code_text(answer)) for answer, response in zip(gold, responses)) / len(gold)
    if task in ("NumGLUE-cm", "NumGLUE-ds"):
        return 100 * sum(_math_answer(response) == answer for answer, response in zip(gold, responses)) / len(gold)
    return _sari(gold, responses, prompts)


def score_one(task_id: int, gold: str, response: str, prompt: str) -> float:
    task = TASKS[task_id]
    if task in ("C-STANCE", "FOMC", "ScienceQA"):
        return 100.0 if response and gold[:1] == response[:1] else 0.0
    if task == "MeetingBank":
        from rouge_score import rouge_scorer

        return 100 * rouge_scorer.RougeScorer(["rougeL"]).score(gold, response)["rougeL"].fmeasure
    if task == "Py150":
        from fuzzywuzzy import fuzz

        return float(fuzz.ratio(_code_text(response), _code_text(gold)))
    if task in ("NumGLUE-cm", "NumGLUE-ds"):
        return 100.0 if _math_answer(response) == gold else 0.0
    return _sari([gold], [response], [prompt])


def generation_length(task_id: int) -> int:
    return 1 if TASKS[task_id] in ("C-STANCE", "FOMC") else 512


def sdft_loss_tokens_to_skip(task_id: int) -> int:
    # Preserve at least one supervised token for TRACE's one-token classifiers.
    return min(SDFT_LOSS_TOKENS_TO_SKIP, generation_length(task_id) - 1)


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


def make_replay_buffer(tokenizer, stage: int, seed: int) -> list[dict]:
    selected = []
    for task_id, keep in enumerate(allocations(BUFFER_SIZE, stage)):
        rows = read_jsonl(DATA / TASKS[task_id] / "train.jsonl")
        indices = list(range(len(rows)))
        random.Random(seed * 1000 + task_id).shuffle(indices)
        kept = 0
        for index in indices:
            row = rows[index]
            if len(apply_template(tokenizer, row["prompt"], row["answer"])) > MAX_LENGTH:
                continue
            selected.append({**row, "source_task": TASKS[task_id], "source_index": index})
            kept += 1
            if kept == keep:
                break
        if kept != keep:
            raise RuntimeError(f"{TASKS[task_id]} has only {kept} eligible replay records; expected {keep}")
    assert len(selected) == BUFFER_SIZE
    return selected


def sdft_teacher_prompt(prompt: str, answer: str) -> str:
    return (
        f"\n{prompt}\n\n"
        "This is an example for a response to the question:\n"
        f"{answer}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )


def stage_inference(args) -> None:
    if args.evaluation is None and args.next_support is None:
        raise ValueError("stage-inference needs --evaluation and/or --next-support")
    if args.next_support is not None and args.stage == len(TASKS) - 1:
        raise ValueError("the final stage has no next-task support")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = None
    if args.evaluation is not None:
        # Load the official scorers before vLLM initializes CUDA worker processes.
        official_tools()
        llm = make_llm(args.checkpoint, args.seed)
        task_ids = list(range(args.stage + 1)) if args.stage == len(TASKS) - 1 else [args.stage]
        evaluation = {
            "checkpoint": str(args.checkpoint),
            "stage": args.stage,
            "task_order": list(TASKS),
            "task_scores": evaluate_tasks(llm, tokenizer, task_ids),
        }
        write_json(args.evaluation, evaluation)
    if args.next_support is not None:
        next_stage = args.stage + 1
        if args.method == "replay":
            rows = make_replay_buffer(tokenizer, next_stage, args.seed)
        elif args.method == "opr":
            if llm is None:
                official_tools()
                llm = make_llm(args.checkpoint, args.seed)
            rows = make_opr_buffer(llm, tokenizer, next_stage)
        elif args.method == "cagd":
            if llm is None:
                llm = make_llm(args.checkpoint, args.seed)
            rows = make_cagd_anchors(llm, tokenizer, next_stage, args.seed)
        else:
            raise ValueError(f"{args.method} does not construct replay support")
        write_jsonl(args.next_support, rows)


def train_sft(args) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if args.command == "train-sequential" and args.support is not None:
        raise ValueError("Sequential training does not accept replay support")
    if args.stage > 0 and args.command in ("train-replay", "train-opr") and args.support is None:
        raise ValueError(f"{args.command} requires --support after Stage 0")
    started = time.time()
    # Serialize cold imports because the current environment is shared across workers.
    with Path("/tmp/trace_opr_cagd_python_import.lock").open("a") as import_lock:
        fcntl.flock(import_lock, fcntl.LOCK_EX)
        import torch
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        apply_template(tokenizer, "")
    rows = read_jsonl(DATA / TASKS[args.stage] / "train.jsonl")
    if args.support is not None:
        rows += read_jsonl(args.support)
    encoded = [
        example
        for row in rows
        if (example := encode_training_example(tokenizer, row["prompt"], row["answer"])) is not None
    ]

    class EncodedDataset:
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(features):
        length = max(len(row["input_ids"]) for row in features)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] + [pad_id] * (length - len(row["input_ids"])) for row in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row["input_ids"]) + [0] * (length - len(row["input_ids"])) for row in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [row["labels"] + [-100] * (length - len(row["labels"])) for row in features], dtype=torch.long
            ),
        }

    student = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="sdpa"
    )
    trainable = select_trainable_parameters(student, args.trainable)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class SFTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = student
            self.ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        def forward(self, input_ids, attention_mask, labels):
            hidden = self.student.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            target = labels[:, 1:]
            mask = target != -100
            head = self.student.get_output_embeddings()
            loss = self.ce(head.weight, hidden[:, :-1][mask], target[mask], getattr(head, "bias", None))
            return {"loss": loss}

    deepspeed_config = {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": MICRO_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "train_batch_size": 128,
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
    }
    training_args = TrainingArguments(
        output_dir=str(args.output / "trainer"),
        num_train_epochs=EPOCHS[args.stage],
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
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
        group_by_length=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(model=SFTModel(), args=training_args, train_dataset=EncodedDataset(), data_collator=collate)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device=trainer.accelerator.device)
    torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    if trainer.accelerator.is_main_process:
        output_model = args.output / "model"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        unwrapped.student.config.use_cache = True
        unwrapped.student.save_pretrained(output_model, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output_model)
        write_json(
            args.output / "stage_result.json",
            {
                "method": "shared" if args.stage == 0 else {
                    "train-sequential": "sequential",
                    "train-replay": "vanilla-replay",
                    "train-opr": "opr-ru",
                }[args.command],
                "stage": args.stage,
                "task": TASKS[args.stage],
                "task_order": list(TASKS),
                "checkpoint": str(output_model),
                "elapsed_seconds": time.time() - started,
                "peak_gpu_bytes": int(peak.item()),
                "eligible_training_examples": len(encoded),
                "current_examples_before_filter": 5000,
                "support_examples_before_filter": 0 if args.support is None else len(read_jsonl(args.support)),
                "trainable_scope": args.trainable,
                "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            },
        )
    trainer.accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


class PairedDataset:
    def __init__(self, current: list[dict], anchors: list[dict]):
        self.current = current
        self.anchors = anchors

    def __len__(self):
        return len(self.current)

    def __getitem__(self, index):
        current = self.current[index]
        anchor = self.anchors[index % len(self.anchors)]
        # input_ids is exposed only so Trainer's native length sampler can avoid excessive padding.
        return {"current": current, "anchor": anchor, "input_ids": current["input_ids"]}


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
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not args.smoke and args.support is None:
        raise ValueError("formal CAGD training requires --support")
    started = time.time()
    # Serialize cold imports because the current environment is shared across workers.
    with Path("/tmp/trace_opr_cagd_python_import.lock").open("a") as import_lock:
        fcntl.flock(import_lock, fcntl.LOCK_EX)
        import torch
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        from torch import nn
        from torch.nn import functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        apply_template(tokenizer, "")
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
    trainable = select_trainable_parameters(student, args.trainable)
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
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
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
        group_by_length=True,
        length_column_name="length",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=CAGDModel(),
        args=training_args,
        train_dataset=PairedDataset(current, anchors),
        data_collator=paired_collator(tokenizer),
    )
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
                "task": TASKS[args.stage],
                "task_order": list(TASKS),
                "checkpoint": None if args.smoke else str(output_model),
                "smoke": args.smoke,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_bytes": int(peak.item()),
                "eligible_current_examples": len(current),
                "anchor_examples": len(anchors),
                "trainable_scope": args.trainable,
                "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            },
        )
    trainer.accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def train_sdft(args) -> None:
    """SDFT: on-policy student rollouts with a demonstration-conditioned EMA teacher."""
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    started = time.time()
    # This implements the published SDFT core with the TRACE task interface.
    with Path("/tmp/trace_opr_cagd_python_import.lock").open("a") as import_lock:
        fcntl.flock(import_lock, fcntl.LOCK_EX)
        import torch
        from torch import nn
        from torch.nn import functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        apply_template(tokenizer, "")

    max_new_tokens = generation_length(args.stage)
    loss_tokens_to_skip = sdft_loss_tokens_to_skip(args.stage)
    max_prompt_tokens = MAX_LENGTH - max_new_tokens
    rows = load_eligible(tokenizer, args.stage, "train")
    if args.smoke:
        rows = rows[:128]
    encoded = []
    for row in rows:
        student_ids = apply_template(tokenizer, row["prompt"])[-max_prompt_tokens:]
        teacher_ids = apply_template(tokenizer, sdft_teacher_prompt(row["prompt"], row["answer"]))[-max_prompt_tokens:]
        if student_ids and teacher_ids:
            encoded.append({"student_ids": student_ids, "teacher_ids": teacher_ids, "input_ids": student_ids})

    class SDFTDataset:
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def left_pad(sequences):
        width = max(len(sequence) for sequence in sequences)
        return (
            torch.tensor([[pad_id] * (width - len(sequence)) + sequence for sequence in sequences], dtype=torch.long),
            torch.tensor([[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences], dtype=torch.long),
        )

    def collate(features):
        student_ids, student_mask = left_pad([row["student_ids"] for row in features])
        teacher_ids, teacher_mask = left_pad([row["teacher_ids"] for row in features])
        return {
            "student_input_ids": student_ids,
            "student_attention_mask": student_mask,
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": teacher_mask,
        }

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
    student = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    trainable = select_trainable_parameters(student, args.trainable)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class SDFTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = student
            # The frozen teacher is already resident on this rank and must not enter DeepSpeed's optimizer/state dict.
            object.__setattr__(self, "teacher", teacher)

        def train(self, mode=True):
            super().train(mode)
            self.teacher.eval()
            return self

        @staticmethod
        def completion_mask(completion):
            eos_id = tokenizer.eos_token_id
            mask = completion.eq(eos_id).cumsum(dim=1).le(1)
            if pad_id != eos_id:
                mask &= completion.ne(pad_id)
            return mask.long()

        def forward(
            self,
            student_input_ids,
            student_attention_mask,
            teacher_input_ids,
            teacher_attention_mask,
        ):
            was_training = self.student.training
            self.student.eval()
            with torch.no_grad():
                generated = self.student.generate(
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            self.student.train(was_training)
            completion = generated[:, student_input_ids.shape[1] :]
            completion_mask = self.completion_mask(completion)
            loss_mask = completion_mask.clone()
            loss_mask[:, :loss_tokens_to_skip] = 0
            student_full = torch.cat((student_input_ids, completion), dim=1)
            student_full_mask = torch.cat((student_attention_mask, completion_mask), dim=1)
            teacher_full = torch.cat((teacher_input_ids, completion), dim=1)
            teacher_full_mask = torch.cat((teacher_attention_mask, completion_mask), dim=1)

            student_hidden = self.student.model(
                input_ids=student_full, attention_mask=student_full_mask, use_cache=False
            ).last_hidden_state[:, student_input_ids.shape[1] - 1 : -1]
            with torch.no_grad():
                teacher_hidden = self.teacher.model(
                    input_ids=teacher_full, attention_mask=teacher_full_mask, use_cache=False
                ).last_hidden_state[:, teacher_input_ids.shape[1] - 1 : -1]

            student_head = self.student.get_output_embeddings()
            teacher_head = self.teacher.get_output_embeddings()
            loss_sum = student_hidden.reshape(-1)[0].float() * 0.0
            token_count = loss_mask.sum().clamp(min=1)
            for start in range(0, completion.shape[1], SDFT_KL_CHUNK):
                stop = min(start + SDFT_KL_CHUNK, completion.shape[1])
                valid = loss_mask[:, start:stop].bool()
                if not valid.any():
                    continue
                student_logits = F.linear(
                    student_hidden[:, start:stop][valid],
                    student_head.weight,
                    getattr(student_head, "bias", None),
                ).float()
                with torch.no_grad():
                    teacher_logits = F.linear(
                        teacher_hidden[:, start:stop][valid],
                        teacher_head.weight,
                        getattr(teacher_head, "bias", None),
                    ).float()
                    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                loss_sum = loss_sum + F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_log_probs,
                    reduction="sum",
                    log_target=True,
                )
            return {"loss": loss_sum / token_count}

    class EMATeacher(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            with torch.no_grad():
                for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
                    teacher_parameter.mul_(1.0 - SDFT_EMA_RATE).add_(student_parameter, alpha=SDFT_EMA_RATE)

    deepspeed_config = {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": SDFT_MICRO_BATCH,
        "gradient_accumulation_steps": SDFT_GRADIENT_ACCUMULATION,
        "train_batch_size": 128,
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
    }
    training_args = TrainingArguments(
        output_dir=str(args.output / "trainer"),
        num_train_epochs=1 if args.smoke else EPOCHS[args.stage],
        max_steps=1 if args.smoke else -1,
        per_device_train_batch_size=SDFT_MICRO_BATCH,
        gradient_accumulation_steps=SDFT_GRADIENT_ACCUMULATION,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
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
        group_by_length=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=SDFTModel(),
        args=training_args,
        train_dataset=SDFTDataset(),
        data_collator=collate,
        callbacks=[EMATeacher()],
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device=trainer.accelerator.device)
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
                "method": "sdft",
                "stage": args.stage,
                "task": TASKS[args.stage],
                "task_order": list(TASKS),
                "checkpoint": None if args.smoke else str(output_model),
                "smoke": args.smoke,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_bytes": int(peak.item()),
                "eligible_training_examples": len(encoded),
                "trainable_scope": args.trainable,
                "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
                "student_sampling_temperature": 1.0,
                "student_sampling_top_p": 1.0,
                "student_sampling_top_k": 0,
                "maximum_generation_tokens": max_new_tokens,
                "loss_tokens_to_skip": loss_tokens_to_skip,
                "teacher_ema_rate": SDFT_EMA_RATE,
                "teacher_update_frequency": "after every optimizer step",
                "distillation": "token-level forward KL on student rollouts",
                "generation_source": "student conditioned only on the original prompt",
                "teacher_context": "original prompt plus its paired expert demonstration",
                "importance_sampling_correction": "not applicable: generation and optimization use the same model",
            },
        )
    trainer.accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def summarize(args) -> None:
    method_root = args.run / args.method
    diagonal = []
    for stage in range(len(TASKS)):
        root = args.run / ("shared" if stage == 0 and args.method != "sdft" else args.method) / f"stage{stage}"
        evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        diagonal.append(evaluation["task_scores"][TASKS[stage]]["mean"])
    final_eval = json.loads((method_root / "stage7/evaluation.json").read_text(encoding="utf-8"))
    final = [final_eval["task_scores"][task]["mean"] for task in TASKS]
    summary = {
        "method": args.method,
        "task_order": list(TASKS),
        "diagonal": dict(zip(TASKS, diagonal)),
        "final": dict(zip(TASKS, final)),
        "ACC": sum(final) / len(final),
        "BWT": sum(final[index] - diagonal[index] for index in range(len(TASKS) - 1)) / (len(TASKS) - 1),
    }
    write_json(method_root / "summary.json", summary)


def summarize_comparison(args) -> None:
    rows = []
    for method in args.methods:
        summary = json.loads((args.run / method / "summary.json").read_text(encoding="utf-8"))
        rows.append(summary)
    result = {
        "schema_version": 2,
        "status": "ok",
        "experiment": "trace_comparison",
        "task_order": list(TASKS),
        "rows": rows,
    }
    output = args.run / "summary.json"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError(f"existing comparison summary differs: {output}")
        return
    write_json(output, result)


def self_check() -> None:
    assert allocations(50, 3) == [17, 17, 16]
    assert allocations(50, 7) == [8, 7, 7, 7, 7, 7, 7]
    assert dict(zip(TASKS, EPOCHS)) == dict(zip(CANONICAL_TASKS, CANONICAL_EPOCHS))
    current = [{"input_ids": list(range(index + 1)), "labels": []} for index in range(3)]
    anchors = [{"input_ids": [1], "labels": []}, {"input_ids": [1, 2], "labels": []}]
    toy = PairedDataset(current, anchors)
    assert toy[2] == {"current": current[2], "anchor": anchors[0], "input_ids": current[2]["input_ids"]}
    assert generation_length(TASKS.index("C-STANCE")) == 1
    assert generation_length(TASKS.index("ScienceQA")) == 512
    assert sdft_loss_tokens_to_skip(TASKS.index("C-STANCE")) == 0
    assert sdft_loss_tokens_to_skip(TASKS.index("MeetingBank")) == 3
    assert _code_text("<NUM_LIT> <STR_LIT:x>") == "0 x"
    assert _math_answer("work 1.0, then -2.5") == "-2.5"
    assert score_rows(TASKS.index("C-STANCE"), ["A", "B"], ["Answer", ""], ["", ""]) == 50.0
    teacher_prompt = sdft_teacher_prompt("question", "answer")
    assert teacher_prompt.startswith("\nquestion\n\nThis is an example")
    assert "\nanswer\n\n" in teacher_prompt
    assert teacher_prompt.endswith("including the thinking process.\n")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        value = torch.tensor([dist.get_rank() + 1.0], device=torch.cuda.current_device())
        dist.all_reduce(value)
        assert value.item() == dist.get_world_size() * (dist.get_world_size() + 1) / 2
        dist.destroy_process_group()
    print(
        json.dumps(
            {"self_check": "ok", "world_size": int(os.environ.get("WORLD_SIZE", "1")), "task_order": TASKS}
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-order", choices=("canonical", "reverse"), default="canonical")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--model", type=Path, default=MODEL)
    inventory.add_argument("--output", type=Path, required=True)

    inference = sub.add_parser("stage-inference")
    inference.add_argument("--method", choices=("sequential", "replay", "sdft", "opr", "cagd"), required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--stage", type=int, choices=range(8), required=True)
    inference.add_argument("--seed", type=int, default=3407)
    inference.add_argument("--evaluation", type=Path)
    inference.add_argument("--next-support", type=Path)

    for command in ("train-sequential", "train-replay", "train-opr"):
        sft = sub.add_parser(command)
        sft.add_argument("--checkpoint", type=Path, required=True)
        sft.add_argument("--stage", type=int, choices=range(8), required=True)
        sft.add_argument("--seed", type=int, default=3407)
        sft.add_argument("--support", type=Path)
        sft.add_argument("--trainable", choices=("last_block", "all"), default="all")
        sft.add_argument("--output", type=Path, required=True)

    cagd = sub.add_parser("train-cagd")
    cagd.add_argument("--checkpoint", type=Path, required=True)
    cagd.add_argument("--stage", type=int, choices=range(1, 8), required=True)
    cagd.add_argument("--seed", type=int, default=3407)
    cagd.add_argument("--support", type=Path)
    cagd.add_argument("--trainable", choices=("last_block", "all"), default="all")
    cagd.add_argument("--smoke", action="store_true")
    cagd.add_argument("--output", type=Path, required=True)

    sdft = sub.add_parser("train-sdft")
    sdft.add_argument("--checkpoint", type=Path, required=True)
    sdft.add_argument("--stage", type=int, choices=range(8), required=True)
    sdft.add_argument("--seed", type=int, default=3407)
    sdft.add_argument("--trainable", choices=("last_block", "all"), default="all")
    sdft.add_argument("--smoke", action="store_true")
    sdft.add_argument("--output", type=Path, required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--run", type=Path, required=True)
    summary.add_argument("--method", choices=("sequential", "replay", "sdft", "opr", "cagd"), required=True)
    comparison = sub.add_parser("summarize-comparison")
    comparison.add_argument("--run", type=Path, required=True)
    comparison.add_argument(
        "--methods", nargs="+", choices=("sequential", "replay", "sdft", "opr", "cagd"), required=True
    )
    return result


def main() -> None:
    args = parser().parse_args()
    configure_task_order(args.task_order)
    if args.command == "self-check":
        self_check()
    elif args.command == "inventory":
        write_json(args.output, model_inventory(args.model))
    elif args.command == "stage-inference":
        stage_inference(args)
    elif args.command in ("train-sequential", "train-replay", "train-opr"):
        train_sft(args)
    elif args.command == "train-cagd":
        train_cagd(args)
    elif args.command == "train-sdft":
        train_sdft(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "summarize-comparison":
        summarize_comparison(args)


if __name__ == "__main__":
    main()

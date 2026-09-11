# TRACE comparison: CAGD versus OPR-RU

Status: frozen before pilot training on 2026-09-10.

## Question and reporting gate

This experiment asks whether CAGD is competitive with the default rule-scored
On-Policy Replay method (OPR-RU) on the standard eight-task TRACE stream under
full-parameter adaptation of the same 4B instruction model.

Seed 3407 is run first as a paired systems pilot.  The remaining seeds 3408
and 3409 are authorized when the pilot is not clearly dominated: CAGD must be
within 2 percentage points of OPR-RU on both final average accuracy and BWT.
The three-seed result may enter Section 4.4 after the existing Table 4 only if
CAGD either exceeds OPR-RU, or is within 1 point on both mean final accuracy
and mean BWT.  Otherwise the manuscript is not changed.

## Locked benchmark and model

- TRACE `LLM-CL-Benchmark_5000`, downloaded from the official release linked
  by both TRACE and OPR.  Every task has 5,000 training examples; official test
  sets are used without subsampling.
- Canonical order: C-STANCE, FOMC, MeetingBank, Py150, ScienceQA,
  NumGLUE-cm, NumGLUE-ds, and 20Minuten.
- Qwen3-4B-Instruct-2507 snapshot
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Full-parameter bf16 adaptation on eight A100 40GB GPUs.
- Seeds: 3407, 3408, and 3409.

## Shared training and evaluation

Both methods use AdamW (`beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`) with learning rate `1e-5`, cosine decay, no warmup,
zero weight decay, gradient clipping at 1, max sequence length 2,048, and
global batch size 128.  The per-task epoch schedule is
`[5, 3, 7, 5, 3, 5, 5, 7]`.  Because the available GPUs have 40GB rather than
the OPR paper's 80GB, a microbatch of four with four gradient-accumulation
steps preserves the same global batch size.  Gradient checkpointing and ZeRO are engineering
changes shared by both methods, not experimental factors.

Training and evaluation use the Qwen chat template with thinking disabled.
Loss is computed only on assistant tokens.  Examples whose complete templated
sequence exceeds 2,048 tokens are filtered consistently.  Evaluation uses the
official OPR task scorers: first-character accuracy for C-STANCE, FOMC, and
ScienceQA; ROUGE-L F1 for MeetingBank; fuzzy edit similarity for Py150;
numeric exact match for NumGLUE-cm/ds; and SARI for 20Minuten.  Decoding uses
temperature 0.1 and eight repetitions, following OPR.  We report the final
mean of the eight task scores (ACC) and BWT over the seven past tasks,
`mean_i<8(a_i,8 - a_i,i)`; higher is better for both.

## Method-specific state

The permanent replay budget is 1% of one task, or 50 records total, divided as
evenly as possible over prior tasks at every stage.

- **OPR-RU:** use the implementation at commit
  `1384d8823b250acf2725dc983aea6ac6e64a4283`.  Roll out the latest checkpoint
  once on every eligible historical training prompt, score each response with
  its TRACE task metric, retain the highest-scoring allocation, and concatenate
  those 50 prompt--response pairs with the next task for ordinary SFT.
- **CAGD:** retain only 50 deterministically sampled historical prompts.  At
  each stage boundary, the frozen previous checkpoint greedily generates one
  completion of at most 512 tokens per anchor.  The fixed teacher logits on
  those fixed AR trajectory states are cached exactly once at the stage
  boundary.  During every current-task update, a separate task-balanced anchor
  minibatch receives forward token KL from that cache on answer positions,
  with temperature 1 and weight 1.  This cache is mathematically identical to
  repeating the frozen-teacher forward pass; it changes runtime, not the loss.
  The teacher cache and generated completions are stage-local.

The methods are matched on backbone, task data and order, optimizer schedule,
current-task exposures, permanent-record count, and evaluation.  They are not
FLOP matched: CAGD deliberately pays a frozen-teacher forward pass during
updates, whereas OPR pays large stage-boundary rollouts and ordinary replay
SFT.  Wall time and peak memory are therefore recorded alongside quality.

## Source discrepancies handled by the runner

The released OPR repository contains an undefined `checkpoint_dir`, uses a
linear scheduler in `scripts/train.sh` although the paper specifies cosine,
and selects generation length from the current stage rather than the task
being evaluated.  The third-party checkout remains unchanged.  The comparison
runner follows the published protocol and imports/reuses the released task
scorers and OPR-RU selection rule.  Ordinary current-task and OPR replay SFT
use the same assistant-only CE/ZeRO runner as the CAGD student, avoiding a
training-backend confound; this does not alter OPR's generated replay or
rule-scored selection.  Every correction is recorded in result provenance.

## Full-baseline extension

Status: specified on 2026-09-12 before running the added baselines.

The canonical three-seed matrix additionally contains the following methods.
All share the model, data, evaluation, optimizer, epoch schedule, and global
batch size above.

- **Sequential:** ordinary assistant-token SFT on the current task, with no
  old-task signal.
- **Vanilla Replay:** ordinary SFT on the current task plus 50 stored gold
  prompt--answer records, divided evenly over prior tasks. Records are sampled
  deterministically within task and seed.
- **SDFT:** a clean-room implementation of the public algorithm at
  `Continual-Intelligence/Self-Distillation` commit
  `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e`. At every update, the student
  samples one on-policy completion from the original query at temperature 1,
  top-p 1, with top-k disabled. The EMA teacher receives the original query
  plus its paired expert demonstration and scores that same student-generated
  completion. Token-level forward KL from the teacher distribution to the
  student distribution is minimized, and the teacher is updated after every
  optimizer step as $\phi\leftarrow0.01\theta+0.99\phi$. Because generation
  uses the same Hugging Face student that is optimized, rather than a separate
  vLLM copy, no inference-engine importance correction is needed. Task-specific
  maximum generation lengths match evaluation (one token for C-STANCE and
  FOMC, 512 otherwise). Following the released implementation, the first three
  completion tokens are excluded from the loss on long-form tasks; this is set
  to zero for the one-token classification tasks so their loss remains defined.
  SDFT trains Stage 0 independently; Sequential, Vanilla Replay, OPR-RU, and
  CAGD share the identical Stage-0 SFT checkpoint.

The SDFT teacher receives one user message constructed exactly as follows,
where the student receives only `<query>`:

```text
<query>

This is an example for a response to the question:
<expert demonstration>

Now answer with a response of your own, including the thinking process.
```

The SDFT source is used as a specification rather than vendored: its released
trainer is task-specific, has no TRACE adapter, and carries no explicit license.
The local runner records the SDFT sampling, EMA, teacher-context, and loss
settings in every stage result.

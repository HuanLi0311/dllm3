# Qwen3-0.6B full-parameter diagnostic results

Status: complete on 2026-09-10. The manuscript was not changed.

## Paired result

Both runs use seed 3407 and the same four-task forward factual stream and
hyperparameters as the existing Qwen scale experiment. Lower loss and
forgetting are better; higher answer-token accuracy is better.

| Trainable parameters | Method | Final average loss | Past-task forgetting | Answer-token accuracy | Final-task loss |
|---|---|---:|---:|---:|---:|
| All 596,049,920 | Sequential | 3.2795 | 0.5800 | 0.7401 | 0.4838 |
| All 596,049,920 | CAGD (AR/GD) | 1.9361 | -0.7607 | 0.8603 | 0.5050 |
| Paired change | CAGD - Sequential | -1.3434 | -1.3407 | +0.1202 | +0.0212 |

CAGD lowers final average held-out loss by 41.0%. Its negative forgetting is
backward improvement under the preregistered definition: each of the three
past-task final losses is no higher than its loss when first learned. The
final-task loss changes only from 0.4838 to 0.5050, while final-task answer-token
accuracy changes from 0.9673 to 0.9691, so the retention gain is not explained
by a failure to acquire the final task.

## Comparison with the existing last-block run

The existing seed-3407 last-block endpoints are read from
`formal_summary_3scale.json`.

| Trainable scope | Method | Final average loss | Past-task forgetting | Answer-token accuracy |
|---|---|---:|---:|---:|
| Last block | Sequential | 3.9849 | 2.0275 | 0.7496 |
| Last block | CAGD (AR/GD) | 2.7477 | 0.3713 | 0.8521 |
| All parameters | Sequential | 3.2795 | 0.5800 | 0.7401 |
| All parameters | CAGD (AR/GD) | 1.9361 | -0.7607 | 0.8603 |

Full-parameter Sequential is already substantially less forgetful than its
last-block counterpart, so the result does not support a claim that ordinary
full-parameter adaptation is intrinsically unstable. It does support the
narrower paper claim: under the same full-parameter protocol, CAGD adds a large
retention and final-loss benefit beyond Sequential, and the paired direction
matches the existing last-block result.

This is a one-seed sensitivity check. It establishes direction for deciding
whether a full three-seed cell is worthwhile; it does not provide a variance
estimate or justify replacing the existing three-seed result.

## Audit

- Trainable tensors: 310; trainable parameters: 596,049,920 in both runs.
- Task order: `d2p_8-11 -> p2d_12-15 -> d2p_16-19 -> p2d_20-23`.
- CAGD replay counts by stage: 0, 64, 128, and 192; every old task contributes
  64 prompts, balanced at 16 prompts per fact.
- Sequential result SHA-256:
  `35d25f5cac4240e76f585c2221bbde1fbcbf3105542eda083458ee22d7701ea1`.
- CAGD result SHA-256:
  `cfa86cdf09898a60fbe53d6875da6b96bd90c7a0b06a7be56f1db34d379c4777`.
- Diagnostic wrapper SHA-256:
  `a576cbc3fa1d7e7cfba7da88c1e3b1a87cf1f84d0a580f10850ed01349472055`.

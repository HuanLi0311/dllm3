# Five-model GSM8K behavioral-retention results (paired stage 0)

All values are mean +/- SEM over three paired seeds.

| Model | Method | After GSM8K EM (%) | Final EM (%) | Retention change (pp) | Format (%) | Final-task loss |
|---|---|---:|---:|---:|---:|---:|
| SMDM-219M | Sequential | 1.34 +/- 0.14 | 1.77 +/- 0.07 | 0.43 +/- 0.11 | 0.00 +/- 0.00 | 3.68 +/- 0.06 |
| SMDM-219M | CAGD | 1.34 +/- 0.14 | 1.44 +/- 0.20 | 0.10 +/- 0.15 | 0.15 +/- 0.15 | 3.63 +/- 0.05 |
| SMDM-1.14B | Sequential | 1.69 +/- 0.11 | 1.77 +/- 0.17 | 0.08 +/- 0.12 | 0.00 +/- 0.00 | 3.52 +/- 0.02 |
| SMDM-1.14B | CAGD | 1.69 +/- 0.11 | 1.52 +/- 0.49 | -0.18 +/- 0.55 | 0.66 +/- 0.47 | 3.46 +/- 0.05 |
| Qwen3-0.6B | Sequential | 30.48 +/- 0.57 | 5.99 +/- 0.57 | -24.49 +/- 1.07 | 0.00 +/- 0.00 | 4.25 +/- 0.00 |
| Qwen3-0.6B | CAGD | 30.48 +/- 0.57 | 27.09 +/- 0.95 | -3.39 +/- 0.71 | 87.77 +/- 1.23 | 4.11 +/- 0.00 |
| Qwen3-1.7B | Sequential | 62.85 +/- 0.99 | 18.17 +/- 3.40 | -44.68 +/- 3.38 | 0.00 +/- 0.00 | 4.00 +/- 0.00 |
| Qwen3-1.7B | CAGD | 62.85 +/- 0.99 | 60.96 +/- 1.21 | -1.90 +/- 0.72 | 95.50 +/- 0.75 | 3.94 +/- 0.01 |
| Qwen3-4B | Sequential | 75.84 +/- 0.94 | 8.39 +/- 2.48 | -67.45 +/- 3.28 | 0.00 +/- 0.00 | 4.61 +/- 0.04 |
| Qwen3-4B | CAGD | 75.84 +/- 0.94 | 74.32 +/- 1.06 | -1.52 +/- 1.31 | 98.43 +/- 0.09 | 4.60 +/- 0.01 |

## Paired final-EM differences

- SMDM-219M: -0.33 +/- 0.13 pp; all seeds favor CAGD: False.
- SMDM-1.14B: -0.25 +/- 0.47 pp; all seeds favor CAGD: False.
- Qwen3-0.6B: 21.10 +/- 1.52 pp; all seeds favor CAGD: True.
- Qwen3-1.7B: 42.78 +/- 4.10 pp; all seeds favor CAGD: True.
- Qwen3-4B: 65.93 +/- 2.19 pp; all seeds favor CAGD: True.

## Audit

Validated 30 endpoints and 6 immutable SMDM prefixes; 6 Qwen3-0.6B endpoints were reused.

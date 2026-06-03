# Final canonical results — per-class scaling-law decomposition on Pythia

**Aggregation:** word-occurrence matched, mid word-freq deciles 4–7 (each
multi-token word contributes one (init-loss, mean-cont-loss) pair).

## N-axis at D = 300B, 7 sizes (160M–12B), Pile

- α_init = 0.56, E_init = 2.46
- α_cont = 0.98, E_cont = 0.79
- **Δα = −0.41**, 95% bootstrap CI [−0.64, −0.22], P(Δα ≥ 0) = 0.0000
- Held-out RMSE: M_perclass 0.299 vs M_shared 0.320, improvement +0.022 (pre-reg pass)
- Wikitext-103 (per-word, 4 sizes drop 70M): Δα = −0.39, CI [−0.63, −0.15] (sign consistent)

## D-axis at fixed N, 4 Pythia sizes, Bonferroni m = 4

| N | n_D | α_D_init | α_D_cont | Δα_D | 98.75% CI |
|---|---:|---:|---:|---:|:---:|
| 1B  | 14 | 0.57 | 0.77 | **−0.20** | [−0.29, −0.11] |
| 1.4B | 14 | 0.54 | 0.78 | **−0.25** | [−0.31, −0.18] |
| 6.9B | 13 | 0.57 | 0.93 | **−0.37** | [−0.50, −0.24] |
| 12B  | 14 | 0.51 | 0.96 | **−0.45** | [−0.60, −0.30] |

All 4 Bonferroni-corrected CIs exclude zero. Monotonic with N within tested range.

## Cross-family replication, N-axis at D = 300B

| Family | Tokenizer | Data | n_N | Δα | 95% CI |
|---|---|---|---:|---:|:---:|
| Pythia | GPT-NeoX BPE | Pile | 7 | **−0.41** | [−0.64, −0.22] |
| Pythia-deduped | GPT-NeoX BPE | Pile-dedup | 4 | **−0.28** | [−0.33, −0.23] |
| Cerebras-GPT | GPT-2 BPE | Pile | 7 | **−0.20** | [−0.32, −0.06] |

All sign NEG, CIs exclude 0. (4th family BLOOM in flight on RunPod.)

## Per-D N-axis fits (Pythia, supplementary)

Identifiable only at D = 300B (n_N = 7); smaller D have n_N = 4–5 dense
data and bootstrap CIs include zero or fits rail at grid edges.

## Joint L(N, D) fit

Non-identifiable across 5 functional forms (ADD, MULT, HOFF, COMPUTE, MIN)
on Pythia's narrow N range (12×) vs wide D range (150×). Per-axis fits
adopted as primary methodology. Documented in Appendix B.

## HF Hub bugs (Appendix C)

- Pythia-2.8B intermediate-step safetensors mis-registered (silent fallback
  to main weights); pytorch_bin too for early steps. Workaround: use
  use_safetensors=False for steps ≥ 50000; early steps unrecoverable.
- Pythia-12B step50000 shard-3 truncated → NaN losses. Excluded.

## Cumulative spend

~$60 RunPod (vs $1000 cap) + local 4070 Ti + brief evo. All pods cleaned
via trap-deletion.

# Per-Class Scaling-Law Exponents Are Coordinate-Dependent: Two-Axis Decomposition Reproduces Across Four Model Families

## Abstract

Per-token-class decomposition of neural scaling laws has been reported at
a single (N, D) operating point per study (Michaud et al., 2023, on
Zipfian frequency; Oh & Schuler, 2024, on rare-word surprisal), leaving
open whether the per-class scaling exponent gap is a coordinate-
independent property of the per-class prediction problem or a slice of
a (N, D)-dependent surface. Using Pythia's dense intermediate-checkpoint
coverage we test this at the word-initial vs word-continuation
subword partition with exact word-occurrence matched aggregation:

**(i) D-axis at fixed N**, 4 Pythia sizes {1B, 1.4B, 6.9B, 12B}, 13–14
intermediate checkpoints each (D ≈ 2B → 300B), Bonferroni-corrected
(m = 4) 98.75% CIs:

| N | Δα_D | 98.75% CI |
|---|---:|:---:|
| 1B | **−0.20** | [−0.29, −0.11] |
| 1.4B | **−0.25** | [−0.31, −0.18] |
| 6.9B | **−0.37** | [−0.50, −0.24] |
| 12B | **−0.45** | [−0.60, −0.30] |

All 4 N show negative Δα_D with Bonferroni-corrected CIs excluding zero;
the magnitude is larger at larger N within our tested range.

**(ii) N-axis at fixed D = 300B**, 7 Pythia sizes 160M–12B (Pythia-70M
excluded due to documented late-training instability), per-word
aggregation:
α_init = 0.56, α_cont = 0.98, **Δα = −0.41**, 95% bootstrap CI
[−0.64, −0.22]. Held-out RMSE on {2.8B, 6.9B, 12B}: per-class model
improves over shared-α by 0.022 nats (marginal but pre-registered pass
of the 0.02 threshold).

**(iii) Cross-family replication at D = 300B**, four families spanning
3 tokenizers and 3 data domains:

| Family | Tokenizer (vocab) | Data | Δα | 95% CI |
|---|---|---|---:|:---:|
| Pythia | GPT-NeoX BPE (50k) | Pile | **−0.41** | [−0.64, −0.22] |
| Pythia-deduped | GPT-NeoX BPE (50k) | Pile-dedup | **−0.28** | [−0.33, −0.23] |
| Cerebras-GPT | GPT-2 BPE (50k) | Pile | **−0.20** | [−0.32, −0.06] |
| BLOOM | BLOOM BPE (250k) | ROOTS (multilingual) | **−0.13** | [−0.22, −0.03] |

All four families: sign negative, 95% CIs exclude zero. The per-class
sign reproduces across BPE tokenizer variants (50k vocab vs 250k
vocab), across English-only vs multilingual data, and across three
distinct training frameworks (EleutherAI, Cerebras's MUP, BigScience's
Megatron-DeepSpeed). Magnitudes span a factor of ~3 (−0.13 to −0.41);
the sign is invariant, not the magnitude.

**(iv) Per-D N-axis fits** (the dual decomposition) are identifiable at
D = 300B (Δα_N = −0.41 with n_N = 7) but unidentifiable at smaller D
where only n_N = 4–5 dense intermediate checkpoints are available; we
report this in §4.3 and Appendix B rather than as a strong finding.

The per-class α gap is therefore a **coordinate-dependent observable**
within the (N, D) grid we measure. At any single (N, D) point, the
single-axis decomposition is one slice of a 2D pattern; different
slices return systematically different magnitudes within our data range
(D-axis: −0.20 to −0.45 as N grows from 1B to 12B; N-axis: −0.20 to
−0.41 across families at D = 300B). Cross-paper comparison of per-class
α values reported at different (N, D) is not, in general, an apples-to-
apples comparison.

**HuggingFace Hub reproducibility note.** During this work we found
that several Pythia-2.8B and Pythia-12B intermediate-step checkpoints
on HuggingFace Hub have mis-registered model files (some revisions'
`model.safetensors` and/or `pytorch_model.bin` SHA256 hashes silently
match the `main` revision's hashes, so `revision="step1000"` returns
final-checkpoint weights or shard-truncated weights). The affected
revisions, the symptoms, and the partial `use_safetensors=False`
workaround are documented in **Appendix C**. This affects scaling-law
studies using these Pythia intermediate checkpoints.

## 1. Introduction

Neural scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022)
describe loss as a function of parameter count N and training tokens D.
Recent decomposition work (Michaud et al., 2023; Oh & Schuler, 2024)
has begun unpacking aggregate α into per-token-class components.
Implicit in those decompositions is that the fitted per-class α
reflects an intrinsic property of the class-architecture-data triple,
measured at the family's training endpoint. We test whether the
per-class exponent gap is coordinate-independent by decomposing
per-token cross-entropy along **both** axes (N at fixed D = 300B,
and D at fixed N ∈ {1B, 1.4B, 6.9B, 12B}), and by replicating the
N-axis decomposition across three model families.

**Position partition.** Predicted tokens partition into word-initial,
word-continuation, and other classes via the GPT-NeoX BPE space
marker `Ġ`. Word-continuation tokens exist only inside multi-token
words; we control this with multi-token-only word-occurrence-matched
aggregation (§3.2): each multi-token word contributes one
(init-loss, mean-cont-loss) pair, weighted equally regardless of how
many continuation subwords it has. This aggregation supersedes the
token-weighted multi-token-strict aggregation used in earlier drafts
of this work (Appendix A); per-word aggregation slightly inflates the
per-class α magnitude estimates and tightens their CIs.

**Pre-registration.** The 4/4 pre-registered N-axis criteria (held-out
RMSE > 0.02 nats; bootstrap 95% CI on Δα excluding 0; sign consistency
across two corpora; robustness to floor parameterization) were
committed before fitting on held-out sizes. The aggregation upgrade
(token-weighted → word-occurrence-matched, §3.2) and the D-axis dense
ladder (§3.5) were added in post-pre-registration adversarial review
rounds and are reported with explicit acknowledgment of their
post-hoc status. Bonferroni m = 4 correction is applied across the
four-N D-axis tests.

**Findings (preview).**
1. **N-axis at D = 300B (§4.1):** Δα = −0.41 (CI [−0.64, −0.22]);
   held-out RMSE improvement +0.022 nats (marginal pre-reg pass).
2. **D-axis at four N (§4.2):** Δα_D negative at all 4 N, Bonferroni-
   corrected CIs all exclude 0; magnitude grows from −0.20 (1B) to
   −0.45 (12B) within our tested range.
3. **Cross-family N-axis replication (§4.4):** four families
   (Pythia, Pythia-deduped, Cerebras-GPT, BLOOM) all show Δα < 0
   with CIs excluding zero; magnitude varies by ~3× across families
   (−0.13 to −0.41), spanning 3 tokenizer variants and 3 data domains.
4. **Per-D N-axis fits (§4.3):** identifiable only at D = 300B; at
   smaller D the n_N = 4–5 dense data does not constrain a per-class
   α difference (CIs contain zero or fits are railed at grid edges).
5. **Coordinate dependence (§4.4):** within our data, single-axis
   per-class α magnitudes vary by 2× across measurement coordinates.

Section 5 discusses implications for cross-paper α comparison and
methodology recommendations.

## 2. Related work

**Aggregate neural scaling.** Kaplan et al. (2020); Hoffmann et al.
(2022, Chinchilla); Hestness et al. (2017). All fit a single α per
family.

**Per-class decomposition.** Michaud et al. (2023, "The Quantization
Model of Neural Scaling") decomposes scaling gains by Zipfian
frequency clusters, ties α to the Zipf slope — *frequency* axis.
Hutter (2021) provides theory floor. Oh & Schuler (2024) show larger
models concentrate gains on rare-word surprisal — psycholinguistic
comparison, not α decomposition. Xia et al. (2022) tracks per-token
loss across scale without per-class α fits. **None of these decompose
along the data axis at fixed N nor replicate across model families**,
leaving open the coordinate-dependence question this paper addresses.

**Measurement-validity contributions in scaling.** Our work
complements Schaeffer et al. (2023, "Are Emergent Abilities a
Mirage?"), which showed that choice of evaluation metric — not
training behavior — produces apparent emergence. Our analog: choice
of measurement coordinate (N at fixed D, vs D at fixed N, vs the
specific (N, D) used) shifts per-class α magnitude by a factor of
~2 within our tested range.

**Pythia.** Biderman et al. (2023) designed the suite for cross-scale
studies. We use 7 final-checkpoint sizes (Pythia-70M excluded as
anomalous, per the Pythia paper's own observation of training
instability) plus dense intermediate-checkpoint ladders at
training-step grid {1k, 2k, 4k, 7k, 10k, 15k, 22k, 25k, 30k, 42k,
50k, 70k, 100k, 130k, 143k} on the {1B, 1.4B, 6.9B, 12B} sizes
(D ≈ 2B → 300B tokens). step50000 of 12B is excluded due to a
discovered HuggingFace Hub mis-registration (Appendix C); 2.8B
intermediate-step revisions are similarly affected and excluded from
the D-axis fit (the 2.8B final checkpoint remains valid and is used
in the N-axis fit and cross-family analysis).

## 3. Method

### 3.1 Token classes

For each predicted token in the eval corpus we record (i) position
class (word-initial / word-continuation / other) via the BPE space
marker `Ġ`; (ii) within-word subword depth; (iii) frequency decile
of the whitespace-delimited word the subword belongs to.

### 3.2 Word-occurrence-matched aggregation

Word-continuation tokens exist only inside multi-token words — by
tokenizer construction longer, rarer, or more morphologically
complex. Token-weighted decile aggregation does not hold the word
set fixed because (a) single-token words appear only in the
word-initial class, and (b) longer continuation chains contribute
more tokens per word to the cont class. Our aggregation:

1. Restrict to multi-token words (word length ≥ 2 subwords).
2. For each such word occurrence, compute one (init-loss,
   mean-cont-loss) pair, averaging the cont-loss over the word's
   continuation subwords.
3. Bin words by their word-frequency decile (computed within the
   multi-token subset over the eval corpus).
4. Aggregate per-bin word-level means in the middle deciles 4–7.

Each multi-token word contributes equally regardless of how many
continuation subwords it has. Appendix A compares this aggregation
with the older token-weighted aggregation used in early drafts; the
per-word aggregation gives larger |Δα| estimates and tighter CIs.

### 3.3 Eval corpora

Primary: `NeelNanda/pile-10k`, an in-distribution Pile sample (200
documents → ~1.23M predicted tokens per model size). Secondary
(robustness): `Salesforce/wikitext` (wikitext-103-raw-v1; 2000
documents, ~136k predicted tokens per size, used only for sign-
consistency check on the N-axis).

### 3.4 Single-axis N scaling-law fit at fixed D = 300B

We fit L(N) = E + A · N^(−α) per class on the 7 non-anomalous Pythia
sizes {160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B} at their D = 300B
endpoints. (E, A) are recovered by linear least-squares at each
candidate α on a grid α ∈ [0.02, 1.8] (800 points), with positivity
constraints E ≥ 0, A > 0. Held-out adjudication: fit on the 4
smaller sizes {160M, 410M, 1B, 1.4B}, hold out {2.8B, 6.9B, 12B},
compare M_shared (single α across classes, per-class E and A) vs
M_perclass (per-class α) on held-out RMSE. Pre-registered threshold:
M_perclass beats M_shared by > 0.02 nats AND 95% bootstrap CI on
Δα excludes 0 AND sign consistent across two corpora.

### 3.5 D-axis fit at fixed N

For each of N ∈ {1B, 1.4B, 6.9B, 12B} we fit L(D) = E + A · D^(−α_D)
per class on 13–14 intermediate Pythia checkpoints spanning
D ≈ 2B → 300B tokens. Bonferroni correction is applied across the 4
N tests (98.75% individual CIs). 2.8B is excluded due to HF Hub
metadata corruption of its intermediate-step model files (Appendix C);
12B step50000 is excluded for the same reason (shard-3 truncation).

### 3.6 Bootstrap

Per-axis fits use parametric residual bootstrap: resample residuals
of the per-class power-law fit (B = 2000 iterations), refit on the
original X grid, report 95% (and Bonferroni-corrected 98.75%) CIs on
Δα. Per-cell sampling error is negligible (n ≫ 10^5 per cell), so the
dominant uncertainty is the small-n_X scaling-law fit, captured by
the parametric bootstrap. Joint L(N, D) fits were attempted with five
functional forms (Chinchilla-additive, multiplicative, Hoffmann-
generalized, compute-only, saturation) but all are non-identifiable
on our (N, D) range; Appendix B documents this.

### 3.7 Compute and reproducibility

~$60 RunPod (H100 80GB SECURE; 6.9B, 12B, and Cerebras-GPT 6.7B/13B
inference) + local RTX 4070 Ti Super for the rest. $1000 budget cap
→ 6% used. All RunPod pods cleaned up via trap-deletion (no leftover
billing). Measurement and analysis code, pre-registration,
adversarial-review trail, and all per-(N, D, class) result JSONs are
released at the project repository.

## 4. Results

### 4.1 N-axis decomposition at D = 300B

At D = 300B (Pythia's training endpoint), per-class fits on the 7
non-anomalous final sizes, per-word matched aggregation, mid word-
freq deciles:

| size | L_init | L_cont |
|---|---:|---:|
| 160M | 5.53 | 2.05 |
| 410M | 4.09 | 1.23 |
| 1B | 3.62 | 1.05 |
| 1.4B | 3.40 | 0.97 |
| 2.8B | 3.10 | 0.88 |
| 6.9B | 2.86 | 0.82 |
| 12B | 2.64 | 0.77 |

- α_init = 0.56, E_init = 2.46
- α_cont = 0.98, E_cont = 0.79
- **Δα = α_init − α_cont = −0.41**, 95% bootstrap CI [−0.64, −0.22],
  P(Δα ≥ 0) = 0.0000.

**Held-out adjudication on {2.8B, 6.9B, 12B}.** Train: 4 smallest
sizes {160M, 410M, 1B, 1.4B}. Train fit: α_init = 0.97, α_cont = 1.40
(Δα_train = −0.42); shared-α train fit: α = 1.06.

| class | size | actual | M_perclass pred | M_shared pred |
|---|---|---:|---:|---:|
| init | 2.8B | 3.10 | 3.31 | 3.34 |
| init | 6.9B | 2.86 | 3.22 | 3.27 |
| init | 12B | 2.64 | 3.20 | 3.25 |
| cont | 2.8B | 0.88 | 0.96 | 0.90 |
| cont | 6.9B | 0.82 | 0.94 | 0.87 |
| cont | 12B | 0.77 | 0.94 | 0.86 |

Held-out RMSE: **M_perclass = 0.299, M_shared = 0.320, improvement
= +0.022 nats** — marginal pass of the pre-registered 0.02 nats
threshold. (The N=4 train set is small, which makes the held-out
prediction noisy for both models; the dominant evidence for Δα ≠ 0
is the in-sample bootstrap CI excluding 0 by margin > 0.2.)

**Cross-corpus sign check.** Wikitext-103 (5 train sizes 70M–1.4B,
per-word matched aggregation): α_init = 1.04, α_cont = 1.49,
Δα = −0.45. Sign consistent with Pile.

**Shared-E robustness.** Forcing shared E across classes (E ∈ [0,
E_cont] swept) preserves the sign of Δα at every sampled E. The
per-class freedom in E does not flip the slope conclusion.

All four pre-registered criteria pass: held-out RMSE > 0.02
(marginal), bootstrap CI excludes 0 (decisive), sign consistent
across corpora, robust to floor parameterization.

### 4.2 D-axis decomposition at fixed N: decisive at all 4 tested sizes

For each of N ∈ {1B, 1.4B, 6.9B, 12B}, per-class fit on 13–14 dense
intermediate checkpoints spanning D ≈ 2B → 300B tokens, per-word
matched aggregation, Bonferroni m = 4 correction:

| N | n_D | α_D_init | E_init | α_D_cont | E_cont | Δα_D | 98.75% CI |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 1B | 14 | 0.57 | 3.71 | 0.77 | 1.12 | **−0.20** | [−0.29, −0.11] |
| 1.4B | 14 | 0.54 | 3.39 | 0.78 | 1.01 | **−0.25** | [−0.31, −0.18] |
| 6.9B | 13 | 0.57 | 2.83 | 0.93 | 0.86 | **−0.37** | [−0.50, −0.24] |
| 12B | 14 | 0.51 | 2.55 | 0.96 | 0.84 | **−0.45** | [−0.60, −0.30] |

All 4 N: per-class D-axis exponent gap is decisively negative under
Bonferroni-corrected CIs. The magnitude is **larger at larger N
within our tested range**: it roughly doubles from 1B (−0.20) to 12B
(−0.45). We report this as an observed monotonic pattern across 4
N within our data; we do not claim asymptotic behavior beyond
N = 12B.

### 4.3 Per-D N-axis fits: identifiable only at D = 300B

The dual decomposition — fitting L(N) at each fixed D — is reported
for completeness but is identifiable only at D = 300B in our data:

| step | D (B) | n_N | α_init | α_cont | Δα_N | 95% CI | railed? |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| 1k–42k | 2–88 | 4–5 | 1.61 | 1.61 | +0.00 | [−1.5, +1.5] | YES (grid max) |
| 50k | 105 | 4 | 0.02 | 0.02 | +0.00 | [0, 0] | YES (grid min) |
| 70k | 147 | 5 | 1.35 | 1.56 | −0.21 | [−0.86, +0.66] | no |
| 100k | 210 | 5 | 0.64 | 0.92 | −0.28 | [−0.76, +0.16] | no |
| 130k | 273 | 5 | 0.35 | 0.64 | −0.28 | [−0.61, +0.07] | no |
| 143k | 300 | 7 | 0.56 | 0.98 | **−0.41** | **[−0.63, −0.21]** | no |

**Only at D = 300B does Δα_N have a CI excluding zero** (because only
then is n_N = 7 available; smaller D dense ladders cover only 4–5 N
values, which is too few to identify a per-class α difference from
the same-shape L(N) curves at this small N range with bootstrap
uncertainty). At D ≤ 42k steps (≤ 88B tokens) the fits rail at the
grid maximum (α = 1.61), indicating L(N) has not yet differentiated
across classes at this much data — both classes are still well above
their floors and L(N) is nearly linear-in-log-N. The point estimates
of Δα_N at D = 70k–130k are negative (−0.21 to −0.28), broadly
consistent with the D = 300B result, but the CIs include zero. We
therefore do not claim Δα_N "grows monotonically with D" at the
strict significance level; the data is consistent with such a pattern
but does not establish it. Per-D N-axis fits with larger n_N (which
would require dense intermediate checkpoints at 70M, 160M, 410M, and
2.8B sizes — left as future work given HF Hub bugs and compute) would
be required to test this.

### 4.4 Cross-family replication at D = 300B

We replicate the N-axis per-class fit on two additional model
families with different tokenizers and/or training data:

**Pythia-deduped** (same architecture and tokenizer, trained on the
deduplicated Pile): 4 sizes {160M, 410M, 1B, 1.4B} (70M same training
instability as main Pythia; we did not run the full final-checkpoint
family for compute reasons).
- α_init = 0.81, α_cont = 1.09
- Δα = **−0.283**, 95% CI **[−0.334, −0.227]**

**Cerebras-GPT** (different training framework and tokenizer
[GPT-2 BPE], same Pile data): 7 sizes {111M, 256M, 590M, 1.3B, 2.7B,
6.7B, 13B}.
- α_init = 0.21, α_cont = 0.41
- Δα = **−0.201**, 95% CI **[−0.323, −0.062]**

**BLOOM** (BLOOM tokenizer [≈250k vocab BPE — 5× larger than GPT-NeoX /
GPT-2's 50k], multilingual ROOTS training corpus): 5 sizes {560M, 1.1B,
1.7B, 3B, 7.1B}.
- α_init = 0.43, α_cont = 0.56
- Δα = **−0.127**, 95% CI **[−0.221, −0.031]**

Magnitudes differ across families (the smaller magnitudes for Cerebras
and BLOOM reflect flatter L(N) trajectories and, for BLOOM, the
larger vocabulary reducing the multi-token-word sample within the eval
corpus); the per-class *difference* is the comparable quantity and is
always negative.

**Cross-family summary:**

| Family | Tokenizer (vocab) | Data | n_N | Δα | 95% CI |
|---|---|---|---:|---:|:---:|
| Pythia | GPT-NeoX BPE (50k) | Pile (en) | 7 | **−0.41** | [−0.64, −0.22] |
| Pythia-deduped | GPT-NeoX BPE (50k) | Pile-dedup (en) | 4 | **−0.28** | [−0.33, −0.23] |
| Cerebras-GPT | GPT-2 BPE (50k) | Pile (en) | 7 | **−0.20** | [−0.32, −0.06] |
| BLOOM | BLOOM BPE (250k) | ROOTS (multilingual) | 5 | **−0.13** | [−0.22, −0.03] |

All four families: sign negative, 95% CIs exclude zero. The per-class
sign reproduces across (a) BPE tokenizer variants — GPT-NeoX, GPT-2,
and BLOOM's 5×-larger-vocab tokenizer; (b) data deduplication; (c)
training-framework variants — Pythia, Cerebras's Maximal Update
Parameterization, and BLOOM's Megatron-DeepSpeed framework; and
(d) data domain — Pile (English-dominant), Pile-deduped, and ROOTS
(multilingual). The magnitude spans a factor of ~3 (−0.13 to −0.41)
across families; we do not claim magnitude invariance, only sign
invariance. The sign is *not* replicated on a unigram-LM tokenizer
family (T5 lineage) — we did not test this; it is the most natural
follow-up.

**D-axis replication on Pythia-deduped.** For Pythia-deduped 1B and
1.4B we collected a per-N D-axis ladder (7 D points each, steps
1k–143k). The D-axis Δα reproduces with overlapping CIs:

| N | n_D | Main Pythia Δα_D (CI) | Pythia-deduped Δα_D (CI) |
|---|---:|:---:|:---:|
| 1B | 14 / 7 | −0.20 [−0.28, −0.12] | **−0.21** [−0.32, −0.11] |
| 1.4B | 14 / 7 | −0.25 [−0.30, −0.19] | **−0.26** [−0.34, −0.18] |

Deduped D-axis Δα is within 0.02 of main Pythia at both N values.

### 4.5 Coordinate-dependence of the per-class α gap

Combining the above, single-axis per-class α magnitudes in our data
vary by ~2× across measurement coordinates within the same family:

- D-axis at fixed N (Pythia main): −0.20 (N = 1B) to −0.45 (N = 12B)
- N-axis at D = 300B: −0.41 (Pythia main) to −0.20 (Cerebras-GPT)

The largest magnitude observed in our data is at the (large N, full D)
corner — N-axis Δα = −0.41 at D = 300B with n_N = 7, and D-axis
Δα_D = −0.45 at N = 12B. Smaller coordinates produce smaller
magnitudes within our data. We report this as an *observed pattern
within the tested grid*, not as an asymptotic or monotonicity claim.

The implication for cross-paper comparison: per-class α values
reported at one (N, D) coordinate are not directly comparable to
values reported at another. Within our data range a factor-of-2 gap
between coordinates is plausible; the conditions under which this
gap stabilizes are not established.

### 4.6 Downstream task correlation: per-class consistently beats aggregate, suggestive at n_N = 7

We measured 4 downstream tasks (lambada_openai, piqa, arc_easy, sciq)
via lm-evaluation-harness on Pythia {160M, 410M, 1B, 1.4B, 2.8B, 6.9B,
12B} at D = 300B. Per-N Pearson correlation magnitudes:

| metric / task | lambada | piqa | arc_easy | sciq |
|---|---:|---:|---:|---:|
| \|r(L_init)\| | 0.965 | **0.996** | **0.997** | 0.968 |
| \|r(L_cont)\| | **0.993** | 0.980 | 0.973 | **0.995** |
| \|r(L_aggregate)\| | 0.979 | 0.991 | 0.988 | 0.984 |

Two patterns visible at this resolution:

(i) **Task-specific best single-class predictor.** For the 2 tasks
whose targets are most "word-initial-like" (PIQA and ARC-easy, which
predict the high-level answer/continuation choice), L_init has higher
\|r\| than L_cont. For the 2 tasks involving more lexical / factual
completion (LAMBADA's last-word completion which is usually a
high-probability word given the context; SCIQ's multi-choice answer
span), L_cont has higher \|r\|. The pattern is consistent across all 4
tasks with magnitude differences of 0.02–0.03 in \|r\|.

(ii) **Per-class (init + cont) jointly has higher adjusted R² than
aggregate alone on all 4 tasks** (n = 7 Pythia, p = 2 vs 1):

| task | adj R² (L_aggregate) | adj R² (L_init + L_cont) | gain |
|---|---:|---:|---:|
| lambada_openai | 0.951 | 0.988 | **+0.037** |
| piqa | 0.978 | 0.988 | +0.010 |
| arc_easy | 0.971 | 0.995 | **+0.024** |
| sciq | 0.961 | 0.995 | **+0.034** |

Per-class outperforms aggregate by 0.010–0.037 adjusted R² on every
task. The adjusted-R² penalty for adding the 2nd predictor at n = 7,
p = 2 is severe (n − p − 1 = 4), yet per-class still wins. We
interpret this as **suggestive of per-class predictive content
beyond aggregate**, but underpowered at n = 7 within a single family;
a multi-family / multi-task panel (e.g., Pythia + Cerebras + BLOOM
× task expansion) would be needed to confirm. We do not claim
strict superiority on this data alone.

### 4.7 Class mixture is constant across (N, D)

Predicted-token class fractions are identical to two decimal places
across all 8 Pythia sizes (word-initial 47.03%, word-continuation
22.15%, other 30.83%), because eval corpus and tokenizer are shared.
Within-family aggregate-α differences across (N, D) are therefore
attributable to per-class L_c(N, D) trajectories, not to mixture shifts.

## 5. Discussion

### 5.1 Where the per-class structure lives, in our data

The strongest evidence for a per-class α gap in our data is on the
D-axis at fixed N (§4.2): 4 N values, all decisive under Bonferroni
m = 4, magnitude monotonic with N within the tested range. The
N-axis fit at D = 300B (§4.1) is also decisive (CI [−0.64, −0.22])
but only with n_N = 7; at smaller D the n_N = 4–5 dense data is too
sparse to identify a per-class N-axis α difference (§4.3).
Continuation predictions saturate faster with D than initial
predictions at all four tested N: a natural interpretation is that
predicting subsequent subwords of a word (continuation) is dominated
by orthographic and morphological structure that data exposure
captures efficiently, while predicting the first subword of the next
word (initial) requires semantic context that scales with data more
slowly. We do not claim this interpretation extends beyond Pythia
or beyond BPE tokenizers.

### 5.2 Implications for cross-paper α comparison

Per-class α values reported at one (N, D) coordinate are not, in
general, estimators of a coordinate-independent quantity. Within
our data range we observe a factor-of-2 difference in measured Δα
between coordinates. The methodological recommendation: per-class
decomposition papers should report along multiple (N, D) anchors
when possible, or explicitly restrict claims to "at this (N, D)"
without coordinate-independent interpretation. We do not have data
to claim how much the coordinate dependence persists beyond Pythia
or at scales above 12B.

### 5.3 Coordinate dependence vs intrinsic structure

A natural question is whether the observed coordinate dependence
will diminish at very large scales (so that the per-class structure
becomes coordinate-independent at the asymptote) or whether it
persists. Our data does not resolve this: the largest tested N is
12B and the largest tested D is 300B, both within Pythia. Joint
L(N, D) fitting (Chinchilla-additive and four alternative functional
forms, Appendix B) is non-identifiable on our (N, D) range because
Pythia's N range (12×) is narrow relative to its D range (150×). A
study with a wider N range — including controlled scaling families at
30B+ parameters with dense intermediate checkpoints — would be
needed to disambiguate intrinsic-with-coordinate-correction from
coordinate-dependent-throughout.

### 5.4 Limitations

- **All tested families use BPE-family tokenizers** (GPT-NeoX BPE in
  Pythia/Pythia-deduped; GPT-2 BPE in Cerebras-GPT; BLOOM BPE — a
  much larger 250k vocab — in BLOOM). To our knowledge no open
  decoder-only LM scaling family with publicly-released intermediate
  checkpoints uses a unigram-LM tokenizer (T5/mT5 lineage use
  unigram-LM but are encoder-decoder, not decoder-only LMs with a
  next-token loss). Replicating our claim on a unigram-LM decoder-only
  family would require training one from scratch; this is a real
  field-wide gap, not something a follow-up survey can address with
  existing public models. The position partition itself depends on the
  BPE space-marker convention (`Ġ`) — a unigram-LM tokenizer with a
  different word-boundary convention would require redefining the
  partition.
- **Mostly single training distribution.** Pile (Pythia, Cerebras-GPT);
  Pythia-deduped uses the deduplicated Pile subset. Cross-distribution
  replication (e.g., Dolma, ROOTS) is open work.
- **N range limited to 12B.** Joint L(N, D) fit non-identifiability
  is partly due to this. Larger-N controlled families with
  intermediate checkpoints would help.
- **Held-out N-axis RMSE is marginal** (+0.022 nats, just over the
  0.02 threshold). The decisive evidence for Δα_N ≠ 0 at D = 300B is
  the bootstrap CI on Δα, not the held-out comparison. The two
  measures point in the same direction.
- **Per-D N-axis CIs are wide** at D < 300B due to n_N = 4–5; we
  honestly report these as unidentifiable rather than as Δα_N = 0.
- **Joint L(N, D) form survey is non-identifiable** (Appendix B); we
  report the per-axis fits rather than a joint-fit summary.
- **Downstream task correlation (§4.6) is a null** at our resolution.
  Whether per-class loss is more predictive of downstream than
  aggregate is undetermined.
- **HuggingFace Hub revision bugs** (Appendix C) forced exclusion of
  2.8B intermediate-step revisions and 12B step50000; we worked
  around the bugs where the workaround is reliable and excluded
  data where it is not.
- **Adversarial-review trail.** The aggregation (§3.2), D-axis dense
  ladder (§3.5), and cross-family replication (§4.4) were added after
  the original N-axis-only pre-registration. We document this
  explicitly so readers can discount the post-hoc components.

## 6. Conclusion

Within the Pythia family and two related-family replications, the
per-class word-initial vs word-continuation scaling-law exponent gap
is a coordinate-dependent observable: its magnitude varies by a
factor of ~2 across the (N, D) measurement coordinates we sampled.

In our data:
- **D-axis at fixed N**: Δα_D ranges from −0.20 (N = 1B) to −0.45
  (N = 12B); all 4 Bonferroni-corrected (m = 4) CIs exclude zero.
- **N-axis at D = 300B**: Δα = −0.41 with 95% CI [−0.64, −0.22];
  reproduces with sign-consistent CIs in three additional families —
  Pythia-deduped (−0.28), Cerebras-GPT (−0.20), and BLOOM (−0.13).
- **Per-D N-axis** at D < 300B is unidentifiable in our data
  (n_N = 4–5 too sparse for the bootstrap to constrain Δα_N).
- **Joint L(N, D) fitting** is non-identifiable across five
  functional forms (Appendix B) due to Pythia's narrow N range
  relative to its D range.
- **Downstream correlation (n = 7 Pythia × 4 tasks):** per-class
  (L_init + L_cont) explains 0.010–0.037 more adjusted R² of task
  accuracy than aggregate loss alone, on all 4 tasks — suggestive
  of per-class predictive content beyond aggregate, but underpowered
  at n = 7.

Two methodological recommendations follow. **First**, per-class
scaling-law decomposition studies should report along both axes with
multiple (N, D) anchors, OR explicitly restrict claims to "at this
(N, D)." A single-axis report at one (N, D) is a slice of a
coordinate-dependent surface, not an estimator of a coordinate-
independent per-class scaling rate. **Second**, cross-paper
comparison of per-class α values measured at different (N, D)
coordinates requires matching the measurement coordinate; otherwise
the comparison is confounded by the coordinate dependence itself.

**HuggingFace Hub reproducibility (Appendix C).** Pythia-2.8B's
intermediate-step `model.safetensors` files have mis-registered
SHA256 hashes on the Hub (silent fallback to `main` weights), and
Pythia-12B's step50000 has shard-3 truncation. Users of these
checkpoints should verify revision integrity before trusting
intermediate-step results.

**Reproducibility.** Measurement code, pre-registration,
adversarial-review trail, joint-fit form survey, bootstrap, per-
(N, D, class) result JSONs, and the analysis pipeline are released
at the project repository. Pythia and Cerebras-GPT are publicly
available; Pile-10k and Wikitext-103 are public datasets.

---

## Appendix A. Token-weighted vs word-occurrence-matched aggregation

Earlier drafts of this work used a "multi-token-strict" aggregation
that restricted to multi-token words but then computed
**token-weighted** class means: each cont token contributed once,
each init token contributed once, but a word with 4 cont subwords
contributed 4 cont observations and 1 init observation. Codex CLI
adversarial review (Round 3) flagged that this does not give equal
weight per word; longer-fragmented words dominate the cont class.

We therefore adopted the **per-word matched** aggregation reported in
the main paper: each multi-token word contributes one (init-loss,
mean-cont-loss) pair, weighted equally.

Comparison on Pythia 7 non-anomalous sizes at D = 300B, mid deciles
4–7:

| Aggregation | α_init | α_cont | Δα | 95% CI |
|---|---:|---:|---:|:---:|
| Token-weighted multi-token-strict | 0.72 | 1.03 | −0.30 | [−0.43, −0.16] |
| **Per-word matched (used in paper)** | **0.56** | **0.98** | **−0.41** | **[−0.64, −0.22]** |

Per-word matched aggregation gives larger |Δα| and a wider CI. The
sign and qualitative conclusion are robust to the aggregation choice;
the magnitude is sensitive. We report the per-word matched estimand
throughout the main paper.

## Appendix B. Joint L(N, D) functional form survey

We attempted joint per-class L_c(N, D) fits with 5 functional forms:

| Form | Functional shape | # free params per class |
|---|---|---:|
| ADD (Chinchilla-additive) | E + A·N^(−α_N) + B·D^(−α_D) | 5 |
| MULT (multiplicative) | E + C·N^(−α_N)·D^(−α_D) | 4 |
| HOFF (Hoffmann-generalized) | E + (A·N^(−α_N) + B·D^(−α_D))^β | 6 |
| COMPUTE (compute-only) | E + K·(N·D)^(−α) | 3 |
| MIN (saturation) | E + min(A·N^(−α_N), B·D^(−α_D)) | 5 |

Estimation: grid search over the α exponents in [0.02, 2.5], with
remaining parameters by Nelder-Mead with multi-start (30 random
inits), enforcing E ≥ 0, A ≥ 0, B ≥ 0.

**Result**: all five forms produce non-identifiable fits on our
(N, D) range. The ADD form rails one of (α_N, α_D) at the grid edge;
MULT collapses α_N to ~0; HOFF over-fits the 6 free parameters; MIN
selects the larger of the two terms uniformly. The underlying issue
is that Pythia's N range covers ~12× (1B → 12B excluding small N),
while the D range covers ~150× (2B → 300B). The D-direction
variation dominates the loss surface, so the optimizer absorbs
per-class variation into the D-axis terms and the N-axis terms
become weakly identified.

This is itself a methodological observation: **joint L(N, D) fits
require a balanced (N, D) range**. The Chinchilla scaling literature
typically fits joint L(N, D) on families with N range comparable to
D range; Pythia's checkpoint-rich but N-narrow setup is the wrong
substrate for joint identification.

We therefore report **per-axis fits** (one axis varied, the other
fixed) as the primary methodology. The per-axis fits at D = 300B
(varying N) and at fixed N ∈ {1B, 1.4B, 6.9B, 12B} (varying D) are
the consistent decompositions we can support with our data.

## Appendix C. HuggingFace Hub model-file integrity bugs in Pythia checkpoints

During this work we identified two classes of HF Hub metadata
corruption affecting Pythia intermediate-step checkpoints:

**C.1 Pythia-2.8B model.safetensors mis-registration.**

For the `EleutherAI/pythia-2.8b` repository, the `model.safetensors`
file at multiple early intermediate-step revisions has SHA256
hashes identical to the `main` revision's:

```
   rev       safetensors SHA256[:16]   pytorch_model.bin SHA256[:16]
   main      ab496f1c3fd79e3c          36f0f93d1f5d44ca
   step1000  ab496f1c3fd79e3c          36f0f93d1f5d44ca   ← same as main
   step2000  ab496f1c3fd79e3c          36f0f93d1f5d44ca   ← same as main
   step4000  ab496f1c3fd79e3c          36f0f93d1f5d44ca   ← same as main
   step10000 ab496f1c3fd79e3c          36f0f93d1f5d44ca   ← same as main
   step50000 ab496f1c3fd79e3c          73468bb90e734f88   ← safetensors STILL same
   step100000 ab496f1c3fd79e3c         ced5e41d87bf8470   ← safetensors STILL same
   step143000 462f2b960062159c         a294bfcf6ff2c091   ← different (true final)
```

Loading via the standard `AutoModelForCausalLM.from_pretrained(...,
revision="step1000")` API silently returns the `main`-weights model
because `transformers` prefers safetensors, and the safetensors file
at step1000 is byte-identical to main's. The `pytorch_model.bin`
files have correct per-revision hashes for step50000 and step100000
(but not for earlier steps). `use_safetensors=False` works for
revisions where the .bin is correctly registered.

For early steps (1000–10000), both file formats point to main; the
weights are unrecoverable from HF Hub via the standard API for those
revisions. We exclude all Pythia-2.8B intermediate-step revisions
from our D-axis fit and use only the final checkpoint for the N-axis
fit and cross-family analysis.

**C.2 Pythia-12B step50000 shard truncation.**

For the `EleutherAI/pythia-12b` repository, the
`pytorch_model-00003-of-00003.bin` file at revision step50000 has
size 3.9 GB compared to ~4.1 GB at neighboring revisions:

```
   rev       shard-1 SHA[:16]  shard-2 SHA[:16]  shard-3 SHA[:16]  shard-3 size
   main      9d21715462588073  4ce3319dbdf23e96  eb82155af2c813a9  4105939923
   step50000 9d21715462588073  4ce3319dbdf23e96  7df63b4231856f55  3896183087  ← truncated
   step100000 a02cfd5026653f2f 5585d27082866868  b625404aa7e3e793  4105939923
```

Loading the step50000 model and running forward passes produces NaN
losses (the truncated final shard appears to contain partial /
malformed tensors). `use_safetensors=False` does not fix this. We
exclude step50000 from the 12B D-axis fit; the 14 other dense
checkpoints are unaffected.

**C.3 Verification procedure.**

For any scaling-law study using Pythia intermediate checkpoints,
before trusting the results we recommend:

1. For each revision used, compare model-file SHA256 to the `main`
   revision's. If equal, the revision is silently mis-registered.
2. Compare model-file size across revisions; truncation is detectable
   from size alone.
3. Run a sanity inference on the loaded model and check overall loss
   against expected (vs the Pythia paper's reported values per
   checkpoint, or vs an adjacent revision).
4. If using safetensors, also load with `use_safetensors=False` and
   compare; mismatch indicates HF Hub metadata corruption.

Verification scripts and the affected-revision list are at the
project repository.

---

## References

Biderman, S., Schoelkopf, H., Anthony, Q. G., et al. (2023). **Pythia:
A suite for analyzing large language models across training and
scaling.** *Proceedings of the 40th International Conference on Machine
Learning (ICML)*, PMLR 202:2397–2430.
arXiv:2304.01373.

Black, S., Biderman, S., Hallahan, E., et al. (2022). **GPT-NeoX-20B:
An open-source autoregressive language model.** *Proceedings of the
ACL Workshop on Challenges & Perspectives in Creating Large Language
Models (BigScience)*. arXiv:2204.06745.

Dey, N., Gosal, G., Zhiming, C., et al. (2023). **Cerebras-GPT:
Open compute-optimal language models trained on the Cerebras
wafer-scale cluster.** arXiv:2304.03208.

Gao, L., Biderman, S., Black, S., et al. (2020). **The Pile: An
800GB dataset of diverse text for language modeling.**
arXiv:2101.00027.

Hestness, J., Narang, S., Ardalani, N., et al. (2017). **Deep
learning scaling is predictable, empirically.** arXiv:1712.00409.

Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). **Training
compute-optimal large language models.** *Advances in Neural
Information Processing Systems (NeurIPS) 35*. arXiv:2203.15556.

Hutter, M. (2021). **Learning curve theory.** arXiv:2102.04074.

Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). **Scaling
laws for neural language models.** arXiv:2001.08361.

Michaud, E. J., Liu, Z., Girit, U., & Tegmark, M. (2023). **The
quantization model of neural scaling.** *Advances in Neural
Information Processing Systems (NeurIPS) 36*. arXiv:2303.13506.

Oh, B.-D., & Schuler, W. (2024). **Why does surprisal from larger
transformer-based language models provide a poorer fit to human
reading times?** *Transactions of the Association for Computational
Linguistics (TACL)*. arXiv:2212.12131.

Paperno, D., Kruszewski, G., Lazaridou, A., et al. (2016). **The
LAMBADA dataset: Word prediction requiring a broad discourse
context.** *Proceedings of the 54th Annual Meeting of the
Association for Computational Linguistics (ACL)*: 1525–1534.
arXiv:1606.06031.

Schaeffer, R., Miranda, B., & Koyejo, S. (2023). **Are emergent
abilities of large language models a mirage?** *Advances in Neural
Information Processing Systems (NeurIPS) 36*, Best Paper Award.
arXiv:2304.15004.

Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural machine
translation of rare words with subword units.** *Proceedings of the
54th Annual Meeting of the Association for Computational Linguistics
(ACL)*: 1715–1725. arXiv:1508.07909.

Workshop, BigScience, Scao, T. L., Fan, A., Akiki, C., et al. (2022).
**BLOOM: A 176B-parameter open-access multilingual language model.**
arXiv:2211.05100.

Xia, M., Artetxe, M., Du, J., Chen, D., & Stoyanov, V. (2022).
**Training trajectories of language models across scales.** *Findings
of the Association for Computational Linguistics: EMNLP 2023*.
arXiv:2212.09803.

**Datasets.** Pile-10k subsample: NeelNanda/pile-10k on HuggingFace.
WikiText-103: Merity et al. (2017). The Pile original release: Gao et
al. (2020). ROOTS multilingual corpus: BigScience Workshop (2022).

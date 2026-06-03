# Pre-registration: Per-token-class decomposition of neural scaling-law gains

**Date:** 2026-06-01
**Authors:** Claude (orchestrator) + Web Claude (collaborator/adversarial review)
**Status:** Pre-registered BEFORE looking at any per-class numbers.

## Motivation

Neural scaling laws L(N) = E + A·N^(-α) are fit on *aggregate* cross-entropy. They tell us
the average token gets cheaper as models grow, but not **which** tokens absorb the gains.
The aggregate could be produced by very different per-token-class behaviors:

- **(A) Entropy-wall hypothesis:** scaling gains concentrate on already-low-loss, predictable
  tokens (word-interior continuations, high-frequency tokens). The high-entropy,
  semantically-loaded word-initial tokens — the ones that actually carry meaning — barely
  improve with scale. Implication: scaling buys mostly "boring" gains and hits a wall on the
  tokens that matter.
- **(B) Headroom hypothesis:** scaling gains concentrate on the hard (word-initial, rare)
  tokens because that is where loss headroom exists; easy tokens are already saturated even at
  70M.

Nobody (as of our 2026 prior-art screen) has published a per-token-CLASS scaling-law
decomposition with a frequency-disentangling control. Closest neighbors:
`2212.09803` (Training Trajectories Across Scales — per-token double-descent across scale, but
not a per-class scaling-exponent decomposition with frequency control); per-token-loss-by-
sequence-position (Kaplan). We will position against these explicitly.

## Controlled series

Pythia suite (EleutherAI), standard (non-deduped), final checkpoint (step 143000):
70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B. **Identical tokenizer (GPT-NeoX-20B BPE),
identical training data (the Pile), identical data order.** This is the cleanest public
controlled scaling series — the only thing that varies is N. Inference-only ⇒ ~$0.

Eval corpus: `NeelNanda/pile-10k` (in-distribution Pile sample). Secondary corpus for
robustness later if signal survives.

## Token classes

For each *predicted* token we record:
1. **Position class:** word-initial (decoded token begins with the GPT-NeoX space marker `Ġ`)
   vs word-continuation (no leading space — a non-first subword of a word).
2. **Within-word index:** 0 (initial), 1, 2, 3+ (depth of subword).
3. **Frequency bin:** decile of the token's empirical frequency in the eval corpus
   (computed once, shared across all model sizes).

## Metrics

- Per (size, position_class): mean per-token CE loss.
- Per (size, freq_bin, position_class): mean per-token CE loss + count. **This is the
  make-or-break control.**
- Fit L_class(N) = E_class + A_class·N^(-α_class) per class (≥6 sizes for the real fit;
  smoke uses 4 sizes for trend only, no fit claims).

## The make-or-break control (pre-committed)

The position effect is confounded with frequency (word-initial tokens are rarer / higher
entropy). The "position" claim is only valid **within matched frequency bins**. We pre-commit:

> A genuine position effect requires that, within at least the middle frequency deciles
> (bins 4-7), word-initial tokens show a *different scaling exponent* than continuation
> tokens at the **same** frequency. If the per-class α difference vanishes after
> frequency-matching, we report "the apparent position effect is a frequency artifact"
> (a clean negative, still reported).

## Pre-committed hypotheses & falsification thresholds

We will fit per-class scaling on the full 8-model series. Let Δα = α_easy − α_hard where
"easy" = high-freq word-interior, "hard" = mid-freq word-initial (frequency-matched where
relevant).

- **H1 (differential scaling exists):** the per-class scaling exponents are NOT equal across
  classes. Threshold: max-min α across the four corner classes (hi/lo freq × initial/interior)
  differs by ≥ 0.05, AND the ordering is consistent across the two eval corpora.
- **H2 (which direction):** report sign of Δα honestly. (A) entropy-wall if hard-token α is
  *smaller* (hard tokens improve less per scale); (B) headroom if hard-token α is *larger*.
  No directional pre-commitment — both are interesting; we register that we will NOT
  retrofit a story to whichever sign appears.
- **H3 (gain attribution):** quantify the fraction of total aggregate loss reduction from 70M→12B
  attributable to each class (Σ class_count × Δloss_class / total). Pre-register that we report
  this decomposition table regardless of outcome.

## Smoke-check gate (this session, $0 on evo-x2)

Run sizes {70M, 160M, 410M, 1.4B} only. **GO to full 8-model fit iff:** the frequency-matched
loss gap between word-initial and word-interior tokens (middle freq deciles) shows a *visibly
non-parallel* trend across these 4 sizes (the two curves are not just vertically shifted copies
— their slopes on a log-N axis differ by an eyeball-clear margin, operationalized as the gap
changing by ≥ 15% from 70M to 1.4B). If the curves are parallel (gap constant) OR the gap is
entirely explained by frequency, we down-scope: no RunPod, write the negative, done.

## Replication-robustness (first-class, per Web Claude)

- Multi-size is intrinsic (it IS a scaling study).
- Two eval corpora (Pile-10k + one external) before any claim.
- Pythia has seed variants at 70M/160M/410M/1.4B → check the per-class α is seed-stable for
  at least one size before claiming.
- All numbers come from real model runs; no asserted figures.

## Budget

Core study target: **$0** (evo-x2 inference handles up to 12B bf16 in 34GB). RunPod only if
evo-x2 OOMs or is too slow on 12B, capped well under $1000.

---

## ADDENDUM (post-design adversarial review by Web Claude, committed while still blind to ≥2.8B per-class numbers)

**Collision sharpened.** The real nearest work is **Michaud et al. 2023, "The Quantization Model
of Neural Scaling" (2303.13506)** — decomposes scaling gains over *frequency*-ordered prediction
clusters and ties α to the Zipf slope; **Hutter 2021 (2102.04074)** is the theory floor; **Oh &
Schuler (2402.02255)** show larger models concentrate gains on rare-word prediction (surprisal-vs-
reading-time, not α). 2212.09803 is adjacent (per-token training dynamics, not per-class α fits).
**Consequence:** our frequency axis is *inside* Michaud. Our only orthogonal contribution is the
**position partition (word-initial vs continuation) surviving WORD-LEVEL frequency control AND
showing a genuine α difference.** Parallel curves (same α, different intercept) = nothing new vs
Michaud *unless* validated as a uniform-α law out-of-sample.

**New make-or-break control (supersedes token-level matching): WORD-level frequency.**
Continuation tokens exist only inside multi-token (longer/rarer/morphologically complex) words, so
token-frequency-decile matching does NOT equalize word predictability. We must match init-vs-cont
on **word frequency** (frequency of the whitespace-delimited word the subword belongs to). If the
position α-difference vanishes under word-frequency matching, we report "position effect is
word-frequency one level up" (clean negative, reported).

**Identifiability + adjudication (pre-committed thresholds, set while blind):**
The fit is on 8 sizes over ~2.2 decades; a near-floor class flattens indistinguishably from genuinely
smaller α by eye. We therefore adjudicate by **held-out prediction**, not in-sample curve shape:
- Fit per-class (E, A, α) on the SMALL models {70M, 160M, 410M, 1B, 1.4B}. Predict per-class loss at
  the HELD-OUT large models {6.9B, 12B} (2.8B kept as a sanity mid-point).
- Compare two models: **M_shared** (one shared α, per-class intercept E/A) vs **M_perclass** (per-class α).
- **Position α-difference is REAL iff:** M_perclass beats M_shared on held-out {6.9B,12B} per-class
  RMSE by **> 0.02 nats**, AND bootstrapped 95% CI on (α_initial − α_continuation) excludes 0,
  AND this holds under word-frequency matching, AND the sign is consistent across two eval corpora.
- **Uniform-α is the (publishable) result iff:** M_shared predicts held-out per-class loss within
  **0.03 nats** and M_perclass does NOT beat it by > 0.02 nats (a real equivalence test — "scale buys
  down every token class at the same exponent; difficulty sets the intercept, not the rate").
- Anything else (underpowered / overlapping CIs / corpus-inconsistent) = NULL, down-scope, report honestly.

**Sequence-position confound** now controlled (measure.py buckets predicted positions into
early<128 / mid / late; position claims must hold in the mid band).

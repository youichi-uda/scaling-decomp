# R2 Self Review (post-R1 fixes, Round 2)

Reviewer: Self (second adversarial pass after R1 fixes).
File reviewed: `paper/main.tex`, with cross-checks against `README.md`,
`final_perN_Daxis.json`, `final_Naxis_perword.json`, `paper/make_figures.py`,
and Appendix A.

## Verification of R1 fixes

| R1 finding | Status | Notes |
|---|---|---|
| C1 (Table 1 E values) | FIXED | `tab:daxis` now matches JSON to 2 dp: 3.49/3.18/2.68/2.36 and 1.05/0.96/0.85/0.82. Codex R1 verification independently passed. |
| C2 (P=0.0000) | FIXED | Body line 311-312 and Fig 1 caption line 330-331 both use "$0/2000$ draws, $P < 5\times 10^{-4}$". Consistent. |
| C3 (Bonferroni scope) | FIXED | §1 lines 118-128 now explicitly justifies why Bonferroni $m{=}4$ is applied only to the D-axis family and not to the cross-family or per-D tables. |
| C4 (2D-surface framing) | FIXED | Abstract now says "within our tested grid... single-axis $\alpha$ varies by $\sim 2\times$" and explicitly notes "only sampled densely at the $D{=}300\text{B}$ column." |
| I1 (Bootstrap IID assumption) | PARTIALLY FIXED | §3.6 line 280-283 now states the homoscedasticity assumption and acknowledges it; no paired-bootstrap robustness check added. Acceptable hedge. |
| I2 (PIQA +0.010 noise floor) | FIXED | §4.6 lines 668-675 and §6 conclusion bullet line 879-880 both flag PIQA as at the noise floor. |
| I3 (monotonicity language) | PARTIALLY FIXED — see R2 N1 | §1 preview line 137 and §6 conclusion bullet line 858-861 are softened, but §6.1 line 696 still says "magnitude monotonic with $N$ within the tested range" without the rank-correlation hedge. |
| I4 (HF Hub demoted) | FIXED | Abstract closing sentence is now a hedged "we report... acknowledge ruling out the resolution-layer caching... would require direct LFS-blob testing we did not perform." Broader Impact statement reframed. |
| I5 (50k railed footnote) | NOT FIXED | `tab:perD` row "50k" still shows $\alpha=0.02$ rail with no explanatory footnote/parenthetical; reader still sees apparent flip 1.61 → 0.02 between 42k and 50k. |
| I6 (marginal in §6 bullet) | FIXED | §6 conclusion line 864 now reads "marginal pre-registered pass". |
| I7 (Wikitext 70M inclusion) | FIXED | §4.1 lines 346-354 explain why 70M is in the Wikitext sign-check but not in the Pile main fit. |
| M1 (Bonferroni forward-pointer) | FIXED | §3.5 line 266 and §3.6 line 277 both reintroduce the correction. |
| M2 (Ġ U+0120 marker) | FIXED | Lines 100 and 824 now use `\texttt{\char192}` with parenthetical "(U+0120, GPT-NeoX/GPT-2 leading-space marker)". |
| M4 (sennrich2016bpe uncited) | FIXED | Line 204 cites it. |
| M5/M6/M7 (README issues) | FIXED | README TL;DR notes 70M exclusion; family list corrected; "Coming soon" removed. |
| M8 (Fig 1 caption $P=0$) | FIXED | Caption uses $<5\times 10^{-4}$. |
| Persona A 1 (floor–slope) | FIXED (in Limitations) | New floor-slope bullet at lines 787-794; new §4.1 floor-slope caveat at lines 313-321. |
| Persona A 2 (Michaud Zipf) | PARTIALLY FIXED — see R2 C1 | Limitations bullet at 795-808 adds the Zipf-coupling caveat but **over-attributes** the prediction "$\alpha \to 1$" to Michaud 2023. |
| Persona A 3 (per-axis projection) | FIXED | Bullet at 809-816 acknowledges joint non-identifiability leaks into per-axis fits. |
| Persona A 5 (Cerebras undertrained) | FIXED | §4.4 lines 512-523 add the compute-frontier alternative explanation. |
| Persona C 1, 4 (HF Hub locus, Pile-10k size) | PARTIALLY FIXED | Limitations bullets added; the Pile-10k bullet does not address Persona C's specific concern that BLOOM (ROOTS-trained) on Pile-10k is *out-of-distribution*, only that the sample is small. See R2 I1. |
| Persona C 5 (per-family class mixture) | NOT FIXED | Limitations bullet at 823-832 acknowledges the gap but no per-family mixture numbers are reported. Acceptable as honest deferral. |

## New Critical (block TMLR; must fix R2)

### R2-C1. §1 line 107-108 contradicts Appendix A line 958 on whether per-word aggregation tightens or widens the CI

§1: "per-word aggregation slightly inflates the per-class $\alpha$ magnitude estimates **and tightens their CIs**."
Appendix A: "Per-word matched aggregation gives larger $|\dalpha|$ **and a wider CI**."

The JSON numbers in App A confirm: per-word CI $[-0.64, -0.22]$ has width $0.42$ vs token-weighted $[-0.43, -0.16]$ width $0.27$ — per-word is unambiguously **wider**, not tighter. The Appendix A text and the table are correct; the §1 sentence is incorrect.

Fix: change line 107-108 to "per-word aggregation gives larger $|\dalpha|$ estimates and somewhat wider CIs (Appendix~\ref{app:agg}); both the sign and significance survive." A TMLR reviewer who compares §1 to App A will spot this contradiction on a literal read.

### R2-C2. Over-attribution of "$\alpha \to 1$" prediction to Michaud 2023 (Limitations bullet, lines 795-798)

Current text: "\citet{michaud2023quantization} predicts per-class $\alpha \to 1$ when the per-class prediction problem collapses to its unigram-conditioned distribution."

Michaud's quantization model predicts $\alpha \propto$ Zipf slope (the cluster-frequency exponent) — it does not literally predict $\alpha \to 1$ at the unigram floor as a closed-form result. Persona A's review paraphrased this as "$\alpha \to 1$ as the cluster frequency approaches the unigram floor"; the R1 fix took that paraphrase as the source rather than going back to Michaud. The current sentence misrepresents what Michaud actually argues, and a referee who knows the Michaud paper will flag it as an over-attribution that the paper's own footnote chain cannot support.

Fix: rephrase to "\citet{michaud2023quantization} ties per-class $\alpha$ to the Zipf slope of the per-class frequency distribution; per-class problems dominated by their unigram-conditioned distribution would have $\alpha$ values consistent with the high-frequency end of that mapping (loosely, $\alpha$ near 1 for the most-predictable classes)." This preserves the Persona A intuition without putting an unsupported claim in Michaud's mouth. Alternatively, drop the explicit attribution and write "predictions in the Zipf-coupling literature would place $\acont$ on the side of the partition dominated by unigram-conditioned completion" without naming a numerical target.

## New Important (should fix this round)

### R2-I1. Pile-10k is OOD for BLOOM and the new Limitations bullet on Pile-10k does not address this

Line 236: "Primary: `NeelNanda/pile-10k`, an **in-distribution** Pile sample". This is accurate for Pythia (trained on Pile), Pythia-deduped (Pile-dedup), and Cerebras-GPT (Pile). It is **not** in-distribution for BLOOM, which was trained on ROOTS (multilingual). The new Limitations bullet at line 817-822 covers Pile-10k *sample size* but does not flag that BLOOM is evaluated OOD on Pile-10k. The Persona C 4 concern was that this OOD evaluation, not just sample size, is part of why BLOOM's $|\dalpha|$ is smallest.

Fix: extend the Limitations bullet to one extra sentence: "Pile-10k is in-distribution for Pythia, Pythia-deduped, and Cerebras-GPT (all Pile-trained); for BLOOM it is out-of-distribution (BLOOM was trained on ROOTS). BLOOM's smaller $|\dalpha|$ may therefore reflect OOD evaluation as well as the tokenizer effect." Alternatively, qualify line 236 explicitly: "in-distribution for Pythia, Pythia-deduped, and Cerebras-GPT; out-of-distribution for BLOOM, which was trained on ROOTS."

### R2-I2. §6.1 line 696 still says "magnitude monotonic with $N$" without the rank-correlation hedge

R1 I3 was applied in §1 preview and §6 conclusion bullets but missed §6.1 (Discussion). The phrase "magnitude monotonic with $N$ within the tested range" at line 696 anchors a stronger reading than the §6 bullet's "rank-correlation $\rho{=}+1$ on $n{=}4$, suggestive but not decisive".

Fix: change line 696 to "magnitude rank-correlated with $N$ across the four sampled values" — same wording as line 411 and §6 conclusion bullet line 860, so all three monotonicity-touching sentences match.

### R2-I3. Abstract magnitude claim is inconsistent with §4.4 new alternative-explanation paragraph

Abstract line 59-60 says "magnitudes span $\sim$3$\times$ ($-0.13$ to $-0.41$), the sign is invariant, the magnitude is not." §4.4 lines 512-523 now correctly distinguishes (a) coordinate dependence of $\dalpha$ *within* Pythia from (b) magnitude differences *across* families that are partly attributable to compute-budget choices, and explicitly says "the magnitude is comparable only within (a)." But the abstract treats the cross-family magnitude span as part of the coordinate-dependence story, which §4.4 now disavows.

Fix: split the abstract finding (iii). Either (a) demote the magnitude span to a robustness check — "the sign is invariant across families; magnitude differences across families are partly compute-frontier confounded (§4.4) and are not the headline coordinate-dependence claim" — or (b) move the magnitude claim out of the abstract entirely and reserve "coordinate-dependent" for the within-Pythia D-axis result.

### R2-I4. Limitations bullet on Pythia-deduped omission of $\geq 2.8\text{B}$ sizes is still under-justified

§4.4 line 487-489 says "we did not run the full final-checkpoint family for compute reasons" — but Pythia-deduped 2.8B/6.9B/12B final checkpoints exist on HF Hub and are inference-only (no training cost). A reviewer who checks the Hub will see this and ask why the cross-family Pythia-deduped fit is restricted to $n_N{=}4$ when 7 sizes are available, which makes the $[-0.33, -0.23]$ CI suspiciously tight.

Fix: either run Pythia-deduped on the remaining 3 sizes and rerun the fit (cost: <$5 RunPod for 6.9B+12B; the compute is essentially the same as one cross-family entry), or add one explicit sentence on why $n_N{=}4$ for Pythia-deduped is sufficient — e.g., "we restricted Pythia-deduped to the four smaller sizes to match the held-out adjudication train-set in §4.1." That second framing at least makes the choice principled rather than ad-hoc.

## New Minor / Nits

### R2-N1. The "1/12 under the null" calculation in §4.2 (line 415) is correct but undated

The text says "a strict-monotone ordering of 4 points happens by chance with probability $1/12$ under the null." Strictly, this is $2/4! = 2/24 = 1/12$ counting either strictly increasing or strictly decreasing as monotone. The factor of 2 (two-sided) is implicit. A reader who computes $1/4! = 1/24$ will pause; one extra word — "two-sided strict-monotone ordering" — closes the gap.

### R2-N2. Use of $\rho$ for rank correlation without naming Spearman

Line 412 and line 860 use "$\rho = +1$" — the standard Spearman symbol. The paper does not state "Spearman" anywhere. A pedantic reviewer may ask whether this is Pearson on ranks (which is Spearman) or Kendall $\tau$ (also $\rho$-ish in some notations). One word ("Spearman $\rho = +1$") resolves it.

### R2-N3. The new Limitations bullet on the "shared-$E$ sweep" implicitly promises a procedure described only in three lines of §4.1

§4.1 paragraph "Shared-E robustness" at lines 356-359 says "Forcing shared $E$ across classes ($E \in [0, E_\mathrm{cont}]$ swept) preserves the sign of $\dalpha$ at every sampled $E$." The number of sample points, whether the held-out RMSE was re-evaluated under shared-$E$, and whether the per-class $\alpha$ magnitudes were re-reported are not specified. The Limitations bullet at 791-793 then leans on this sweep ("The shared-$E$ sweep preserves the sign of $\dalpha$ but does not pin its magnitude"). Adding one sentence in §4.1 — "we swept $E$ across 11 evenly-spaced values in $[0, 0.79]$; at each $E$ we refit per-class $A$ and $\alpha$ by linear least-squares and verified $\dalpha < 0$ across all sampled $E$" — would make the sweep concrete.

### R2-N4. The §6 conclusion still describes Pythia-deduped+Cerebras+BLOOM as "three additional families" while the abstract says four

Line 866 says "Sign-consistent CIs in three additional families --- Pythia-deduped, Cerebras-GPT, and BLOOM." The abstract (line 56-59) counts Pythia + 3 additional = 4 families total. This is consistent — "three additional" refers to non-Pythia — but a fast reader bouncing between §6 and abstract may pause. Consider "three additional families (four families total, see Table~\ref{tab:families})" for clarity.

### R2-N5. `arxiv_submission.tar.gz` in repo root is stale relative to the R1-fixed `main.tex`

`paper/arxiv_submission.tar.gz` exists and is presumably from the pre-R1 build. A reader who downloads the tar instead of building from `main.tex` will get the wrong Table 1 E values, the wrong $P=0.0000$ string, the wrong abstract framing, etc. Either rebuild the tar from the current `main.tex` or delete it from the repo and let users build it locally.

### R2-N6. README "main.pdf (14 pages, 4 figures)" — verify after R1 additions whether page count is still 14

R1 added the floor-slope caveat, Michaud Limitations bullet, Cerebras Chinchilla bullet, Pile-10k bullet, tokenizer-partition bullet, HF Hub locus bullet, the cross-corpus 70M paragraph, and the §4.4 alternative-explanation paragraph. Page count may have grown. Re-check after compilation and update README if needed.

---

**Recommendation.** R1 fixes landed substantively; R2 surfaces two genuine new critical items (R2-C1 internal contradiction on CI width; R2-C2 Michaud over-attribution) that emerged from the R1 additions themselves rather than being missed in R1, plus four important items where the R1 additions partially under-covered the original concern. None of the R2 critical fixes are research-effort items (all are wording fixes against existing data); R2-I4 is the only one that could justify a small additional measurement run ($<\$5$). The paper after R2 fixes would be honestly TMLR-ready in the sense that no reviewer reading carefully should find a literal numerical or attribution error.

# R3 Self Review (final pass before arXiv)

Reviewer: Self (third adversarial pass).
File reviewed: `paper/main.tex` (1172 lines, 16 PDF pages), cross-checked
against `floor_robust.json`, `hub_blob_audit.json`, `final_perN_Daxis.json`,
`final_Naxis_perword.json`, and the R1/R2 review trail.

## Regression check (R1/R2 edits)

* **Orphan labels `sec:nfit`, `sec:results`.** Confirmed removed. `grep -n` on
  both labels in `main.tex` returns nothing; the R2 commit cleanup also
  removed any `\ref{}` that might have pointed at them. No dangling refs.
* **Cross-references.** Audited every `\ref{...}`: `app:joint`, `app:hub`,
  `app:agg`, `sec:agg`, `sec:dfit`, `sec:daxis`, `sec:families`, `sec:perD`,
  `sec:naxis`, `sec:coord`, `sec:downstream`, `sec:disc`, `sec:limits`,
  `sec:mixture`, `fig:naxis`, `fig:daxis`, `fig:families`,
  `fig:downstream`, `tab:daxis`, `tab:perD`, `tab:families` -- every one has
  a matching `\label`. Clean.
* **floor_robust.json numbers vs §4.1 text.** JSON canonical_free has
  $\alpha_\mathrm{init}{=}0.561$, $\alpha_\mathrm{cont}{=}0.976$,
  $E_\mathrm{init}{=}2.460$, $E_\mathrm{cont}{=}0.792$, free-fit
  $\dalpha{=}-0.414$; shared-$\alpha$ $F(1,8){=}4.47$, $p{=}0.0674$.
  Paper rounds these to $0.56/0.98/2.46/0.79/-0.41$ and $F(1,8){=}4.47$,
  $p{=}0.067$ -- exact match.
* **Shared-E sweep range.** JSON has 12 feasible $E$ points spanning
  $0.0 \to 1.834$ with $\dalpha$ from $-0.082$ (near 0) to $-1.530$ (large $E$).
  Paper §4.1 says "varies from $-0.08$ (near $E{=}0$) to $-1.5$" -- match.
  The grid description "$[0, 1.2 \cdot E_\mathrm{init}^\mathrm{free}]$" is a
  faithful description of the linspace upper bound in
  `floor_robust.py:92-97`; the realized max (1.834) is below the nominal
  $1.2 \cdot 2.46 = 2.95$ only because of the three-segment grid
  construction. Minor: a reader checking will see the nominal vs realized
  gap, but it's not load-bearing. Acceptable.
* **Internal consistency: abstract vs §4.1 floor caveat vs §6 conclusion.**
  Abstract still reports "$\dalpha = -0.41$, 95\% CI $[-0.64, -0.22]$"
  without the floor-conditional caveat in the same sentence; §4.1 adds the
  caveat one paragraph later; §6 opens with "magnitude is floor-conditional
  on the N-axis." The three places are not mutually contradictory but they
  do peak at different hedge strengths. **No literal contradiction**; a
  reader on a fast pass takes "$-0.41$" from the abstract, a reader on a
  slow pass gets the floor caveat in §4.1 and §6. This is the
  abstract-vs-Limitations gradient that B-Crit-R2-1 already flagged in R2;
  R3 confirms it remains but is no longer hidden -- §6 explicitly says
  "magnitude floor-conditional on the N-axis (varies by a factor of
  $\sim 20$ across feasible shared-$E$ values)" in the opening paragraph.
* **Abstract claim "all four families show $\dalpha < 0$ with 95\% CIs
  excluding zero".** Verified against §4.4 numbers and `tab:families`:
  Pythia $[-0.64, -0.22]$, Pythia-deduped $[-0.33, -0.23]$, Cerebras-GPT
  $[-0.32, -0.06]$, BLOOM $[-0.22, -0.03]$. All four exclude zero;
  the abstract claim is literally true. The floor-conditional caveat
  applies to the *magnitude* $-0.41$, not to the sign claim.
* **R2-introduced text consistency.** §6 opening paragraph correctly
  cross-refs §4.1; §6 bullet 2 still reports $-0.41$ with the
  marginal-RMSE caveat but **does not** restate the shared-$\alpha$
  $F$-test $p{=}0.067$ in the conclusion bullet. That is the one
  surviving asymmetry between §4.1 (mentions both $p{=}0.067$ and
  RMSE $+0.022$) and §6 (mentions only RMSE $+0.022$).

## Residual Critical (must fix before arXiv)

**None.** All four R1 Criticals (C1 table values, C2 P=0, C3 Bonferroni
scope, C4 2D-surface framing) and R2-C1 (App A vs §1 CI-width
contradiction), R2-C2 (Michaud over-attribution) are FIXED. The §4.1
floor caveat and Conclusion reframe address Persona A R2-Crit-1's
"magnitude not pinned" concern without retreating from the sign claim.
Persona C R2-Crit-1 (Hub bug locus) is handled by the new
`hub_blob_audit.json` + `verify_checkpoints.py` artifact and the
Appendix C wording change to "Hub-side ... independent of the
\texttt{transformers} or \texttt{huggingface\_hub} client resolution
layer" -- this is now a substantive claim, not a hedge.

## Residual Important (should fix; acceptable to submit and revise)

**R3-I1. §6 conclusion bullet 2 omits the $F$-test $p{=}0.067$ that §4.1
calls out as the second marginal adjudicator.** Bullet 2 reports
"held-out RMSE improvement $+0.022$~nats is a marginal pre-registered
pass" but stops there. §4.1 says both held-out RMSE AND the
shared-$\alpha$ $F$-test are marginal ($p{=}0.067$); §6.1 also
restates both. Conclusion bullet 2 should be one phrase longer: "...is a
marginal pre-registered pass ($0.022$~nats vs $0.020$~nats threshold;
the shared-$\alpha$ $F$-test on the same N-axis fit gives
$p{=}0.067$, also marginal)." This brings the conclusion-bullet
language flush with the §4.1 / §6.1 wording. **One-line fix.**

**R3-I2. §4.6 downstream framing did not receive a floor-confound
revisit.** The §4.6 conclusion paragraph reads "We interpret this as
suggestive of per-class predictive content beyond aggregate, but
underpowered at $n{=}7$". After the new §4.1 floor-conditional caveat,
a careful reader will ask: if the per-class $\alpha$ freedom is
statistically marginal (shared-$\alpha$ $p{=}0.067$), then the
"per-class regressor beats aggregate on adj $R^2$" comparison is
implicitly leaning on the freedom of $\acont$ to track $L(N)$
differently than $\ainit$ -- the same freedom whose statistical status
is marginal. The downstream test does not directly use the per-class
$\alpha$ values (it uses $L_\mathrm{init}$ and $L_\mathrm{cont}$ as
regressors, not $\ainit$ and $\acont$), so the dependence is indirect,
but it would be honest to note this in one sentence at end of §4.6:
"The per-class regressor's advantage uses per-class loss trajectories
$L_\mathrm{init}(N)$ vs $L_\mathrm{cont}(N)$, not the fitted per-class
$\alpha$ values; so this finding is not directly subject to the
$\alpha$-freedom $F$-test caveat in §4.1." **One-sentence fix
clarifying scope.**

**R3-I3. The $F(1,8)$ reference distribution is approximate at
$n_\mathrm{obs}{=}14$.** The shared-$\alpha$ $F$-test compares a 5-param
nested model to a 6-param model on 14 observations, giving $F$ with
$df_1{=}1$, $df_2{=}8$. The $F$-distribution assumption requires
Gaussian residuals and homoscedasticity (the same assumption the
bootstrap is acknowledged to need in §3.6). At $n_\mathrm{obs}{=}14$
the asymptotic $F$ is only approximate; a permutation-based or
parametric-bootstrap $p$ would be more defensible. The paper does not
flag this. A TMLR statistical reviewer will note that the $p{=}0.067$
threshold depends on a distributional assumption the paper itself
admits is unverified. **Fix: one sentence in §4.1 after the $F$-test
paragraph: "The $F(1,8)$ reference distribution assumes Gaussian
residuals under the null; with $n_\mathrm{obs}{=}14$ this is asymptotic
rather than exact, and the same IID-residual caveat that applies to the
bootstrap (§3.6) applies here."** Acceptable to submit and revise.

**R3-I4. R2 Persona B's "Limitations is 12 bullets" remains.** §5
Limitations is now 13 bullets (one was added for the verification
artifacts). This is honest but the Limitations-to-Results length ratio
has grown. R3 will not block on this for arXiv (TMLR can do its own
length judgement on revision); flagging only.

**R3-I5. arxiv_submission.tar.gz is 3 hours stale.** The `paper/`
directory has `arxiv_submission.tar.gz` from 11:13 while `main.tex` was
last modified at 14:16. If the arXiv upload uses this tarball it will
ship pre-R2 content (no floor-robust paragraph, no Hub LFS audit).
**Rebuild the tarball from current `main.tex` + `references.bib` +
`figs/` + `tmlr.sty` + `tmlr.bst` + `fancyhdr.sty` before uploading.**
This is the one literal blocker for today's arXiv submission.

## Polish (optional)

* §4.1 floor-slope adjudication paragraph could note the realized
  upper bound (1.83) of the shared-$E$ grid vs the nominal upper bound
  $1.2 \cdot E_\mathrm{init}^\mathrm{free} = 2.95$. A reader who runs
  `floor_robust.py` and sees the 12 printed rows stop at 1.83 won't be
  confused but will pause briefly.
* Abstract's "magnitudes span $\sim$3$\times$" for the four-family
  result vs §4.4's note that part of the cross-family magnitude gap is
  compute-budget-driven -- this was R2-I3, and the R2 fix added the
  caveat to §4.4 but did not touch the abstract sentence. The abstract
  still asserts "the sign is invariant, the magnitude is not" without
  flagging that part of "is not" is confounded. Acceptable on a fast
  read; pedantic on a slow one.
* The bibliography has 17 entries (`grep -c "^@" references.bib`). All
  load-bearing citations have entries; no obvious missing references.
* PDF is 16 pages (`pdfinfo`), one page longer than R2 noted (15 pages),
  consistent with the R2 floor-robust paragraph + Hub audit text
  additions.
* LICENSE and LICENSE-paper present.

## Adversarial sweep for a future TMLR reviewer

* **Held-out RMSE threshold = 0.020 is somewhat arbitrary.** The paper
  pre-registered this threshold but never argues why $0.020$ rather than
  $0.010$ or $0.030$. With the observed $0.022$ being a 10% margin,
  a TMLR reviewer will ask "what would have happened at $0.025$?". The
  honest answer is "we chose $0.020$ before fitting"; the paper says
  this in §3.4. R3 verdict: acceptable -- pre-registration is the
  defense against threshold-shopping, and the threshold was committed
  before measurement.
* **Bonferroni $m{=}4$ vs uncorrected cross-family.** §1 paragraph 4
  explicitly justifies the choice ("each family is treated as an
  independent replication rather than as members of a joint
  rejection-decision family"). A statistically conservative reviewer
  could still ask whether the $m{=}4$ correction should apply to the
  union $\{$D-axis 4 N$\} \cup \{$cross-family 4 fams$\}$. The paper
  scopes the test families separately; this is defensible but
  worth being prepared to defend.
* **$F$-test reference distribution** -- see R3-I3 above.

## Verdict

**ARXIV-READY (after one literal blocker fix).**

The one literal blocker is R3-I5: rebuild `arxiv_submission.tar.gz`
from current `main.tex` before uploading -- the stale tarball would
ship pre-R2 content. After that rebuild, the paper is
arXiv-submittable today. R3-I1 (conclusion bullet 2 should mention
$p{=}0.067$) and R3-I2 (downstream scope sentence) and R3-I3 ($F$-test
distributional caveat) are all one-sentence revisions that would
strengthen the paper but do not block arXiv submission; they can be
made for the TMLR camera-ready or in an arXiv v2.

The paper's load-bearing claims (sign of $\dalpha$ across 4 families,
D-axis Bonferroni-decisive at 4 $N$, $\dalpha$ magnitude is
floor-conditional and coordinate-conditional) are all defended at the
$n$-restricted resolution available. The §6 reframe and the §4.1
floor-slope adjudication are mutually consistent and honest. The
shared-$\alpha$ $F$-test $p{=}0.067$ is fairly reported as marginal,
not as a rejection. The Hub appendix now stakes the Hub-side claim on
`x-linked-etag` evidence rather than only on transformers-side
symptoms, which addresses Persona C R2-Crit-1.

**Recommend: rebuild tarball, upload to arXiv today, file R3-I1/2/3
as a v2 amendment after a day or two.**

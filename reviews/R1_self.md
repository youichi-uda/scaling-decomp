# R1 Self Review (post-LaTeX, Round 1)

Reviewer: Self (adversarial pass).
File reviewed: `paper/main.tex` (commit-time content), with cross-checks against `final_Naxis_perword.json`, `final_perN_Daxis.json`, `paper/make_figures.py`, `README.md`, and `PREREGISTRATION.md`.

## Summary

The paper makes a focused, well-hedged measurement claim — per-class word-initial vs word-continuation α gap is coordinate-dependent, replicates in four families, and is decisive on the D-axis. The pre-registration trail, honest non-identifiability reporting on per-D N-axis fits and joint $L(N,D)$ forms, and the HF Hub appendix are real strengths. However, several issues would damage credibility under TMLR review: **(a) Table 1 D-axis $E_\mathrm{init}$ and $E_\mathrm{cont}$ values do not match the canonical JSON** (this is a literal numerical inconsistency that any reviewer running the code will spot); **(b) the bootstrap procedure's IID-residual assumption is unstated and unverified** on n=7/n=13–14 power-law residuals; **(c) $P=0.0000$ for a B=2000 bootstrap is a reporting error** (should be $P<5\times10^{-4}$); **(d) Bonferroni $m=4$ is applied only to the D-axis family but not to the cross-family or per-D N-axis tables**, which is methodologically inconsistent given the headline framing; **(e) the "coordinate-dependent observable" claim is structurally two 1D slices with different $n_N$ counts rather than a 2D surface**, which the discussion partially admits but the title and abstract over-sell. Several minor LaTeX issues (uncited `sennrich2016bpe`, missing `\Gamma`/`Ġ` for the BPE space marker in §3.1) round out the punch list.

## Critical (block TMLR accept; fix this round)

### C1. Table 1 (`tab:daxis`, main.tex:362–365) — $E$ values do not match `final_perN_Daxis.json`

The canonical JSON `final_perN_Daxis.json` has:

- 1B: `E_init=3.486`, `E_cont=1.049`
- 1.4B: `E_init=3.184`, `E_cont=0.962`
- 6.9B: `E_init=2.678`, `E_cont=0.854`
- 12B: `E_init=2.358`, `E_cont=0.824`

The paper Table 1 prints:

- 1B: `E_init=3.71`, `E_cont=1.12`
- 1.4B: `E_init=3.39`, `E_cont=1.01`
- 6.9B: `E_init=2.83`, `E_cont=0.86`
- 12B: `E_init=2.55`, `E_cont=0.84`

The $\alpha_D$ and CI columns DO match the JSON. So only the $E$ columns drifted, almost certainly from an older fit pass that was not re-rendered. PAPER_DRAFT.md:309 shares the same (wrong) numbers, so it's a pre-LaTeX inherited error, not a LaTeX typo.

Fix: regenerate Table 1 directly from `final_perN_Daxis.json` (`E_init` and `E_cont` rounded to two decimals: 3.49, 3.18, 2.68, 2.36 and 1.05, 0.96, 0.85, 0.82). Either (i) re-run the canonical $D$-axis fit to confirm the JSON is correct and update Table 1, or (ii) replace the JSON with the latest fit if Table 1 actually reflects the canonical pipeline. Pick one source of truth and propagate. A TMLR reviewer running `final_perN_Daxis.py` will see the discrepancy in under five minutes.

### C2. main.tex:292 — $P(\dalpha\geq 0)=0.0000$ misrepresents bootstrap resolution

`final_Naxis_perword.json` reports `P_nonneg: 0.0` from $B=2000$ bootstraps. With $B=2000$ the minimum representable upper bound on $P(\dalpha\geq 0)$ is $1/2000 = 5\times 10^{-4}$, **not** $0$. Reporting `P=0.0000` invites a reviewer comment that the author confused "no positive draws observed" with a finite-precision probability.

Fix: rewrite §4.1 and the Figure 1 caption as `$P(\dalpha \geq 0) < 5 \times 10^{-4}$ (0/2000 bootstrap draws)` or simply "no bootstrap draws had $\dalpha\geq 0$ out of 2000". Same change to fig1 caption.

### C3. Bonferroni $m=4$ scope is inconsistent across the headline claims

The paper applies Bonferroni $m=4$ to the D-axis table (98.75% CIs) but uses uncorrected 95% CIs for:

- the cross-family forest plot (Pythia/dedup/Cerebras/BLOOM; Table 2, Fig 3) — also $m=4$ independent tests of $\dalpha<0$,
- the per-D N-axis table (`tab:perD`) — 6 D-points, all reported with 95% CIs,
- the N-axis main fit (one test, 95% CI).

This is methodologically defensible (each is a separate family of hypotheses), but the paper does not say so. A TMLR statistical reviewer will flag this. Either (a) apply $m=4$ to the cross-family table as well — under $m=4$ the BLOOM CI $[-0.22, -0.03]$ at 98.75% becomes wider and would need rechecking — or (b) explicitly state §3.4 the rationale: "Bonferroni is applied only to the D-axis four-N family because that is where the headline 'all CIs exclude zero' claim is sensitive to the joint test; the cross-family table reports per-family CIs because each family is an independent replication that we do not combine into a single rejection decision."

The honest move is (b); silently using different correction policies for tables that share visual prominence is the wrong signal to send.

### C4. "Coordinate-dependent observable" is structurally two 1D slices with different $n_N$, not a 2D surface

The headline asks the reader to picture an $(N,D)$ surface. What the data actually supports is:

- one well-identified D-axis ladder at each of four $N$ (the cleanest evidence),
- one well-identified N-axis fit at $D=300\mathrm{B}$ with $n_N=7$,
- per-D N-axis fits that are not identifiable at any other $D$ (§4.3 honestly admits this).

So the empirical "surface" has only one cell with both axes identifiable. The §6 conclusion already softens this ("within our data..."), but the abstract sentence "single-axis decomposition at one (N,D) point recovers one slice of a 2D pattern" implies the 2D pattern is mapped. It isn't — only one column of the matrix is mapped on both axes.

Fix: rephrase the abstract claim to: "within our tested grid, single-axis per-class $\alpha$ varies by a factor of $\sim 2$ depending on whether $N$ or $D$ is the varied axis and the operating point of the held axis. We are not able to fit a joint $L_c(N,D)$ surface (Appendix B), so 'coordinate dependence' here means the single-axis estimates depend on the axis and on the held value." Same tightening in the §6 first paragraph.

## Important (block higher venues / weaken impact; should fix)

### I1. Bootstrap procedure assumes IID residuals across $N$ (or $D$); never verified

§3.6 says "parametric residual bootstrap: resample residuals of the per-class power-law fit." This is fine **only if** residuals are exchangeable across the X grid — homoscedastic and IID. With $n=7$ Pythia sizes, residual structure is essentially unmeasurable, but power-law fits typically have larger absolute residuals at the small-$N$ end (unsaturated regime) and at the floor end. If residuals are heteroscedastic, the bootstrap CI on $\dalpha$ is too narrow at the unsaturated end and too wide at the saturated end.

Fix: at minimum, add a sentence in §3.6 stating the homoscedasticity assumption explicitly and acknowledging it as a limitation given $n_N=7$. Better: add a paired bootstrap (resample $(N_i, L_\mathrm{init}(N_i), L_\mathrm{cont}(N_i))$ tuples) as a robustness check and report whether the CI changes. With $n=7$ the paired bootstrap will have only $\binom{13}{7}=1716$ distinct samples but still gives a meaningful sanity check. Even one sentence — "we verified the paired-bootstrap CI is within ±0.03 of the residual-bootstrap CI" — would close this hole.

### I2. §4.6 downstream: "+0.010–0.037 adj R² gain across all 4 tasks" is over-sold at $n=7$, $p=2$

With $n=7$ and going from $p=1$ to $p=2$, the adj-$R^2$ penalty term changes from $(1-R^2)\cdot 6/5$ to $(1-R^2)\cdot 6/4$ — i.e., adding a useless second regressor would drop adj-$R^2$ by about $(1-R^2)\cdot 0.3$. At $R^2 \sim 0.97$, the penalty is only $\sim 0.009$. So "+0.010 on PIQA" is **at the threshold of measurement noise** — the gain is essentially the inverse of the penalty rather than evidence that init+cont contains information beyond aggregate.

The paper does hedge ("suggestive, not decisive", "underpowered at n=7"), which is good. But the **conclusion bullet** still reports it as a positive finding without re-stating that PIQA's +0.010 is at the noise floor. Also: there is no script in the repo that computes these adj-$R^2$ values from `lmeval_*` JSONs — they appear hardcoded in `make_figures.py:349`. A reviewer who tries to reproduce will not find a `compute_downstream.py`.

Fix: (a) split PIQA into a separate sentence noting +0.010 is within the adj-$R^2$ adjustment noise floor at $n=7$; (b) commit the `compute_downstream_adjr2.py` script to the repo (currently absent — verified by `grep -rln "adj_R\|adjusted" *.py`), so the +0.037/+0.034/+0.024/+0.010 numbers have a reproducible source.

### I3. §4.2 "magnitude grows from −0.20 (1B) to −0.45 (12B)" is monotonic on $n=4$ — call it a trend, not a pattern

The four points $\{-0.20, -0.25, -0.37, -0.45\}$ are strictly monotonic in $|\dalpha_D|$ with $N$. With $n=4$, a strictly monotonic sequence happens by chance with probability $2/4! = 1/12 \approx 8\%$ even under the null of no relationship. The paper says "observed monotonic pattern across 4 N within our data; we do not claim asymptotic behavior beyond $N=12\mathrm{B}$" — the hedge is correct but the word "monotonic" still anchors a stronger reading.

Fix: replace "monotonic pattern" with "rank-correlated with $N$ ($\rho=+1$ across $n=4$, which is suggestive but underpowered)" or "the magnitude appears to grow with $N$ across our four tested values; we make no claim of monotonicity in $N$ at scales we did not test." Same edit in §6 conclusion bullet.

### I4. Appendix C (HF Hub bugs) as a standalone "contribution" claim is over-billed

The abstract closes with "We additionally document HuggingFace Hub mis-registration bugs..." — implying it as a contribution alongside the coordinate-dependence finding. The bugs are real and useful to document, but as a paper contribution they are a **data-hygiene observation** that affects a specific tool (HF Hub) at a specific time; they are not a methodological or empirical contribution per se. The Broader Impact statement ("main practical impact is reproducibility") leans even harder on this framing.

Fix: demote Appendix C from "contribution" to "reproducibility note" in the abstract (1 sentence, last position, no bold or "additionally"). The Broader Impact should not call the Hub bug the "main practical impact" — the cross-family replication and methodological recommendation are bigger. Suggested rewrite: "Broader Impact: this is a methodological cautionary study; the main practical impact is the recommendation to report per-class $\alpha$ values with their measurement coordinate. We also flag two HuggingFace Hub mis-registration bugs found during this work (Appendix C)."

### I5. §4.3 "step 50k railed at $\alpha=0.02$" is suspicious and unexplained

`tab:perD` row "50k": $\ainit=\acont=0.02$, CI $[0,0]$, "yes (grid min)". This means at $D=105\mathrm{B}$ the per-class fit collapsed both classes to the smallest grid value. That is a 4-point fit with too few points to identify the shape, but it should rail at the **same** grid edge for both classes (which it does), so it's not strictly wrong. However, immediately above (row "1k-42k": railed at $\alpha=1.61$ grid max) and immediately below (row 70k: $\ainit=1.35$) shows a non-monotonic sweep. The reader will ask "did 50k actually have data and what does $\alpha=0.02$ mean here?"

Fix: add a one-sentence footnote/parenthetical clarifying that the railed fits should be read as "fit is not identifiable; the grid edge is reported as a flag, not as a point estimate." This is implicit but a TMLR reviewer will flag the apparent flip from $\alpha=1.61$ at 42k to $\alpha=0.02$ at 50k.

### I6. Held-out RMSE: $n=4$ train, $n=3$ test, "improvement $+0.022$" is at threshold

The pre-registered threshold was $>0.02$ nats; observed is $0.0216$ nats. That's a 3% margin. The paper does call this "marginal" twice (Abstract, §4.1, §5 Limitations) which is honest. However the §6 conclusion **omits the "marginal" qualifier** when stating "held-out RMSE improvement +0.022 nats" in the bullet list. Consistency.

Fix: in §6 conclusion bullet on N-axis: add "marginal pre-registered pass" after the held-out number, matching the abstract phrasing.

### I7. Wikitext sign-check uses different size set ($n=5$ including 70M) than the main N-axis fit ($n=7$ excluding 70M)

§4.1 "Cross-corpus sign check" uses Wikitext-103 on "5 train sizes 70M–1.4B". The main fit excludes 70M as anomalous. Including it in the cross-corpus check is inconsistent: either (a) 70M is anomalous and should be excluded everywhere, in which case the Wikitext check has $n=4$ (and is even less powered), or (b) 70M is fine for sign-direction even if anomalous for endpoint fitting — which is plausible but not stated.

Fix: add one sentence explaining why 70M is included in the Wikitext sign-check but excluded from the Pile main fit. The honest answer is "we needed at least 5 sizes to fit a curve at all and 2.8B/6.9B/12B were not measured on Wikitext for compute reasons"; say that explicitly.

## Minor (polish)

### M1. main.tex:113 — Bonferroni $m=4$ note is buried in the pre-registration paragraph

The Bonferroni correction is introduced in the §1 pre-registration paragraph (line 113-114) but is methodologically relevant to §3.5 and §4.2. Add a forward pointer or restate the $m=4$ rationale at the start of §3.5.

### M2. main.tex:95 — BPE space marker is rendered as `\verb|G|` rather than `\verb|Ġ|`

§3.1 and §1 both refer to "the GPT-NeoX BPE space marker `G`" — this is the Unicode `Ġ` (U+0120), commonly seen in GPT-2/GPT-NeoX. The verbatim `G` renders as ASCII 'G' which is misleading for any reader familiar with the tokenizer. Either show `Ġ` directly (works in standard pdfTeX with `\usepackage[utf8]{inputenc}` if not already in tmlr.sty) or switch to `\texttt{<U+0120>}` or describe it as "the leading-space marker (`Ġ`)" with an explanatory parenthetical.

### M3. main.tex:117–137 — "Findings (preview)" duplicates the abstract almost verbatim

The numbered findings list in §1 repeats the abstract content in nearly identical language. Two options: (a) cut the §1 preview list, since the abstract already covers it; (b) keep it and add the §1 preview the section/subsection numbers (e.g., "see §4.1 for the bootstrap and §4.2 for the held-out check") to add value over the abstract.

### M4. References — `sennrich2016bpe` is in `.bib` but never cited

`grep -n "sennrich"` on main.tex returns nothing. Either cite it (natural place is §3.1 when introducing BPE) or remove from references.bib to keep the bib clean.

### M5. README.md "TL;DR" matches paper claims but uses "n=7 Pythia" without flagging the 70M exclusion

The README says "(7 Pythia sizes)" without noting that 70M is excluded. A reader who counts Pythia model sizes will count 8. Add "(70M excluded as anomalous)" in the README TL;DR, matching the paper's wording.

### M6. README.md:42 lists `analyze_families.py ... (Pythia/Cerebras/OPT/GPT-2)` but the paper uses Pythia/Pythia-deduped/Cerebras/BLOOM

This is a stale README line. OPT and GPT-2 are not in the paper. Update to match.

### M7. README.md:92 mentions `verify_checkpoints.py *Coming soon*` — promising a script that does not exist

The HF Hub appendix would benefit from this script being committed. Either commit a stub `verify_checkpoints.py` that runs the SHA256 comparison described in Appendix C.3, or remove the "Coming soon" promise. As is, a TMLR reviewer who pulls the repo will not find what the README promises.

### M8. main.tex:298 / fig caption — Figure 1 caption ends with "$P(\dalpha\geq 0) = 0$" while body uses "$0.0000$"

Inconsistent. Plus this is the C2 issue (should be $<5\times 10^{-4}$).

### M9. main.tex:170 — "Pythia-70M excluded as anomalous, per the Pythia paper's own observation of training instability"

Verify Biderman et al. 2023 actually flags 70M as having training instability. If the paper instead documents this on 14M/31M but not 70M, the citation framing is wrong. (My recollection is that the Pythia paper does flag the very small sizes, but 70M may or may not be in that set — worth a 30-second check before submitting.)

### M10. Cross-family discussion sentence (main.tex:460-465) "BLOOM, the larger vocabulary reducing the multi-token-word sample within the eval corpus" — should quantify

"Larger vocabulary reducing the multi-token-word sample" is plausible but unquantified. One sentence: "BLOOM's 250k vocab reduces the multi-token-word fraction in pile-10k from $X\%$ (GPT-NeoX BPE) to $Y\%$" would close the loop. If $X, Y$ aren't easy to extract, just say "approximately halves" or similar order-of-magnitude.

## Nits (typos / formatting)

### N1. main.tex:31–32 — `\Mp` and `\Ms` are defined but never used in the main text

`\newcommand{\Mp}{M_\mathrm{per\text{-}class}}` and `\newcommand{\Ms}{M_\mathrm{shared}}` are defined at line 31-32 but the body uses `$\Mp$` and `$\Ms$` only at lines 240-242 and 311. That's actually fine — they ARE used. Ignore.

### N2. main.tex:96 — "via the GPT-NeoX BPE space marker \verb|G|" → see M2.

### N3. main.tex:272 — `\$60` should probably be `\$60`-`\$65` to match the README and memory note (`\$65` is in the memory)

Memory note records `$65` for top-main push. Paper says `~$60`. Either is fine; just pick one and be consistent across paper + README + project memory.

### N4. main.tex:354 — Table 1 caption "$98.75\%$" should arguably be "$98.75\%$ (= $95\%$ Bonferroni-corrected with $m=4$)" for self-containment

A reader landing on Table 1 from a back-reference shouldn't have to scroll up to figure out where the 98.75% came from.

### N5. main.tex:497–514 — Inline table without `\begin{table}` wrapper

The Pythia-deduped D-axis replication table at §4.4 uses raw `\begin{center}...\end{center}` instead of a `\begin{table}\caption{}` floating env. This is fine for prose, but for camera-ready, a `table` env with caption "Pythia-deduped D-axis replication at $N\in\{1\mathrm{B}, 1.4\mathrm{B}\}$" gives it a label for cross-referencing. Same for the per-token-class correlation table at §4.6 (lines 550–561), the downstream adj-R² table (lines 591–603), and the Appendix A aggregation comparison table (lines 817–827). Five "naked" tables in a paper with four numbered ones is a stylistic inconsistency a reviewer may comment on.

### N6. main.tex:556 — `$|r(\Linit)|$` etc. uses bold for `$\mathbf{0.996}$` to highlight max-per-row

The bolding pattern (max in each row gets bold) is not stated. Add one sentence in the caption: "Bold marks the larger of init/cont magnitude per task." Otherwise reader has to infer.

### N7. main.tex:23 — `\openreview{\url{...}}` is defined but `\openreview` is never `\renewcommand{}`-d, and the URL placeholder is `forum?id=TBD`

If the paper is going to arXiv before TMLR acceptance, the TBD URL is fine; just confirm the TMLR template doesn't auto-render `\openreview` in the camera-ready layout. (`grep -n openreview tmlr.sty` would confirm.)

### N8. main.tex:545 — "lm-evaluation-harness" should probably be cited (it's a tool, not just a name)

If a `gao2023lmeval` or `eleutherai_lmeval_2023` cite is appropriate, add it. Otherwise add a footnote URL.

### N9. main.tex:792 — `\bibliography{references}` followed by `\bibliographystyle{tmlr}` — order is fine but conventionally `\bibliographystyle` comes first

No effect on output; cosmetic.

---

**Recommendation:** Critical issues C1 (numerical Table 1 inconsistency) and C2 (P=0.0000 misreporting) must be fixed before TMLR submission — a reviewer will spot both within minutes of opening the repo. C3 (Bonferroni scope) and C4 (2D-surface framing) should be addressed in the next pass to forestall predictable reviewer pushback. Important items I1–I7 are all defensible with one-paragraph additions; together they would raise the paper from "honest measurement note" to "robust methodological contribution." The minor and nit items are polish-pass material.

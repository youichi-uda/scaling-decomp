# R2 — Three-Persona Adversarial Review (Round 2)

Paper: *Per-Class Scaling-Law Exponents Are Coordinate-Dependent: Two-Axis Decomposition Reproduces Across Four Model Families*
File: `paper/main.tex` (TMLR preprint, 10pt, 1 column, 15 pages)

R1 reference: `reviews/R1_persona.md`. Author's R1 self-review: `reviews/R1_self.md`.

---

## === Persona A R2 — Methodology Expert ===

### R1 follow-up

**R1-Critical-1 (floor–slope confound).** *Acknowledged in §5 Limitations bullet "Floor–slope confound" (main.tex:787–794) and re-flagged inline at §4.1 (main.tex:313–321).* The author has added a "Floor–slope caveat" paragraph at the head of the N-axis result section and a Limitations bullet calling out the asymmetric $E_\mathrm{init}=2.46$ vs $E_\mathrm{cont}=0.79$ ridge. **The profile-likelihood contour I asked for is explicitly listed as "the right follow-up" but not run.**

**R1-Critical-2 (Zipf coupling, $\acont \to 1$).** *Acknowledged in §5 Limitations bullet "$\acont \approx 1.0$ is consistent with Zipf-coupled unigram-LM behaviour" (main.tex:795–808).* The author explicitly cites Michaud's prediction that per-class $\alpha \to 1$ when the class collapses to its unigram-conditioned distribution, names the unigram-LM-subtraction test as the adjudicator, and admits **"We did not run this test and flag it as the most important open question."**

**R1-Critical-3 (per-axis projection inherits joint non-id).** *Acknowledged in §4.2 inline (main.tex:417–421) and in §5 Limitations bullet "Per-axis projection inherits the joint non-identifiability" (main.tex:809–816).* The "$|\dalpha_D|$ grows with $N$" pattern is now explicitly flagged as "consistent with an $N$-axis confound rather than with a genuine $N$-dependent $D$-axis slope." Profile-likelihood test fixing $\alpha_N$ from the 300B fit not run.

**R1-Major-4 (monotonicity stats on n=4).** *Addressed inline §4.2 (main.tex:411–414): "rank-correlated with $N$ across the four tested values ($\rho = +1$ on $n{=}4$, which is suggestive but not decisive — a strict-monotone ordering of 4 points happens by chance with probability $1/12$ under the null)."* This is exactly the calibration I asked for.

**R1-Major-5 (Cerebras-as-undertrained vs coordinate-dependent).** *Addressed §4.4 (main.tex:512–523): "Alternative explanation for the magnitude gap. Cerebras-GPT was trained at Chinchilla-optimal compute … its flatter $L(N)$ and smaller $|\dalpha|$ are also consistent with Cerebras-GPT being on a different point of the $(N, D)$ compute-optimal frontier."* The author now explicitly distinguishes (a) within-family coordinate dependence from (b) across-family compute-budget differences. **Good fix.**

**Minor 6 (grid rail) and 7 (parametric bootstrap)** both addressed: §3.6 (main.tex:280–283) now explicitly states the IID-residual homoscedasticity assumption and acknowledges $n_N=7$ makes it "essentially unverifiable in-sample."

### New Critical

**A-Crit-R2-1: "Moving the three biggest confounds to Limitations" is not equivalent to fitting them.** The paper's central quantitative claim is **"$\dalpha = -0.41$ with 95% CI $[-0.64, -0.22]$"** at $D{=}300\text{B}$ (abstract, Conclusion bullet 2). The author has now told the reader, *in the same paper*: (i) the magnitude is not pinned independently of the per-class floor under the long $(E,\alpha)$ ridge, (ii) the $\acont \approx 1.0$ value is exactly what the Zipf-unigram null predicts, and (iii) the "$|\dalpha_D|$ grows with $N$" Table 1 pattern is consistent with the per-axis projection inheriting the joint non-identifiability. **If all three of those are live, then the magnitude estimate is not a load-bearing claim, and the paper should either retire the magnitude from the headline or run at least one of {profile-likelihood contour, unigram-LM subtraction, shared-$\alpha$ HOFF refit} before being accepted.** "We acknowledge but do not test" is a defensible move for a *secondary* concern; here it covers the headline. The sign claim survives (the shared-$E$ sweep preserves it; the cross-family forest plot is sign-only); the magnitude claim does not.

**Concrete ask:** at minimum, run the **unigram-LM subtraction** sanity check on Pythia 12B + the smallest size. Cost: $0 (already-downloaded checkpoints, log-prob computation is one pass). If $\acont$ drops by more than 0.2 after the subtraction, the headline reframes from "per-class $\alpha$ varies" to "$\Lcont$ is dominated by the unigram floor at large $N$ in our data." That is a different and weaker contribution than the current framing.

**A-Crit-R2-2: The Conclusion section still asserts the magnitude as a finding.** Conclusion bullets (main.tex:857–880) report "$\dalpha_D$ ranges from $-0.20$ to $-0.45$", "$\dalpha = -0.41$ with 95% CI $[-0.64, -0.22]$", and "Per-class … $0.010$–$0.037$ more adjusted $R^2$" without re-stating the confounds. A reader who reads only abstract + introduction + conclusion (which is most readers) will take away the magnitudes as established. **The R1 fixes have hedged §3, §4, and §5 thoroughly but left the Conclusion intact.** Either (i) duplicate one sentence per Limitations bullet into the corresponding Conclusion bullet, or (ii) explicitly demote the magnitude in the Conclusion to "magnitude is reported but the floor–slope and Zipf-coupling confounds are unresolved."

### New Important

**A-Imp-R2-1: The "marginal pre-registered pass" $+0.022$ nats now sits inside a more honest paper, which makes it look worse.** R1 noted this was marginal; the R1 fix correctly added "marginal pre-registered pass" in the abstract (main.tex:55) and Conclusion (main.tex:864). But the paper now also acknowledges the floor–slope confound and the Zipf-coupling issue *in the same family of fits.* The reader's natural next step is: "given the train-fit is $n=4$, the floor is unconstrained, and $\acont \approx 1$ is the Zipf-null, is the $+0.022$ nat margin distinguishable from a per-class floor-difference artifact?" The current text does not address this. Add one sentence: "We did not verify the held-out improvement is invariant to forcing a shared per-class floor at the train-set minimum."

**A-Imp-R2-2: The R1 floor–slope caveat is now in §4.1 but Table 1's $E$ column still carries the asymmetry.** Table 1 (main.tex:402–405) shows $E_\mathrm{init}/E_\mathrm{cont}$ at 1B as $3.49/1.05$, at 12B as $2.36/0.82$. The $E_\mathrm{cont}$ values are well below the smallest observed $L_\mathrm{cont}$ at each $N$ (similarly for $E_\mathrm{init}$). The caveat is verbal; the table reports values that the caveat says shouldn't be trusted as point estimates. Add a footnote: "$E$ values are presented but are not separately identified from $\alpha$ at $n_D = 13$–$14$; see §5 Limitations."

### Final recommendation: **revise** (major revision, accept-with-changes)

The paper is methodologically careful and the R1 hedges are good faith. **But the three confessions in §5 Limitations (floor–slope, Zipf-coupling, per-axis projection) collectively undercut the magnitude claim that the abstract and Conclusion still feature.** The sign claim (negative $\dalpha$ in 4 families, all CIs exclude zero) is robust and is publishable as-is. The magnitude claim ("$-0.41$", "varies by factor of 2") requires at least one of: profile-likelihood contour on $(E,\alpha)$, unigram-LM subtraction, or a shared-$\alpha$ HOFF refit. The unigram-LM subtraction is the cheapest and the most pre-registered-test-shaped of the three. **Without it, accept the paper as a sign-only contribution and retitle.**

---

## === Persona B R2 — TMLR Action Editor ===

### R1 follow-up

**R1-Critical-1 (retitle / reframe).** *Partially addressed.* Title still reads "Per-Class Scaling-Law Exponents Are Coordinate-Dependent." Abstract was softened — "single-axis per-class $\alpha$ varies by a factor of $\sim$2" and "we are not able to fit a joint $L_c(N, D)$ surface" — but the headline framing still asserts a coordinate-dependence claim as the contribution. The "Measurement-validity contributions in scaling" paragraph in Related Work (main.tex:175–181) adds the Schaeffer parallel I suggested; that's a good fix. **The retitle didn't happen and the author's R1 self-review didn't list it as a fix to make.** I'm willing to live with the title.

**R1-Critical-2 (unactionable §6 recommendations).** *Not addressed in the way I suggested.* §6.2 (main.tex:710–720) still has the two recommendations — "report along multiple $(N, D)$ anchors" and "explicitly restrict claims to 'at this $(N, D)$'." No concrete checklist was added. The §6 frame is unchanged.

**R1-Critical-3 (cut §4.6 downstream).** *Not addressed in the way I suggested; partially mitigated.* §4.6 is still there (main.tex:599–675). The author added an explicit "PIQA $+0.010$ is at the adj-$R^2$ adjustment-noise floor" hedge (main.tex:669–675) — that's a concession to my R1 critique without doing the cut. Limitations bullet for §4.6 was added (main.tex:774–777). **The author kept the section in the paper despite acknowledging in the self-review that it is "suggestive, not decisive" at the noise floor.**

**R1-Major-4 (Appendix C as separate note).** *Not addressed.* Appendix C is still in the paper, intact. Abstract was softened ("we also flag two HuggingFace Hub mis-registration bugs") to demote it from "contribution" to "flag." That's the right direction but the appendix itself is unchanged. New Limitations bullet "HF Hub bug locus" (main.tex:833–842) admits the author did not test whether the locus is Hub-side or client-side — Persona C will be unhappy about this (correctly).

**R1-Major-5 (length).** *Worse, not better.* Paper grew from 14 to 15 pages. The R1 additions (Limitations bullets, floor–slope caveat, alternative-explanation paragraph, Zipf-coupling discussion, per-axis projection caveat, Pile-10k subset caveat, tokenizer-partition portability caveat, HF Hub bug locus caveat) are all *correct* additions but they add cumulative length without removing the §4.6 / Appendix C bulk I asked be cut.

**Minors 6 (abstract softening) and 7 (term unification).** Minor 6 was addressed (the marginal pass is now flagged as "marginal pre-registered pass" in the abstract, not given first-position prominence). Minor 7 partial — "coordinate-dependent observable" is now consistently within-family in §6.3 and across-family is renamed "magnitude differences across families" in §4.4. Acceptable.

### New Critical

**B-Crit-R2-1: The R1 additions have over-corrected. The paper now reads as a list of caveats.** Limitations §5 was a 4-bullet list pre-R1; it is now a 12-bullet list (BPE tokenizers, single training distribution, $N$ range limited to 12B, held-out RMSE marginal, per-D CIs wide, joint $L(N,D)$ non-identifiable, downstream suggestive, HF Hub bugs, adversarial-review trail, floor–slope confound, Zipf coupling, per-axis projection, Pile-10k subset size, tokenizer-partition portability, HF Hub bug locus). At 12 bullets, the Limitations section is approaching the length of the Results section. Reviewers comparing the in-text claims to the Limitations list will conclude that the paper itself agrees the load-bearing claims have multiple untested confounds. **This is honest but it is also publication-detrimental at TMLR's bar.** TMLR explicitly evaluates whether claims are supported by evidence; if §5 lists three untested confounds for the headline magnitude claim, the action editor's natural reading is "the author has told me the magnitude isn't pinned down, so I cannot accept the magnitude claim, so the contribution reduces to sign + cross-family replication + the Hub appendix." That is a 5-page TMLR note, not a 15-page paper.

**B-Crit-R2-2: The disagreement with my R1 recommendations is acceptable but should be explicit.** The author kept §4.6 (against my "cut" recommendation), kept Appendix C (against my "extract" recommendation), kept the title (against my "retitle" recommendation), and grew the paper by 1 page (against my "target 6 pages" recommendation). **This is the author's prerogative**, but the response-to-reviewers letter (if there is one) should explicitly acknowledge the disagreement and the rationale. For TMLR I will let these stand as author choices; my R1 was advisory.

### New Important

**B-Imp-R2-1: 15 pages of main text + 4 appendices is long for TMLR but not out of bounds.** TMLR has no hard page limit but the editorial signal is "as long as needed and no longer." 15 pages for what survives a confident reading (sign claim + cross-family replication + the Hub appendix) is approximately 2× too long. I am not going to reject on length alone but I will note it.

**B-Imp-R2-2: The Conclusion now over-promises relative to the Limitations.** Conclusion bullet 5 ("Per-class … 0.010–0.037 more adjusted $R^2$, on all 4 tasks — suggestive of per-class predictive content beyond aggregate") is correct as written but the Limitations bullet ("Downstream task correlation is suggestive, not decisive") says less. **The Conclusion bullet should match the Limitations bullet's language more closely**, e.g., "suggestive at $n{=}7$; the PIQA case is at the adj-$R^2$ noise floor."

### Final recommendation: **accept with minor revision**

Despite my R1 cuts not landing, the paper as it stands is publishable at TMLR. The sign claim is robust, the cross-family replication is well-executed, the HF Hub appendix is genuinely useful, and the methodology is documented in painful detail. The over-hedging is a problem (B-Crit-R2-1) but TMLR's "honest-and-supported" bar is met. **My recommendation is accept with minor revisions:** (i) tighten the Conclusion bullets to match the Limitations language, (ii) commit to either cutting §4.6 or compressing it to one paragraph, (iii) commit to one of the three confound tests Persona A is asking for (the unigram-LM subtraction is cheapest). If the author refuses all three, I still recommend accept — the paper is good enough on the sign claim alone.

---

## === Persona C R2 — Reproducibility / EleutherAI Internals ===

### R1 follow-up

**R1-Critical-1 (Hub-side vs transformers-side bug locus).** *Acknowledged, not tested.* The author added a Limitations bullet "HF Hub bug locus" (main.tex:833–842): *"We did not perform a direct `git clone + git checkout step1000 + sha256sum` of the underlying LFS blob, so we cannot distinguish whether the bug is in the Hub's revision-to-blob mapping or in the `huggingface_hub` client resolution layer."* Abstract was also softened (main.tex:73–77) to acknowledge the resolution-layer uncertainty. **Admission is helpful but it does not resolve my R1 ask.** The R1 ask was: run `git clone + git checkout step1000 + sha256sum` and report the LFS blob OID. Cost: 1 command, ~10 minutes of compute (the 2.8B repo is ~12GB). It would *either* confirm the Hub-side bug (LFS blob OID at step1000 ≠ main) or pin it as a transformers-side bug (LFS blob OID at step1000 = main). **As stated, the appendix gives a debugging recipe (§C.3 verification procedure) but does not establish the bug class.** A reader filing a HF issue based on this appendix will not know whether to file against `huggingface/transformers` or against HF Hub's repository service. That is a reproducibility-actionability gap.

**R1-Critical-2 (step50000 direct LFS fetch).** *Not addressed.* The author still reports the 3.9GB-vs-4.1GB size difference as evidence of truncation. The R1 ask was: report the SHA256 of `pytorch_model-00003-of-00003.bin` obtained via low-level `hf_hub_download` vs high-level `from_pretrained`. **Not done.** Same admission via the Limitations bullet covers this too — but again, the admission is not a fix.

**R1-Critical-3 (bootstrap seed / canonical JSON provenance).** *Partially addressed.* §3.6 now states (main.tex:274–275): *"seed $20260603$ in the released code."* This is good — it tells the reader where the released code's seed lives. **But it does not address my actual R1 concern**, which was that the canonical `final_Naxis_perword.json` reporting CI $[-0.64, -0.22]$ was produced by a *separate* script that does not record its seed in the JSON itself, and a reader running `make_figures.py` will re-bootstrap and may get a different CI. The Limitations section does not mention this. **The R1 fix mentioned the seed for the released code but the canonical JSON is still loaded without provenance.** A reproducibility reviewer who follows the trail — `make_figures.py` → reads `final_Naxis_perword.json` → wants to regenerate it — will not find a single regeneration script with a documented seed.

**Concrete ask:** add 2 sentences to §3.6: (a) "The canonical `final_Naxis_perword.json` was produced by `final_Naxis_perword.py` with seed $20260603$, identical to the seed used in `make_figures.py`." (b) "Running `make_figures.py` on the released measurement JSONs reproduces the reported CI to within $\pm 0.01$." If (b) is not true, that is the actual reproducibility problem.

**R1-Major-4 (Pile-10k subset size).** *Addressed.* New Limitations bullet "Pile-10k subset size" (main.tex:817–822) explicitly admits the 200-doc / 1.23M-token subset is small and the four-family fit was not re-run on a larger corpus. **Good acknowledgment, no test.**

**R1-Major-5 (tokenizer partition portability).** *Partially addressed.* New Limitations bullet "Tokenizer-partition portability" (main.tex:823–832) admits the BLOOM 250k-vocab case is not quantified and that a per-family class-mixture table is the right follow-up. **Good acknowledgment, no test.** The R1 ask was: report per-family (word-initial, word-continuation, other) fractions. The §4.7 (`sec:mixture`) table reports these for Pythia only (47.03% / 22.15% / 30.83%) but not for Cerebras-GPT or BLOOM. The "BLOOM ~halves continuation fraction" claim in R1 Minor M10 also remains unquantified.

**Minor 6 (pre-registration date).** Addressed implicitly in §1.2 (main.tex:114–118) noting the aggregation upgrade and D-axis ladder were added in post-pre-registration adversarial-review rounds.

**Minors 7–8 (JSON consolidation, repo layout).** Not addressed in the paper; presumably the repo README handles this. I cannot verify without checking the repo state, which the R1 task did not ask me to do.

### New Critical

**C-Crit-R2-1: "Admitting we didn't test it" does not address either of my R1 Critical 1 or 2.** Both R1 Criticals were about *evidence that the Hub bug is what the appendix claims it is.* The R1 fix added a Limitations bullet saying "we don't know the locus." That is honest but it converts a *contribution* (the Hub bug report) into a *symptom report* (we saw something weird via the high-level API). **TMLR's reproducibility-track standard for a bug-report appendix is that the bug class is established.** If the appendix had said "we filed HF issue #XXXX; see https://github.com/huggingface/transformers/issues/XXXX for the maintainer response identifying the bug locus," that would close it. As written, the appendix is now a less-confident version of the R1 appendix. The author has admitted weakness without strengthening the contribution.

**Concrete ask:** Either (a) run `git clone https://huggingface.co/EleutherAI/pythia-2.8b && cd pythia-2.8b && git checkout step1000 && git lfs ls-files` and report whether the LFS pointer OID for `model.safetensors` at step1000 equals the OID at main; that test takes <10 minutes and decisively pins the bug class. Or (b) file the issue upstream (EleutherAI and/or HuggingFace) and reference the issue URL in Appendix C. Either fixes my R1 Critical 1 + 2. The admission as written does not.

**C-Crit-R2-2: The canonical-JSON-provenance gap is still open.** As described under R1 follow-up: §3.6 reports the seed used in `make_figures.py` but does not verify that running `make_figures.py` reproduces the canonical CI. If the seed in `final_Naxis_perword.json`'s generating script differs (or is undocumented), the abstract's CI is not reproducible from the released code without manual seed-matching. **This is a 30-second commit-time check (re-run, compare CI to ≤0.01) but the paper does not claim it.** The Limitations should add a bullet "Reproducibility: running `make_figures.py` on released JSONs reproduces the abstract CI to within ±X; canonical JSONs were generated with the same seed."

### New Important

**C-Imp-R2-1: The per-family class-mixture is still unquantified for Cerebras-GPT and BLOOM.** The new Limitations bullet says it would be the right follow-up; that's an acknowledgment, not a fix. **This is cheap — one pandas groupby over already-computed token-level JSONs.** If BLOOM's word-continuation fraction is materially smaller than Pythia's (likely true given the 250k vocab), then the cross-family "$\dalpha$ sign-invariance" claim is comparing estimands measured on differently-sized continuation populations, and that should be quantified.

**C-Imp-R2-2: Wikitext-103 cross-corpus sign check uses $n_N = 5$ including 70M while main fit uses $n_N = 7$ excluding 70M.** Now explicitly addressed inline §4.1 (main.tex:349–354) with the rationale: "the larger Pythia checkpoints ($\geq 2.8\text{B}$) were not measured on Wikitext for compute reasons, and we needed $n_N \geq 5$ to fit a power-law curve at all." Acceptable. Resolves R1 Important I7.

### Final recommendation: **revise** (major revision)

The paper's reproducibility documentation is *better* after R1 than at R1 — the seed is now in §3.6, the marginal pass is consistently flagged, the corpus rationale is explained. But the two R1 Criticals about Hub bug locus and the canonical-JSON provenance are both unresolved. The author chose to admit rather than test. **For the Hub appendix's status as a "contribution" claim (still present in the broader-impact statement: "We also flag two HuggingFace Hub mis-registration bugs"), the bug class needs to be established or the contribution claim needs to drop to "we observed the following symptoms via the standard transformers API."** As written, the appendix is symptom-only and the conclusion still calls it a "bug." That mismatch is the load-bearing reproducibility issue I cannot wave through.

**Required for acceptance:**
1. Either run the `git lfs ls-files` test (10 min, decisive) or file the upstream issue and cite it. Without this, demote "Hub mis-registration bugs" to "observed symptoms" throughout.
2. Add one sentence verifying `make_figures.py` reproduces the canonical CI from the released JSONs. If it doesn't, fix the seed first.

The per-family mixture table (C-Imp-R2-1) would strengthen the cross-family claim but is not blocking.

---

**Total: ~2470 words across three personas.**

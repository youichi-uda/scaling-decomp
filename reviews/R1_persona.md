# R1 — Three-Persona Adversarial Review

Paper: *Per-Class Scaling-Law Exponents Are Coordinate-Dependent: Two-Axis Decomposition Reproduces Across Four Model Families*
File: `paper/main.tex` (TMLR preprint format, 10pt, 1 column)

---

## === Persona A: Methodology Expert (scaling-laws / Hoffmann / Kaplan / Hestness school) ===

### Summary
The paper documents an empirical regularity (sign of Δα between word-initial and word-continuation classes) that is robust across 4 families and 4 N values on the D-axis, but the methodological framing — "coordinate-dependent observable, joint L(N,D) is non-identifiable so we do per-axis fits" — is a confession of weakness dressed as a contribution. The α magnitudes vary not because the per-class problem is coordinate-dependent in any deep sense, but because the per-axis fit absorbs the floor E into α whenever the floor is poorly constrained. The "headline" Δα at 300B is mostly a story about E_init = 2.46 vs E_cont = 0.79, not about α.

### Critical (must address for TMLR accept)

1. **Floor–slope confound is not addressed seriously.** (`main.tex` §3.4 / `final_Naxis_perword.json`)
   With only n_N = 7 points covering ~1.9 decades (160M → 12B), L = E + A·N^(-α) is notoriously under-determined: you can trade off E against α along a long ridge. The fitted E_init = 2.46 is suspiciously close to the largest model's L_init value itself, which means α_init is being driven almost entirely by the *initial-side curvature* with the floor latched near the data minimum. Conversely E_cont = 0.79 is well below the smallest L_cont in the data, so α_cont is constrained by the asymptote. The "Δα = −0.41" headline is therefore not comparing slopes at matched parameterization; it is comparing one slope with a saturating floor to another slope with a free floor. The §4.1 "shared-E robustness" check (sweep E ∈ [0, E_cont] and verify sign) is too weak — it only checks sign, not whether the *magnitude* is an artifact of the floor latching. **Fix:** report the full (E, α) likelihood profile per class as a 2D contour, plus a shared-(E, α) test with per-class A only. If Δα survives with shared α as a Welch-like test under matched E parameterization, say so quantitatively. Otherwise reframe the headline as "L_init and L_cont have different trajectories" without the α-decomposition language.

2. **α_cont ≈ 1.0 is on the boundary of "trivial Zipf coupling."** (`main.tex` §4.1, line 290–292)
   Michaud 2023's quantization model predicts α ∝ Zipf-slope, with α → 1 as the cluster frequency approaches the unigram floor. α_cont = 0.98 at D = 300B and α_cont = 0.93–0.96 at large N on the D-axis are *exactly* what one would predict if continuation tokens are essentially being modeled by their unigram-conditioned distribution (orthographic completion of a known word). The paper notes this verbally in §6.1 ("orthographic and morphological structure that data exposure captures efficiently") but does not test it. **Fix:** add an explicit comparison against a per-class unigram-LM baseline. If subtracting the unigram log-prob from L_cont collapses α_cont toward α_init, then the Δα signal IS the Zipf coupling Michaud already named, and the "novel" partition reduces to a renamed slice of Michaud's frequency partition.

3. **The "joint L(N,D) is non-identifiable, so per-axis fits are the methodology" framing is backwards.** (`main.tex` §3.6 line 263–268, Appendix B)
   Appendix B reports that all 5 functional forms (ADD, MULT, HOFF, COMPUTE, MIN) fail to identify on Pythia's (N, D) range, and the paper concludes "per-axis fits are the consistent decompositions we can support." But the per-axis fits are *projections* of whichever joint surface is true — the projection inherits the identifiability problem, it doesn't escape it. If a Hoffmann-style L = E + A·N^(-α_N) + B·D^(-α_D) cannot be fit, then the per-N D-slice α_D(N) is not estimating α_D of the joint surface; it is estimating an effective exponent that mixes A·N^(-α_N) into E. The "monotonic increase of |Δα_D| with N" reported in Table 1 is consistent with exactly this confound: as N grows, A·N^(-α_N) shrinks, so more of the curvature on the D-axis becomes "real" α_D rather than residual N-axis curvature absorbed into E. **Fix:** either drop the "Δα_D grows with N" finding (which is mostly an N-axis artifact), or run a profile-likelihood test fixing α_N from the 300B fit and re-fitting α_D per class; the residual gap is the genuinely-D-axis component.

### Major (must address for higher-impact reroute)

4. **Only 4 N points on the D-axis monotonic claim.** (Table 1, line 359–367)
   −0.20, −0.25, −0.37, −0.45 across {1B, 1.4B, 6.9B, 12B}. Bonferroni m=4 controls the per-N CI but says nothing about the monotonicity-vs-N pattern. With 4 ordered points, a strict-monotone test (Page's L or Spearman) is the right adjudicator and would have been pre-registered if monotonicity were a real claim; instead the paper just eyeballs it ("roughly doubles from 1B to 12B"). **Fix:** report the rank-correlation p-value, or restate as "all 4 are negative" and drop the monotonicity language. Note Table 1 in main.tex gives 6.9B as −0.37 with CI [−0.50, −0.24], while final_perN_Daxis.json gives −0.365 with the same CI — the rounding is fine but the difference between −0.37 and −0.20 across 6× N change is well within the floor-confound uncertainty raised in finding (1).

5. **Cerebras α magnitudes (0.21 / 0.41) vs Pythia (0.56 / 0.98) — Cerebras is undertrained, not "coordinate-dependent."** (§4.4, line 449–452)
   The paper presents Cerebras-GPT's flatter exponents as part of the "coordinate dependence" story. But Cerebras-GPT was trained at Chinchilla-optimal compute (≈20 tokens/parameter), which is roughly 5× less data than Pythia at 12B. The flatter L(N) curves and smaller |α| are exactly what Hoffmann 2022 would predict for an undertrained family. Calling this "coordinate-dependent" rather than "the family is on a different point of the (N, D) Pareto frontier" misuses the language. **Fix:** rephrase §4.4 and §6.3 to distinguish (a) genuine coordinate-dependence of an estimand (the gap measured at different (N, D) of *the same family*) from (b) trivially-different-α-between-families because one family is undertrained. The Pythia D-axis Table 1 result is (a); the Pythia-vs-Cerebras comparison in Table 2 is (b) and is much less interesting.

### Minor (polish)

6. **Grid search α ∈ [0.02, 1.8] with 800 points** (`make_figures.py:71`, `fit_scaling`) — the 0.02 lower bound is hit at step 50k (Table 2, "α = 0.02 grid min"), confirming the railing. State the rail explicitly in caption.

7. **Per-axis CI reported as parametric residual bootstrap** (§3.5) but residuals are 7 points; the parametric assumption (i.i.d. residuals on a power-law fit at n=7) is heroic. Report the empirical-distribution check or switch to a leave-one-out CI for honesty.

---

## === Persona B: TMLR Action Editor ===

### Summary
This is a methodologically careful, honestly-reported negative-leaning study with a useful HF Hub bug discovery embedded. The actual paper-worthy contribution is two-thirds of a TMLR note: (i) the D-axis decisive result + cross-family sign replication, and (iii) the HF Hub bug appendix. The "coordinate-dependent" framing in the title is a marketing overreach for what the data shows. The downstream §4.6 should be cut. At 10 pages it is too long for what survives an honest scrub; a 5–6 page TMLR note would land better.

### Critical (must address for TMLR accept)

1. **Is the headline "coordinate-dependent" novel, or is it a restating of "scaling laws depend on (N, D) regime"?** (Title, abstract)
   Every working scaling-laws practitioner already believes per-class α is regime-dependent — that's why Chinchilla had to refit. The novel finding here is *quantitative*: within Pythia, the same Δα estimand varies by 2× across (N, D) anchors. The paper buries this under "coordinate-dependent observable" language that sounds either trivial (to scaling-laws people) or grandiose (to outsiders who may think a deeper claim is being made). **Fix:** retitle to something like *"Per-Class Scaling-Law Exponents Vary by 2× Across (N, D) Measurement Anchors: A Cautionary Study on Pythia and 3 Replications,"* and frame the contribution as a measurement-validity audit, parallel to Schaeffer 2023's emergence-vs-metric paper that §2 already cites.

2. **The two §6 recommendations are unactionable.** (`main.tex` §6 line 755–764)
   "Report along both axes with multiple (N, D) anchors" requires a dense intermediate-checkpoint suite at the family being studied — which only Pythia provides for open decoder-only LMs. "Or explicitly restrict claims to 'at this (N, D)'" is just "be honest about what you measured" — nobody disagrees with that, but it isn't a method. **Fix:** convert §6 into a concrete checklist: (a) report which N-anchor you fit at, (b) report your D-anchor, (c) report a sensitivity sweep across one neighboring anchor if possible. Or drop §6 to a single paragraph.

3. **Downstream §4.6 should be cut.** (`main.tex` §4.6 line 541–610)
   The author writes "we interpret this as suggestive of per-class predictive content beyond aggregate, but underpowered at n=7 within a single family; a multi-family / multi-task panel would be needed to confirm." This is correct and honest, but if it's underpowered and won't change a reader's prior, why is it in the paper? Adjusted R² gains of 0.010–0.037 with 4 task points and n=7 — with the dependent variable being the same Pythia loss that already entered the per-class fit — has no inferential weight. It is the kind of section that invites reviewers to attack the strongest part of the paper (the D-axis result) on the grounds that the weak section reveals the authors' priors. **Fix:** delete §4.6 and Figure 4, mention in one sentence in §6 "preliminary downstream-correlation analysis is in the repo."

### Major (must address for higher-impact reroute)

4. **HF Hub Appendix C could be its own EleutherAI bug report / GitHub issue + a 1-paragraph reference in the paper.** (`main.tex` Appendix C, line 884–974)
   The SHA256 forensics on Pythia-2.8B (all early steps share the `main` safetensors hash `ab496f1c3fd79e3c`) is a great piece of reproducibility-infrastructure work, and probably more useful to the community as a HuggingFace Hub issue thread or an EleutherAI mailing-list post than as Appendix C of a scaling-laws paper. Including it gives this paper a split personality. **Fix:** move the full appendix to a companion technical note on the repo, file the bug with HF/EleutherAI with a permalink, and keep one paragraph in the main paper that says "we found and documented these bugs; see [URL]."

5. **Length: 10 pages of main content for what reduces to "Δα is consistently negative, magnitude depends on (N, D), here's the D-axis table."**
   A TMLR-style note at ~5–6 pages would be more impactful and easier to accept. The Related Work, Method, and Discussion are individually fine but together inflate the paper without adding evidence. **Fix:** target 6 pages main + appendices.

### Minor (polish)

6. **Abstract reports "marginal pre-registered pass" of held-out RMSE (+0.022 nats) prominently** (line 54–55) — this is honest but also the weakest evidence in the paper; it should not be in the first sentence-cluster. Move it to the §4.1 detail.

7. **"Coordinate-dependent observable" is used inconsistently** — sometimes meaning "varies with (N, D)," sometimes "varies with family." These are different things (Persona A finding 5). Unify the term.

---

## === Persona C: Reproducibility Reviewer (EleutherAI / HF internals) ===

### Summary
The reproducibility claims are mostly well-supported by released JSON and code, but several load-bearing details are under-documented. The HF Hub bug write-up (Appendix C) is genuinely useful but does not actually demonstrate that the bug is *Hub-side* rather than *transformers-side*; the authors appear to have only tried the high-level `AutoModelForCausalLM.from_pretrained` API. The 4-family claim leans on Pile-10k (200 docs, ~1.23M tokens), which is small enough that the cross-family magnitudes could be Pile-10k-subset artifacts. The tokenizer-partition definition is described only for GPT-NeoX BPE; whether the same partition is applied identically to GPT-2 BPE and BLOOM 250k BPE is not shown in the paper or the released `measure.py`.

### Critical (must address for TMLR accept)

1. **HF Hub bug: is it Hub-side, or transformers-side?** (`main.tex` Appendix C.1, line 891–929)
   The forensics show that `model.safetensors` at revisions step1000…step10000 of Pythia-2.8B all have SHA256 `ab496f1c3fd79e3c`, identical to `main`. The paper concludes "the standard `from_pretrained(..., revision="step1000")` API silently returns the `main`-weights model because `transformers` prefers safetensors, and the safetensors file at step1000 is byte-identical to main's." That diagnosis is plausible but the paper does not show evidence the authors ran `git lfs fetch --all` directly against the Hub git repo and SHA-checked the *raw* LFS blob. If `huggingface_hub` is downloading the safetensors file from a CDN-cached pointer that resolves to `main`'s blob, the bug could be in `huggingface_hub`'s revision-resolution logic, not in the Hub's actual file storage. **Fix:** add a `git clone https://huggingface.co/EleutherAI/pythia-2.8b && git checkout step1000 && sha256sum model.safetensors` invocation in the verification procedure (Appendix C.3), and report whether the LFS pointer at that ref resolves to a distinct blob OID. Right now the bug report cannot be replicated by a reader who only has access to the Hub UI; they will read your hashes, believe them, and not be able to verify.

2. **"step50000 of 12B is unrecoverable" — did you try direct LFS fetch?** (Appendix C.2, line 930–954)
   The shard-3 truncation (3.9GB vs 4.1GB) is documented well. But the paper says "Loading the step50000 model and running forward passes produces NaN losses" and "`use_safetensors=False` does not fix this." That's consistent with a truncated blob on the Hub, but if the underlying LFS blob is intact and `transformers` / `huggingface_hub` is fetching a partial copy from a stale CDN, a direct `huggingface-cli download EleutherAI/pythia-12b --revision step50000 pytorch_model-00003-of-00003.bin` could yield a different file. **Fix:** report the SHA256 of the file obtained by direct `hf_hub_download` (low-level API) vs `from_pretrained` (high-level API) at step50000. If they differ, the bug is in the resolution layer; if they're identical and both truncated, it really is Hub blob corruption.

3. **Bootstrap seed and per-family CI determinism.** (`make_figures.py:100` `bootstrap_diff(... seed=20260603)`)
   The seed `20260603` is hard-coded in `make_figures.py` and used for the 4-family forest plot. Good. But the canonical N-axis Δα CI [−0.64, −0.22] in `final_Naxis_perword.json` was produced by a separate script (the file is loaded but not regenerated by `make_figures.py`), and the seed used to produce *that* CI is not documented in the JSON. A reader running `make_figures.py` will re-bootstrap Pythia inside `fig3_families` and get a CI that may differ from the canonical [−0.64, −0.22] reported in the abstract, depending on float-order issues across numpy versions. **Fix:** (a) regenerate `final_Naxis_perword.json` from a script in the released repo and record the seed in the JSON itself; (b) verify `make_figures.py`'s `fig3` Pythia CI matches the canonical one to ≤0.01.

### Major (must address for higher-impact reroute)

4. **Pile-10k is 200 docs / ~1.23M predicted tokens per model.** (`main.tex` §3.3, line 220–224)
   That is small enough that the 4-family cross-replication could be Pile-10k-subset-specific. BLOOM in particular was trained on ROOTS (multilingual), so its eval on a 200-doc English Pile sample is doubly exposed: small sample × out-of-distribution. The paper does note this in §4.4 ("the smaller magnitudes for Cerebras and BLOOM reflect flatter L(N) trajectories and, for BLOOM, the larger vocabulary reducing the multi-token-word sample within the eval corpus") but does not test sensitivity. **Fix:** rerun the 4-family Δα fit on at least one of {full Pile validation subset, C4 validation, the BLOOM ROOTS validation set if released} and report whether the sign + magnitude survive. Wikitext-103 is used for Pythia sign-check but not for the cross-family comparison; extend it.

5. **Tokenizer partition: does Ġ semantics carry across to Cerebras and BLOOM?** (`main.tex` §3.1, line 188–191; §6.4 limitations)
   The paper defines word-initial via "the GPT-NeoX BPE space marker Ġ." That works for GPT-NeoX BPE and for GPT-2 BPE (same convention). For BLOOM's 250k-vocab BPE, the byte-level encoding is similar but the vocabulary contains many whole-word tokens (no continuation needed), which is exactly why §4.4 notes "the larger vocabulary reducing the multi-token-word sample." That observation should be quantified: what fraction of predicted tokens in Pile-10k are classified as word-initial / word-continuation / other under each tokenizer? If BLOOM's word-continuation fraction is, say, 5% vs Pythia's 22%, then the BLOOM Δα = −0.13 is a *different estimand* than Pythia's Δα = −0.41, and the "cross-family sign reproduction" claim is comparing apples to small apples. **Fix:** add a per-family table of (word-initial fraction, word-continuation fraction, other fraction) in §4.4 or Appendix; if BLOOM's continuation fraction is materially smaller, qualify the cross-family claim.

### Minor (polish)

6. **Pre-registration document is dated 2026-06-01.** (`PREREGISTRATION.md` line 3) That's two days before the paper's `\year{2026}\month{06}`. The pre-registration ADDENDUM section (line 107–142) substantially revised the hypotheses (introducing the held-out RMSE > 0.02 nat threshold, the M_shared vs M_perclass comparison). Note in §1.2 whether the ADDENDUM was committed before *any* per-class fits had been run on ≥2.8B models, as it claims. The repository's git history should show this.

7. **Released JSON files** (`final_Naxis_perword.json`, `final_perN_Daxis.json`) contain raw fit results but not the input loss values per checkpoint. A reader who wants to refit with a different α-grid or a different bootstrap procedure has to also fetch all `v2mt_*.json` files. Document this in the README and consider releasing a single consolidated `losses_per_(N,D,class).csv`.

8. **`measure_v2.py` and the multiple `v2mt_*.json` checkpoints** — there are ~50 such files in the repo root. A `data/` subdirectory with a manifest would make the release substantially more navigable for reproduction. The current layout works but invites the reviewer to wonder which file was actually used for which table.

---

**Total: ~2350 words across three personas.**

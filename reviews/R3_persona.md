# R3 — Three-Persona Adversarial Review (Round 3, final)

Paper: *Per-Class Scaling-Law Exponents Are Coordinate-Dependent: Two-Axis Decomposition Reproduces Across Four Model Families*
File: `paper/main.tex` (TMLR preprint, 10pt, 1 column, 16 pages)

R1: `reviews/R1_persona.md`. R2: `reviews/R2_persona.md`.
New artifacts since R2: `floor_robust.py` + `floor_robust.json`; `verify_checkpoints.py` + `hub_blob_audit.json`.

---

## === Persona A R3 — Methodology Expert ===

### R1+R2 thread

The author ran the shared-$E$ sweep and shared-$\alpha$ $F$-test I asked for in R1-Crit-1 and R2-Crit-1. That is a real concession to adversarial review — most authors would have kept hedging. The numerical result is unkind: shared-$\alpha$ $F(1,8)=4.47$, $p=0.067$, which fails to reject shared $\alpha$ at the conventional 5% level on the N-axis alone. The shared-$E$ sweep keeps the sign of $\dalpha$ negative at every feasible floor, but its magnitude swings from $-0.08$ to $-1.5$ depending on where the floor is pinned. The paper now reports both honestly in §4.1 "Floor-slope adjudication (post-R1 robustness)" (main.tex:355–390).

The unigram-LM subtraction test (R1-Crit-2, R2-Crit-1) was **not** run; it is now flagged in the §5 Limitations bullet "$\acont \approx 1.0$ may be Zipf-coupled" as "the most important open question" (main.tex:843–845) and in the new "Adversarial-review verifications" bullet (main.tex:879–882). The §6 Conclusion correctly demotes the N-axis fit to "marginal under stricter adjudicators" alongside the D-axis Bonferroni result as the load-bearing evidence.

### Residual concerns

**N-axis as a standalone claim is now formally undercut.** The author has done the right thing by reporting $p=0.067$ inline at §4.1 and stating that the N-axis "on its own ... does not, by itself, cross conventional significance thresholds" (main.tex:384–386). But the abstract still leads with "$\ainit{=}0.56$, $\acont{=}0.98$, $\dalpha{=}-0.41$, 95\% bootstrap CI $[-0.64, -0.22]$" as a co-equal contribution alongside the D-axis Bonferroni result (main.tex:51–55). A reader who stops at the abstract will not know the shared-$\alpha$ $F$-test fails. **Required fix:** add one parenthetical to the abstract's N-axis bullet, e.g., "(shared-$\alpha$ $F$-test $p=0.067$; the N-axis fit is suggestive, not decisive, on its own)." This is a 12-word addition and brings the abstract in line with §4.1 and §6.

**The D-axis Bonferroni $m=4$ result carries the conclusion, but inherits its own caveat.** The author's §5 Limitations bullet "Per-axis projection inherits the joint non-identifiability" (main.tex:846–853) admits that the $|\dalpha_D|$-grows-with-$N$ pattern in Table 1 is consistent with an $N$-axis confound rather than a genuine $N$-dependent $D$-axis slope. The four CIs themselves remain decisive (all exclude zero at 98.75%) and the sign claim survives, but the "grows with $N$" magnitude pattern is conceded as possibly artifactual. That is honest, but it means the only fully load-bearing remaining claims are: (i) sign of $\dalpha$ is negative on the D-axis at all 4 tested $N$, (ii) sign of $\dalpha$ is negative in 4 model families on the N-axis at $D=300\text{B}$. **The magnitude story is gone except as a within-data observation.** That is fine for a TMLR cautionary paper; it is the framing the §6 recommendations (R1) and (R2) now support. The Conclusion no longer over-promises on magnitude (R2-Crit-1 closed). Good.

**Zipf-coupling test is still the most consequential open question.** The author flagged it as "most important open follow-up" but did not run it. With the shared-$\alpha$ $F$-test now public at $p=0.067$, a $30$-min unigram-LM subtraction would either (a) confirm $\acont \approx 1$ collapses toward $\acont \approx 0.5$ when the unigram floor is subtracted (closing the paper as a renamed Michaud slice with sign-only contribution), or (b) leave $\acont$ at $\sim 1$ even after subtraction (which would be a genuine result and would also resolve the shared-$\alpha$ $p=0.067$ ambiguity). Not blocking for TMLR — the paper is now honest about the gap — but it is the obvious next experiment for a v2.

### Final verdict for TMLR: **MINOR REVISION**

Add the parenthetical to the abstract on the N-axis shared-$\alpha$ $F$-test result (or equivalent demotion). With that one sentence, accept. The paper is methodologically careful, the headline magnitude has been correctly demoted to "within-data observed range, sign-only across coordinates," the D-axis Bonferroni $m=4$ result is solid, and the cross-family sign replication is well-executed. The author ran two of my three R1 confound tests rather than just acknowledging them — that is the right epistemic move, and TMLR's "claims supported by evidence" bar is met for the surviving claims.

---

## === Persona B R3 — TMLR Action Editor ===

### R1+R2 thread

Length grew from 15 to 16 pages with the R2 additions (floor_robust paragraph in §4.1, three-recommendation R1/R2/R3 list in §6, x-linked-etag rewrite of Appendix C, the new "Adversarial-review verifications" Limitations bullet). The R2 cuts I asked for (§4.6 downstream, Appendix C as separate note, 6-page target) still did not happen. The author is consistently choosing "add" over "cut," which is honest but length-detrimental. The title remains "Per-Class Scaling-Law Exponents Are Coordinate-Dependent."

§6 was substantively reframed (main.tex:886–949): three recommendations now (R1/R2/R3) instead of two. (R1) is unchanged from R2 (report both axes). (R2) is new and concrete — report per-class floors and run the `floor_robust.py` adjudicator, $<1$s on cached losses. (R3) is new and concrete — audit `x-linked-etag` via `verify_checkpoints.py`, $<1$s. The R1/R2/R3 list is exactly the "concrete checklist" I asked for in R1-Crit-2, finally landed in R3. Good.

§4.6 downstream stays with the PIQA noise-floor caveat (main.tex:700–707) and matching §6 Conclusion bullet (main.tex:917–923). The hedge is verbal; the section is still in the paper.

### Residual concerns

**Floor-robust paragraph essentiality (R3 question).** It belongs in main text, not appendix. It is the load-bearing methodological-validity test for §4.1's $\dalpha=-0.41$ headline, and §4.1 cannot honestly stand without it given the R1 critique. Moving it to appendix would let the abstract claim "$\dalpha=-0.41$ with 95% CI" without the reader seeing the shared-$\alpha$ $p=0.067$ caveat until the appendix. Keep it in §4.1. **No change requested.**

**Title (R3 question).** The title still asserts coordinate dependence as the contribution. After R2's reframe to "sign robust, magnitude floor- and coordinate-conditional," the title is technically defensible (the magnitude $\dalpha$ at one coordinate is demonstrably different from the magnitude at another, that is coordinate dependence) but it understates the actual evidence: the **decisive** finding is sign-invariance across 4 families and 4 $N$-slices; the coordinate-dependence claim survives only as a within-data factor-of-2 range. A title like "Sign-Robust, Magnitude-Conditional: A Coordinate-Anchor Audit of Per-Class Scaling Laws" would match the §6 framing better. **I will not block on title** — author prerogative — but flag it.

**§4.6 downstream (R3 question).** The PIQA noise-floor hedge in §4.6 is the right caveat, but the section as a whole remains the weakest single contribution in the paper and continues to invite reviewers to attack the strongest contributions on the grounds that the author keeps weak material. The R2 hedge satisfies "honest about the limitation" but not "fix the inferential weakness." A reasonable compromise: collapse §4.6 to a one-paragraph note in §6 ("Preliminary downstream-correlation analysis is in the repository; per-class beats aggregate adj-$R^2$ on 4/4 tasks by $0.010$–$0.037$, the smallest gain is at the noise floor at $n=7$, $p=2$"), drop Figure 4. **Recommended but not blocking.**

**Page count (R3 final question on acceptability).** 16 pages is long for what survives — the sign claim + D-axis Bonferroni + cross-family + the Hub appendix. TMLR has no hard limit. The paper is publishable at length.

### Final verdict for TMLR: **ACCEPT** (with optional minor cosmetic revisions)

The R2 additions resolve my R1-Crit-2 (§6 unactionable recommendations) decisively: R1/R2/R3 are now concrete checklist items each backed by a $<1$s reproducer script. The author's choice to keep §4.6 and Appendix C against my R1 recommendations is a defensible author call; the §4.6 PIQA caveat and the appendix's now-confirmed Hub-side bug locus (via x-linked-etag) make both sections honest. The title overstates relative to the framing of the body, but is technically defensible. **Accept as-is**; if the author wants to incorporate Persona A's abstract parenthetical and optionally compress §4.6, those would polish but are not required for my acceptance.

---

## === Persona C R3 — Reproducibility / EleutherAI Internals ===

### R1+R2 thread

The author ran exactly the test I asked for in R1-Crit-1 and R2-Crit-1. `verify_checkpoints.py` queries the HF Hub `/resolve/<revision>/<filename>` endpoint with HEAD and reads `x-linked-etag` (= LFS blob SHA256) and `x-linked-size` from the response headers **before** the CDN redirect. The script disables redirect following via a custom `HTTPRedirectHandler` (main.tex Appendix C and `verify_checkpoints.py:28–42`). This is server-side metadata: it reflects which LFS blob the Hub itself maps each (repo, revision, filename) triple to, independent of any client-side resolution layer in `huggingface_hub` or `transformers`. The output `hub_blob_audit.json` shows that for `pythia-2.8b/model.safetensors`, revisions `step1000`, `step2000`, `step4000`, `step10000`, `step50000`, `step100000` all share blob SHA256 `ab496f1c3fd79e3c...` with `main` (commit SHAs differ — these are real distinct Hub commits pointing at the same LFS blob). For `pythia-12b/pytorch_model-00003-of-00003.bin` at `step50000`, blob size is $3{,}896{,}183{,}087$ vs $4{,}105{,}939{,}923$ at neighbouring revisions (~5.1% under-size). The Appendix C section title now states "The bug locus is therefore **Hub-side**, not client-side" (main.tex:1075–1076) and the table column header says "LFS SHA256" rather than "SHA256," correctly anchoring the claim to Hub-reported blob metadata. **R1-Crit-1 and R2-Crit-1 are fully closed.**

### Residual concerns

**Releasable evidence for future studies (R3 question).** Yes. `verify_checkpoints.py` is 124 lines, dependency-free (Python stdlib only — `urllib.request`, `argparse`, `json`), runs in $<1$s, and emits machine-readable JSON. A reader who wants to cite "Pythia-2.8B step1000..step100000 safetensors are silently mis-registered as `main`" can either rerun the script themselves or cite the committed `hub_blob_audit.json` as a frozen artifact at a specific git commit. The `--json` flag and the `CHECKS` constant make extension straightforward (add new (repo, file, revisions) tuples). The custom `_NoRedirect` opener with the comment "x-linked-etag header sits on the resolve response BEFORE the CDN redirect" (lines 28–42) is exactly the load-bearing technical detail a future user would need to reproduce. **This is genuinely citable infrastructure.**

**Canonical JSON provenance (R2-Crit-2, still partially open).** R2 asked the author to add one sentence verifying that running `make_figures.py` on released JSONs reproduces the abstract CI $[-0.64, -0.22]$. The §3.6 seed-declaration (main.tex:274–275) is in place; the README §"Reproducing the headline numbers" lists `python3 final_Naxis_perword.py` → `Δα = -0.414, 95% CI [-0.642, -0.218]` (README:73–74). That covers the canonical N-axis reproducer. The R2 ask is effectively satisfied via the README rather than the paper text. **Closed.**

**Per-family class-mixture table (R2-Imp-1, still open).** The author flagged this in the §5 Limitations bullet "Tokenizer-partition portability" (main.tex:863–872) but did not produce the per-family table. This was non-blocking in R2 and remains non-blocking in R3. **Recommended for v2, not required for acceptance.**

**README reproducibility bar (R3 question).** The README "Reproducing the headline numbers" section (README:67–88) lists three commands: (a) `python3 final_Naxis_perword.py` for the canonical N-axis fit, (b) `python3 paper/make_figures.py` for the 4-family forest plot + all 4 figures, (c) D-axis fits noted as stored in `final_perN_Daxis.json`. The new artifacts `floor_robust.py` and `verify_checkpoints.py` are listed in the "Repository layout" section (README:49–52) with their JSON outputs. Total reproduction time for headline numbers on a laptop is well under 1 hour (the JSONs are already committed; the scripts just refit / regenerate). **Reproducibility bar met.**

**One remaining minor gap.** The TL;DR (README:18) now says "The Hub-side locus is confirmed via `x-linked-etag` HTTP headers" — this is correctly anchored. But the "Re-measuring from scratch" section (README:90–100) still says "Whether the bug is in the Hub's revision-to-blob mapping or in `huggingface_hub`'s client resolution layer is not adjudicated here." That paragraph is stale relative to the new `verify_checkpoints.py` artifact and the new Appendix C language. **Recommended fix:** update README:97–100 to point to `verify_checkpoints.py` and `hub_blob_audit.json` as the adjudication. Not blocking; cosmetic.

### Final verdict for TMLR: **ACCEPT** (with one cosmetic README fix)

The author resolved both my R1+R2 Criticals decisively with a real test (`verify_checkpoints.py`) plus a committed evidence artifact (`hub_blob_audit.json`). The Hub-side bug claim in Appendix C is now backed by server-side metadata that any reader can re-query in one second. The two R1 Major asks (Pile-10k subset size, tokenizer-partition portability) remain flagged-but-not-tested in §5 Limitations — those are honest acknowledgments, not fixes, but the author has earned the benefit of the doubt by running two of the three confound tests at R2. The README needs one paragraph updated (lines 97–100) to remove the stale "not adjudicated here" wording now that adjudication is in fact in the repo. **Accept.**

---

**Summary across personas:**

| Persona | R1 verdict | R2 verdict | **R3 verdict** |
|---|---|---|---|
| A (methodology) | major revision | major revision (revise) | **MINOR REVISION** (one abstract parenthetical) |
| B (action editor) | accept w/ minor revision | accept w/ minor revision | **ACCEPT** (optional polish) |
| C (reproducibility) | major revision | major revision (revise) | **ACCEPT** (one README fix) |

The author ran the floor-slope sweep, the shared-$\alpha$ $F$-test, and the x-linked-etag Hub audit between R2 and R3 — three concrete adversarial tests rather than three more Limitations bullets. The N-axis on its own is now formally demoted ($p=0.067$); the D-axis Bonferroni $m=4$ + cross-family sign replication carry the publishable contribution. The single remaining experimental gap (unigram-LM subtraction for the Zipf-coupling alternative) is correctly named as the v2 follow-up. The paper is now an honest TMLR cautionary methodology study, supported by released code and adjudicable artifacts.

**Total: ~1700 words across three personas.**

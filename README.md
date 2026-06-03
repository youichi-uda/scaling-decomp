# Per-class scaling-law exponents are coordinate-dependent

Code, raw measurements, pre-registration, and analysis pipeline for the paper:

> **Per-Class Scaling-Law Exponents Are Coordinate-Dependent: Two-Axis Decomposition Reproduces Across Four Model Families**
> Yoichi Uda (Independent Researcher), 2026.
> Preprint: arXiv:TBD.

## TL;DR

For the word-initial vs word-continuation per-class scaling-law exponent gap (Δα):

- **D-axis at fixed N** (4 Pythia sizes, 13–14 dense checkpoints each, Bonferroni m=4): Δα<sub>D</sub> ∈ {−0.20, −0.25, −0.37, −0.45}, all 98.75 % CIs exclude zero; magnitude grows with N.
- **N-axis at D=300 B** (7 Pythia sizes): Δα = −0.41, 95 % CI [−0.64, −0.22]; held-out RMSE improvement +0.022 nats (marginal pre-registered pass).
- **Cross-family at D=300 B** (Pythia, Pythia-deduped, Cerebras-GPT, BLOOM): all four families Δα<0 with CIs excluding zero; magnitudes span ~3× (−0.13 to −0.41) but the sign is invariant.
- The per-class α gap is therefore a **coordinate-dependent observable**, not an intrinsic property of the per-class prediction problem.

Plus two **HuggingFace Hub mis-registration bugs** affecting Pythia-2.8B intermediate-step revisions and Pythia-12B step50000 (Appendix C).

## Repository layout

```
paper/
  main.tex                   TMLR/arXiv submission source (preprint option)
  main.pdf                   compiled PDF (14 pages, 4 figures)
  references.bib             16-entry bibliography
  figs/                      4 PDF figures (regenerable from make_figures.py)
  make_figures.py            generates fig1-4 from JSON measurements
  tmlr.{sty,bst}             TMLR style files
  arxiv_submission.tar.gz    self-contained arXiv upload bundle
PAPER_DRAFT.md               original markdown draft (long-form notes)
PREREGISTRATION.md           pre-registered N-axis criteria (4/4) + addenda
RESULTS.md                   numerical summary

measure_v2.py                per-class loss measurement (per-word matched aggregation)
analyze_v2.py                N-axis bootstrap (D=300B)
analyze_daxis.py             single-N D-axis fit
analyze_multiN_daxis.py      D-axis fits across {1B, 1.4B, 6.9B, 12B}
analyze_perD_Nfit.py         per-D N-axis fits (identifiable only at D=300B)
analyze_joint_forms.py       5-form joint L(N,D) survey (non-identifiable on Pythia)
analyze_joint_ND.py          ADD-form joint fit prototype
analyze_families.py          cross-family per-class α fit (Pythia/Cerebras/OPT/GPT-2)
bootstrap_ci.py              parametric residual bootstrap
bootstrap_joint_ND.py        bootstrap for the joint ADD form
final_Naxis_perword.py       canonical N-axis fit + held-out adjudication
final_perN_Daxis.py          canonical D-axis fits per N (this script's output is
                              final_perN_Daxis.json + final_perD_Nfit_bootstrap.json)

v2mt_<size>.json             raw per-class per-decile measurements (Pythia 70m-12b)
v2mt_<size>d.json            Pythia-deduped
v2mt_<size>_step<S>.json     intermediate-checkpoint measurements (D-axis ladder)
v2mt_cerebras-<size>.json    Cerebras-GPT
v2mt_bloom-<size>.json       BLOOM
final_Naxis_perword.json     canonical N-axis result (alpha, E, A, CI, held-out)
final_perN_Daxis.json        per-N D-axis fits with Bonferroni CIs
final_perD_Nfit_bootstrap.json  per-D N-axis bootstrap (shows non-identifiability)
multiN_daxis_dense.json      dense D-axis ladder data
joint_ND_fit.json            joint ADD form attempt (non-identifiable on Pythia)
lmeval_EleutherAI_pythia-*/  downstream task evaluations (lambada/piqa/arc_easy/sciq)
```

## Reproducing the headline numbers

Requires Python 3.10+ with `numpy`, `scipy`, `matplotlib`.

```bash
# Canonical N-axis fit (Pythia 7 sizes, per-word matched, D=300B)
python3 final_Naxis_perword.py
# → Δα = -0.414, 95 % CI [-0.642, -0.218], held-out improvement +0.022 nats

# Cross-family replication (Pythia / Pythia-deduped / Cerebras-GPT / BLOOM)
python3 paper/make_figures.py
# → writes paper/figs/fig{1,2,3,4}.pdf
# Console output replays the 4-family Δα table:
#   Pythia            -0.414 [-0.642, -0.218]
#   Pythia-deduped    -0.283 [-0.334, -0.229]
#   Cerebras-GPT      -0.201 [-0.323, -0.062]
#   BLOOM             -0.127 [-0.221, -0.031]

# D-axis fits (4 N) and their Bonferroni-corrected 98.75 % CIs are stored in
#   final_perN_Daxis.json
# and replotted in paper/figs/fig2_daxis.pdf.
```

## Re-measuring from scratch

To recompute the raw `v2mt_*.json` measurements from Pythia / Cerebras-GPT /
BLOOM checkpoints, see `measure_v2.py` and the per-family driver scripts under
`runpod_*.sh`. Note the HuggingFace Hub mis-registration bugs documented in
Appendix C of the paper — `use_safetensors=False` is required for some
Pythia-2.8B revisions, and Pythia-12B step50000 is unrecoverable from the Hub
(see `verify_checkpoints.py` *Coming soon* for the hash-comparison procedure).

## Pre-registration

`PREREGISTRATION.md` contains the 4 N-axis criteria committed before
held-out fitting (held-out RMSE > 0.02 nats; bootstrap 95 % CI excluding 0;
sign consistency across two corpora; robustness to floor parameterization).
The D-axis dense ladder (§3.5) and cross-family replication (§4.4) were
added in post-pre-registration adversarial-review rounds and are documented
as such in the paper.

## Building the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# → main.pdf (14 pages)

python3 make_figures.py  # regenerate figs/fig*.pdf from JSONs
```

## Citation

Preprint metadata will be updated once the arXiv ID is assigned. For now:

```bibtex
@misc{uda2026perclass,
  author = {Uda, Yoichi},
  title  = {Per-Class Scaling-Law Exponents Are Coordinate-Dependent:
            Two-Axis Decomposition Reproduces Across Four Model Families},
  year   = {2026},
  note   = {Preprint; under submission to TMLR}
}
```

## License

MIT for code; CC BY 4.0 for the paper text and figures.
See `LICENSE` and `LICENSE-paper`.

## Acknowledgments

Compute: ~$60 RunPod (H100 80 GB SECURE) + local RTX 4070 Ti Super.
Pythia checkpoints: EleutherAI. Cerebras-GPT: Cerebras Systems.
BLOOM: BigScience. Pile-10k subsample: NeelNanda on HuggingFace.
Wikitext-103: Salesforce on HuggingFace.

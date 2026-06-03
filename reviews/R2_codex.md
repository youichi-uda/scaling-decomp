# R2 Codex Review (post-R1)

## 1. New §5.4 Limitations claims

- **FAIL:** “shared-E sweep preserves the sign of Δα” is supported in §4.1: lines around the “Shared-E robustness” paragraph explicitly sweep shared `E in [0, E_cont]` and say sign is preserved.
- **FAIL:** Michaud claim is over-attributed. §5.4 says `michaud2023quantization` “predicts per-class α → 1 when the per-class prediction problem collapses to its unigram-conditioned distribution.” The body (§6.1/§5.4 text) only gives an interpretation that continuation tokens may be orthographic/morphological completion and proposes a unigram-subtraction test. It does not show Michaud specifically predicts this tokenizer-position/unigram-LM coupling.
- **FAIL:** Cerebras “Chinchilla-optimal (~20 tokens/param)” is asserted in §4.4 without verification in the paper body. It is plausible from the Cerebras paper title/claim, but this manuscript treats it as fact and gives no local evidence/calculation.

## 2. Abstract vs conclusion coordinate language

- **FAIL:** `grep -n "coordinate" paper/main.tex` shows old framing remains in Conclusion: line 851 “is a coordinate-dependent observable” and lines 887-888 “coordinate-dependent surface, not an estimator...”. This conflicts with the softened abstract that now emphasizes ~2x variation rather than “coordinate-dependent observable.”

## 3. R1 citation entries

- **PASS with minor bib cleanup:** `gao2024lmeval` includes Zenodo publisher and DOI `10.5281/zenodo.10256836`. The canonical citation appears to be Dec 2023/version v0.4.0, while the key/year says 2024.
- **FAIL minor:** `sennrich2016bpe` pages `1715--1725` are correct, but booktitle omits “Volume 1: Long Papers” (ACL Anthology canonical venue: Proceedings of ACL 2016, Volume 1: Long Papers).

## 4. Tokenizer-partition portability table

- **PASS as limitation / FAIL as referenced evidence:** §5.4 correctly says the per-family word-continuation fraction is “not quantified in this paper.” §4.7/§`sec:mixture` only reports Pythia fractions (47.03/22.15/30.83) because eval corpus/tokenizer are shared. No Appendix A or §4.7 per-family table exists.

## 5. Broken refs/labels

- **PASS:** Requested greps show every `\ref{...}` has a matching `\label{...}`. Orphan labels not referenced: `sec:nfit`, `sec:results`.

## 6. Section flow / redundancy

- **FAIL minor:** Flow still hangs together, but R1 additions created redundancy: coordinate-dependence cautions appear in §4.5, §6.2, §6.3, §5.4, and Conclusion; the conclusion repeats the old stronger framing. The Limitations list has become a catch-all adversarial-response dump and now includes several paragraphs that read like mini-discussion subsections rather than terse limitations.

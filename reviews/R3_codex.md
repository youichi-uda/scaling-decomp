# R3 Codex Review

## 1. Numbers re-verify

- **PASS:** §4.1 matches `floor_robust.json`: shared-alpha test reports `F(1, 8) = 4.47`, `p = 0.067`, shared `alpha = 0.62`; JSON has `F=4.4730739863776465`, `p_value=0.06735099314369442`, `alpha_shared=0.6215018773466834`.
- **PASS:** shared-`E` sweep text says `dalpha < 0` throughout and ranges from `-0.08` to `-1.5`; JSON sweep is `-0.0824 ... -1.5305` over feasible fixed floors. Free fit is `-0.4144`.
- **PASS:** Appendix C matches `hub_blob_audit.json`: Pythia-2.8B `model.safetensors` `main`, `step1000`, `step2000`, `step4000`, `step10000`, `step50000`, and `step100000` all share blob `ab496f1c...`; Pythia-12B `step50000` shard 3 size is `3,896,183,087`.

## 2. Citations

- **PASS:** `dey2023cerebras` is the bib key in `paper/references.bib`, and §4.4 cites it in both the Cerebras paragraph and the alternative-explanation paragraph.
- **PASS:** `sennrich2016bpe` remains cited in §3.1 at the byte-level BPE space-marker sentence.

## 3. Cross-references

- **PASS:** requested ref/label greps are balanced. Missing labels for refs: none. Unreferenced labels: none.

## 4. Abstract vs conclusion consistency

- **PASS:** Abstract still reports `dalpha=-0.41`, 95% CI `[-0.64, -0.22]`; Conclusion repeats the same N-axis value and CI.
- **PASS:** Framing is now consistent: both emphasize sign robustness/invariance while treating magnitude as coordinate-dependent/non-invariant; Conclusion adds the stronger floor-conditional caveat from §4.1 without contradicting the Abstract.

## 5. New artifacts committed

- **FAIL minor:** All four files exist at repo root: `floor_robust.py`, `floor_robust.json`, `verify_checkpoints.py`, `hub_blob_audit.json`.
- **FAIL minor:** `main.tex` references `floor_robust.py`, `verify_checkpoints.py`, and `hub_blob_audit.json`, but I found no literal reference to `floor_robust.json`.

## 6. Page count / length

- **PASS:** `paper/main.pdf` is 16 pages.
- **PASS:** Length is on the long side for main-text TMLR style, but no new redundant block jumps out as arXiv-blocking. The Hub appendix is long, but evidence-dense and directly supports the checkpoint exclusions.

NEEDS-FIX

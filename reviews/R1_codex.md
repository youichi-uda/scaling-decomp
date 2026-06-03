# R1 Codex Review (focused)

## A. Table 1 vs final_perN_Daxis.json
- PASS. `tab:daxis` matches `final_perN_Daxis.json` to 2 decimals for `E_init`, `E_cont`, and alpha values.
- 1B: alpha init/cont `0.57/0.77`, E init/cont `3.49/1.05`; matches JSON rounded.
- 1.4B: `0.54/0.78`, `3.18/0.96`; matches.
- 6.9B: `0.57/0.93`, `2.68/0.85`; matches.
- 12B: `0.51/0.96`, `2.36/0.82`; matches.
- Exact discrepancies: none.

## B. Abstract delta-alpha and CI
- PASS. Abstract says `\dalpha=-0.41`, 95% CI `[-0.64, -0.22]`.
- JSON has `diff=-0.41436795994993736`, `CI95=[-0.6416020025031289, -0.218322903629537]`, matching to 2 decimals.

## C. Held-out RMSE improvement
- PASS. Text says held-out RMSE improvement `+0.022` nats.
- JSON has `held_out.improvement=0.02158360825903305`, rounding to `+0.022`.

## D. Citation keys
- FAIL. `references.bib` is not present at the requested path, so keys from `grep -oE '\\cite[pt]?\\{[^}]+\\}' paper/main.tex` cannot be verified as defined there.
- Affected cited keys include: `michaud2023quantization`, `oh2024surprisal`, `kaplan2020scaling`, `hoffmann2022chinchilla`, `hestness2017deep`, `hutter2021learning`, `xia2023training`, `schaeffer2023emergent`, `biderman2023pythia`, `sennrich2016bpe`, `black2022gptneox`, `paperno2016lambada`, `gao2024lmeval`, `gao2020pile`, `merity2017wikitext`.

## E. Refs and labels
- PASS. Every `\ref{X}` in `paper/main.tex` has a matching `\label{X}`.

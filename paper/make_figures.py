#!/usr/bin/env python3
"""Generate Fig 1-4 for the per-class scaling-law paper (TMLR/arXiv submission).

Outputs:
  paper/figs/fig1_naxis.pdf
  paper/figs/fig2_daxis.pdf
  paper/figs/fig3_families.pdf
  paper/figs/fig4_downstream.pdf
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    }
)

COL_INIT = "#1f77b4"
COL_CONT = "#d62728"


def midwf(d, lo=4, hi=7):
    """Word-frequency-decile mid-band aggregation (deciles 4-7), per-word matched.

    Prefers the `_per_word` field (n_words counts, matches paper canonical method).
    Falls back to token-weighted if per_word is unavailable.
    """
    fp = d.get("by_wordfreqdecile_position_per_word")
    if fp is not None:
        n_key = "n_words"
    else:
        fp = d["by_wordfreqdecile_position"]
        n_key = "n"
    iw = cw = inn = cnn = 0.0
    for b in range(lo, hi + 1):
        a = fp[str(b)]["word_initial"]
        c = fp[str(b)]["word_continuation"]
        if a["mean_loss"] is not None and a.get(n_key, 0):
            iw += a["mean_loss"] * a[n_key]
            inn += a[n_key]
        if c["mean_loss"] is not None and c.get(n_key, 0):
            cw += c["mean_loss"] * c[n_key]
            cnn += c[n_key]
    return (iw / inn if inn else np.nan, cw / cnn if cnn else np.nan)


def fit_scaling(N, L, grid=None):
    """Fit L = A * N^(-alpha) + E. Positivity-constrained on A and E (matches paper)."""
    if grid is None:
        grid = np.linspace(0.02, 1.8, 800)
    N = np.asarray(N, float)
    L = np.asarray(L, float)
    best = None
    for al in grid:
        x = N ** (-al)
        B = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(B, L, rcond=None)
        if coef[1] <= 0 or coef[0] < 0:
            continue
        pred = B @ coef
        sse = float(np.sum((pred - L) ** 2))
        if best is None or sse < best[3]:
            best = (al, float(coef[0]), float(coef[1]), sse)
    if best is None:
        # fall back to unconstrained best
        for al in grid:
            x = N ** (-al)
            B = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(B, L, rcond=None)
            pred = B @ coef
            sse = float(np.sum((pred - L) ** 2))
            if best is None or sse < best[3]:
                best = (al, float(coef[0]), float(coef[1]), sse)
    return best  # (alpha, E, A, sse)


def bootstrap_diff(N, init, cont, B=2000, seed=20260603):
    fi = fit_scaling(N, init)
    fc = fit_scaling(N, cont)
    Ni = np.asarray(N, float)
    pred_i = fi[2] * Ni ** (-fi[0]) + fi[1]
    pred_c = fc[2] * Ni ** (-fc[0]) + fc[1]
    ri = np.asarray(init) - pred_i
    rc = np.asarray(cont) - pred_c
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        Li = pred_i + rng.choice(ri, size=len(Ni), replace=True)
        Lc = pred_c + rng.choice(rc, size=len(Ni), replace=True)
        diffs[b] = fit_scaling(Ni, Li)[0] - fit_scaling(Ni, Lc)[0]
    q025, q50, q975 = np.quantile(diffs, [0.025, 0.5, 0.975])
    return fi, fc, (float(q025), float(q50), float(q975))


# ---------------------------------------------------------------------------
# Figure 1: N-axis at D = 300B (Pythia, 7 sizes)
# ---------------------------------------------------------------------------
def fig1_naxis():
    sizes = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    N = [1.6e8, 4.1e8, 1.0e9, 1.4e9, 2.8e9, 6.9e9, 1.2e10]
    init, cont = [], []
    for s in sizes:
        with open(ROOT / f"v2mt_{s}.json") as fp:
            li, lc = midwf(json.load(fp))
        init.append(li)
        cont.append(lc)

    with open(ROOT / "final_Naxis_perword.json") as fp:
        canon = json.load(fp)
    a_init, E_init, A_init = canon["alpha_init"], canon["E_init"], canon["A_init"]
    a_cont, E_cont, A_cont = canon["alpha_cont"], canon["E_cont"], canon["A_cont"]
    diff = canon["diff"]
    ci_lo, ci_hi = canon["CI95"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    Ngrid = np.logspace(np.log10(1.4e8), np.log10(1.4e10), 200)
    ax.plot(
        Ngrid, A_init * Ngrid ** (-a_init) + E_init, "-", color=COL_INIT, lw=1.4,
        label=fr"$L_\mathrm{{init}}$ fit: $\alpha_\mathrm{{init}}={a_init:.2f}$",
    )
    ax.plot(
        Ngrid, A_cont * Ngrid ** (-a_cont) + E_cont, "-", color=COL_CONT, lw=1.4,
        label=fr"$L_\mathrm{{cont}}$ fit: $\alpha_\mathrm{{cont}}={a_cont:.2f}$",
    )
    ax.scatter(N, init, s=28, c=COL_INIT, marker="o", zorder=5, edgecolors="white", lw=0.5)
    ax.scatter(N, cont, s=28, c=COL_CONT, marker="s", zorder=5, edgecolors="white", lw=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters $N$ (non-embed)")
    ax.set_ylabel("Per-word matched loss (nats)")
    ax.set_title(
        fr"Pythia, $D=300\mathrm{{B}}$: $\Delta\alpha = {diff:+.2f}$  [{ci_lo:+.2f}, {ci_hi:+.2f}]"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # Annotate sizes
    label_map = {"160m": "160M", "410m": "410M", "1b": "1B", "1.4b": "1.4B",
                 "2.8b": "2.8B", "6.9b": "6.9B", "12b": "12B"}
    for s, n, li in zip(sizes, N, init):
        ax.annotate(label_map[s], (n, li), textcoords="offset points",
                    xytext=(0, -10), fontsize=6, ha="center", color="#444")

    fig.tight_layout()
    fig.savefig(FIGS / "fig1_naxis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig1] wrote fig1_naxis.pdf")


# ---------------------------------------------------------------------------
# Figure 2: D-axis at fixed N (4 panels for N in {1B, 1.4B, 6.9B, 12B})
# ---------------------------------------------------------------------------
def fig2_daxis():
    Ns = ["1b", "1.4b", "6.9b", "12b"]
    N_label = {"1b": "1B", "1.4b": "1.4B", "6.9b": "6.9B", "12b": "12B"}

    # Pythia checkpoint step list -> approximate D (tokens) = step * (2097152 batch tokens)
    BATCH_TOK = 2_097_152
    step_pattern_candidates = [1000, 2000, 4000, 7000, 10000, 15000, 22000, 25000,
                               30000, 42000, 50000, 70000, 100000, 130000]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=False)

    with open(ROOT / "final_perN_Daxis.json") as fp:
        fits = json.load(fp)

    for ax, n in zip(axes, Ns):
        steps, init, cont = [], [], []
        for st in step_pattern_candidates:
            p = ROOT / f"v2mt_{n}_step{st}.json"
            if not p.exists():
                continue
            with open(p) as fp:
                d = json.load(fp)
            li, lc = midwf(d)
            if math.isnan(li) or math.isnan(lc):
                continue
            steps.append(st)
            init.append(li)
            cont.append(lc)
        # Add final (300B) point
        p_final = ROOT / f"v2_{n}.json"
        if p_final.exists():
            with open(p_final) as fp:
                d = json.load(fp)
            li, lc = midwf(d)
            steps.append(143000)
            init.append(li)
            cont.append(lc)

        D = np.array(steps) * BATCH_TOK

        ax.scatter(D, init, s=22, c=COL_INIT, marker="o", zorder=5, edgecolors="white", lw=0.4)
        ax.scatter(D, cont, s=22, c=COL_CONT, marker="s", zorder=5, edgecolors="white", lw=0.4)

        fit = fits.get(n, {})
        if fit and len(D) >= 3:
            # Refit to get A coefficient (final_perN_Daxis.json stores alpha+E only).
            fi_loc = fit_scaling(D, init)
            fc_loc = fit_scaling(D, cont)
            ai = fit["alpha_D_init"]; ei = fit["E_init"]; Ai = fi_loc[2]
            ac = fit["alpha_D_cont"]; ec = fit["E_cont"]; Ac = fc_loc[2]
            Dgrid = np.logspace(np.log10(D.min() * 0.7), np.log10(D.max() * 1.4), 200)
            ax.plot(Dgrid, Ai * Dgrid ** (-ai) + ei, "-", color=COL_INIT, lw=1.2)
            ax.plot(Dgrid, Ac * Dgrid ** (-ac) + ec, "-", color=COL_CONT, lw=1.2)

            diff = fit["diff"]
            ci_lo, ci_hi = fit["CI_bonf"]
            ax.set_title(
                fr"$N={N_label[n]}$, $\Delta\alpha_D={diff:+.2f}$" + "\n" +
                fr"98.75\% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]",
                fontsize=9,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training tokens $D$")
        if ax is axes[0]:
            ax.set_ylabel("Per-word matched loss (nats)")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # Shared legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color=COL_INIT, label=r"$L_\mathrm{init}$"),
            plt.Line2D([0], [0], marker="s", color=COL_CONT, label=r"$L_\mathrm{cont}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "fig2_daxis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig2] wrote fig2_daxis.pdf")


# ---------------------------------------------------------------------------
# Figure 3: 4-family forest plot of Δα at D=300B with bootstrap 95% CIs
# ---------------------------------------------------------------------------
FAMILIES = {
    "Pythia": [
        ("160m", 1.6e8), ("410m", 4.1e8), ("1b", 1.0e9), ("1.4b", 1.4e9),
        ("2.8b", 2.8e9), ("6.9b", 6.9e9), ("12b", 1.2e10),
    ],
    "Pythia-deduped": [
        ("160md", 1.6e8), ("410md", 4.1e8), ("1bd", 1.0e9), ("1.4bd", 1.4e9),
    ],
    "Cerebras-GPT": [
        ("cerebras-111M", 1.11e8), ("cerebras-256M", 2.56e8), ("cerebras-590M", 5.9e8),
        ("cerebras-1.3B", 1.3e9), ("cerebras-2.7B", 2.7e9), ("cerebras-6.7B", 6.7e9),
        ("cerebras-13B", 1.3e10),
    ],
    "BLOOM": [
        ("bloom-560m", 5.6e8), ("bloom-1b1", 1.1e9), ("bloom-1b7", 1.7e9),
        ("bloom-3b", 3.0e9), ("bloom-7b1", 7.1e9),
    ],
}


def _load_family(name, sizes):
    init, cont, N = [], [], []
    for s, n in sizes:
        p = ROOT / f"v2mt_{s}.json"
        if not p.exists():
            continue
        with open(p) as fp:
            d = json.load(fp)
        li, lc = midwf(d)
        if math.isnan(li) or math.isnan(lc):
            continue
        init.append(li); cont.append(lc); N.append(n)
    return N, init, cont


def fig3_families():
    results = []
    for fam, sizes in FAMILIES.items():
        N, init, cont = _load_family(fam, sizes)
        if len(N) < 3:
            print(f"  [warn] {fam} only n={len(N)}, skipping")
            continue
        fi, fc, (lo, med, hi) = bootstrap_diff(N, init, cont, B=2000, seed=20260603)
        diff = fi[0] - fc[0]
        results.append({"family": fam, "n": len(N), "diff": diff, "lo": lo, "hi": hi})
        print(f"  [fig3] {fam:18s} n={len(N)} Δα={diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # Sort: keep insertion order (Pythia, dedup, Cerebras, BLOOM)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(results))[::-1]
    diffs = [r["diff"] for r in results]
    los = [r["diff"] - r["lo"] for r in results]
    his = [r["hi"] - r["diff"] for r in results]
    labels = [f"{r['family']}\n($n={r['n']}$)" for r in results]

    ax.errorbar(
        diffs, y, xerr=[los, his],
        fmt="o", color="#222", ecolor="#666", elinewidth=1.2, capsize=3, ms=5,
    )
    ax.axvline(0, color="#999", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta\alpha = \alpha_\mathrm{init} - \alpha_\mathrm{cont}$  (95\% bootstrap CI)")
    ax.set_title("Per-class $\\Delta\\alpha$ across four model families (D=300B)")
    ax.grid(True, axis="x", alpha=0.3, lw=0.5)

    # Annotate values to the right of each error bar
    for yi, r in zip(y, results):
        ax.annotate(
            fr"${r['diff']:+.2f}\ [{r['lo']:+.2f}, {r['hi']:+.2f}]$",
            xy=(r["hi"], yi), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=7, color="#333",
        )
    ax.set_xlim(min(r["lo"] for r in results) - 0.08, 0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_families.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig3] wrote fig3_families.pdf")
    return results


# ---------------------------------------------------------------------------
# Figure 4: downstream adj R² per-class vs aggregate
# ---------------------------------------------------------------------------
def fig4_downstream():
    # Canonical numbers from PAPER_DRAFT.md §4.6
    tasks = ["lambada_openai", "piqa", "arc_easy", "sciq"]
    agg = [0.951, 0.978, 0.971, 0.961]
    pcl = [0.988, 0.988, 0.995, 0.995]

    x = np.arange(len(tasks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    b1 = ax.bar(x - width / 2, agg, width, label=r"$L_\mathrm{aggregate}$", color="#7f7f7f", edgecolor="white", lw=0.5)
    b2 = ax.bar(x + width / 2, pcl, width, label=r"$L_\mathrm{init} + L_\mathrm{cont}$", color="#1f77b4", edgecolor="white", lw=0.5)
    for bar, v in zip(b1, agg):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(b2, pcl):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for xi, a, p in zip(x, agg, pcl):
        ax.annotate(
            f"+{p - a:.3f}", xy=(xi + width / 2, p), xytext=(0, 14),
            textcoords="offset points", ha="center", fontsize=7,
            color="#1a6f1a", weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=0)
    ax.set_ylim(0.93, 1.02)
    ax.set_ylabel("Adjusted $R^2$ ($n=7$, Pythia)")
    ax.set_title("Downstream task fit: per-class vs aggregate loss")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_downstream.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig4] wrote fig4_downstream.pdf")


if __name__ == "__main__":
    os.chdir(str(ROOT))
    fig1_naxis()
    fig2_daxis()
    fig3_families()
    fig4_downstream()
    print("All figures written to", FIGS)

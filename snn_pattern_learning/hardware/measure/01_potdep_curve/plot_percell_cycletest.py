#!/usr/bin/env python3
"""Shared vs per-cell LinearStep parameters on the cycle test."""
import ast
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fit_percell_cycletest as F

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
MEAS, SHARED, PC = "#2a78d6", "#eb6834", "#2e9e6b"
N = 5


def main():
    path = "2026-08-10_14-29_CycleTest_Data.csv"
    diff, steps = F.load_cycle(path)
    n_cy = diff.shape[0]
    cfg = json.load(open("aihwkit_linearstep_params.json", encoding="utf-8"))
    L, lsb = cfg["linearstep"], cfg["lsb_per_w"]
    rows = list(csv.DictReader(open("percell_cycletest_params.csv",
                                    encoding="utf-8")))

    meas = [np.concatenate([diff[cy, :, r, c] for cy in range(n_cy)])
            for r in range(N) for c in range(N)]
    w0s = [m[0] - (m[1] - m[0]) for m in meas]
    ub, db = min(L["up_down"], 0.0), max(-L["up_down"], 0.0)
    p_sh = np.array([(ub + 1.0) * L["dw_min"] * lsb,
                     (db + 1.0) * L["dw_min"] * lsb, 1.0, 1.0,
                     L["w_max"] * lsb, L["w_min"] * lsb])
    ppr = 5

    fig, axes = plt.subplots(N, N, figsize=(20, 15), sharex=True, sharey=True)
    lim = 20 * int(np.ceil(max(abs(min(m.min() for m in meas)),
                               max(m.max() for m in meas)) / 20 + 0.5))
    for i in range(N * N):
        r, c = divmod(i, N)
        ax = axes[r][c]
        m, w0, rec = meas[i], w0s[i], rows[i]
        p_pc = np.array([float(rec[k]) for k in
                         ("scale_up", "scale_down", "gamma_up", "gamma_down",
                          "w_max", "w_min")])
        x = np.arange(len(m))
        ax.plot(x, m, marker=".", ms=3, lw=1.2, color=MEAS, label="measured",
                zorder=3)
        ax.plot(x, F.run(p_sh, w0, n_cy, steps, ppr), lw=1.5, color=SHARED,
                alpha=0.8, label=f"shared ({float(rec['rms_shared']):.0f})",
                zorder=2)
        ax.plot(x, F.run(p_pc, w0, n_cy, steps, ppr), lw=1.5, color=PC,
                ls="--", alpha=0.9,
                label=f"per-cell ({float(rec['rms_percell']):.0f})", zorder=2)
        ax.set_title(f"Cell ({r+1}, {c+1})", fontsize=10)
        ax.grid(alpha=0.3, ls=":")
        ax.set_ylim(-lim, lim)
        for k in range(1, n_cy):
            ax.axvline(k * 2 * steps - 0.5, color="k", lw=1)
        for k in range(n_cy):
            ax.axvline(k * 2 * steps + steps - 0.5, color="gray", ls="--",
                       lw=0.8)
        ax.legend(fontsize=7, loc="lower left")

    sh = np.sqrt(np.mean([float(r["rms_shared"]) ** 2 for r in rows]))
    pc = np.sqrt(np.mean([float(r["rms_percell"]) ** 2 for r in rows]))
    fig.suptitle(f"Cycle test: one shared LinearStep for all cells "
                 f"(rms {sh:.0f} LSB) vs per-cell parameters "
                 f"(rms {pc:.0f} LSB)\n"
                 f"per-cell fitting removes {100*(1-pc/sh):.0f}% of the error, "
                 f"so most of the gap was cell-to-cell spread, not the model form",
                 fontsize=15)
    fig.supxlabel("Total Measurement Steps (Cycle 1 -> Cycle 2 -> ...)",
                  fontsize=12)
    fig.supylabel("Differential ADC Value (N5 - N6)", fontsize=12)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.95])
    fig.savefig("cycletest_percell.png", dpi=100)
    print("saved -> cycletest_percell.png")

    # parameter spread, the thing that would become dtod in aihwkit
    fig2, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, key, lab in zip(
            axs, ["scale_up", "scale_down", "gamma_up", "w_max"],
            ["scale_up (LSB)", "scale_down (LSB)", "gamma_up", "w_max (LSB)"]):
        v = np.array([float(r[key]) for r in rows])
        ax.hist(v, bins=10, color="#7fa8d9", edgecolor="white")
        ax.axvline(v.mean(), color=INK, lw=2,
                   label=f"mean {v.mean():.2f}\ncv {v.std()/abs(v.mean()):.3f}")
        ax.set_xlabel(lab)
        ax.set_ylabel("cells")
        ax.legend(fontsize=8.5)
        ax.grid(color=GRID, lw=0.8, axis="y")
    fig2.suptitle("Per-cell parameter spread — this is what dw_min_dtod / "
                  "w_max_dtod have to represent", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.9])
    fig2.savefig("cycletest_percell_spread.png", dpi=110)
    print("saved -> cycletest_percell_spread.png")


if __name__ == "__main__":
    main()

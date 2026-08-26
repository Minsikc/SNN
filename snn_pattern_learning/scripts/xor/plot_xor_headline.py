#!/usr/bin/env python3
"""Temporal XOR headline figure: accuracy trajectories + best accuracy.

A two-panel cut of plot_xor_summary.py (its panels 1 and 3), for reporting.
Reads the four canonical result JSONs in results/xor/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
DIG, ANA, FRZ, BPTT = "#2a78d6", "#eb6834", "#b8b6ae", "#2e9e6b"
HERE = os.path.join("results", "xor")

COND = [("xor_bptt_seed0.json", "BPTT", BPTT),
        ("xor_digital_res_seed0.json", "e-prop digital", DIG),
        ("xor_analog_res_seed0.json", "e-prop analog", ANA),
        ("xor_frozen_res_seed0.json", "$W_{out}$ frozen", FRZ)]


def main():
    runs = []
    for fn, lab, col in COND:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            runs.append((lab, col, json.load(open(p, encoding="utf-8"))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(wspace=0.3)

    # ---- accuracy over training ------------------------------------------
    for lab, col, d in runs:
        ax1.plot(np.arange(1, len(d["accs"]) + 1), d["accs"], color=col,
                 lw=1.5, alpha=0.85, label=lab)
    ax1.axhline(0.25, color=MUTED, lw=0.8, ls=":")
    ax1.set_ylim(0, 1.08)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy (4 patterns)")
    ax1.set_title("accuracy over training", fontsize=11)
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.grid(color=GRID, lw=0.8)

    # ---- best accuracy ----------------------------------------------------
    labs = [l for l, _, _ in runs]
    accs = [d["best_acc"] for _, _, d in runs]
    cols = [c for _, c, _ in runs]
    y = np.arange(len(runs))
    ax2.barh(y, accs, color=cols, height=0.6)
    for i, a in enumerate(accs):
        ax2.text(a + 0.02, i, f"{a:.2f}", va="center", fontsize=9)
    ax2.set_yticks(y, labs, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.15)
    ax2.axvline(0.25, color=MUTED, lw=1, ls=":")
    ax2.set_xlabel("best accuracy")
    ax2.set_title("best accuracy", fontsize=11)
    ax2.grid(color=GRID, lw=0.8, axis="x")

    fig.suptitle("Temporal XOR, seed 0, 60 epochs", fontsize=12.5)
    fig.savefig("xor_headline.png", dpi=130, bbox_inches="tight")
    print("saved -> xor_headline.png")


if __name__ == "__main__":
    main()

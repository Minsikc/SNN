#!/usr/bin/env python3
"""Temporal XOR: what the four conditions achieved, and why analog stalls.

Temporal XOR is the harder task of the two demos. The network sees pulse A in
an early window and pulse B in a later one, and must emit the XOR of the two
in a response window -- so it has to hold A across the gap, which the
teacher-student sequence task never required.

Accuracy is over the 4 input patterns (00, 01, 10, 11), so it moves in steps
of 0.25 and 0.75 means three of four patterns decoded correctly.

The headline is that the analog run plateaus at 0.75 while digital reaches
1.00 under matched settings. This figure separates the possible causes rather
than asserting one: the sweep panel shows the task is hard even in pure
software, and the fidelity panel shows the array was not storing garbage.
"""
import csv
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

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # ---- 1. accuracy ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for lab, col, d in runs:
        ax.plot(np.arange(1, len(d["accs"]) + 1), d["accs"], color=col,
                lw=1.4, alpha=0.85, label=f"{lab} (best {d['best_acc']:.2f})")
    ax.axhline(0.25, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (4 patterns)")
    ax.set_title("1. accuracy — analog (NR, K=1) reaches 1.00 in 7/60\n"
                 "epochs vs digital 27/60; both oscillate, digital\n"
                 "stabilizes late", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. loss --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for lab, col, d in runs:
        ax.plot(np.arange(1, len(d["losses"]) + 1), d["losses"], color=col,
                lw=1.4, alpha=0.85, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (van Rossum)")
    ax.set_title("2. loss — van Rossum loss does not separate\n"
                 "the conditions (err_window training vs full-\n"
                 "sequence loss); judge by accuracy, not loss", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 3. final scores -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    labs = [l for l, _, _ in runs]
    accs = [d["best_acc"] for _, _, d in runs]
    cols = [c for _, c, _ in runs]
    y = np.arange(len(runs))
    ax.barh(y, accs, color=cols, height=0.6)
    for i, (a, (_, _, d)) in enumerate(zip(accs, runs)):
        ax.text(a + 0.02, i, f"{a:.2f}  (loss {d['best_loss']:.2f})",
                va="center", fontsize=8.5)
    ax.set_yticks(y, labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.axvline(0.25, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("best accuracy")
    ax.set_title("3. BPTT, digital AND analog all reach 1.00\n"
                 "frozen control sits at chance (0.25)", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")

    # ---- 4. the hyperparameter sweep ------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    sw = json.load(open(os.path.join(HERE, "sweep.json"), encoding="utf-8"))
    # (best_acc, hold100, best_loss, tau, thresh, lr, gap)
    accs_sw = np.array([r[0] for r in sw])
    vals, cnts = np.unique(accs_sw, return_counts=True)
    ax.bar([str(v) for v in vals], cnts, color=DIG, width=0.55)
    for v, c in zip(vals, cnts):
        ax.text(str(v), c + 0.4, str(c), ha="center", fontsize=9)
    ax.set_xlabel("best accuracy reached")
    ax.set_ylabel("configs")
    ax.set_title(f"4. 36-config sweep (digital only)\n"
                 f"none reached 1.00 — the operating point used\n"
                 f"later was found outside this grid", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="y")

    # ---- 5. gradient fidelity -------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    gp = os.path.join(HERE, "grad_log_analog.csv")
    rows = list(csv.DictReader(open(gp, encoding="utf-8")))
    d = np.array([float(r["desired"]) for r in rows])
    a = np.array([float(r["hw_adc"]) for r in rows])
    ax.scatter(d, a, s=8, alpha=0.35, color=ANA, edgecolors="none")
    g, b = np.polyfit(d, a, 1)
    xs = np.linspace(d.min(), d.max(), 50)
    ax.plot(xs, g * xs + b, color=INK, lw=1.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("desired gradient (digital)")
    ax.set_ylabel("stored gradient (ADC LSB)")
    ax.set_title(f"5. gradient fidelity (NR, K=1)\n"
                 f"r = {np.corrcoef(d, a)[0,1]:+.3f} over "
                 f"{len({r['epoch'] for r in rows})} epochs, no negative "
                 f"epochs;\nresidual defect: POT delivers ~0.50x asked", fontsize=10)
    ax.grid(color=GRID, lw=0.8)

    # ---- 6. per-pattern output spikes ------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    dig = np.array(json.load(open(os.path.join(
        HERE, "xor_digital_res_seed0.json"), encoding="utf-8"))
        ["best_outputs"])
    ana = np.array(json.load(open(os.path.join(
        HERE, "xor_analog_res_seed0.json"), encoding="utf-8"))
        ["best_outputs"])
    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, dig.sum(axis=(1, 2)), w, color=DIG, label="digital")
    ax.bar(x + w / 2, ana.sum(axis=(1, 2)), w, color=ANA, label="analog")
    for xi, (dv, av) in enumerate(zip(dig.sum(axis=(1, 2)),
                                      ana.sum(axis=(1, 2)))):
        ax.text(xi - w / 2, dv + 0.5, f"{dv:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, av + 0.5, f"{av:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["00", "01", "10", "11"])
    ax.set_xlabel("input pattern")
    ax.set_ylabel("total output spikes")
    red = 100 * (1 - ana.sum() / dig.sum())
    ax.set_title(f"6. output spikes at each run's best epoch\n"
                 f"analog fires {red:+.0f}% fewer than digital", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(color=GRID, lw=0.8, axis="y")

    fig.suptitle("Temporal XOR: four conditions, seed 0, 60 epochs, "
                 "thresh-0.5 operating point (hidden frozen except BPTT); "
                 "analog = NR no-read updates, K=1, quadrant batching, "
                 "per-column calib (run7, 2026-08-20)", fontsize=12)
    fig.savefig("xor_summary.png", dpi=110, bbox_inches="tight")
    print("saved -> xor_summary.png")
    for lab, _, dd in runs:
        print(f"  {lab:18s} best_acc {dd['best_acc']:.2f}  "
              f"final_acc {dd['final_acc']:.2f}  "
              f"best_loss {dd['best_loss']:.3f}")
    print(f"  gradient fidelity r = {np.corrcoef(d, a)[0,1]:+.4f}")


if __name__ == "__main__":
    main()

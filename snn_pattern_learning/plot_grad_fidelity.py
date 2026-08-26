#!/usr/bin/env python3
"""How faithfully did the array store the gradient during training?

Every epoch the demo logged, for all 25 cells, the digitally computed gradient
(`desired`) and what came back off the array (`hw_adc`, in ADC LSB).  These
logs were written but never plotted.  They answer the question the loss curves
cannot: the loss says learning worked, this says whether it worked *because*
the analog gradient was faithful or *despite* it being noisy.

`desired` is in gradient units and `hw_adc` in LSB, so the two are compared by
correlation and by a fitted scale, never by absolute difference.
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
ACC, ACC2 = "#2a78d6", "#eb6834"
HERE = os.path.join("results", "eprop_grad_log")

RUNS = [("gradient_log_seed0.csv", "seed 0\n(n_seq=1, T=20)"),
        ("grad_n1_T40.csv", "n_seq=1, T=40"),
        ("grad_n1_T80.csv", "n_seq=1, T=80"),
        ("grad_n2_T20.csv", "n_seq=2, T=20"),
        ("grad_n5_T20.csv", "n_seq=5, T=20")]


def load(fn):
    rows = list(csv.DictReader(open(os.path.join(HERE, fn),
                                    encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("epoch", "row", "col", "desired", "hw_adc")}


def main():
    runs = [(lab, load(fn)) for fn, lab in RUNS
            if os.path.exists(os.path.join(HERE, fn))]
    base_lab, base = runs[0]

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # ---- 1. pooled scatter ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    d, a = base["desired"], base["hw_adc"]
    ax.scatter(d, a, s=7, alpha=0.3, color=ACC, edgecolors="none")
    g, b = np.polyfit(d, a, 1)
    xs = np.linspace(d.min(), d.max(), 50)
    ax.plot(xs, g * xs + b, color=INK, lw=1.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("desired gradient (digital)")
    ax.set_ylabel("stored gradient (ADC LSB)")
    ax.set_title(f"1. {base_lab.splitlines()[0]}: all 50 epochs x 25 cells\n"
                 f"r = {np.corrcoef(d, a)[0,1]:+.4f}, "
                 f"scale {g:.0f} LSB per unit", fontsize=10)
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. fidelity over epochs ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for lab, r in runs:
        eps = sorted(set(r["epoch"].astype(int)))
        rr = []
        for e in eps:
            m = r["epoch"] == e
            rr.append(np.corrcoef(r["desired"][m], r["hw_adc"][m])[0, 1]
                      if r["desired"][m].std() > 1e-12 else np.nan)
        ax.plot(eps, rr, lw=1.3, alpha=0.85,
                label=lab.replace("\n", " "))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("r(desired, stored) within epoch")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("2. per-epoch fidelity\nnoisy early, when gradients are "
                 "small", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(color=GRID, lw=0.8)

    # ---- 3. fidelity vs gradient magnitude ------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for lab, r in runs:
        eps = sorted(set(r["epoch"].astype(int)))
        mag, rr = [], []
        for e in eps:
            m = r["epoch"] == e
            if r["desired"][m].std() < 1e-12:
                continue
            mag.append(np.abs(r["desired"][m]).mean())
            rr.append(np.corrcoef(r["desired"][m], r["hw_adc"][m])[0, 1])
        ax.scatter(mag, rr, s=14, alpha=0.55,
                   label=lab.replace("\n", " "))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("mean |desired gradient| that epoch")
    ax.set_ylabel("r(desired, stored)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("3. fidelity is set by gradient size\n"
                 "small gradients fall under the device noise floor",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 4. per-cell fidelity -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            m = (base["row"] == i + 1) & (base["col"] == j + 1)
            if m.sum() > 3 and base["desired"][m].std() > 1e-12:
                M[i, j] = np.corrcoef(base["desired"][m],
                                      base["hw_adc"][m])[0, 1]
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if M[i, j] > 0.75 or M[i, j] < 0.25
                    else "black")
    ax.set_xticks(range(5), [f"c{c+1}" for c in range(5)], fontsize=8)
    ax.set_yticks(range(5), [f"r{r+1}" for r in range(5)], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("r", fontsize=8)
    n_nan = int(np.isnan(M).sum())
    ax.set_title(f"4. per-cell fidelity, {base_lab.splitlines()[0]}\n"
                 f"mean {np.nanmean(M):.3f}, range "
                 f"{np.nanmin(M):.2f}-{np.nanmax(M):.2f}"
                 + (f"\n{n_nan} blank: desired gradient never varied "
                    f"(dead unit)" if n_nan else ""), fontsize=10)

    # ---- 5. saturation check --------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    labs = [l.replace("\n", " ") for l, _ in runs]
    mx = [np.abs(r["hw_adc"]).max() for _, r in runs]
    mn = [np.abs(r["hw_adc"]).mean() for _, r in runs]
    x = np.arange(len(runs))
    ax.bar(x - 0.2, mx, 0.4, color=ACC2, label="max |stored|")
    ax.bar(x + 0.2, mn, 0.4, color=ACC, label="mean |stored|")
    ax.axhline(455, color=INK, lw=1.6, ls="--", label="array rail 455 LSB")
    for xi, v in zip(x, mx):
        ax.text(xi - 0.2, v + 8, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labs, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("|stored gradient| (LSB)")
    ax.set_ylim(0, 520)
    n_over = sum(int((np.abs(r["hw_adc"]) > 455).sum()) for _, r in runs)
    n_tot = sum(len(r["hw_adc"]) for _, r in runs)
    ax.set_title(f"5. mean usage stays under 15% of the rail\n"
                 f"only {n_over} of {n_tot} samples reach it, so range is "
                 f"not\nwhat limits these runs", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8, axis="y")

    # ---- 6. fidelity by run ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    rs = [np.corrcoef(r["desired"], r["hw_adc"])[0, 1] for _, r in runs]
    ax.barh(range(len(runs)), rs, color=ACC, height=0.6)
    for i, v in enumerate(rs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(runs)), labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("r(desired, stored), pooled over the run")
    ax.set_title(f"6. fidelity is {min(rs):.2f}-{max(rs):.2f} across runs\n"
                 "with no trend in task size: T=80 scores highest,\n"
                 "n_seq=5 lowest", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")

    fig.suptitle("Teacher-student e-prop: how faithfully the array stored "
                 "each epoch's gradient", fontsize=13)
    fig.savefig("eprop_grad_fidelity.png", dpi=110, bbox_inches="tight")
    print("saved -> eprop_grad_fidelity.png")
    for lab, r in runs:
        print(f"  {lab.replace(chr(10), ' '):22s} "
              f"r={np.corrcoef(r['desired'], r['hw_adc'])[0,1]:+.4f}  "
              f"max|adc|={np.abs(r['hw_adc']).max():.0f}")


if __name__ == "__main__":
    main()

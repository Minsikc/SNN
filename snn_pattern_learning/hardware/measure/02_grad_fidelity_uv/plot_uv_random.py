#!/usr/bin/env python3
"""Desired vs actual update for random signed u, v.

Top: one representative trial as 5x5 maps -- what was asked for, what the
array did, and the residual. Bottom: the pooled scatter over all trials,
plus the per-exposure fit.

Two "desired" definitions are distinguished throughout:
    ideal     u_r * v_c * L      what the algorithm asked for
    realized  sign * (C_p+C_d)   what the pulses actually delivered
The gap between them is stochastic sampling, which is not the device's
fault, so the device is judged against `realized`.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5
INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
ACC, ACC2 = "#2a78d6", "#eb6834"


def grid_of(rows, key):
    M = np.zeros((N, N))
    for r in rows:
        M[int(r["cell_row"]) - 1, int(r["cell_col"]) - 1] = float(r[key])
    return M


def draw(ax, M, title, cmap, fig, label, fmt="{:+.0f}"):
    lim = max(abs(M.min()), abs(M.max())) or 1
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(N), [f"c{c+1}" for c in range(N)], fontsize=7)
    ax.set_yticks(range(N), [f"r{r+1}" for r in range(N)], fontsize=7)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.6 * lim else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(label, fontsize=7)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("*_uv_random_signed.csv"))[-1]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    trial = int(sys.argv[2]) if len(sys.argv) > 2 else 31
    sub = [r for r in rows if int(r["trial"]) == trial]

    col = lambda k, rs=rows: np.array([float(r[k]) for r in rs])
    d, des, idl = col("delta"), col("desired"), col("ideal")
    g, b = np.polyfit(des, d, 1)
    res = d - (g * des + b)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38,
                          wspace=0.3)

    # ---- one trial, as maps ----
    D = grid_of(sub, "desired")
    A = grid_of(sub, "delta")
    R = A - (g * D + b)
    u = [float(r["u"]) for r in sub[:N * N:N]]
    v = [float(sub[j]["v"]) for j in range(N)]

    draw(fig.add_subplot(gs[0, 0]), D,
         f"trial {trial}: desired (realized)\n"
         f"u = {np.round(u,2)}\nv = {np.round(v,2)}",
         "coolwarm", fig, "coincidences")
    draw(fig.add_subplot(gs[0, 1]), A,
         f"actual (r = {np.corrcoef(D.ravel(), A.ravel())[0,1]:+.3f})",
         "coolwarm", fig, "LSB")
    draw(fig.add_subplot(gs[0, 2]), R,
         f"residual = actual - fit\nsd {R.std():.1f} LSB "
         f"= {R.std()/abs(g):.2f} coincidences",
         "PuOr", fig, "LSB")

    # ---- pooled scatter, realized vs ideal side by side ----
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(des, d, s=7, alpha=0.32, color=ACC, edgecolors="none")
    xs = np.linspace(des.min(), des.max(), 50)
    ax.plot(xs, g * xs + b, color=INK, lw=2)
    ax.set_xlabel("desired (realized coincidences, signed)")
    ax.set_ylabel("measured delta (LSB)")
    ax.set_title(f"pooled, {len(rows)} cells\n"
                 f"r = {np.corrcoef(des, d)[0,1]:+.4f}, "
                 f"gain {g:.1f} LSB/coinc", fontsize=10)
    ax.grid(color=GRID, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(idl, d, s=7, alpha=0.32, color=ACC2, edgecolors="none")
    ax.set_xlabel("ideal (u x v x L, signed)")
    ax.set_ylabel("measured delta (LSB)")
    ax.set_title(f"vs the ASKED-FOR value\nr = "
                 f"{np.corrcoef(idl, d)[0,1]:+.4f} — the drop from "
                 f"{np.corrcoef(des, d)[0,1]:.3f}\nis stochastic sampling, "
                 f"not the device", fontsize=10)
    ax.grid(color=GRID, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)

    # ---- per-exposure contributions ----
    ax = fig.add_subplot(gs[1, 2])
    names = ["C_pot", "C_dep", "H_N1", "H_N2", "H_N3", "H_N4"]
    Amat = np.column_stack([col(k) for k in names] + [np.ones(len(rows))])
    coef, *_ = np.linalg.lstsq(Amat, d, rcond=None)
    cols = [ACC if c > 0 else ACC2 for c in coef[:6]]
    ax.barh(range(6), coef[:6], color=cols)
    ax.set_yticks(range(6), names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1)
    for i, c in enumerate(coef[:6]):
        ax.text(c + (1.0 if c > 0 else -1.0), i, f"{c:+.2f}",
                va="center", ha="left" if c > 0 else "right", fontsize=8.5)
    ax.set_xlabel("LSB per exposure")
    ax.set_title("coincidence dominates; every half-select\n"
                 f"term is under {max(abs(coef[2:6])):.1f} LSB "
                 f"({100*max(abs(coef[2:6]))/abs(coef[0]):.0f}% of one "
                 f"coincidence)", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")
    ax.set_xlim(min(coef[:6]) * 1.35, max(coef[:6]) * 1.35)

    fig.suptitle("Random signed u, v ~ U(-1,1): desired vs actual update on "
                 "the 5x5 array", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "uv_random_signed.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)
    print(f"\npooled r(realized) {np.corrcoef(des,d)[0,1]:+.4f}  "
          f"r(ideal) {np.corrcoef(idl,d)[0,1]:+.4f}")
    print(f"gain {g:.2f} LSB/coinc, residual {res.std()/abs(g):.2f} coinc")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The fitted nonlinear update model against the measurements."""
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("m", "fit_update_model.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
POT, DEP, NEU = "#2a78d6", "#eb6834", "#b8b6ae"


def main():
    d = M.load()
    p = M.fit(M.m_sat_c, [14.5, .05, 15.5, .05, 1.5, -1.5, -1.5, 0.], d)
    gp, kp, gd, kd, ap, ar, ad, c = p
    pl = M.fit(M.m_linear, [13., 0.], d)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    # ---- 1. burst curve --------------------------------------------------
    ax = axes[0]
    for lab, key, other, col, sign in (("POT", "C_pot", "C_dep", POT, +1),
                                       ("DEP", "C_dep", "C_pot", DEP, -1)):
        sel = d[other] == 0
        xs, ys, es = [], [], []
        for n in range(1, 11):
            s = sel & (d[key] == n)
            if s.sum() >= 8:
                xs.append(n)
                ys.append(d["delta"][s].mean())
                es.append(d["delta"][s].std() / np.sqrt(s.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, color=col, capsize=3,
                    label=f"{lab} measured")
        cc = np.linspace(0, 10, 100)
        g_, k_ = (gp, kp) if sign > 0 else (gd, kd)
        ax.plot(cc, sign * g_ * M.sat(cc, k_), color=col, lw=2, alpha=0.75)
        ax.plot(cc, sign * abs(pl[0]) * cc, color=col, lw=1, ls=":", alpha=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("coincidences in the burst, C")
    ax.set_ylabel("delta (LSB)")
    ax.set_title("1. the burst saturates\nsolid = fitted, dotted = linear "
                 f"({pl[0]:.1f} LSB/coinc)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. per-pulse step ----------------------------------------------
    ax = axes[1]
    nn = np.arange(1, 11)
    ax.plot(nn, gp * np.exp(-kp * (nn - 1)), "o-", color=POT, lw=2, ms=6,
            label=f"POT  {gp:.1f} -> {gp*np.exp(-kp*9):.1f} LSB")
    ax.plot(nn, -gd * np.exp(-kd * (nn - 1)), "o-", color=DEP, lw=2, ms=6,
            label=f"DEP  {-gd:.1f} -> {-gd*np.exp(-kd*9):.1f} LSB")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pulse index within the burst")
    ax.set_ylabel("that pulse's contribution (LSB)")
    ax.set_title("2. later pulses in a burst move the cell less\n"
                 f"DEP fades faster (k={kd:.3f}) than POT (k={kp:.3f})",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(color=GRID, lw=0.8)

    # ---- 3. half-select amplitudes + model comparison --------------------
    ax = axes[2]
    names = ["HS pot-driving", "HS reset-driving", "HS dep-driving"]
    vals = [ap, ar, ad]
    cols = [POT if v > 0 else DEP for v in vals]
    ax.barh(range(3), vals, color=cols, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + (0.06 if v > 0 else -0.06), i, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9)
    ax.set_yticks(range(3), names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("LSB per half-select pair")
    ax.set_xlim(min(vals) * 1.6, max(vals) * 1.6 + 0.4)
    ax.set_title("3. half-select leak, per ordered pair\n"
                 f"|leak| <= {max(abs(np.array(vals))):.1f} LSB vs "
                 f"{gp:.0f} LSB for one coincidence", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")

    fig.suptitle("Nonlinear update model: saturating coincidence burst + "
                 "half-select leak", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("update_model.png", dpi=110)
    print("saved -> update_model.png")


if __name__ == "__main__":
    main()

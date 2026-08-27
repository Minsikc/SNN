#!/usr/bin/env python3
"""Refit read disturbance using long (500-read) traces.

Why this supersedes the 60-read fit: those traces only decayed ~42% of the
way to their apparent asymptote, so the floor was an extrapolation from
curvature. It gave V_inf = 106 +/- 0.7 LSB and looked well constrained
(profile SSE rose 3-30x within +/-20 LSB), but a direct 500-read run walked
straight through it -- 79 LSB and still falling. The lesson is that a decay
must be followed for several time constants before its asymptote means
anything.

Models compared here on the long traces:
    1exp : V = Vinf + (V0 - Vinf) * f**n
    2exp : V = A*fa**n + B*fb**n + C
The first pass's 0.90 %/read is a blend of two rates, not one constant.
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
INK, GRID = "#0b0b0b", "#e1e0d9"


def one_exp(n, vinf, v0, f):
    return vinf + (v0 - vinf) * f ** n


def two_exp(n, a, fa, b, fb, c):
    return a * fa ** n + b * fb ** n + c


def load_long(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for r in rows:
        out.setdefault(int(r["n_prog"]), {}).setdefault(
            int(r["read_idx"]), []).append(float(r["level"]))
    traces = {}
    for npg, d in out.items():
        idx = sorted(d)
        traces[npg] = (np.array(idx, float),
                       np.array([np.mean(d[i]) for i in idx]))
    return traces


def load_short(pattern="*_disturb_exp_model.csv"):
    hits = sorted(glob.glob(pattern))
    if not hits:
        return {}
    rows = [r for r in csv.DictReader(open(hits[-1], encoding="utf-8"))
            if r["exp"] == "D"]
    out = {}
    for r in rows:
        out.setdefault(int(r["n_prog"]), {}).setdefault(
            int(r["read_idx"]), []).append(float(r["level"]))
    return {npg: (np.array(sorted(d), float),
                  np.array([np.mean(d[i]) for i in sorted(d)]))
            for npg, d in out.items()}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("*_disturb_long500.csv"))[-1]
    long_tr = load_long(path)
    short_tr = load_short()
    charges = sorted(long_tr)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fits = {}

    # ---- panel 1: long traces + both model fits ----
    ax = axes[0][0]
    for ci, npg in enumerate(charges):
        n, y = long_tr[npg]
        col = RAMP[-1 - ci]
        ax.plot(n, y, "o", ms=2.5, color=col, alpha=0.55,
                label=f"charge x{npg}  V$_0$={y[0]:.0f}")
        p1, _ = curve_fit(one_exp, n, y, p0=[60, y[0], 0.995], maxfev=200000)
        p2, _ = curve_fit(two_exp, n, y,
                          p0=[y[0] * 0.2, 0.98, y[0] * 0.7, 0.997, 50],
                          maxfev=400000)
        s1 = ((y - one_exp(n, *p1)) ** 2).sum()
        s2 = ((y - two_exp(n, *p2)) ** 2).sum()
        fits[npg] = (p1, s1, p2, s2, y[0])
        ax.plot(n, one_exp(n, *p1), "--", lw=1.6, color=col, alpha=0.9)
        ax.plot(n, two_exp(n, *p2), "-", lw=2.0, color=INK, alpha=0.75)
    ax.set_xlabel("array reads")
    ax.set_ylabel("state (LSB, N5$-$N6, 15 devices)")
    ax.set_title("500-read decay traces\n"
                 "dashed = single exponential + floor, black = double "
                 "exponential", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    # ---- panel 2: residuals, the reason 2exp wins ----
    ax = axes[0][1]
    for ci, npg in enumerate(charges):
        n, y = long_tr[npg]
        p1, s1, p2, s2, _ = fits[npg]
        col = RAMP[-1 - ci]
        ax.plot(n, y - one_exp(n, *p1), "--", lw=1.5, color=col,
                label=f"x{npg} 1exp (SSE {s1:.0f})")
        ax.plot(n, y - two_exp(n, *p2), "-", lw=1.8, color=col,
                label=f"x{npg} 2exp (SSE {s2:.0f})")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("array reads")
    ax.set_ylabel("residual (LSB)")
    ax.set_title("Fit residuals — the single exponential bows "
                 "systematically", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # ---- panel 3: what the 60-read fit predicted vs reality ----
    ax = axes[1][0]
    npg = charges[0]
    n, y = long_tr[npg]
    ax.plot(n, y, "o", ms=3, color=RAMP[3], label="measured (500 reads)")
    old = 106.0 + (y[0] - 106.0) * 0.99100 ** n
    ax.plot(n, old, lw=2, color="#eb6834",
            label="60-read fit extrapolated (V$_\\infty$=106, 0.90%/read)")
    ax.axhline(106.0, color="#eb6834", ls=":", lw=1.5)
    p1, _, p2, _, _ = fits[npg]
    ax.plot(n, two_exp(n, *p2), lw=2, color=INK, label="refit (double exp)")
    if short_tr:
        ax.axvspan(0, 60, color="#e1e0d9", alpha=0.55, zorder=0)
        ax.text(30, y[0] * 0.55, "range of the\noriginal fit",
                ha="center", fontsize=8, color="#52514e")
    ax.set_xlabel("array reads")
    ax.set_ylabel("state (LSB)")
    ax.set_title("Extrapolating a 42%-complete decay overshoots the floor",
                 fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    # ---- panel 4: instantaneous rate, showing two regimes ----
    ax = axes[1][1]
    for ci, npg in enumerate(charges):
        n, y = long_tr[npg]
        p1, _, p2, _, _ = fits[npg]
        floor = p2[4]
        above = np.clip(y - floor, 1e-6, None)
        # local %/read from a rolling log-slope
        w = 21
        rate = np.full(n.size, np.nan)
        for i in range(w, n.size - w):
            sl = np.polyfit(n[i - w:i + w], np.log(above[i - w:i + w]), 1)[0]
            rate[i] = (1 - np.exp(sl)) * 100
        ax.plot(n, rate, lw=2, color=RAMP[-1 - ci], label=f"charge x{npg}")
    ax.set_xlabel("array reads")
    ax.set_ylabel("instantaneous loss (% per read)")
    ax.set_title("Rate is not one constant: fast early, slower later",
                 fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Read disturbance refit on 500-read traces — "
                 "15 good devices (cols 3-5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "disturb_long500_refit.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)

    print("\n%-8s %-42s %8s" % ("charge", "single exp + floor", "SSE"))
    for npg in charges:
        p1, s1, p2, s2, v0 = fits[npg]
        print("%-8d Vinf=%6.1f V0=%6.1f f=%.5f (%.3f%%/rd) %8.1f"
              % (npg, p1[0], p1[1], p1[2], (1 - p1[2]) * 100, s1))
    print("\n%-8s %-42s %8s" % ("charge", "double exp", "SSE"))
    for npg in charges:
        _, _, p2, s2, _ = fits[npg]
        print("%-8d A=%5.0f fa=%.5f (%.2f%%) B=%5.0f fb=%.5f (%.3f%%) C=%5.1f  %6.1f"
              % (npg, p2[0], p2[1], (1 - p2[1]) * 100,
                 p2[2], p2[3], (1 - p2[3]) * 100, p2[4], s2))


if __name__ == "__main__":
    main()

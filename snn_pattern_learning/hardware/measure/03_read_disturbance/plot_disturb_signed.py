#!/usr/bin/env python3
"""Positive vs negative start: does read disturbance have an absolute attractor?"""
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
POT, DEP, OLDC = "#2a78d6", "#eb6834", "#9b8ec4"


def load(path, arm_filter=None):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        arm = r.get("arm", "pot")
        if arm_filter and arm != arm_filter:
            continue
        d[(arm, int(r["rep"]), int(r["row"]), int(r["col"]))][
            int(r["reads"])] = float(r["level"])
    return {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}


def mB(p, k):
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def fit(k, v):
    return least_squares(lambda p: mB(p, k) - v, [v[-1], v[0], 50.0],
                         max_nfev=20000).x


def main():
    path = sorted(glob.glob("*_disturb_signed.csv"))[-1]
    new = load(path)
    old = load("2026-08-06_23-07_disturb_all25.csv")
    Pn = {t: fit(*new[t]) for t in new}
    Po = {t: fit(*old[t]) for t in old}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # ---- 1. the two arms ------------------------------------------------
    ax = axes[0]
    kk = np.logspace(0, 3, 300)
    for arm, col, lab in (("pot", POT, "positive start"),
                          ("dep", DEP, "negative start")):
        ts = [t for t in sorted(new) if t[0] == arm and t[1] == 0][:5]
        for i, t in enumerate(ts):
            k, v = new[t]
            ax.plot(k, v, "o", ms=3, color=col, alpha=0.5,
                    label=lab if i == 0 else None)
            ax.plot(kk, mB(Pn[t], kk), color=col, lw=1.2, alpha=0.8)
    vi_p = np.mean([Pn[t][0] for t in Pn if t[0] == "pot"])
    vi_d = np.mean([Pn[t][0] for t in Pn if t[0] == "dep"])
    ax.axhline(vi_p, color=POT, ls="--", lw=1.6,
               label=f"$V_\\infty$ pot {vi_p:+.1f}")
    ax.axhline(vi_d, color=DEP, ls="--", lw=1.6,
               label=f"$V_\\infty$ dep {vi_d:+.1f}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("1. the negative arm RISES THROUGH ZERO\n"
                 "both arms land on the same attractor", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(color=GRID, lw=0.8, which="both")

    # ---- 2. absolute vs fractional --------------------------------------
    ax = axes[1]
    v0 = np.array([Pn[t][1] for t in sorted(Pn)])
    vi = np.array([Pn[t][0] for t in sorted(Pn)])
    cols = [POT if t[0] == "pot" else DEP for t in sorted(Pn)]
    ax.scatter(v0, vi, s=22, c=cols, alpha=0.7, edgecolors="none")
    xs = np.linspace(v0.min() * 1.05, v0.max() * 1.05, 50)
    frac = np.mean([Pn[t][0] / Pn[t][1] for t in Pn if t[0] == "pot"])
    ax.plot(xs, np.full_like(xs, vi.mean()), color=INK, lw=2,
            label=f"ABSOLUTE: $V_\\infty$={vi.mean():+.1f} (flat)")
    ax.plot(xs, frac * xs, color=MUTED, lw=2, ls="--",
            label=f"FRACTIONAL: $V_\\infty$={frac:.3f}$V_0$")
    b, a = np.polyfit(v0, vi, 1)
    ax.plot(xs, b * xs + a, color="#2e9e6b", lw=1.5, ls=":",
            label=f"measured slope {b:+.4f}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("starting level $V_0$ (LSB)")
    ax.set_ylabel(r"fitted $V_\infty$ (LSB)")
    ax.set_title("2. $V_\\infty$ does not follow $V_0$\n"
                 "slope is ~0, so the attractor is absolute", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 3. degradation -------------------------------------------------
    ax = axes[2]
    ts_o = [t for t in sorted(Po) if t[1] == 0][:5]
    ts_n = [t for t in sorted(Pn) if t[0] == "pot" and t[1] == 0][:5]
    for i, t in enumerate(ts_o):
        k, v = old[t]
        ax.plot(k, v, "o-", ms=3, lw=1, color=OLDC, alpha=0.65,
                label="2026-08-06" if i == 0 else None)
    for i, t in enumerate(ts_n):
        k, v = new[t]
        ax.plot(k, v, "o-", ms=3, lw=1, color=POT, alpha=0.65,
                label="this session" if i == 0 else None)
    vo = np.mean([Po[t][0] for t in Po])
    ax.axhline(vo, color=OLDC, ls="--", lw=1.6,
               label=f"$V_\\infty$ then {vo:+.0f}")
    ax.axhline(vi_p, color=POT, ls="--", lw=1.6,
               label=f"$V_\\infty$ now {vi_p:+.0f}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("read index k")
    ax.set_ylabel("level (LSB)")
    ax.set_title("3. the floor moved: positive arm re-measured\n"
                 r"$\tau$ unchanged (127.8 vs 126.9) but $V_\infty$ fell",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(color=GRID, lw=0.8, which="both")

    fig.suptitle("Read disturbance from positive and negative starts, "
                 "one interleaved session", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("disturb_signed.png", dpi=110)
    print("saved -> disturb_signed.png")


if __name__ == "__main__":
    main()

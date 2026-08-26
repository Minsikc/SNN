#!/usr/bin/env python3
"""Fit and plot read disturbance / leakage as exponential decays.

Everything is plotted as a FRACTION of the initial state, because both
processes are multiplicative: a fixed LSB-per-read number is an artifact of
looking at one charge level. Semi-log panels are included since an
exponential is a straight line there -- that is the visual test of the model,
not the fit statistic.

Reads:  V_n = Vinf + (V0 - Vinf) * f**n        f = per-read retention
Leak:   V(t) = Vinf + (V0 - Vinf) * exp(-t/tau)
"""
import csv
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# one hue, two shades (validated: #6da7ec / #1c5cab pass the palette checks);
# charge levels use a light->dark sequential ramp
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def decay_reads(n, vinf, v0, f):
    return vinf + (v0 - vinf) * f ** n


def decay_time(t, vinf, v0, tau):
    return vinf + (v0 - vinf) * np.exp(-t / tau)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("*_disturb_exp_model.csv"))[-1]
    rows = load(path)
    D = [r for r in rows if r["exp"] == "D"]
    E = [r for r in rows if r["exp"] == "E"]
    charges = sorted({int(r["n_prog"]) for r in D})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # A global fit is used rather than one per charge level: the decay goes
    # to a common nonzero FLOOR, and only the part above that floor decays
    # exponentially. Fitting each level independently hides that -- it makes
    # the per-read % look state-dependent (0.33%/read at V0=177 vs
    # 0.62% at V0=409) when in fact one (Vinf, f) pair explains all levels
    # (SSE 8 vs 675 for a floorless model).
    traces = []
    for npg in charges:
        idx = sorted({int(r["read_idx"]) for r in D if int(r["n_prog"]) == npg})
        y = np.array([np.mean([float(r["level"]) for r in D
                               if int(r["n_prog"]) == npg
                               and int(r["read_idx"]) == i]) for i in idx])
        traces.append((npg, np.array(idx, float) - 1, y))

    def joint(_, vinf, f, *v0s):
        return np.concatenate([vinf + (v0s[k] - vinf) * f ** n
                               for k, (_, n, _) in enumerate(traces)])

    ally = np.concatenate([y for _, _, y in traces])
    p0 = [80.0, 0.99] + [y[0] for _, _, y in traces]
    gp, _ = curve_fit(joint, np.arange(ally.size), ally, p0=p0, maxfev=80000)
    VINF, F = gp[0], gp[1]

    # ---------- panel 1: normalized to the DECAYING part ----------
    ax = axes[0][0]
    fits = []
    for ci, (npg, n, y) in enumerate(traces):
        v0 = y[0]
        fits.append((npg, v0, (VINF, v0, F)))
        ax.plot(n, (y - VINF) / (v0 - VINF), "o", ms=4, color=RAMP[ci],
                markerfacecolor="white", markeredgewidth=1.4, zorder=4,
                label=f"charge x{npg}  V$_0$={v0:.0f}")
    nn = np.linspace(0, max(n.max() for _, n, _ in traces), 200)
    ax.plot(nn, F ** nn, color=INK, lw=2, ls="--", zorder=5,
            label=f"shared fit  f={F:.5f}  ({(1-F)*100:.2f}%/read)")
    ax.set_xlabel("array reads")
    ax.set_ylabel(r"(V $-$ V$_\infty$) / (V$_0$ $-$ V$_\infty$)")
    ax.set_title("Read disturbance, normalized to the decaying part\n"
                 f"all charge levels collapse once the V$_\\infty$="
                 f"{VINF:.0f} LSB floor is removed", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=7.5, frameon=False)

    # ---------- panel 2: semi-log ----------
    ax = axes[0][1]
    for ci, (npg, n, y) in enumerate(traces):
        ax.semilogy(n, (y - VINF) / (y[0] - VINF), "o", ms=3.5,
                    color=RAMP[ci], label=f"charge x{npg}")
    ax.semilogy(nn, F ** nn, color=INK, lw=2, ls="--", label="shared fit")
    ax.set_xlabel("array reads")
    ax.set_ylabel(r"(V $-$ V$_\infty$)/(V$_0$ $-$ V$_\infty$)  (log)")
    ax.set_title("Semi-log: straight and parallel = one exponential rate",
                 fontsize=11)
    ax.grid(color=GRID, lw=0.8, which="both")
    ax.legend(fontsize=8, frameon=False)

    # ---------- panel 3: leakage vs time ----------
    ax = axes[1][0]
    waits = sorted({float(r["wait"]) for r in E})
    ratio = np.array([np.mean([float(r["level"]) / float(r["ref"])
                               for r in E if float(r["wait"]) == w])
                      for w in waits])
    err = np.array([np.std([float(r["level"]) / float(r["ref"])
                            for r in E if float(r["wait"]) == w], ddof=1)
                    for w in waits])
    t = np.array(waits)
    # divide out the probe read's own cost (the t=0 point is pure probe)
    probe = ratio[0] if t[0] == 0 else 1.0
    corr = ratio / probe
    ax.errorbar(t, corr, yerr=err / probe, fmt="o", ms=6, color=RAMP[3],
                markerfacecolor="white", markeredgewidth=1.6, capsize=3,
                zorder=4, label="measured (probe cost divided out)")
    try:
        p, _ = curve_fit(decay_time, t, corr, p0=[0.97, 1.0, 10],
                         maxfev=40000)
        tt = np.linspace(0, t.max(), 300)
        ax.plot(tt, decay_time(tt, *p), color=RAMP[1], lw=2, zorder=3,
                label=f"fit: floor={p[0]:.4f}, tau={p[2]:.1f}s")
        leak_total = (1 - p[0]) * 100
    except Exception:
        leak_total = float("nan")
    ax.set_xlabel("wait before the probe read (s)")
    ax.set_ylabel("retained fraction")
    ax.set_title(f"Leakage — decays to a floor, total loss ~{leak_total:.1f}%",
                 fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    # ---------- panel 4: which dominates, per unit of use ----------
    ax = axes[1][1]
    if fits:
        fmean = np.mean([p[2] for _, _, p in fits])
        n = np.arange(0, 41)
        read_loss = (1 - fmean ** n) * 100
        ax.plot(n, read_loss, lw=2.5, color=RAMP[3],
                label=f"reads ({(1-fmean)*100:.2f}%/read)")
    tsec = np.linspace(0, 40 * 0.215, 200)     # same wall-clock as 40 reads
    try:
        leak_curve = (1 - decay_time(tsec, *p)) * 100
        ax.plot(tsec / 0.215, leak_curve, lw=2.5, color=RAMP[1],
                label="leakage over the same wall-clock")
    except Exception:
        pass
    ax.set_xlabel("array reads  (lower axis doubles as elapsed time)")
    ax.set_ylabel("cumulative loss (%)")
    ax.set_title("Read count, not time, is the budget", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("6T1C state retention: read disturbance vs leakage — "
                 "15 good devices (cols 3-5), charged then probed",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "disturb_leakage_exponential.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)

    print(f"\nshared model:  V_n = {VINF:.1f} + (V0 - {VINF:.1f}) * "
          f"{F:.5f}**n")
    print(f"  floor V_inf = {VINF:.1f} LSB   "
          f"per-read loss above floor = {(1-F)*100:.3f}%")
    print("\napparent per-read % if the floor is (wrongly) ignored:")
    for npg, v0, _ in fits:
        idx = [t for t in traces if t[0] == npg][0]
        y = idx[2]
        n_last = idx[1][-1]
        app = (1 - (y[-1] / y[0]) ** (1 / n_last)) * 100
        print(f"  charge x{npg}  V0={v0:6.1f} -> {app:5.3f}%/read")
    print("  (this spread is an artifact of the floor, not real state "
          "dependence)")


if __name__ == "__main__":
    main()

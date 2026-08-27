#!/usr/bin/env python3
"""Cycle test (full-range POT/DEP sweep) overlaid with the aihwkit LinearStep fit.

This is the decisive test of the fit.  The LinearStep parameters were fitted to
random-signed u,v bursts that never left -162..+39 LSB, so everything outside
that band was extrapolation.  The cycle test drives the cell rail to rail, so
it measures exactly the region the fit had to guess.

Two device variants are drawn:

  fitted    gamma from the burst decay (1.78 / 1.71).  Reproduces the short
            burst curve, but its step vanishes near +255/-260 LSB.
  gamma=1   'soft bounds': the step vanishes exactly at the measured rails.

Whichever tracks the cycle test is the one to use for full-range simulation.

The measurement gives one ADC read per step-block, and each block is a fixed
number of update pulses.  That pulse count is not recorded in the CSV, so it is
calibrated once from the data (a single scalar, shared by all 25 cells and both
variants) rather than assumed.
"""
import ast
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
MEAS, FIT, SOFT = "#2a78d6", "#eb6834", "#2e9e6b"
N = 5


def load_cycle(path):
    """-> diff[cycle, read_event, row, col] in LSB, and reads per half-cycle.

    GLOBAL_PD_SEQ_READ runs `update_num` potentiation pulses, calling
    Read_all_rows_sequentially every `read_period` pulses, then does the same
    for depression.  Each read event emits FIVE csv fields, one per array row.
    So the field axis is (read_event x row), not (step x half-block): reshaping
    it any other way interleaves the potentiation and depression ramps.
    """
    rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
    raw = np.array([[ast.literal_eval(f) for f in r] for r in rows]).astype(int)
    n_cy, n_fields = raw.shape[0], raw.shape[1]
    n_ev = n_fields // N
    b = raw.reshape(n_cy, n_ev, N, 10)
    return b[:, :, :, 0:N] - b[:, :, :, N:2 * N], n_ev // 2


def series(diff, r, c, steps):
    """One continuous trace: every cycle's POT ramp then DEP ramp."""
    return np.concatenate([diff[cy, :, r, c] for cy in range(diff.shape[0])])


class LinearStep:
    """aihwkit LinearStepDevice, noiseless, in LSB units."""

    def __init__(self, L, lsb, gamma_up=None, gamma_down=None):
        ub, db = min(L["up_down"], 0.0), max(-L["up_down"], 0.0)
        s_up = (ub + 1.0) * L["dw_min"]
        s_dn = (db + 1.0) * L["dw_min"]
        gu = L["gamma_up"] if gamma_up is None else gamma_up
        gd = L["gamma_down"] if gamma_down is None else gamma_down
        # work directly in LSB
        self.s_up = s_up * lsb
        self.s_dn = s_dn * lsb
        self.wmax = L["w_max"] * lsb
        self.wmin = L["w_min"] * lsb
        self.sl_up = -gu * s_up / L["w_max"]
        self.sl_dn = -gd * s_dn / L["w_min"]

    def pulse(self, w, positive):
        if positive:
            return min(w + self.sl_up * w + self.s_up, self.wmax)
        return max(w - (self.sl_dn * w + self.s_dn), self.wmin)

    def run(self, w0, n_cycles, steps, pulses_per_step):
        """Replay the cycle-test protocol, sampling once per step-block."""
        out, w = [], w0
        for _ in range(n_cycles):
            for _ in range(steps):                 # potentiation half
                for _ in range(pulses_per_step):
                    w = self.pulse(w, True)
                out.append(w)
            for _ in range(steps):                 # depression half
                for _ in range(pulses_per_step):
                    w = self.pulse(w, False)
                out.append(w)
        return np.array(out)


def calibrate(dev, meas_list, w0_list, n_cy, steps, hi=40):
    """One shared scalar: pulses per read interval, chosen by total rms.

    The CSV does not record read_period, so this recovers it.  It is a single
    integer shared by all 25 cells and never refitted per cell or per variant
    beyond this one search.
    """
    best, bestn = None, 1
    for n in range(1, hi + 1):
        err = []
        for m, w0 in zip(meas_list, w0_list):
            err.append(dev.run(w0, n_cy, steps, n) - m)
        e = float(np.sqrt(np.mean(np.square(np.concatenate(err)))))
        if best is None or e < best:
            best, bestn = e, n
    return bestn, best


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else "2026-08-10_14-29_CycleTest_Data.csv")
    diff, steps = load_cycle(path)
    n_cy = diff.shape[0]
    cfg = json.load(open("aihwkit_linearstep_params.json", encoding="utf-8"))
    L, lsb = cfg["linearstep"], cfg["lsb_per_w"]

    meas = [series(diff, r, c, steps) for r in range(N) for c in range(N)]
    w0s = [m[0] - (m[1] - m[0]) for m in meas]     # back off one step

    dev_fit = LinearStep(L, lsb)
    dev_soft = LinearStep(L, lsb, gamma_up=1.0, gamma_down=1.0)
    n_fit, e_fit = calibrate(dev_fit, meas, w0s, n_cy, steps)
    n_soft, e_soft = calibrate(dev_soft, meas, w0s, n_cy, steps)

    print(f"{path}: {n_cy} cycles x {steps} steps per half")
    print(f"measured range {min(m.min() for m in meas):.0f} .. "
          f"{max(m.max() for m in meas):.0f} LSB")
    print(f"  fitted gamma ({L['gamma_up']:.2f}/{L['gamma_down']:.2f}): "
          f"{n_fit} pulses/step, rms {e_fit:.1f} LSB")
    print(f"  gamma = 1.0 (soft bounds):     {n_soft} pulses/step, "
          f"rms {e_soft:.1f} LSB")

    fig, axes = plt.subplots(N, N, figsize=(20, 15), sharex=True, sharey=True)
    lim = 20 * int(np.ceil(max(abs(min(m.min() for m in meas)),
                               max(m.max() for m in meas)) / 20 + 0.5))
    for i in range(N * N):
        r, c = divmod(i, N)
        ax = axes[r][c]
        m = meas[i]
        x = np.arange(len(m))
        ax.plot(x, m, marker=".", ms=3, lw=1.2, color=MEAS, label="measured",
                zorder=3)
        ax.plot(x, dev_fit.run(w0s[i], n_cy, steps, n_fit), lw=1.6,
                color=FIT, alpha=0.85,
                label=f"LinearStep, fitted $\\gamma$", zorder=2)
        ax.plot(x, dev_soft.run(w0s[i], n_cy, steps, n_soft), lw=1.6,
                color=SOFT, alpha=0.85, ls="--",
                label="LinearStep, $\\gamma$=1", zorder=2)
        ax.set_title(f"Cell ({r+1}, {c+1})", fontsize=10)
        ax.grid(alpha=0.3, ls=":")
        ax.set_ylim(-lim, lim)
        for k in range(1, n_cy):
            ax.axvline(k * 2 * steps - 0.5, color="k", lw=1)
        for k in range(n_cy):
            ax.axvline(k * 2 * steps + steps - 0.5, color="gray", ls="--",
                       lw=0.8)
        if i == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        f"Cycle test vs fitted aihwkit LinearStepDevice\n"
        f"fitted $\\gamma$ rms {e_fit:.0f} LSB ({n_fit} pulses/step)   |   "
        f"$\\gamma$=1 rms {e_soft:.0f} LSB ({n_soft} pulses/step)",
        fontsize=15)
    fig.supxlabel("Total Measurement Steps (Cycle 1 -> Cycle 2 -> ...)",
                  fontsize=12)
    fig.supylabel("Differential ADC Value (N5 - N6)", fontsize=12)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.955])
    fig.savefig("cycletest_vs_aihwkit.png", dpi=100)
    print("\nsaved -> cycletest_vs_aihwkit.png")

    # per-cell breakdown, so a good total rms cannot hide a bad cell
    print(f"\n{'cell':>6s} {'fitted':>9s} {'gamma=1':>9s}")
    for i in range(N * N):
        r, c = divmod(i, N)
        a = np.sqrt(np.mean((dev_fit.run(w0s[i], n_cy, steps, n_fit)
                             - meas[i]) ** 2))
        b = np.sqrt(np.mean((dev_soft.run(w0s[i], n_cy, steps, n_soft)
                             - meas[i]) ** 2))
        print(f"({r+1},{c+1}) {a:9.1f} {b:9.1f}")


if __name__ == "__main__":
    main()

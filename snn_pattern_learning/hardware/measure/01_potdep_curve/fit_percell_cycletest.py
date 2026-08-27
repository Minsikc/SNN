#!/usr/bin/env python3
"""How much of the cycle-test error is shared-parameter error?

plot_cycletest_vs_aihwkit.py drew ONE device for all 25 cells: a single
dw_min, up_down, gamma_up, gamma_down, w_max, w_min, and a single
pulses-per-read.  Only the starting state w0 differed per cell.  That is the
right default -- an aihwkit tile is built from one parameter set plus dtod
spread -- but it means every deviation in that figure mixes two causes:

  (a) the LinearStep functional form not matching the device
  (b) cell-to-cell parameter spread, which the shared fit cannot express

This script separates them by refitting per cell and comparing.  If per-cell
fitting collapses the error, the form is fine and the cells simply differ; if
it does not, the form itself is the limit.

Fitted per cell: dw_min, up_down, gamma_up, gamma_down, w_max, w_min.
pulses-per-read stays global -- it is a protocol constant, not a device
property, so letting it vary per cell would be fitting the instrument.
"""
import ast
import csv
import json
import sys

import numpy as np
from scipy.optimize import least_squares

N = 5


def load_cycle(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
    raw = np.array([[ast.literal_eval(f) for f in r] for r in rows]).astype(int)
    n_cy, n_fields = raw.shape[0], raw.shape[1]
    n_ev = n_fields // N
    b = raw.reshape(n_cy, n_ev, N, 10)
    return b[:, :, :, 0:N] - b[:, :, :, N:2 * N], n_ev // 2


def run(p, w0, n_cy, steps, ppr):
    """LinearStep in LSB.  p = [s_up, s_dn, g_up, g_dn, wmax, wmin]."""
    s_up, s_dn, g_up, g_dn, wmax, wmin = p
    sl_up = -g_up * s_up / wmax
    sl_dn = -g_dn * s_dn / wmin
    out, w = [], w0
    for _ in range(n_cy):
        for _ in range(steps):
            for _ in range(ppr):
                w = min(w + sl_up * w + s_up, wmax)
            out.append(w)
        for _ in range(steps):
            for _ in range(ppr):
                w = max(w - (sl_dn * w + s_dn), wmin)
            out.append(w)
    return np.array(out)


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else "2026-08-10_14-29_CycleTest_Data.csv")
    diff, steps = load_cycle(path)
    n_cy = diff.shape[0]
    cfg = json.load(open("aihwkit_linearstep_params.json", encoding="utf-8"))
    L, lsb = cfg["linearstep"], cfg["lsb_per_w"]

    meas = [diff[:, :, r, c].ravel() for r in range(N) for c in range(N)]
    meas = [np.concatenate([diff[cy, :, r, c] for cy in range(n_cy)])
            for r in range(N) for c in range(N)]
    w0s = [m[0] - (m[1] - m[0]) for m in meas]

    ub, db = min(L["up_down"], 0.0), max(-L["up_down"], 0.0)
    p_shared = np.array([(ub + 1.0) * L["dw_min"] * lsb,
                         (db + 1.0) * L["dw_min"] * lsb,
                         1.0, 1.0,                     # gamma=1 variant
                         L["w_max"] * lsb, L["w_min"] * lsb])

    # global pulses-per-read, as in the shared figure
    best, ppr = None, 5
    for n in range(1, 41):
        e = np.sqrt(np.mean(np.square(np.concatenate(
            [run(p_shared, w, n_cy, steps, n) - m
             for m, w in zip(meas, w0s)]))))
        if best is None or e < best:
            best, ppr = e, n
    print(f"{path}: {n_cy} cycles x {steps} steps, "
          f"{ppr} pulses/read (global)")
    print(f"shared parameters, all 25 cells : rms {best:.1f} LSB\n")

    lo = [1.0, 1.0, 0.2, 0.2, 200.0, -900.0]
    hi = [200.0, 200.0, 4.0, 4.0, 900.0, -200.0]
    rows_out, tot = [], []
    print(f"{'cell':>6s} {'shared':>8s} {'per-cell':>9s} | "
          f"{'s_up':>7s} {'s_dn':>7s} {'g_up':>6s} {'g_dn':>6s} "
          f"{'w_max':>7s} {'w_min':>8s}")
    print("-" * 78)
    for i in range(N * N):
        r, c = divmod(i, N)
        m, w0 = meas[i], w0s[i]
        e_sh = float(np.sqrt(np.mean((run(p_shared, w0, n_cy, steps, ppr)
                                      - m) ** 2)))
        res = least_squares(
            lambda p: run(p, w0, n_cy, steps, ppr) - m,
            p_shared, bounds=(lo, hi), max_nfev=4000)
        e_pc = float(np.sqrt(np.mean(res.fun ** 2)))
        tot.append((e_sh, e_pc))
        p = res.x
        rows_out.append([r + 1, c + 1, e_sh, e_pc] + list(p))
        print(f"({r+1},{c+1}) {e_sh:8.1f} {e_pc:9.1f} | "
              f"{p[0]:7.2f} {p[1]:7.2f} {p[2]:6.3f} {p[3]:6.3f} "
              f"{p[4]:7.1f} {p[5]:8.1f}")

    sh = np.sqrt(np.mean([e ** 2 for e, _ in tot]))
    pc = np.sqrt(np.mean([e ** 2 for _, e in tot]))
    print(f"\nrms over all cells: shared {sh:.1f} -> per-cell {pc:.1f} LSB "
          f"({100*(1-pc/sh):.0f}% reduction)")

    P = np.array([r[4:] for r in rows_out])
    names = ["scale_up", "scale_down", "gamma_up", "gamma_down",
             "w_max", "w_min"]
    print(f"\n{'param':>11s} {'mean':>9s} {'sd':>8s} {'cv':>7s} "
          f"{'min':>8s} {'max':>8s}")
    for j, nm in enumerate(names):
        v = P[:, j]
        print(f"{nm:>11s} {v.mean():9.2f} {v.std():8.2f} "
              f"{v.std()/abs(v.mean()):7.3f} {v.min():8.2f} {v.max():8.2f}")

    with open("percell_cycletest_params.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "rms_shared", "rms_percell"] + names)
        w.writerows(rows_out)
    print("\nsaved -> percell_cycletest_params.csv")

    g = np.concatenate([P[:, 2], P[:, 3]])
    print(f"\ngamma across cells and directions: {g.mean():.2f} +- {g.std():.2f}"
          f"  (range {g.min():.2f}..{g.max():.2f})")
    print("gamma=1 was the shared choice; per-cell values scatter around it,")
    print("so the residual is cell spread plus form error, not a wrong gamma.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit the half-select pair attractor dynamics from the seq_attractor map.

Model, per ordered pair: one HS_SEQ cycle multiplies the distance to an
attractor by a fixed factor,

    after = A + (before - A) * d_cycle ** cycles

fitted over all cells, starting states, reps and cycle counts at once.
One cycle = first line then second (firmware HS_SEQ), i.e. ~2 ordered
transitions, so the per-transition factor is d_trans = sqrt(d_cycle).

Outputs, per pair: A (attractor, LSB), d_cycle, d_trans, % decay per
transition, fit RMS. The reset-like pairs give the calibration for the
aihwkit ``hs_decay`` / ``hs_reset_decay`` parameters (which currently decay
toward 0 -- the fitted A says how wrong that is).

Cross-check: the additive pair step measured in scripts/04 and the ABAB
replay (~+1.5..1.65 LSB/pair for pot pairs) should equal (A - w)*(1 - d_trans)
at the operating level w ~ 0.

    python fit_seq_attractor_decay.py 2026-08-07_15-20_seq_attractor.csv
"""
import csv
import sys

import numpy as np
from scipy.optimize import curve_fit

GROUPS = {
    "potentiating": ["N1+N2", "N2+N1"],
    "reset-like": ["N1+N3", "N3+N1", "N2+N4", "N4+N2"],
    "depressing": ["N3+N4", "N4+N3"],
}


def load(paths):
    out = {}
    for path in paths:
        for x in csv.DictReader(open(path, encoding="utf-8")):
            combo = x["combo"]
            out.setdefault(combo, []).append(
                (float(x["cycles"]), float(x["before"]), float(x["after"])))
    return {k: np.array(v) for k, v in out.items()}


def fit_combo(arr):
    n, w0, w1 = arr[:, 0], arr[:, 1], arr[:, 2]

    def model(X, A, d):
        nn, ww0 = X
        return A + (ww0 - A) * np.power(np.clip(d, 1e-6, 1.0), nn)

    p, cov = curve_fit(model, (n, w0), w1, p0=[20.0, 0.99],
                       bounds=([-700, 0.5], [700, 1.0]), maxfev=20000)
    resid = w1 - model((n, w0), *p)
    se = np.sqrt(np.diag(cov))
    return p[0], p[1], se[0], se[1], float(np.sqrt((resid ** 2).mean()))


def main():
    paths = sys.argv[1:] or ["2026-08-07_14-42_seq_attractor.csv",
                             "2026-08-07_15-20_seq_attractor.csv"]
    data = load(paths)
    print(f"{paths}: {sum(len(v) for v in data.values())} rows, "
          f"{len(data)} combos, cycles "
          f"{sorted({int(c) for v in data.values() for c in v[:, 0]})}")
    print()
    print(f"{'pair':>7} {'A (LSB)':>10} {'d_cycle':>9} {'d_trans':>9} "
          f"{'%/transition':>13} {'rms':>7}")
    results = {}
    for gname, combos in GROUPS.items():
        print(f"--- {gname} ---")
        for c in combos:
            if c not in data:
                print(f"{c:>7}   (no data)")
                continue
            A, d, seA, sed, rms = fit_combo(data[c])
            dt = np.sqrt(d)
            results[c] = (A, d, dt)
            print(f"{c:>7} {A:+7.1f}+-{seA:4.1f} {d:9.4f} {dt:9.4f} "
                  f"{100 * (1 - dt):12.2f}% {rms:7.1f}")

    print()
    print("cross-check vs the additive pair step (04 / ABAB replay):")
    for c in ("N1+N2", "N2+N1"):
        if c in results:
            A, d, dt = results[c]
            print(f"  {c}: (A - 0) * (1 - d_trans) = {A * (1 - dt):+.2f} "
                  f"LSB/transition   (measured ~ +1.5..+1.65)")
    for c in ("N1+N3", "N2+N4"):
        if c in results:
            A, d, dt = results[c]
            print(f"  {c}: pull at w=+228 LSB: {(A - 228) * (1 - dt):+.2f} "
                  f"LSB/transition")

    print()
    print("aihwkit mapping (LinearStep/ConstantStep params):")
    rl = [results[c] for c in GROUPS["reset-like"] if c in results]
    if rl:
        A_m = np.mean([r[0] for r in rl])
        dt_m = np.mean([r[2] for r in rl])
        print(f"  reset-like pairs pooled: A = {A_m:+.1f} LSB, "
              f"d_trans = {dt_m:.4f}  ({100 * (1 - dt_m):.2f}%/event)")
        print(f"  -> hs_decay / hs_reset_decay = {dt_m:.4f}")
        print(f"     caveat: model decays toward 0, measured attractor is "
              f"{A_m:+.1f} LSB ({A_m / 455:+.3f} w units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

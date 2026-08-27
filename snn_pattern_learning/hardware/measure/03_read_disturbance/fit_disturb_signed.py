#!/usr/bin/env python3
"""Absolute attractor or fixed fraction?  Decided by the negative-start arm.

Model B is V_k = V_inf + (V_0 - V_inf) exp(-k/tau).  Fitted only to positive
starts it cannot say what V_inf means, because two readings coincide there:

    ABSOLUTE    V_inf is a fixed level the cell relaxes toward, the same
                number regardless of where it started
    FRACTIONAL  V_inf = f * V_0, so the cell keeps a fixed share of its
                starting level and the "floor" flips sign with the start

From a negative start they predict opposite things.  With V_0 ~ -390 LSB:

    ABSOLUTE    V_inf ~ +67 LSB   (the same attractor; the cell rises past 0)
    FRACTIONAL  V_inf ~ -66 LSB   (0.17 x -390; the cell stays negative)

So the sign of the fitted V_inf in the depression arm decides it outright.

Both arms come from one interleaved session, so the positive arm here is the
degradation-matched control for the negative arm -- not the 2026-08-06 run.
The old positive numbers are printed alongside only to show how much the
device moved, never as the comparison baseline.
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

OLD = dict(tau=126.87, tau_d2d=5.80, tau_c2c=3.82,
           vinf=67.35, vinf_d2d=21.87, vinf_c2c=4.31, v0=394.7)


def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["arm"], int(r["rep"]), int(r["row"]), int(r["col"]))
        d[key][int(r["reads"])] = float(r["level"])
    return {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}


def mB(p, k):
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def fit(k, v):
    return least_squares(lambda p: mB(p, k) - v, [v[-1], v[0], 50.0],
                         max_nfev=20000).x


def decompose(per_cell):
    cells = sorted(per_cell)
    cm = np.array([np.mean(per_cell[c]) for c in cells])
    wi = [np.std(per_cell[c], ddof=1) for c in cells if len(per_cell[c]) > 1]
    c2c = float(np.sqrt(np.mean(np.square(wi)))) if wi else 0.0
    n = np.mean([len(per_cell[c]) for c in cells])
    d2d = float(np.sqrt(max(cm.std(ddof=1) ** 2 - c2c ** 2 / n, 0.0)))
    return float(cm.mean()), d2d, c2c, cm


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else sorted(glob.glob("*_disturb_signed.csv"))[-1])
    data = load(path)
    arms = sorted({k[0] for k in data})
    print(f"{path}: {len(data)} traces, arms {arms}")
    kmax = int(max(data[k][0].max() for k in data))
    print(f"reads 1..{kmax}\n")

    P = {t: fit(*data[t]) for t in data}
    rms = [np.sqrt(np.mean((mB(P[t], data[t][0]) - data[t][1]) ** 2))
           for t in data]
    print(f"model B fit: rms {np.mean(rms):.2f} LSB "
          f"(worst {np.max(rms):.2f})\n")

    res = {}
    for arm in arms:
        keys = [k for k in P if k[0] == arm]
        cells = sorted({k[2:] for k in keys})
        reps = sorted({k[1] for k in keys})
        print("=" * 66)
        print(f"arm: {arm}")
        print("=" * 66)
        r = {}
        for idx, nm, unit in ((1, "V_0", " LSB"), (2, "tau", " reads"),
                              (0, "V_inf", " LSB")):
            per = {c: [P[(arm, rp) + c][idx] for rp in reps
                       if (arm, rp) + c in P] for c in cells}
            m, d2d, c2c, cm = decompose(per)
            r[nm] = (m, d2d, c2c, cm)
            print(f"  {nm:<7s} mean {m:9.2f}{unit}   d2d {d2d:7.2f} "
                  f"(cv {d2d/abs(m):.3f})   c2c {c2c:6.2f} "
                  f"(cv {c2c/abs(m):.3f})")
        # fraction form
        per_f = {c: [P[(arm, rp) + c][0] / P[(arm, rp) + c][1] for rp in reps
                     if (arm, rp) + c in P] for c in cells}
        mf, df, cf, _ = decompose(per_f)
        r["frac"] = (mf, df, cf, None)
        print(f"  {'Vinf/V0':<7s} mean {mf:9.4f}       d2d {df:7.4f} "
              f"(cv {df/abs(mf):.3f})   c2c {cf:6.4f}")
        res[arm] = r
        print()

    # ---- the verdict ---------------------------------------------------
    if "dep" in res and "pot" in res:
        vp, vd = res["pot"]["V_inf"][0], res["dep"]["V_inf"][0]
        v0p, v0d = res["pot"]["V_0"][0], res["dep"]["V_0"][0]
        fp, fd = res["pot"]["frac"][0], res["dep"]["frac"][0]
        pred_abs = vp
        pred_frac = fp * v0d
        print("=" * 66)
        print("VERDICT: what does V_inf mean?")
        print("=" * 66)
        print(f"  positive arm: V_0 {v0p:+8.1f} -> V_inf {vp:+8.1f} LSB "
              f"(fraction {fp:+.3f})")
        print(f"  negative arm: V_0 {v0d:+8.1f} -> V_inf {vd:+8.1f} LSB "
              f"(fraction {fd:+.3f})")
        print()
        print(f"  ABSOLUTE  predicts the negative arm ends at {pred_abs:+.1f} "
              f"LSB  -> error {abs(vd-pred_abs):6.1f}")
        print(f"  FRACTIONAL predicts it ends at            {pred_frac:+.1f} "
              f"LSB  -> error {abs(vd-pred_frac):6.1f}")
        print()
        win = ("ABSOLUTE" if abs(vd - pred_abs) < abs(vd - pred_frac)
               else "FRACTIONAL")
        print(f"  -> {win} wins")
        # a cleaner test: is V_inf's sign the same as the start's?
        print(f"  sign check: V_inf of the negative arm is "
              f"{'NEGATIVE' if vd < 0 else 'POSITIVE'}; absolute requires "
              f"POSITIVE, fractional requires NEGATIVE")

        # pooled fit of V_inf = a + b*V_0 across both arms
        allv0 = np.array([P[t][1] for t in P])
        allvi = np.array([P[t][0] for t in P])
        b, a = np.polyfit(allv0, allvi, 1)
        print(f"\n  pooled regression over BOTH arms: "
              f"V_inf = {b:.3f}*V_0 {a:+.1f}")
        print(f"    b=0 would mean purely absolute; b={fp:.2f} purely "
              f"fractional")

    # ---- degradation check --------------------------------------------
    if "pot" in res:
        print("\n" + "=" * 66)
        print("degradation: this session's positive arm vs 2026-08-06")
        print("=" * 66)
        for nm, old in (("V_0", OLD["v0"]), ("tau", OLD["tau"]),
                        ("V_inf", OLD["vinf"])):
            new = res["pot"][nm][0]
            print(f"  {nm:<7s} {old:8.2f} -> {new:8.2f}  "
                  f"({100*(new-old)/abs(old):+6.1f}%)")
        print("  (the negative arm is compared only against THIS session's")
        print("   positive arm, so any degradation cancels)")

    out = "disturb_signed_params.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "row", "col", "V_inf", "V_0", "tau"])
        for t in sorted(P):
            w.writerow([t[0], t[2], t[3], P[t][0], P[t][1], P[t][2]])
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

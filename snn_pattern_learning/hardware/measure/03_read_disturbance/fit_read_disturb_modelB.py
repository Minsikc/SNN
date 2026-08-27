#!/usr/bin/env python3
"""Model B for read disturbance, with c2c / d2d spread for BOTH parameters.

    V_k = V_inf + (V_0 - V_inf) * exp(-k / tau)

k is the read index.  Each read removes a fixed fraction of the DISTANCE to
V_inf rather than of the level itself, so the cell relaxes toward a non-zero
attractor instead of toward zero.  (Form A, V_0*exp(-k/tau), is the special
case V_inf = 0; it was rejected on this data -- CV 23.1 vs 4.8 LSB -- because
the traces plainly stop at ~60 LSB after 7.9 time constants.)

VARIANCE DECOMPOSITION
----------------------
Each cell was measured over several repeats, so the two spreads separate:

    d2d   sd of the per-cell means            device-to-device
    c2c   within-cell sd across repeats,      cycle-to-cycle
          pooled as sqrt(mean(sd_cell^2))

Note the per-cell means are themselves noisy, so the raw d2d sd is inflated by
c2c/sqrt(n_reps).  The unbiased device variance is

    var_d2d = var(cell means) - var_c2c / n_reps

and both the raw and the corrected value are reported.

A CAVEAT ON V_inf
-----------------
V_inf correlates with the starting level (r = +0.66), so it is not obviously a
fixed absolute floor.  But every trace in this dataset starts POSITIVE
(V_0 = 332..467 LSB), so "an absolute attractor near +67 LSB" and "a fixed
fraction ~0.17 of wherever the cell started" both fit and cannot be told
apart.  The regression V_inf = 0.36*V_0 - 76 sits between the two, favouring
neither cleanly.  Both parameterisations are printed; deciding between them
needs a disturb run started from NEGATIVE levels, which does not exist yet.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

DEFAULT = "2026-08-06_23-07_disturb_all25.csv"


def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d[(int(r["rep"]), int(r["row"]), int(r["col"]))][int(r["reads"])] = \
            float(r["level"])
    return {k: (np.array(sorted(s), float),
                np.array([s[int(i)] for i in sorted(s)]))
            for k, s in d.items()}


def modelB(p, k):
    """p = [V_inf, V_0, tau]"""
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def decompose(per_cell, name, unit=""):
    """-> dict with mean, d2d (raw + corrected), c2c."""
    cells = sorted(per_cell)
    cm = np.array([np.mean(per_cell[c]) for c in cells])
    within = [np.std(per_cell[c], ddof=1) for c in cells
              if len(per_cell[c]) > 1]
    n_rep = np.mean([len(per_cell[c]) for c in cells])
    c2c = float(np.sqrt(np.mean(np.square(within)))) if within else 0.0
    d2d_raw = float(cm.std(ddof=1))
    # remove the sampling noise in each cell mean
    var_corr = max(d2d_raw ** 2 - c2c ** 2 / n_rep, 0.0)
    d2d = float(np.sqrt(var_corr))
    mean = float(cm.mean())
    print(f"  {name:<22s} mean {mean:8.3f}{unit}")
    print(f"  {'':22s} d2d  {d2d:8.3f}  (cv {d2d/abs(mean):.3f})"
          f"   [raw {d2d_raw:.3f}]")
    print(f"  {'':22s} c2c  {c2c:8.3f}  (cv {c2c/abs(mean):.3f})")
    return dict(mean=mean, d2d=d2d, d2d_raw=d2d_raw, c2c=c2c,
                cell_means={c: float(np.mean(per_cell[c])) for c in cells})


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    data = load(path)
    keys = sorted(data)
    cells = sorted({k[1:] for k in keys})
    reps = sorted({k[0] for k in keys})
    print(f"{path}: {len(keys)} traces = {len(cells)} cells x {len(reps)} reps")
    print(f"reads 1..{int(max(data[k][0].max() for k in keys))}\n")

    P, rms = {}, []
    for t in keys:
        k, v = data[t]
        p = least_squares(lambda q: modelB(q, k) - v, [v[-1], v[0], 50.0],
                          max_nfev=20000).x
        P[t] = p
        rms.append(np.sqrt(np.mean((modelB(p, k) - v) ** 2)))
    print(f"model B fit quality: rms {np.mean(rms):.2f} LSB "
          f"(worst trace {np.max(rms):.2f})\n")

    print("=" * 62)
    print("V_k = V_inf + (V_0 - V_inf) * exp(-k/tau)")
    print("=" * 62)

    out = {}
    for idx, nm, unit in ((2, "tau", " reads"), (0, "V_inf", " LSB"),
                          (1, "V_0", " LSB")):
        per = {c: [P[(r,) + c][idx] for r in reps if (r,) + c in P]
               for c in cells}
        out[nm] = decompose(per, nm, unit)
        print()

    # V_inf as a fraction of the starting level
    per_f = {c: [P[(r,) + c][0] / P[(r,) + c][1] for r in reps
                 if (r,) + c in P] for c in cells}
    out["V_inf_frac"] = decompose(per_f, "V_inf / V_0", "")
    print()

    vinf = np.array([P[t][0] for t in keys])
    v0 = np.array([P[t][1] for t in keys])
    sl, ic = np.polyfit(v0, vinf, 1)
    print("=" * 62)
    print("is V_inf absolute or proportional to the starting level?")
    print("=" * 62)
    print(f"  r(V_inf, V_0)      {np.corrcoef(vinf, v0)[0,1]:+.3f}")
    print(f"  regression         V_inf = {sl:.3f}*V_0 {ic:+.1f}")
    print(f"  cv as absolute     {out['V_inf']['d2d']/out['V_inf']['mean']:.3f} (d2d)")
    print(f"  cv as fraction     "
          f"{out['V_inf_frac']['d2d']/out['V_inf_frac']['mean']:.3f} (d2d)")
    print("  every trace starts positive (V_0 = "
          f"{v0.min():.0f}..{v0.max():.0f} LSB), so the two readings cannot")
    print("  be separated here.  A disturb run from NEGATIVE levels would.")

    print("\n" + "=" * 62)
    print("simulation recipe (per cell, drawn once at construction)")
    print("=" * 62)
    t, vi = out["tau"], out["V_inf"]
    print(f"  tau_cell   ~ N({t['mean']:.1f}, {t['d2d']:.1f}^2) reads")
    print(f"  Vinf_cell  ~ N({vi['mean']:.1f}, {vi['d2d']:.1f}^2) LSB")
    print(f"  per read:  V <- Vinf_cell + (V - Vinf_cell)*exp(-1/tau_k)")
    print(f"             with tau_k ~ N(tau_cell, {t['c2c']:.1f}^2) "
          f"redrawn each read (c2c)")
    print(f"  first-read loss = {100*(1-np.exp(-1/t['mean'])):.2f}% of the "
          f"distance to Vinf")

    with open("read_disturb_modelB.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "tau_mean", "tau_sd", "Vinf_mean",
                    "Vinf_sd", "V0_mean", "n_reps"])
        for c in cells:
            tv = [P[(r,) + c][2] for r in reps if (r,) + c in P]
            iv = [P[(r,) + c][0] for r in reps if (r,) + c in P]
            zv = [P[(r,) + c][1] for r in reps if (r,) + c in P]
            w.writerow([c[0], c[1], np.mean(tv),
                        np.std(tv, ddof=1) if len(tv) > 1 else "",
                        np.mean(iv),
                        np.std(iv, ddof=1) if len(iv) > 1 else "",
                        np.mean(zv), len(tv)])
    print("\nsaved -> read_disturb_modelB.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Is read disturbance V_k = V_0 * exp(-k/tau)?  And what is tau's spread?

The proposed form says every read removes the same FRACTION of the remaining
level, so the decay is a pure exponential in read index k with a single time
constant.  Three things have to be checked before quoting a tau:

  1. does a single exponential actually fit, or is a second component needed?
  2. does the decay go to zero, or to a non-zero floor?
  3. is tau stable across cells (d2d) and repeats (c2c)?

Point 2 matters because V_0*exp(-k/tau) forces the asymptote to 0.  If the
cell settles at a non-zero level, forcing it through zero biases tau.  So the
same data is fitted three ways and compared on held-out points:

    A  V_0 exp(-k/tau)                       the proposed form
    B  V_inf + (V_0-V_inf) exp(-k/tau)       single exponential + floor
    C  V_inf + a exp(-k/t1) + b exp(-k/t2)   double exponential + floor

Model selection uses 5-fold CV over READ INDICES (leaving out whole k values),
because neighbouring reads of one trace are not independent.

tau's spread is then reported two ways:
    d2d  spread of per-cell tau (averaged over reps)
    c2c  spread of tau across reps within a cell, pooled
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

DEFAULT = "2026-08-06_23-07_disturb_all25.csv"


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = defaultdict(dict)          # (rep,row,col) -> {k: level}
    for r in rows:
        key = (int(r["rep"]), int(r["row"]), int(r["col"]))
        d[key][int(r["reads"])] = float(r["level"])
    out = {}
    for key, series in d.items():
        k = np.array(sorted(series))
        v = np.array([series[i] for i in k], float)
        out[key] = (k, v)
    return out


# ---- models -----------------------------------------------------------
def mA(p, k):
    return p[0] * np.exp(-k / p[1])


def mB(p, k):
    return p[0] + (p[1] - p[0]) * np.exp(-k / p[2])


def mC(p, k):
    return p[0] + p[1] * np.exp(-k / p[2]) + p[3] * np.exp(-k / p[4])


MODELS = {
    "A: V0*exp(-k/tau)": (mA, lambda v: [v[0], 50.0], 2),
    "B: Vinf+(V0-Vinf)exp(-k/tau)": (mB, lambda v: [v[-1], v[0], 50.0], 3),
    "C: Vinf+2 exponentials": (
        mC, lambda v: [v[-1], (v[0] - v[-1]) * 0.6, 10.0,
                       (v[0] - v[-1]) * 0.4, 200.0], 5),
}


def fit(model, p0, k, v):
    r = least_squares(lambda p: model(p, k) - v, p0, max_nfev=20000)
    return r.x


def cv_rmse(model, p0f, k, v, folds=5):
    idx = np.arange(len(k))
    err = []
    for f in range(folds):
        te = idx % folds == f
        if te.all() or (~te).any() is False or (~te).sum() < len(p0f(v)):
            continue
        try:
            p = fit(model, p0f(v[~te]), k[~te], v[~te])
            err.append(model(p, k[te]) - v[te])
        except Exception:
            return np.nan
    return float(np.sqrt(np.mean(np.square(np.concatenate(err))))) if err \
        else np.nan


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    data = load(path)
    traces = sorted(data)
    kmax = max(data[t][0].max() for t in traces)
    print(f"{path}: {len(traces)} traces, reads 1..{kmax}")
    print(f"  {len({t[1:] for t in traces})} cells x "
          f"{len({t[0] for t in traces})} reps\n")

    # ---- 1. does the proposed form fit? -------------------------------
    print("model comparison, 5-fold CV over read indices (LSB)")
    print(f"{'model':32s} {'params':>6s} {'in-sample':>10s} {'CV':>8s}")
    print("-" * 60)
    summary = {}
    for name, (model, p0f, npar) in MODELS.items():
        ins, cvs = [], []
        for t in traces:
            k, v = data[t]
            try:
                p = fit(model, p0f(v), k, v)
                ins.append(np.sqrt(np.mean((model(p, k) - v) ** 2)))
            except Exception:
                continue
            c = cv_rmse(model, p0f, k, v)
            if not np.isnan(c):
                cvs.append(c)
        summary[name] = (np.mean(ins), np.mean(cvs))
        print(f"{name:32s} {npar:6d} {np.mean(ins):10.2f} "
              f"{np.mean(cvs):8.2f}")

    best = min(summary, key=lambda n: summary[n][1])
    print(f"\nbest by CV: {best}")

    # ---- 2. is the asymptote zero? ------------------------------------
    vinf, v0 = [], []
    for t in traces:
        k, v = data[t]
        p = fit(mB, [v[-1], v[0], 50.0], k, v)
        vinf.append(p[0])
        v0.append(p[1])
    vinf, v0 = np.array(vinf), np.array(v0)
    print(f"\nfitted floor V_inf: {vinf.mean():+.1f} +- {vinf.std():.1f} LSB "
          f"(V_0 {v0.mean():.0f})")
    print(f"  as a fraction of V_0: {np.mean(vinf/v0)*100:+.1f}%")
    if abs(vinf.mean()) > 3 * vinf.std() / np.sqrt(len(vinf)):
        print("  -> the floor is significantly non-zero, so form A "
              "(forced through 0) is mis-specified")

    # ---- 3. tau, and its c2c / d2d spread ------------------------------
    # use the form the CV picked, but always report tau of the single
    # exponential with floor so the number is comparable across cells
    tau = {}
    for t in traces:
        k, v = data[t]
        p = fit(mB, [v[-1], v[0], 50.0], k, v)
        tau[t] = p[2]

    cells = sorted({t[1:] for t in traces})
    reps = sorted({t[0] for t in traces})
    per_cell = {c: [tau[(r,) + c] for r in reps if (r,) + c in tau]
                for c in cells}

    cell_mean = np.array([np.mean(per_cell[c]) for c in cells])
    # c2c: within-cell spread across reps, pooled over cells
    within = [np.std(per_cell[c], ddof=1) for c in cells
              if len(per_cell[c]) > 1]
    c2c = float(np.sqrt(np.mean(np.square(within)))) if within else np.nan

    print(f"\n{'':22s}{'mean':>9s} {'sd':>8s} {'cv':>7s}")
    print(f"{'tau, per-cell mean':22s}{cell_mean.mean():9.2f} "
          f"{cell_mean.std(ddof=1):8.2f} "
          f"{cell_mean.std(ddof=1)/cell_mean.mean():7.3f}   <- d2d")
    print(f"{'tau, within-cell':22s}{'':9s} {c2c:8.2f} "
          f"{c2c/cell_mean.mean():7.3f}   <- c2c")
    print(f"\nd2d sd {cell_mean.std(ddof=1):.2f} reads is "
          f"{cell_mean.std(ddof=1)/c2c:.1f}x the c2c sd {c2c:.2f} reads")
    print(f"tau range across cells: {cell_mean.min():.1f} .. "
          f"{cell_mean.max():.1f} reads")
    print(f"per-read loss = 1-exp(-1/tau) = "
          f"{100*(1-np.exp(-1/cell_mean.mean())):.2f}% of the remaining "
          f"amplitude")

    with open("read_disturb_tau.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "tau_mean", "tau_sd_reps", "n_reps"])
        for c in cells:
            v = per_cell[c]
            w.writerow([c[0], c[1], np.mean(v),
                        np.std(v, ddof=1) if len(v) > 1 else "",
                        len(v)])
    print("\nsaved -> read_disturb_tau.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

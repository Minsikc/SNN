#!/usr/bin/env python3
"""Nonlinear model of the update: coincidence step + half-select leak.

The linear fit delta = g*desired leaves a 9.5 LSB residual.  Two known
nonlinearities should absorb part of it:

  1. the per-coincidence step shrinks as coincidences accumulate WITHIN one
     update -- the cap charges, so later pulses in the same burst move it less
  2. the half-select leak should behave the same way, and its sign depends on
     which attractor the pair pulls toward

Both are written as saturating-exponential approaches to a per-direction
asymptote, which is the standard form for this cell and reduces to the linear
model when the rate is small:

    pot burst of C:   A_p * (1 - exp(-k_p * C))
    dep burst of C:   A_d * (1 - exp(-k_d * C))
    half-select:      per-class amplitude with its own saturation in the count

Models are compared on held-out data (5-fold CV by trial, so the whole 25-cell
array moves together as it does physically) rather than in-sample R^2, because
extra parameters always improve in-sample fit.

A NOTE ON WHAT THIS DATA CANNOT DO
----------------------------------
The earlier characterisation wrote the step as proportional to (477 - V), i.e.
dependent on the cell's absolute state.  That term is NOT identifiable here:
every trial starts from a hard reset, so `before` spans only -162..+39 and is
almost entirely a fixed per-cell offset (sd 34 across cells, 4 within a cell).
Fitting a state term on this data would just relabel cell-to-cell gain spread
as state dependence.  The saturation fitted below is saturation in the
COINCIDENCE COUNT within one burst, which the data does resolve (C spans
0..10 at every cell).  A state-dependence fit needs a dedicated sweep that
sets the starting level on purpose.
"""
import csv
import sys

import numpy as np
from scipy.optimize import least_squares

HS = "halfselect_seq_in_update.csv"
UV = "2026-08-10_14-43_uv_random_signed.csv"


def load():
    a = list(csv.DictReader(open(HS, encoding="utf-8")))
    b = list(csv.DictReader(open(UV, encoding="utf-8")))
    key = lambda r: (int(r["trial"]), int(r["cell_row"]), int(r["cell_col"]))
    m = {key(r): r for r in b}
    d = {}
    col = lambda k, src: np.array([float(src[i][k]) for i in range(len(src))])
    d["trial"] = np.array([int(r["trial"]) for r in a])
    d["cid"] = np.array([(int(r["cell_row"]) - 1) * 5 + int(r["cell_col"]) - 1
                         for r in a])
    d["delta"] = col("delta", a)
    d["desired"] = col("desired", a)
    d["hp"] = col("act_pot", a)
    d["hr"] = col("act_reset", a)
    d["hd"] = col("act_dep", a)
    for k in ("C_pot", "C_dep", "before"):
        d[k] = np.array([float(m[key(r)][k]) for r in a])
    return d


def sat(x, k):
    """(1 - exp(-k x)) / k -> x as k -> 0, so k=0 is the linear model."""
    k = np.asarray(k, float)
    if abs(float(k)) < 1e-9:
        return x
    return (1.0 - np.exp(-k * x)) / k


# ---- model definitions: each returns predicted delta ---------------------
def m_linear(p, d):
    return p[0] * d["desired"] + p[1]


def m_linear_hs(p, d):
    return (p[0] * d["desired"] + p[1] * d["hp"] + p[2] * d["hr"]
            + p[3] * d["hd"] + p[4])


def m_sat_c(p, d):
    """Saturating in coincidence count, separate P and D, linear half-select."""
    gp, kp, gd, kd, ap, ar, ad, c = p
    return (gp * sat(d["C_pot"], kp) - gd * sat(d["C_dep"], kd)
            + ap * d["hp"] + ar * d["hr"] + ad * d["hd"] + c)


def m_sat_both(p, d):
    """Saturating in both the coincidence count and the half-select count."""
    gp, kp, gd, kd, ap, kap, ar, ad, kad, c = p
    return (gp * sat(d["C_pot"], kp) - gd * sat(d["C_dep"], kd)
            + ap * sat(d["hp"], kap) + ar * d["hr"]
            - ad * sat(d["hd"], kad) + c)


MODELS = [
    ("linear (desired only)", m_linear, [13.0, 0.0]),
    ("linear + half-select", m_linear_hs, [13.0, 1.5, -1.5, -1.5, 0.0]),
    ("saturating C + linear HS", m_sat_c,
     [14.5, 0.05, 15.5, 0.05, 1.5, -1.5, -1.5, 0.0]),
    ("saturating C + saturating HS", m_sat_both,
     [14.5, 0.05, 15.5, 0.05, 1.5, 0.05, -1.5, 1.5, 0.05, 0.0]),
]


def fit(model, p0, d, idx=None):
    if idx is None:
        idx = np.ones(len(d["delta"]), bool)
    sub = {k: v[idx] for k, v in d.items()}
    r = least_squares(lambda p: model(p, sub) - sub["delta"], p0,
                      max_nfev=20000)
    return r.x


def main():
    d = load()
    n = len(d["delta"])
    print(f"{n} cell-updates from {HS} + {UV}\n")

    trials = np.unique(d["trial"])
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(trials), 5)

    print(f"{'model':32s} {'params':>6s} {'in-sample':>10s} "
          f"{'CV RMSE':>9s} {'CV r':>8s}")
    print("-" * 70)
    results = {}
    for name, model, p0 in MODELS:
        p_all = fit(model, p0, d)
        rms_in = np.sqrt(np.mean((model(p_all, d) - d["delta"]) ** 2))
        pred = np.zeros(n)
        for f in folds:
            te = np.isin(d["trial"], f)
            p = fit(model, p0, d, ~te)
            sub = {k: v[te] for k, v in d.items()}
            pred[te] = model(p, sub)
        rms_cv = np.sqrt(np.mean((pred - d["delta"]) ** 2))
        r_cv = np.corrcoef(pred, d["delta"])[0, 1]
        results[name] = (p_all, rms_in, rms_cv, r_cv, pred)
        print(f"{name:32s} {len(p0):6d} {rms_in:10.3f} {rms_cv:9.3f} "
              f"{r_cv:+8.4f}")

    best = min(results, key=lambda k: results[k][2])
    print(f"\nbest by cross-validated RMSE: {best}")

    p = results["saturating C + saturating HS"][0]
    gp, kp, gd, kd, ap, kap, ar, ad, kad, c = p
    print("\n--- fitted saturating model ---")
    print(f"  POT coincidence: {gp:.2f} * (1-exp(-{kp:.4f}*C))/{kp:.4f}")
    print(f"       first pulse {gp:+.2f} LSB, 10th pulse "
          f"{gp*np.exp(-kp*9):+.2f} LSB "
          f"({100*np.exp(-kp*9):.0f}% of the first)")
    print(f"  DEP coincidence: {-gd:.2f} * (1-exp(-{kd:.4f}*C))/{kd:.4f}")
    print(f"       first pulse {-gd:+.2f} LSB, 10th pulse "
          f"{-gd*np.exp(-kd*9):+.2f} LSB "
          f"({100*np.exp(-kd*9):.0f}% of the first)")
    print(f"  HS  POT-driving: {ap:+.3f} LSB first, k={kap:.4f}")
    print(f"  HS  reset-driving: {ar:+.3f} LSB (linear)")
    print(f"  HS  DEP-driving: {-ad:+.3f} LSB first, k={kad:.4f}")
    print(f"  offset {c:+.3f} LSB")

    lin = results["linear (desired only)"]
    bst = results[best]
    print(f"\nresidual sd: linear {lin[1]:.2f} -> {best} {bst[1]:.2f} LSB "
          f"({100*(1-bst[1]/lin[1]):.0f}% reduction)")
    print(f"cross-validated: {lin[2]:.2f} -> {bst[2]:.2f} LSB")

    np.save("update_model_params.npy",
            {k: v[0] for k, v in results.items()}, allow_pickle=True)
    with open("update_model_pred.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "cid", "C_pot", "C_dep", "hp", "hr", "hd",
                    "delta", "pred_linear", "pred_best"])
        for i in range(n):
            w.writerow([d["trial"][i], d["cid"][i], d["C_pot"][i],
                        d["C_dep"][i], d["hp"][i], d["hr"][i], d["hd"][i],
                        d["delta"][i], results["linear (desired only)"][4][i],
                        bst[4][i]])
    print("\nsaved -> update_model_pred.csv, update_model_params.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

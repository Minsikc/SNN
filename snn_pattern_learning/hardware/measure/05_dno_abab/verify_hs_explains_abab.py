#!/usr/bin/env python3
"""Can the ABAB NORMAL-vs-DNO difference be explained by half-select pairs?

The ABAB run (abab_dno_test.py, seed 0) used IDENTICAL pulse streams for the
NORMAL and DNO trial of each (grid point, repeat) pair, with a hard reset
before every trial. DNO shields half-selected cells (N3 on undriven rows ->
the N2/N3 pair is inert), NORMAL does not. So within a pair,

    delta_normal - delta_dno = (half-select pair drive) + noise

with the coincidence-driven part cancelling exactly. The pair count P per
cell is computable because the streams regenerate from the seeded RNG.

Tests:
  1. stream regeneration is verified against the stored per-cell C
  2. regress (delta_normal - delta_dno) on P  ->  slope should match the
     +1.65 LSB/pair measured INDEPENDENTLY in halfselect_seq_in_update
     (scripts/04), intercept should be ~0 if pairs are the whole story
  3. Monte-Carlo the paired r(delta, C) difference with and without the
     pair term -> compare to the measured mean(r_dno - r_normal) = +0.023

Pair counting uses the slot rule verified in scripts/04: within a slot the
row line falls last, so the line left active is the ROW whenever the row bit
is 1 (including coincidence slots), else the COLUMN if its bit is 1, else the
previous state persists. A pair fires on every row<->col state transition.

    python verify_hs_explains_abab.py 2026-08-20_12-01_abab_dno_cells.csv
"""
import csv
import sys

import numpy as np
from scipy import stats

N = 5
BL = 10
LEVELS = [0.3, 0.5, 0.7]
REPEATS = 10
SEED = 0

S_PAIR_04 = 1.65   # LSB per potentiating pair, from scripts/04 regression


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for x in rows:
        for k in ("u", "v", "delta", "pre", "post"):
            x[k] = float(x[k])
        for k in ("rep", "C", "cell_row", "cell_col", "order_pos"):
            x[k] = int(x[k])
    return rows


def streams(probs, bl, rng):
    # verbatim from abab_dno_test.py so the RNG consumption matches
    return ["".join(map(str, (rng.random(bl) < p).astype(int))) for p in probs]


def regenerate():
    """Replay the seeded RNG of abab_dno_test.py -> per-trial streams."""
    rng = np.random.default_rng(SEED)
    out = {}
    for u in LEVELS:
        for v in LEVELS:
            for rep in range(REPEATS):
                n1 = streams(np.full(N, u), BL, rng)
                n2 = streams(np.full(N, v), BL, rng)
                a = np.array([[int(c) for c in s] for s in n1])  # rows x BL
                b = np.array([[int(c) for c in s] for s in n2])  # cols x BL
                out[(u, v, rep)] = (a, b)
    return out


def coincidences(a, b):
    return a @ b.T  # (rows x BL) @ (BL x cols)


def pot_pairs(a, b):
    """Ordered N1<->N2 pair count per cell, slot rule from scripts/04."""
    P = np.zeros((N, N), int)
    for i in range(N):
        for j in range(N):
            state, pairs = 0, 0  # 0 none, 1 row(N1), 2 col(N2)
            for k in range(BL):
                if a[i, k]:
                    new = 1          # row falls last, wins any slot it fires
                elif b[j, k]:
                    new = 2
                else:
                    continue
                if state and new != state:
                    pairs += 1
                state = new
            P[i, j] = pairs
    return P


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "2026-08-20_12-01_abab_dno_cells.csv"
    rows = load(path)
    trials = regenerate()

    # ---- 1. verify the stream regeneration against the stored C ----------
    stored = {}
    for x in rows:
        key = (x["u"], x["v"], x["rep"], x["mode"])
        stored.setdefault(key, np.zeros((N, N)))[x["cell_row"] - 1,
                                                 x["cell_col"] - 1] = x["C"]
    n_bad = 0
    for (u, v, rep), (a, b) in trials.items():
        C = coincidences(a, b)
        for mode in ("normal", "dno"):
            if not np.array_equal(C, stored[(u, v, rep, mode)]):
                n_bad += 1
    print("=" * 68)
    print("1. stream regeneration check")
    print("=" * 68)
    print(f"   trials with mismatching C: {n_bad} / {len(trials) * 2}"
          f"   ({'OK - streams exactly recovered' if n_bad == 0 else 'FAIL'})")
    if n_bad:
        return 1

    # ---- 2. paired delta difference vs pair count ------------------------
    deltas = {}
    for x in rows:
        deltas[(x["u"], x["v"], x["rep"], x["mode"],
                x["cell_row"] - 1, x["cell_col"] - 1)] = x["delta"]

    D, P, Cc, uu = [], [], [], []
    for (u, v, rep), (a, b) in sorted(trials.items()):
        pp = pot_pairs(a, b)
        cc = coincidences(a, b)
        for i in range(N):
            for j in range(N):
                D.append(deltas[(u, v, rep, "normal", i, j)]
                         - deltas[(u, v, rep, "dno", i, j)])
                P.append(pp[i, j])
                Cc.append(cc[i, j])
                uu.append((u, v))
    D, P, Cc = np.array(D), np.array(P, float), np.array(Cc, float)

    print()
    print("=" * 68)
    print("2. delta_NORMAL - delta_DNO  vs  half-select pair count")
    print("=" * 68)
    res = stats.linregress(P, D)
    print(f"   n = {len(D)} cell-pairs, pair count mean {P.mean():.2f} "
          f"(range {int(P.min())}-{int(P.max())})")
    print(f"   slope     {res.slope:+.3f} +- {res.stderr:.3f} LSB/pair "
          f"(scripts/04 independent estimate: {S_PAIR_04:+.2f})")
    print(f"   intercept {res.intercept:+.3f} +- {res.intercept_stderr:.3f} LSB "
          f"(0 if pairs explain the whole mode difference)")
    print(f"   r = {res.rvalue:+.4f}   p = {res.pvalue:.2e}")

    # does C add anything once P is in? (it should NOT: same streams)
    X = np.column_stack([P, Cc, np.ones_like(P)])
    beta, *_ = np.linalg.lstsq(X, D, rcond=None)
    resid = D - X @ beta
    dof = len(D) - 3
    cov = np.linalg.inv(X.T @ X) * (resid @ resid / dof)
    se = np.sqrt(np.diag(cov))
    print(f"   joint fit:  pair {beta[0]:+.3f}+-{se[0]:.3f}   "
          f"C {beta[1]:+.3f}+-{se[1]:.3f}   const {beta[2]:+.3f}+-{se[2]:.3f}")
    print("   (a C term != 0 would mean DNO also changes the coincidence "
          "drive itself,")
    print("    i.e. something beyond half-select pairs)")

    print()
    print("   per grid point: mean paired diff vs pair-model prediction")
    print(f"   {'u':>4} {'v':>4} {'meas mean D':>12} {'pred s*P':>10} "
          f"{'pairs/cell':>11}")
    for u in LEVELS:
        for v in LEVELS:
            m = [k == (u, v) for k in uu]
            m = np.array(m)
            print(f"   {u:4.1f} {v:4.1f} {D[m].mean():+12.2f} "
                  f"{res.slope * P[m].mean():+10.2f} {P[m].mean():11.2f}")

    # ---- 3. can the pair term reproduce the measured delta-r? ------------
    print()
    print("=" * 68)
    print("3. Monte-Carlo: does the pair drive reproduce r_dno - r_normal?")
    print("=" * 68)
    # per-session gain and noise, taken from the CLEAN (DNO) arm
    Dd = np.array([deltas[(u, v, rep, "dno", i, j)]
                   for (u, v, rep) in sorted(trials)
                   for i in range(N) for j in range(N)])
    g = stats.linregress(Cc, Dd)
    print(f"   DNO arm calibration: gain {g.slope:+.2f} LSB/coincidence, "
          f"noise sd {np.std(Dd - g.slope * Cc - g.intercept):.2f} LSB")
    sig = np.std(Dd - g.slope * Cc - g.intercept)

    rng = np.random.default_rng(1)
    n_mc = 400
    dr_with, dr_without = [], []
    keys = sorted(trials)
    for _ in range(n_mc):
        dws, dwos = [], []
        for (u, v, rep) in keys:
            a, b = trials[(u, v, rep)]
            cc = coincidences(a, b).ravel()
            pp = pot_pairs(a, b).ravel()
            base_n = g.slope * cc + g.intercept + rng.normal(0, sig, cc.size)
            base_d = g.slope * cc + g.intercept + rng.normal(0, sig, cc.size)
            for s_pair, acc in ((res.slope, dws), (0.0, dwos)):
                dn = base_n + s_pair * pp
                dd = base_d
                if cc.std() > 1e-9 and dn.std() > 1e-9 and dd.std() > 1e-9:
                    acc.append(np.corrcoef(cc, dd)[0, 1]
                               - np.corrcoef(cc, dn)[0, 1])
        dr_with.append(np.mean(dws))
        dr_without.append(np.mean(dwos))
    dr_with, dr_without = np.array(dr_with), np.array(dr_without)

    # measured
    robs = {}
    for x in rows:
        key = (x["u"], x["v"], x["rep"], x["mode"])
        robs.setdefault(key, []).append((x["C"], x["delta"]))
    diffs = []
    for (u, v, rep) in keys:
        def r_of(mode):
            arr = np.array(robs[(u, v, rep, mode)], float)
            return (np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
                    if arr[:, 0].std() > 1e-9 and arr[:, 1].std() > 1e-9
                    else np.nan)
        diffs.append(r_of("dno") - r_of("normal"))
    meas = np.nanmean(diffs)

    print(f"   measured   mean(r_dno - r_normal) = {meas:+.4f}")
    print(f"   model MC   with pair drive:  {dr_with.mean():+.4f} "
          f"+- {dr_with.std():.4f}")
    print(f"   model MC   without (control): {dr_without.mean():+.4f} "
          f"+- {dr_without.std():.4f}")
    z = (meas - dr_with.mean()) / dr_with.std()
    print(f"   measured value sits {z:+.1f} MC-sd from the pair-drive model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

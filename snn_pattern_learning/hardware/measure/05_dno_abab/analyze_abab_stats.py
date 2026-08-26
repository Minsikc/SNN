#!/usr/bin/env python3
"""Paired statistics for an ABAB NORMAL vs DNO run (abab_dno_test.py output).

Unit of analysis: one (grid point, repeat) pair. For each pair, r(delta, C)
is computed over all 25 cells separately for the NORMAL and the DNO trial
(identical pulse streams within the pair, hard reset before each trial,
order alternated by repeat parity). The paired difference D = r_dno -
r_normal is then tested against zero:

  * paired t-test          (parametric)
  * Wilcoxon signed-rank   (distribution-free)
  * sign test              (direction only)

An order-effect check splits D by which mode ran first; if drift mattered,
D would differ between the two orders.

    python analyze_abab_stats.py <abab_cells.csv>
"""
import sys
import csv

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pair_r(path):
    obs = {}
    for x in csv.DictReader(open(path, encoding="utf-8")):
        key = (float(x["u"]), float(x["v"]), int(x["rep"]), x["mode"])
        obs.setdefault(key, []).append((float(x["C"]), float(x["delta"])))

    def r_of(pts):
        C = np.array([a for a, _ in pts])
        d = np.array([b for _, b in pts])
        if C.std() > 1e-9 and d.std() > 1e-9:
            return float(np.corrcoef(C, d)[0, 1])
        return np.nan

    pairs = []
    for (u, v, rep, mode) in sorted(obs):
        if mode != "normal":
            continue
        rn = r_of(obs[(u, v, rep, "normal")])
        rd = r_of(obs[(u, v, rep, "dno")])
        # rep parity encodes the order used by abab_dno_test.py
        first = "normal" if rep % 2 == 0 else "dno"
        pairs.append(dict(u=u, v=v, rep=rep, first=first,
                          r_normal=rn, r_dno=rd, diff=rd - rn))
    return pairs


def report(pairs):
    D = np.array([p["diff"] for p in pairs])
    D = D[np.isfinite(D)]
    n = len(D)
    t_stat, t_p = stats.ttest_rel([p["r_dno"] for p in pairs],
                                  [p["r_normal"] for p in pairs])
    w_stat, w_p = stats.wilcoxon(D)
    n_pos = int((D > 0).sum())
    s_p = stats.binomtest(n_pos, n, 0.5).pvalue
    ci = stats.t.interval(0.95, n - 1, loc=D.mean(),
                          scale=D.std(ddof=1) / np.sqrt(n))

    print(f"pairs: {n}")
    print(f"r_normal median {np.median([p['r_normal'] for p in pairs]):+.4f}   "
          f"r_dno median {np.median([p['r_dno'] for p in pairs]):+.4f}")
    print(f"paired diff (DNO - NORMAL): mean {D.mean():+.4f}  "
          f"median {np.median(D):+.4f}  sd {D.std(ddof=1):.4f}")
    print(f"95% CI of mean diff: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"Cohen's d (paired): {D.mean() / D.std(ddof=1):+.3f}")
    print(f"paired t-test:      t = {t_stat:+.3f}   p = {t_p:.2e}")
    print(f"Wilcoxon signed-rank:              p = {w_p:.2e}")
    print(f"sign test: DNO better in {n_pos}/{n}  p = {s_p:.2e}")

    print("\norder-effect check (drift control):")
    for first in ("normal", "dno"):
        d = np.array([p["diff"] for p in pairs if p["first"] == first])
        print(f"  {first}-first pairs (n={len(d)}): "
              f"mean diff {d.mean():+.4f}  median {np.median(d):+.4f}")
    d_nf = [p["diff"] for p in pairs if p["first"] == "normal"]
    d_df = [p["diff"] for p in pairs if p["first"] == "dno"]
    _, p_order = stats.mannwhitneyu(d_nf, d_df)
    print(f"  order effect Mann-Whitney p = {p_order:.3f} "
          f"(large p = order/drift does not drive the difference)")

    print("\nper grid point (n = repeats each):")
    print(f"  {'u':>4} {'v':>4}  {'mean diff':>10} {'wilcoxon p':>11}  better")
    for u in sorted({p["u"] for p in pairs}):
        for v in sorted({p["v"] for p in pairs}):
            d = np.array([p["diff"] for p in pairs
                          if p["u"] == u and p["v"] == v])
            try:
                _, wp = stats.wilcoxon(d)
            except ValueError:
                wp = np.nan
            print(f"  {u:4.1f} {v:4.1f}  {d.mean():+10.4f} {wp:11.3f}  "
                  f"{int((d > 0).sum())}/{len(d)}")
    return D


def plot(pairs, D, out):
    us = sorted({p["u"] for p in pairs})
    vs = sorted({p["v"] for p in pairs})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels, groups = [], []
    for u in us:
        for v in vs:
            labels.append(f"{u:g},{v:g}")
            groups.append([p["diff"] for p in pairs
                           if p["u"] == u and p["v"] == v])
    ax.axhline(0, color="gray", lw=1)
    ax.boxplot(groups, tick_labels=labels)
    for k, g in enumerate(groups):
        ax.plot(np.full(len(g), k + 1) + np.linspace(-0.15, 0.15, len(g)),
                g, "o", ms=4, alpha=0.6, color="tab:blue")
    ax.set_xlabel("(u, v) grid point")
    ax.set_ylabel("r_dno - r_normal (paired)")
    ax.set_title("Paired difference per grid point")

    ax = axes[1]
    ax.axvline(0, color="gray", lw=1)
    ax.hist(D, bins=15, color="tab:blue", alpha=0.8)
    ax.axvline(D.mean(), color="tab:red", lw=2,
               label=f"mean {D.mean():+.4f}")
    ax.set_xlabel("r_dno - r_normal")
    ax.set_ylabel("pairs")
    ax.set_title(f"All {len(D)} pairs pooled")
    ax.legend()

    fig.suptitle("ABAB paired NORMAL vs DNO, r(delta, C) over all 25 cells — "
                 "reset per trial, identical streams, order alternated",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=110)
    print(f"\nsaved -> {out}")


def main():
    path = sys.argv[1]
    pairs = pair_r(path)
    D = report(pairs)
    plot(pairs, D, path.replace("_cells.csv", "_stats.png"))


if __name__ == "__main__":
    main()

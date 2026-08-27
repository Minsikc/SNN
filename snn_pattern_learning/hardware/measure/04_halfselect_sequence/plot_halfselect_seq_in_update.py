#!/usr/bin/env python3
"""Half-select sequences: which ones move charge, and how often updates fire them.

Left   -- the attractor sweep condensed: where each ordered pair drives the
          cell from all five starting states.  Only pairs whose five endpoints
          collapse are doing anything.
Middle -- how often each pair actually fired per cell-update in the
          random-signed u,v run, split by the direction the update asked for,
          with the cross-quadrant share marked.
Right  -- the consequence: residual (measured minus the desired-update fit)
          against the NET drive, i.e. potentiating pairs minus depressing ones.
"""
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
POT, DEP, NEU = "#2a78d6", "#eb6834", "#b8b6ae"
STARTS = ["full_pot", "mid_pot", "reset", "mid_dep", "full_dep"]


def main():
    arows = []
    for p in sorted(glob.glob("*_seq_attractor.csv")):
        arows += list(csv.DictReader(open(p, encoding="utf-8")))
    info = {}
    for c in sorted({r["combo"] for r in arows if r["combo"] != "control"}):
        sub = [r for r in arows if r["combo"] == c]
        mx = max(int(r["cycles"]) for r in sub)
        ends = [np.mean([float(r["after"]) for r in sub
                         if r["start"] == s and int(r["cycles"]) == mx])
                for s in STARTS]
        info[c] = dict(ends=ends, spread=max(ends) - min(ends),
                       level=float(np.mean(ends)))
        info[c]["active"] = info[c]["spread"] < 250

    urows = list(csv.DictReader(open("halfselect_seq_in_update.csv",
                                     encoding="utf-8")))
    col = lambda k: np.array([float(r[k]) for r in urows])
    d = np.array([r["direction"] for r in urows])
    des, dl = col("desired"), col("delta")
    ap, ar, ad = col("act_pot"), col("act_reset"), col("act_dep")
    net = ap - ad
    g, b = np.polyfit(des, dl, 1)
    res = dl - (g * des + b)

    order = sorted(info, key=lambda k: -info[k]["level"])
    npot, ndep = (d == "pot").sum(), (d == "dep").sum()
    cpot = {k: sum(float(r[k]) for r in urows if r["direction"] == "pot")
            for k in order}
    cdep = {k: sum(float(r[k]) for r in urows if r["direction"] == "dep")
            for k in order}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7))

    # ---- 1. attractor map ------------------------------------------------
    ax = axes[0]
    y = np.arange(len(order))
    for i, k in enumerate(order):
        e, act = info[k]["ends"], info[k]["active"]
        c_ = INK if act else MUTED
        ax.plot([min(e), max(e)], [i, i], color=c_, lw=1.2,
                alpha=1.0 if act else 0.5, zorder=1)
        ax.scatter(e, [i] * len(e), s=26, color=c_, zorder=2,
                   alpha=0.9 if act else 0.55, edgecolors="none")
        if act:
            lv = info[k]["level"]
            ax.scatter([lv], [i], s=95, marker="|", zorder=3,
                       color=POT if lv > 200 else (DEP if lv < -200 else NEU))
    ax.set_yticks(y, [f"{k}{'  *' if info[k]['active'] else ''}"
                      for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("cell state after 300 cycles (LSB)")
    ax.set_title("1. what each ordered pair does\nfive dots = five starting "
                 "states; collapsed = attractor (*)", fontsize=10)
    ax.grid(color=GRID, lw=0.8, axis="x")

    # ---- 2. firing counts ------------------------------------------------
    ax = axes[1]
    h = 0.38
    ax.barh(y - h / 2, [cpot[k] / npot for k in order], height=h, color=POT,
            label=f"update asked POT (n={npot})")
    ax.barh(y + h / 2, [cdep[k] / ndep for k in order], height=h, color=DEP,
            label=f"update asked DEP (n={ndep})")
    for i, k in enumerate(order):
        for v, off in ((cpot[k] / npot, -h / 2), (cdep[k] / ndep, h / 2)):
            if v > 0.005:
                ax.text(v + 0.02, i + off, f"{v:.2f}", va="center",
                        fontsize=8, color=INK)
    ax.set_yticks(y, [f"{k}{'  *' if info[k]['active'] else ''}"
                      for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("times fired per cell-update")
    ax.set_title("2. how often the update run fired each pair\n"
                 "state carried across quadrants, so N1+N3 / N2+N4 / N1+N4 /\n"
                 "N2+N3 appear at the sign change", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, lw=0.8, axis="x")

    # ---- 3. consequence --------------------------------------------------
    ax = axes[2]
    for lab, c_, m in (("POT cells", POT, d == "pot"),
                       ("DEP cells", DEP, d == "dep"),
                       ("zero cells", NEU, d == "zero")):
        xs, ys, es = [], [], []
        for n in sorted({int(x) for x in net[m]}):
            s = m & (net == n)
            if s.sum() < 8:
                continue
            xs.append(n)
            ys.append(res[s].mean())
            es.append(res[s].std() / np.sqrt(s.sum()))
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=6, lw=1.8, color=c_,
                    capsize=3, label=lab)
    A = np.column_stack([des, ap, ar, ad, np.ones(len(urows))])
    c2, *_ = np.linalg.lstsq(A, dl, rcond=None)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("net drive = potentiating pairs - depressing pairs")
    ax.set_ylabel("residual: measured - desired-update fit (LSB)")
    ax.set_title(f"3. residual follows the net half-select drive\n"
                 f"fit: {c2[1]:+.2f} LSB per POT pair, {c2[3]:+.2f} per DEP "
                 f"pair\n= {100*abs(c2[1])/abs(g):.0f}% of one coincidence "
                 f"({g:.1f} LSB)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(color=GRID, lw=0.8)

    fig.suptitle("Half-select sequences that move charge, and how often a real "
                 "update fires them", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("halfselect_seq_in_update.png", dpi=110)
    print("saved -> halfselect_seq_in_update.png")


if __name__ == "__main__":
    main()

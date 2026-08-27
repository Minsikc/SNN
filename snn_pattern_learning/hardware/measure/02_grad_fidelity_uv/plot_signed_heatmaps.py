#!/usr/bin/env python3
"""Desired vs actual update, for one trial of each sign pattern.

Three columns per pattern:
    desired  signed coincidence count, sign(err_r * trc_c) * (C_pot + C_dep)
    actual   measured ADC change in LSB
    residual actual - g*desired, with g fitted on that trial

desired and actual are in different units (coincidences vs LSB), so the two
are drawn on their own scales; the residual column is what shows where the
device departed from the request. Both desired and actual use one diverging
map centred on zero, since sign is the point of these patterns.

The representative trial per pattern is the MEDIAN-r rep, not the best one,
so the panels are not cherry-picked.
"""
import argparse
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5
PATTERNS = ["all_plus", "err_neg", "trc_neg", "both_neg"]
QUAD_LABEL = {(1, 1): "P++", (-1, -1): "P--", (1, -1): "D+-", (-1, 1): "D-+"}


def grid(rows, key):
    M = np.zeros((N, N))
    for r in rows:
        M[int(r["cell_row"]) - 1, int(r["cell_col"]) - 1] = float(r[key])
    return M


def signs(rows):
    es = np.zeros(N, int)
    ts = np.zeros(N, int)
    for r in rows:
        es[int(r["cell_row"]) - 1] = int(r["err_sign"])
        ts[int(r["cell_col"]) - 1] = int(r["trc_sign"])
    return es, ts


def draw(ax, M, title, cmap, vlim=None, fmt="{:+.0f}", fig=None, label=None):
    lim = vlim if vlim else max(abs(M.min()), abs(M.max())) or 1
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks(range(N), [f"c{c+1}" for c in range(N)], fontsize=7)
    ax.set_yticks(range(N), [f"r{r+1}" for r in range(N)], fontsize=7)
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(v) > 0.6 * lim else "black")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        if label:
            cb.set_label(label, fontsize=7)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--mag", type=float, default=0.7)
    args = ap.parse_args()

    path = args.csv or sorted(glob.glob("*_signed_uv_exposure.csv"))[-1]
    allrows = list(csv.DictReader(open(path, encoding="utf-8")))

    picked = {}
    for pat in PATTERNS:
        scored = []
        for rep in sorted({int(r["rep"]) for r in allrows}):
            s = [r for r in allrows if r["pattern"] == pat
                 and float(r["mag"]) == args.mag and int(r["rep"]) == rep]
            if len(s) < N * N:
                continue
            d = np.array([float(x["desired"]) for x in s])
            a = np.array([float(x["delta"]) for x in s])
            if d.std() > 1e-9:
                scored.append((np.corrcoef(d, a)[0, 1], rep, s))
        if scored:
            scored.sort(key=lambda t: t[0])
            picked[pat] = scored[len(scored) // 2]      # median r, not max

    fig, axes = plt.subplots(len(picked), 3,
                             figsize=(13.5, 3.6 * len(picked)),
                             squeeze=False)

    for k, pat in enumerate([p for p in PATTERNS if p in picked]):
        rr, rep, rows = picked[pat]
        des = grid(rows, "desired")
        act = grid(rows, "delta")
        es, ts = signs(rows)
        g = np.polyfit(des.ravel(), act.ravel(), 1)[0]
        resid = act - g * des

        rowlab = "".join("+" if v > 0 else "-" for v in es)
        collab = "".join("+" if v > 0 else "-" for v in ts)

        draw(axes[k][0], des,
             f"{pat}  desired\nerr {rowlab}  trc {collab}",
             "coolwarm", fmt="{:+.0f}", fig=fig, label="coincidences")
        draw(axes[k][1], act,
             f"actual  (r = {rr:+.3f}, g = {g:.1f} LSB/coinc)",
             "coolwarm", fmt="{:+.0f}", fig=fig, label="LSB")
        draw(axes[k][2], resid,
             f"residual = actual - g x desired\n"
             f"sd {resid.std():.1f} LSB = {resid.std()/abs(g):.2f} coincidences",
             "PuOr", fmt="{:+.0f}", fig=fig, label="LSB")

        # mark which quadrant each cell belongs to on the desired panel
        for i in range(N):
            for j in range(N):
                q = QUAD_LABEL[(int(es[i]), int(ts[j]))]
                axes[k][0].text(j, i - 0.34, q, ha="center", va="center",
                                fontsize=5.5, color="#52514e")

    fig.suptitle(f"Desired vs actual update, magnitude {args.mag} — "
                 "one representative (median-r) trial per sign pattern\n"
                 "quadrant of each cell marked above its value; residual is "
                 "what the device got wrong",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"signed_desired_vs_actual_mag{args.mag}.png"
    fig.savefig(out, dpi=110)
    print("saved ->", out)

    print(f"\n{'pattern':>9} {'rep':>4} {'r':>8} {'gain':>7} "
          f"{'resid sd':>9} {'in coinc':>9}")
    for pat in [p for p in PATTERNS if p in picked]:
        rr, rep, rows = picked[pat]
        des, act = grid(rows, "desired"), grid(rows, "delta")
        g = np.polyfit(des.ravel(), act.ravel(), 1)[0]
        res = act - g * des
        print(f"{pat:>9} {rep:>4} {rr:>8.3f} {g:>7.2f} "
              f"{res.std():>9.2f} {res.std()/abs(g):>9.2f}")


if __name__ == "__main__":
    main()

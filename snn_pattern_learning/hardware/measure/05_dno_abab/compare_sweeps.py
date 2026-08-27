#!/usr/bin/env python3
"""Side-by-side comparison of uv_grid sweeps (normal vs DNO vs re-run).

The per-sweep heatmaps each live in their own file, so a condition-vs-condition
read is impossible. This lays the same metric from several sweeps on one row
with a SHARED color scale, plus a paired-difference column, which is the only
way to see the DNO effect and the day-over-day device degradation together.
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = np.round(np.arange(0.1, 0.95, 0.1), 2)

SWEEPS = [
    ("normal 08-05\n(before degradation)", "2026-08-05_19-09_uv_grid_all.csv"),
    ("DNO 08-06",                          "2026-08-06_12-46_uv_grid_dno_all.csv"),
    ("normal 08-06\n(same session as DNO)","2026-08-06_13-51_uv_grid_all.csv"),
]
BL = "10"


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["bit_length"] != BL:
            continue
        out[(int(r["gi"]), int(r["gj"]))] = r
    return out


def grid(d, field):
    M = np.full((len(LEVELS), len(LEVELS)), np.nan)
    for (gi, gj), r in d.items():
        try:
            M[gi, gj] = float(r[field])
        except (ValueError, KeyError):
            pass
    return M


def main():
    data = [(lbl, load(p)) for lbl, p in SWEEPS]
    metrics = [("r_delta_C", "Pearson r (delta vs C)", 0.0, 1.0, "RdYlGn"),
               ("gain", "Gain (LSB / coincidence)", None, None, "viridis"),
               ("crosstalk", "Crosstalk", 0.0, 0.8, "magma_r")]

    ncol = len(data) + 1                      # sweeps + one difference panel
    fig, axes = plt.subplots(len(metrics), ncol,
                             figsize=(4.6 * ncol, 4.3 * len(metrics)),
                             squeeze=False)

    for mi, (field, title, vmin, vmax, cmap) in enumerate(metrics):
        mats = [grid(d, field) for _, d in data]
        if vmin is None:                      # shared robust scale
            allv = np.concatenate([m[np.isfinite(m)] for m in mats])
            vmin, vmax = 0, float(np.percentile(allv, 95))

        for si, ((lbl, _), M) in enumerate(zip(data, mats)):
            ax = axes[mi][si]
            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[0.05, 0.95, 0.05, 0.95], aspect="equal")
            med = np.nanmedian(M)
            ax.set_title(f"{lbl}\n{title}   median={med:.3f}", fontsize=9)
            ax.set_xlabel("v"); ax.set_ylabel("u")
            ax.set_xticks(LEVELS); ax.set_yticks(LEVELS)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046)
            for a in range(len(LEVELS)):
                for b in range(len(LEVELS)):
                    if np.isfinite(M[a, b]):
                        ax.text(LEVELS[b], LEVELS[a], f"{M[a,b]:.2f}",
                                ha="center", va="center", fontsize=5)

        # paired difference: DNO minus same-session normal
        D = mats[1] - mats[2]
        ax = axes[mi][-1]
        lim = float(np.nanpercentile(np.abs(D[np.isfinite(D)]), 95)) or 1e-6
        im = ax.imshow(D, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                       extent=[0.05, 0.95, 0.05, 0.95], aspect="equal")
        ax.set_title(f"DNO - normal (same session)\n{title}   "
                     f"median={np.nanmedian(D):+.3f}", fontsize=9)
        ax.set_xlabel("v"); ax.set_ylabel("u")
        ax.set_xticks(LEVELS); ax.set_yticks(LEVELS)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        for a in range(len(LEVELS)):
            for b in range(len(LEVELS)):
                if np.isfinite(D[a, b]):
                    ax.text(LEVELS[b], LEVELS[a], f"{D[a,b]:+.2f}",
                            ha="center", va="center", fontsize=5)

    fig.suptitle(f"uv grid sweeps, BL={BL} — identical pulse streams (seed 0), "
                 "reset per trial\n"
                 "left-to-right: yesterday's normal, today's DNO, today's normal; "
                 "last column = paired DNO advantage",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "sweep_comparison_normal_vs_dno.png"
    fig.savefig(out, dpi=100)
    print("saved ->", out)


if __name__ == "__main__":
    main()

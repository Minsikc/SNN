"""Desired-vs-hardware gradient correlation, compared across runs.

Outputs a summary table and results/xor/grad_fidelity.png:
  A: per-epoch r trajectories per run
  B: r vs |desired| magnitude (epoch-level scatter)
  C: per-column r per run (the column-gain story)
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = [
    ("teacher s0", "results/eprop_grad_log/gradient_log_seed0.csv", "#999999"),
    ("XOR run1 K=1", "results/xor/grad_log_analog_run1.csv", "#d62728"),
    ("XOR run2 K=3", "results/xor/grad_log_analog_run2.csv", "#ff7f0e"),
    ("XOR run3 K=3+batch", "results/xor/grad_log_analog.csv", "#2ca02c"),
]


def load(path):
    rows = list(csv.DictReader(open(path)))
    eps = sorted({int(r["epoch"]) for r in rows})
    D = np.zeros((len(eps), 5, 5))
    H = np.zeros((len(eps), 5, 5))
    for r in rows:
        e = eps.index(int(r["epoch"]))
        D[e, int(r["row"]) - 1, int(r["col"]) - 1] = float(r["desired"])
        H[e, int(r["row"]) - 1, int(r["col"]) - 1] = float(r["hw_scaled"])
    return D, H


def corr(a, b):
    if a.std() > 0 and b.std() > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return np.nan


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
print(f"{'run':>20} {'pooled r':>8} {'median r':>8} {'IQR':>13} "
      f"{'neg.ep':>6} {'r|low tercile':>13} {'r|high tercile':>14}")

for name, path, color in LOGS:
    D, H = load(path)
    E = len(D)
    rs = np.array([corr(D[e].ravel(), H[e].ravel()) for e in range(E)])
    mags = np.abs(D).mean(axis=(1, 2))
    pooled = corr(D.ravel(), H.ravel())

    q1, q3 = np.nanpercentile(rs, [25, 75])
    neg = int(np.nansum(rs < 0))

    t1, t2 = np.percentile(mags, [33, 67])
    r_low = np.nanmedian(rs[mags <= t1])
    r_high = np.nanmedian(rs[mags >= t2])
    print(f"{name:>20} {pooled:>8.3f} {np.nanmedian(rs):>8.3f} "
          f"[{q1:.2f},{q3:.2f}]  {neg:>4}/{E} {r_low:>13.3f} {r_high:>14.3f}")

    axes[0].plot(np.arange(1, E + 1), rs, color=color, lw=1.2, label=name)
    axes[1].scatter(mags, rs, s=14, color=color, alpha=0.6, label=name)

    col_r = [corr(D[:, :, j].ravel(), H[:, :, j].ravel()) for j in range(5)]
    axes[2].plot(range(5), col_r, "o-", color=color, label=name)

axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("per-epoch r")
axes[0].set_title("A. desired vs HW gradient correlation")
axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_xscale("log")
axes[1].set_xlabel("mean |desired| in epoch (log)")
axes[1].set_ylabel("per-epoch r")
axes[1].set_title("B. fidelity vs gradient magnitude")
axes[1].grid(alpha=0.3)

axes[2].axhline(0, color="k", lw=0.6)
axes[2].set_xticks(range(5))
axes[2].set_xlabel("array column (hidden neuron)")
axes[2].set_ylabel("pooled r per column")
axes[2].set_title("C. per-column fidelity")
axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("results/xor/grad_fidelity.png", dpi=150)
print("\nsaved -> results/xor/grad_fidelity.png")

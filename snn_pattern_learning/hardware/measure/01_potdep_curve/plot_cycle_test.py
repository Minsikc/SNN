"""Replot a CycleTest CSV as the 5x5 continuous time-evolution grid.

Usage:
    python plot_cycle_test.py <csv_path> [out_png]

CSV layout (as written by the array cycle-test routine):
    3 rows            -> 3 cycles
    100 fields/row    -> 10 step-blocks x 10 fields
    10 values/field   -> N5 cols 0-4, then N6 cols 0-4
    Cell(r, c) = N5[r][c] - N6[r][c], 20 steps per cycle (block half 0 then half 1)
"""

import ast
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

N_ROWS = N_COLS = 5
STEPS_PER_HALF = 10


def load(csv_path):
    """Return array of shape (cycle, step_in_block, field, col) of N5-N6 differentials."""
    with open(csv_path) as f:
        rows = list(csv.reader(f))[1:]
    raw = np.array([[ast.literal_eval(field) for field in row] for row in rows]).astype(int)
    n_cycles = raw.shape[0]
    blocks = raw.reshape(n_cycles, STEPS_PER_HALF, 10, 10)
    return blocks[:, :, :, 0:5] - blocks[:, :, :, 5:10]


def cell_series(diff, r, c):
    """Concatenate both half-blocks of every cycle into one continuous trace."""
    return np.concatenate(
        [np.concatenate([diff[cy, :, r, c], diff[cy, :, r + 5, c]]) for cy in range(diff.shape[0])]
    )


def plot(diff, out_png, title="Continuous Time Evolution of 25 Synaptic Cells"):
    n_cycles = diff.shape[0]
    steps_per_cycle = 2 * STEPS_PER_HALF
    # pad past the extremes so no trace is clipped
    lim = 20 * int(np.ceil(max(abs(diff.min()), abs(diff.max())) / 20 + 0.5))

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(20, 20), sharex=True, sharey=True)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax = axes[r][c]
            ax.plot(cell_series(diff, r, c), marker=".", markersize=3)
            ax.set_title(f"Cell ({r + 1}, {c + 1})", fontsize=10)
            ax.grid(alpha=0.3, linestyle=":")
            ax.set_ylim(-lim, lim)
            for k in range(1, n_cycles):  # solid line = cycle boundary
                ax.axvline(k * steps_per_cycle - 0.5, color="k")
            for k in range(n_cycles):  # dashed line = potentiation -> depression
                ax.axvline(k * steps_per_cycle + STEPS_PER_HALF - 0.5, color="gray",
                           linestyle="--", linewidth=0.8)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel("Total Measurement Steps (Cycle 1 -> Cycle 2 -> ...)")
    fig.supylabel("Differential ADC Value (N5 - N6)")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    csv_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else csv_path.rsplit(".", 1)[0] + "_continuous.png"
    diff = load(csv_path)
    plot(diff, out_png)
    print(f"cycles={diff.shape[0]}  steps/cycle={2 * STEPS_PER_HALF}  "
          f"range={diff.min()}..{diff.max()}")

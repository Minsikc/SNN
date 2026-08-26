#!/usr/bin/env python3
"""Figure explaining the temporal XOR task itself.

Draws the actual 4 deterministic samples of TemporalXORDataset: input
spike rasters (10 channels) and target output rasters (5 neurons), with
the A window, memory gap, B window and response window marked. The point
of the figure: the answer window is far from bit A, so the network must
HOLD A across the gap -- membrane decay alone cannot bridge it.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from datasets.customdatasets import TemporalXORDataset

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
C_A, C_B, C_GO = "#dcebfa", "#dff2e6", "#fdeeda"   # window shading
C_IN, C_OUT0, C_OUT1 = "#2a2a2a", "#2a78d6", "#eb6834"

ds = TemporalXORDataset()
data, targets = ds.data.numpy(), ds.targets.numpy()   # (4,T,10), (4,T,5)
T = data.shape[1]
A0, A1 = 1, 4
B0, B1 = 9, 12
R0, R1 = ds.response_window
BITS = [(0, 0), (0, 1), (1, 0), (1, 1)]

fig, axes = plt.subplots(4, 2, figsize=(11.5, 8.2), sharex=True,
                         gridspec_kw=dict(width_ratios=[1.25, 1],
                                          hspace=0.32, wspace=0.16))

def shade(ax):
    ax.axvspan(A0 - 0.5, A1 - 0.5, color=C_A, zorder=0)
    ax.axvspan(B0 - 0.5, B1 - 0.5, color=C_B, zorder=0)
    ax.axvspan(R0 - 0.5, R1 - 0.5, color=C_GO, zorder=0)

def raster(ax, spikes, colors):
    n_ch = spikes.shape[1]
    for ch in range(n_ch):
        ts = np.where(spikes[:, ch] > 0)[0]
        ax.scatter(ts, np.full_like(ts, ch), marker="|", s=90, lw=1.6,
                   color=colors(ch), zorder=3)
    ax.set_ylim(-0.7, n_ch - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, T - 0.5)
    ax.grid(color=GRID, lw=0.6, axis="x")

for i, (a, b) in enumerate(BITS):
    x = a ^ b
    axi, axo = axes[i]
    shade(axi); shade(axo)

    raster(axi, data[i], lambda ch: C_IN)
    axi.set_yticks([1.5, 5.5, 8.5],
                   ["bit=1\nch 0-3", "bit=0\nch 4-7", "go cue\nch 8-9"],
                   fontsize=8)
    axi.axhline(3.5, color=GRID, lw=0.8)
    axi.axhline(7.5, color=GRID, lw=0.8)
    axi.set_ylabel(f"A={a}, B={b}", fontsize=11, rotation=0,
                   ha="right", va="center", labelpad=14, color=INK)

    raster(axo, targets[i], lambda ch: C_OUT1 if ch >= 3 else C_OUT0)
    axo.set_yticks([0.5, 2, 3.5],
                   ["class 0\nn 0-1", "silent\nn 2", "class 1\nn 3-4"],
                   fontsize=8)
    axo.yaxis.set_label_position("right")
    axo.set_ylabel(f"XOR = {x}", fontsize=11, rotation=0,
                   ha="left", va="center", labelpad=14,
                   color=(C_OUT1 if x else C_OUT0))

for ax, lab in [(axes[0, 0], "input (10 channels)"),
                (axes[0, 1], "target output (5 neurons)")]:
    ax.set_title(lab, fontsize=11, pad=34)
for ax in axes[-1]:
    ax.set_xlabel("timestep")

# window labels above the top row, between the axes and the title
top = axes[0, 0]
for x0, x1, lab in [(A0, A1, "bit A"), (B0, B1, "bit B"),
                    (R0, R1, "answer\nwindow")]:
    top.text((x0 + x1 - 1) / 2, -1.3, lab, ha="center", va="bottom",
             fontsize=9, color=INK, clip_on=False)
top.text((A1 + B0 - 1) / 2, -1.3, "gap — hold A\nin memory", ha="center",
         va="bottom", fontsize=8, color=MUTED, clip_on=False)

fig.suptitle("Temporal XOR: answer XOR(A, B) on the go cue — "
             "bit A must be held across the gap",
             fontsize=12.5, y=0.99)
fig.savefig("xor_task.png", dpi=130, bbox_inches="tight")
print("saved -> xor_task.png")

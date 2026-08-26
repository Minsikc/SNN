#!/usr/bin/env python3
"""Loss curves and output rasters for the three learning conditions.

  bptt          exact gradients through time -- the achievable ceiling
  digital       e-prop rule, gradients computed in software
  analog        e-prop, W_out gradient read back from the memristor crossbar
  frozen_wout   W_out held at init; only fc1/recurrent learn

BPTT is plotted on the loss axes only. At 50 epochs it has NOT converged
(best 0.555; it reaches 0.000 by epoch 200 at this lr), so it is a reference
line for what the architecture can ultimately do, not a same-budget rival --
comparing its 50-epoch number against e-prop would understate it.

frozen_wout is the control: if it matched the other two, the hidden layers
would explain the result on their own and the crossbar would not be doing
anything. Same task (single teacher-generated sequence), same lr, same
epoch count, so the three are directly comparable.

The raster marks each output channel's spikes against the target, coloured
by whether they agree, so a channel that is simply silent is visible as a
row of target-only marks rather than hiding inside a distance number.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one hue per condition, plus neutral ink for target/miss marks
COL = {"bptt": "#199e70", "digital": "#2a78d6", "analog": "#eb6834",
       "frozen_wout": "#898781"}
LABEL = {"bptt": "BPTT (exact gradient)", "digital": "e-prop digital (all SW)",
         "analog": "e-prop analog (W$_{out}$ on array)",
         "frozen_wout": "W$_{out}$ frozen"}
ORDER = ["bptt", "digital", "analog", "frozen_wout"]
INK, GRID = "#0b0b0b", "#e1e0d9"
HIT, ONLY_OUT, ONLY_TGT = "#199e70", "#e34948", "#898781"


def load():
    three = json.load(open("results/eprop_grad_log/three_conditions.json"))
    analog = json.load(open("results/eprop_grad_log/analog_curve.json"))
    data = {k: three[k] for k in three}
    data["analog"] = analog
    return data


def raster(ax, out, tgt, title):
    """Mark hits, false positives and misses per output channel."""
    out = np.asarray(out) > 0.5
    tgt = np.asarray(tgt) > 0.5
    T, C = out.shape
    for c in range(C):
        for t in range(T):
            o, g = out[t, c], tgt[t, c]
            if o and g:
                ax.plot([t], [c], "|", ms=13, mew=2.6, color=HIT)
            elif o and not g:
                ax.plot([t], [c + 0.16], "|", ms=10, mew=2.2, color=ONLY_OUT)
            elif g and not o:
                ax.plot([t], [c - 0.16], "|", ms=10, mew=2.2, color=ONLY_TGT)
    hits = int((out & tgt).sum())
    fp = int((out & ~tgt).sum())
    fn = int((~out & tgt).sum())
    ax.set_title(f"{title}\nmatched {hits}, extra {fp}, missed {fn}",
                 fontsize=10)
    ax.set_ylim(-0.6, C - 0.4)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_yticks(range(C), [f"ch{c+1}" for c in range(C)], fontsize=8)
    ax.set_xlabel("time step", fontsize=9)
    ax.grid(color=GRID, lw=0.7, axis="x")


def main():
    data = load()

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42,
                          wspace=0.26)

    # ---- loss curves, all three on one axis ----
    ax = fig.add_subplot(gs[0, :2])
    for cond in ORDER:
        if cond not in data:
            continue
        L = data[cond]["losses"]
        ax.plot(range(1, len(L) + 1), L, lw=2, color=COL[cond],
                label=f"{LABEL[cond]}  (best {min(L):.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Learning curves — same task, lr and epochs", fontsize=12)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=9, frameon=False)

    # ---- summary tile ----
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 0.95, "best loss", fontsize=11, va="top", color=INK)
    y = 0.78
    for cond in [c for c in ORDER if c in data]:
        L = data[cond]["losses"]
        ax.text(0.02, y, f"{LABEL[cond]}", fontsize=9.5, va="top",
                color=COL[cond])
        ax.text(0.98, y, f"{min(L):.3f}", fontsize=11, va="top", ha="right",
                color=COL[cond], family="monospace")
        y -= 0.12
    dig, ana, frz = (min(data[c]["losses"])
                     for c in ("digital", "analog", "frozen_wout"))
    ax.text(0, 0.30, f"analog reaches {ana/dig:.2f}x the digital loss,\n"
                     f"while freezing W$_{{out}}$ leaves it at "
                     f"{frz/dig:.1f}x.",
            fontsize=9.5, va="top", color=INK)
    ax.text(0, 0.08, "so the crossbar-trained output layer is\n"
                     "carrying the learning, not the hidden layers.",
            fontsize=9, va="top", color="#52514e")

    # ---- rasters ----
    for k, cond in enumerate(("digital", "analog", "frozen_wout")):  # 3 panels
        ax = fig.add_subplot(gs[1, k])
        raster(ax, data[cond]["final_spikes"], data[cond]["target_spikes"],
               LABEL[cond])

    handles = [plt.Line2D([], [], color=HIT, marker="|", ls="", ms=12,
                          mew=2.6, label="matched"),
               plt.Line2D([], [], color=ONLY_OUT, marker="|", ls="", ms=10,
                          mew=2.2, label="output only (extra)"),
               plt.Line2D([], [], color=ONLY_TGT, marker="|", ls="", ms=10,
                          mew=2.2, label="target only (missed)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Single teacher-generated spike sequence — BPTT ceiling, "
                 "e-prop digital, e-prop analog, and a frozen-output control",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = "three_conditions_loss_raster.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved ->", out)

    print(f"\n{'condition':>14} {'best loss':>10} {'matched':>8} "
          f"{'extra':>7} {'missed':>7}")
    for cond in [c for c in ORDER if c in data]:
        o = np.asarray(data[cond]["final_spikes"]) > 0.5
        t = np.asarray(data[cond]["target_spikes"]) > 0.5
        print(f"{cond:>14} {min(data[cond]['losses']):>10.4f} "
              f"{int((o&t).sum()):>8} {int((o&~t).sum()):>7} "
              f"{int((~o&t).sum()):>7}")


if __name__ == "__main__":
    main()

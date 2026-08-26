#!/usr/bin/env python3
"""Figure for the temporal-XOR crossbar demo.

Panel A: loss curves, all conditions on one axis.
Panel B: XOR accuracy curves.
Panel C: analog-condition output raster (best epoch) vs target, 4 samples.
Panel D: desired vs hardware gradient scatter from the analog grad log.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results/xor"
CONDITIONS = ["bptt", "digital", "frozen", "analog"]
COLORS = {"bptt": "#888888", "digital": "#1f77b4",
          "frozen": "#d62728", "analog": "#2ca02c"}
LABELS = {"bptt": "BPTT (ceiling)", "digital": "digital e-prop",
          "frozen": "frozen W_out (control)", "analog": "analog W_out (5x5 array)"}
SAMPLE_NAMES = ["A=0,B=0", "A=0,B=1", "A=1,B=0", "A=1,B=1"]


def load(cond, seed=0):
    p = os.path.join(RESULTS, f"xor_{cond}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    results = {c: load(c, seed) for c in CONDITIONS}

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])

    for c, r in results.items():
        if r is None:
            continue
        ep = np.arange(1, len(r["losses"]) + 1)
        ax_loss.plot(ep, r["losses"], color=COLORS[c], label=LABELS[c], lw=1.5)
        # accuracy: show a 5-epoch moving average plus raw dots
        acc = np.array(r["accs"])
        ax_acc.plot(ep, acc, ".", color=COLORS[c], alpha=0.25, ms=4)
        if len(acc) >= 5:
            ma = np.convolve(acc, np.ones(5) / 5, mode="valid")
            ax_acc.plot(ep[4:], ma, color=COLORS[c], label=LABELS[c], lw=1.8)

    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Temporal XOR: training loss")
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("XOR accuracy (4 patterns)")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.axhline(1.0, color="k", ls=":", lw=0.8)
    ax_acc.set_title("XOR accuracy (5-epoch moving average)")
    ax_acc.legend(fontsize=8, loc="lower right"); ax_acc.grid(alpha=0.3)

    # Panel C: analog best-epoch raster (fall back to digital if missing)
    rast_cond = "analog" if results.get("analog") else "digital"
    r = results.get(rast_cond)
    if r is not None:
        out = np.array(r["best_outputs"])   # (4, T, 5)
        tgt = np.array(r["targets"])
        inner = fig.add_gridspec(2, 4, height_ratios=[1, 1.2],
                                 left=0.07, right=0.97, bottom=0.06,
                                 top=0.42, wspace=0.35)
        for i in range(4):
            ax = fig.add_subplot(inner[1, i] if False else inner[:, i])
            T = out.shape[1]
            for n in range(5):
                t_t = np.where(tgt[i, :, n] > 0)[0]
                t_o = np.where(out[i, :, n] > 0)[0]
                ax.scatter(t_t, np.full_like(t_t, n) + 0.18, marker="|",
                           s=90, color="k", label="target" if (n == 0 and i == 0) else None)
                ax.scatter(t_o, np.full_like(t_o, n) - 0.18, marker="|",
                           s=90, color=COLORS[rast_cond],
                           label=rast_cond if (n == 0 and i == 0) else None)
            ax.set_ylim(-0.8, 4.8); ax.set_xlim(-0.5, T - 0.5)
            ax.set_yticks(range(5))
            ax.set_title(f"{SAMPLE_NAMES[i]} (XOR={int(SAMPLE_NAMES[i][2]) ^ int(SAMPLE_NAMES[i][6])})",
                         fontsize=9)
            ax.set_xlabel("t")
            if i == 0:
                ax.set_ylabel("output neuron")
                ax.legend(fontsize=7, loc="upper left")
        fig.text(0.07, 0.44, f"Best-epoch output spikes ({LABELS[rast_cond]}): "
                             "target (black, up) vs model (color, down)",
                 fontsize=10)

    fig.suptitle(f"Temporal XOR on the 5x5 memristor crossbar (seed {seed})",
                 fontsize=13)
    out_path = os.path.join(RESULTS, f"xor_demo_seed{seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")

    # Gradient fidelity stats from the analog grad log
    glog = os.path.join(RESULTS, "grad_log_analog.csv")
    if os.path.exists(glog):
        import csv
        des, hw = [], []
        by_epoch = {}
        with open(glog) as f:
            for row in csv.DictReader(f):
                d, h = float(row["desired"]), float(row["hw_scaled"])
                des.append(d); hw.append(h)
                by_epoch.setdefault(int(row["epoch"]), []).append((d, h))
        des, hw = np.array(des), np.array(hw)
        if des.std() > 0 and hw.std() > 0:
            r_all = np.corrcoef(des, hw)[0, 1]
            rs = []
            for ep, pairs in sorted(by_epoch.items()):
                a = np.array(pairs)
                if a[:, 0].std() > 0 and a[:, 1].std() > 0:
                    rs.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
            print(f"[grad fidelity] pooled r = {r_all:.3f}, per-epoch r: "
                  f"min {min(rs):.3f} / median {np.median(rs):.3f} / "
                  f"max {max(rs):.3f} over {len(rs)} epochs")


if __name__ == "__main__":
    main()

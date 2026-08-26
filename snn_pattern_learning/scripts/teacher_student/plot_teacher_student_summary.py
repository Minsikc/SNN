#!/usr/bin/env python3
"""Teacher-student summary: software baselines vs analog hardware, post-fix.

Three panels, each answering one question.

(a) Does the analog run learn like the software ones?
    Loss per epoch, all conditions on shared axes. The realizable optimum is
    drawn at 0 -- planting the teacher's weights into the student gives loss
    0.000000 and VRD 0.000000 exactly, so a curve that flattens above the line
    has stopped improving, not converged. Before 2026-08-20 the hardware model
    fired at LIF_Node's default 0.5 while the teacher used init_thresh, which
    made the target unreachable and every earlier loss number meaningless.

(b) Does the array hold the gradient it was given?
    Fidelity per epoch = correlation between the gradient asked for and the
    one read back off the array (25 cells), against mean |desired| on a log
    second axis. Both analog runs here use the _NR opcodes, which deliver the
    pulses without the per-command pre/post read: the read-wrapped variant
    cost ~40 array reads/epoch (~27% decay of the stored gradient at
    0.79%/read) and the host discarded every one of them, since it reads the
    accumulated gradient once at the epoch boundary. Dropping those reads
    left ep>=38 fidelity at 0.83 instead of 0.66, and that run is the "analog"
    condition plotted throughout.

(c) Where does each run end up?
    Best loss per condition, so the ordering is readable at a glance.

DNO is the second analog condition and it FAILS, which is itself the result.
In DNO potentiation a row whose pulse bit is 0 gets N3 asserted instead of
idling, so every row is driven every slot -- push-pull rather than one-sided.
Measured single-command strength is ~24% weaker as expected (+111.6 vs +146.8
LSB at u=1010101010). But over a training run the opposing pulses cancel what
was already stored: mean |hw_adc| falls 37.6 -> 1.7 across the run while the
desired gradient holds at ~0.036, so the stored value sinks under the noise
floor and fidelity decays to r ~ 0 and even negative late (ep>=38 mean -0.13).
The array stops holding charge, so DNO is not usable for accumulation here.
"""
import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID = "#0b0b0b", "#e1e0d9"
C_BPTT, C_DIG, C_FROZ = "#104281", "#2a78d6", "#898781"
C_NR, C_DNO = "#eb6834", "#7b3fa0"
NOISE_FLOOR = 0.03      # mean|desired| below which fidelity collapses
LATE = 38               # epoch where the two analog runs diverge


def analog_curve(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    ep = re.findall(r"Epoch (\d+)/\d+, Loss: ([0-9.]+), Metric: ([0-9.]+)", txt)
    return [float(l) for _, l, _ in ep], [float(m) for _, _, m in ep]


def fidelity(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(lambda: ([], []))
    for r in rows:
        by[int(r["epoch"])][0].append(float(r["desired"]))
        by[int(r["epoch"])][1].append(float(r["hw_adc"]))
    eps, rs, mags, adcs = [], [], [], []
    for e in sorted(by):
        d, h = np.array(by[e][0]), np.array(by[e][1])
        if len(d) > 2 and d.std() > 0 and h.std() > 0:
            eps.append(e)
            rs.append(float(np.corrcoef(d, h)[0, 1]))
            mags.append(float(np.abs(d).mean()))
            adcs.append(float(np.abs(h).mean()))
    return (np.array(eps), np.array(rs), np.array(mags), np.array(adcs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sw", default="results/eprop_grad_log/sw_thr04_lr003_ep50.json")
    ap.add_argument("--nr-log", default="_analog_NR_ep50.log")
    ap.add_argument("--nr-grad",
                    default="results/eprop_grad_log/gradient_log_NR_ep50.csv")
    ap.add_argument("--dno-log", default="_analog_DNO_ep50.log")
    ap.add_argument("--dno-grad",
                    default="results/eprop_grad_log/gradient_log_DNO_ep50.csv")
    ap.add_argument("--out",
                    default="results/eprop_grad_log/teacher_student_summary.png")
    args = ap.parse_args()

    sw = json.load(open(args.sw, encoding="utf-8"))
    nr_loss, nr_metric = analog_curve(args.nr_log)
    e_nr, r_nr, m_nr, a_nr = fidelity(args.nr_grad)
    dno_loss, dno_metric = analog_curve(args.dno_log)
    e_dn, r_dn, m_dn, a_dn = fidelity(args.dno_grad)

    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.28)

    # ---- (a) learning curves --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for cond, col in (("bptt", C_BPTT), ("digital", C_DIG),
                      ("frozen_wout", C_FROZ)):
        y = sw[cond]["losses"]
        ax.plot(range(1, len(y) + 1), y, lw=1.6, color=col, alpha=0.95)
    ax.plot(range(1, len(dno_loss) + 1), dno_loss, lw=1.7, color=C_DNO,
            alpha=0.9)
    ax.plot(range(1, len(nr_loss) + 1), nr_loss, lw=2.4, color=C_NR)
    ax.axhline(0, color=INK, lw=1.3, ls="--", alpha=0.8)
    ax.annotate("realizable optimum — planted teacher gives loss 0, VRD 0",
                xy=(0.98, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 7), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("(a) learning curves\nhidden 5, out 5 — thr 0.4, lr 0.03",
                 fontsize=10.5)
    ax.grid(color=GRID, lw=0.8)

    handles = [
        plt.Line2D([], [], color=C_DIG, lw=2,
                   label=f"e-prop digital — {sw['digital']['best_loss']:.3f}"),
        plt.Line2D([], [], color=C_BPTT, lw=2,
                   label=f"BPTT exact — {sw['bptt']['best_loss']:.3f}"),
        plt.Line2D([], [], color=C_NR, lw=2.5,
                   label=f"analog — {min(nr_loss):.3f}"),
        plt.Line2D([], [], color=C_DNO, lw=2,
                   label=f"analog, DNO — {min(dno_loss):.3f}"),
        plt.Line2D([], [], color=C_FROZ, lw=2,
                   label=f"W_out frozen — {sw['frozen_wout']['best_loss']:.3f}"),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")

    # ---- (b) fidelity ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(e_dn, r_dn, "o-", ms=3, lw=1.4, color=C_DNO, alpha=0.85,
            markerfacecolor="white", markeredgewidth=1.0,
            label="DNO")
    ax.axhline(0, color=INK, lw=1.0, alpha=0.5)
    ax.plot(e_nr, r_nr, "o-", ms=3.4, lw=1.8, color=C_NR,
            markerfacecolor="white", markeredgewidth=1.1,
            label="analog")
    ax.axvspan(LATE, e_nr.max(), color=INK, alpha=0.05, zorder=0)
    lo_nr = r_nr[e_nr >= LATE].mean()
    lo_dn = r_dn[e_dn >= LATE].mean()
    ax.annotate(f"ep$\\geq${LATE} mean r —  analog {lo_nr:+.2f}   "
                f"DNO {lo_dn:+.2f}",
                xy=(0.985, 0.965), xycoords="axes fraction", ha="right",
                va="top", fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("fidelity  r(desired, hw_adc)")
    ax.set_title("(b) gradient fidelity — DNO collapses",
                 fontsize=10.5)
    ax.grid(color=GRID, lw=0.8)

    ax2 = ax.twinx()
    ax2.plot(e_nr, m_nr, lw=1.5, color=C_BPTT, alpha=0.65)
    ax2.axhline(NOISE_FLOOR, color=C_BPTT, lw=1.0, ls=":", alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean |desired gradient|", color=C_BPTT, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_BPTT)
    ax2.annotate("device noise floor", xy=(1, NOISE_FLOOR), xytext=(2, -12),
                 textcoords="offset points", fontsize=7, color=C_BPTT)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + [plt.Line2D([], [], color=C_BPTT, lw=1.5, alpha=0.65,
                               label="mean |desired| (right axis)")],
              l1 + ["mean |desired| (right axis)"],
              fontsize=7.5, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, -0.02))

    # ---- (c) best loss ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    names = ["e-prop\ndigital", "BPTT\nexact", "analog",
             "analog\nDNO", "W_out\nfrozen"]
    vals = [sw["digital"]["best_loss"], sw["bptt"]["best_loss"],
            min(nr_loss), min(dno_loss), sw["frozen_wout"]["best_loss"]]
    cols = [C_DIG, C_BPTT, C_NR, C_DNO, C_FROZ]
    bars = ax.bar(names, vals, color=cols, width=0.68)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, lw=1.3, ls="--", alpha=0.8)
    ax.set_ylabel("best loss")
    ax.set_title("(c) best loss reached\n(0 is attainable on this task)",
                 fontsize=10.5)
    ax.grid(color=GRID, lw=0.8, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, max(vals) * 1.18)

    fig.suptitle("Teacher-student on the 5$\\times$5 crossbar — after the "
                 "threshold fix and with read-free updates", fontsize=13)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.05, right=0.96)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print("saved ->", args.out)

    print(f"\n{'condition':>26} {'best loss':>10} {'final metric':>13}")
    print(f"{'e-prop digital (SW)':>26} {sw['digital']['best_loss']:10.3f} "
          f"{sw['digital']['vrds'][-1]:13.3f}")
    print(f"{'BPTT (SW)':>26} {sw['bptt']['best_loss']:10.3f} "
          f"{sw['bptt']['vrds'][-1]:13.3f}")
    print(f"{'analog':>26} {min(nr_loss):10.3f} {nr_metric[-1]:13.3f}")
    print(f"{'analog DNO':>26} {min(dno_loss):10.3f} "
          f"{dno_metric[-1]:13.3f}")
    print(f"{'W_out frozen (SW)':>26} {sw['frozen_wout']['best_loss']:10.3f} "
          f"{sw['frozen_wout']['vrds'][-1]:13.3f}")
    print(f"\nfidelity ep>={LATE}:  analog {lo_nr:+.3f} "
          f"(min {r_nr[e_nr >= LATE].min():+.3f})   "
          f"DNO {lo_dn:+.3f} (min {r_dn[e_dn >= LATE].min():+.3f})")
    print(f"mean |hw_adc| across run:  analog {a_nr.mean():.1f}   "
          f"DNO {a_dn.mean():.1f}"
          f"   (DNO ep0 {a_dn[0]:.1f} -> ep{e_dn[-1]} {a_dn[-1]:.1f})")


if __name__ == "__main__":
    main()

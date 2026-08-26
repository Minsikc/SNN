#!/usr/bin/env python3
"""Measured 6T1C cell vs the fitted aihwkit LinearStepDevice.

Every curve labelled "LinearStep" is produced by replaying aihwkit's OWN
recursion (update_once_add + clip) from the fitted config, not by evaluating
the closed form that was fitted.  So the panels test the parameter mapping,
not just the curve shape.

Panel 5 is the honest one: it shows where the fit stops being a measurement
and starts being an extrapolation.
"""
import csv
import importlib.util
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#0b0b0b", "#e1e0d9", "#898781"
POT, DEP, NEU = "#2a78d6", "#eb6834", "#b8b6ae"


def load_all():
    spec = importlib.util.spec_from_file_location("m", "fit_update_model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    d = M.load()
    cfg = json.load(open("aihwkit_linearstep_params.json", encoding="utf-8"))
    return M, d, cfg


class LinearStep:
    """aihwkit's LinearStepDevice, noiseless, replayed in python.

    Mirrors rpu_pulsed_device.cpp (scale construction) and
    rpu_linearstep_device.cpp (slope construction + update_once_add).
    """

    def __init__(self, L):
        ub, db = min(L["up_down"], 0.0), max(-L["up_down"], 0.0)
        self.s_up = (ub + 1.0) * L["dw_min"]
        self.s_dn = (db + 1.0) * L["dw_min"]
        self.sl_up = -L["gamma_up"] * self.s_up / L["w_max"]
        self.sl_dn = -L["gamma_down"] * self.s_dn / L["w_min"]
        self.wmax, self.wmin = L["w_max"], L["w_min"]

    def step(self, w, positive):
        return (self.sl_up * w + self.s_up if positive
                else -(self.sl_dn * w + self.s_dn))

    def burst(self, w0, n, positive):
        w = w0
        for _ in range(n):
            w = min(max(w + self.step(w, positive), self.wmin), self.wmax)
        return w


def main():
    M, d, cfg = load_all()
    L, lsb, w0 = cfg["linearstep"], cfg["lsb_per_w"], cfg["w0_reset"]
    me = cfg["measured"]
    dev = LinearStep(L)

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # ---- 1. burst curve --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for lab, key, other, col, pos in (("POT", "C_pot", "C_dep", POT, True),
                                      ("DEP", "C_dep", "C_pot", DEP, False)):
        xs, ys, es = [], [], []
        for n in range(1, 11):
            s = (d[key] == n) & (d[other] == 0)
            if s.sum() >= 8:
                xs.append(n)
                ys.append(d["delta"][s].mean())
                es.append(d["delta"][s].std() / np.sqrt(s.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, color=col, capsize=3,
                    label=f"{lab} measured", zorder=3)
        cc = np.arange(0, 11)
        sim = [(dev.burst(w0, int(n), pos) - w0) * lsb for n in cc]
        ax.plot(cc, sim, color=col, lw=2, alpha=0.75,
                label=f"{lab} LinearStep")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("coincidences in the burst, C")
    ax.set_ylabel("delta (LSB)")
    ax.set_title("1. burst response\nfitted device replays the measurement",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 2. per-pulse step ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    nn = np.arange(1, 11)
    for lab, col, pos, A, k in (("POT", POT, True, me["A_pot"], me["k_pot"]),
                                ("DEP", DEP, False, me["A_dep"],
                                 me["k_dep"])):
        sim = []
        w = w0
        for _ in nn:
            st = dev.step(w, pos) * lsb
            sim.append(st)
            w = w + dev.step(w, pos)
        ax.plot(nn, sim, "o-", color=col, lw=2, ms=5,
                label=f"{lab} LinearStep")
        sgn = 1 if pos else -1
        ax.plot(nn, sgn * A * np.exp(-k * (nn - 1)), ls="--", lw=1.4,
                color=col, alpha=0.65, label=f"{lab} measured fit")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pulse index within the burst")
    ax.set_ylabel("that pulse's contribution (LSB)")
    ax.set_title("2. the step shrinks along the burst\n"
                 "device recursion == fitted exponential", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 3. parity -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    pred = np.zeros(len(d["delta"]))
    for i in range(len(pred)):
        cp, cd = int(d["C_pot"][i]), int(d["C_dep"][i])
        w = w0
        if cp:
            w = dev.burst(w, cp, True)
        if cd:
            w = dev.burst(w, cd, False)
        pred[i] = (w - w0) * lsb
    ax.scatter(pred, d["delta"], s=7, alpha=0.3, color=POT,
               edgecolors="none")
    lim = [min(pred.min(), d["delta"].min()) - 8,
           max(pred.max(), d["delta"].max()) + 8]
    ax.plot(lim, lim, color=INK, lw=1.5)
    rms = float(np.sqrt(np.mean((pred - d["delta"]) ** 2)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("LinearStep prediction (LSB)")
    ax.set_ylabel("measured delta (LSB)")
    ax.set_title(f"3. all {len(pred)} cell-updates\n"
                 f"r = {np.corrcoef(pred, d['delta'])[0,1]:+.4f}, "
                 f"rms {rms:.1f} LSB", fontsize=10)
    ax.grid(color=GRID, lw=0.8)

    # ---- 4. per-cell spread ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    gp, gd = [], []
    for cid in range(25):
        s = (d["cid"] == cid) & (d["C_pot"] > 0) & (d["C_dep"] == 0)
        gp.append((d["delta"][s] / d["C_pot"][s]).mean() if s.sum() > 3
                  else np.nan)
        s = (d["cid"] == cid) & (d["C_dep"] > 0) & (d["C_pot"] == 0)
        gd.append((-d["delta"][s] / d["C_dep"][s]).mean() if s.sum() > 3
                  else np.nan)
    gp, gd = np.array(gp), np.array(gd)
    x = np.arange(25)
    ax.plot(x, gp, "o", ms=5, color=POT, label="POT, per cell")
    ax.plot(x, gd, "o", ms=5, color=DEP, label="DEP, per cell")
    mean_all = np.nanmean(np.concatenate([gp, gd]))
    band = L["dw_min_dtod"] * mean_all
    ax.axhline(mean_all, color=INK, lw=1.5,
               label=f"dw_min {L['dw_min']*lsb:.1f} LSB")
    ax.axhspan(mean_all - band, mean_all + band, color=MUTED, alpha=0.22,
               label=f"dw_min_dtod = {L['dw_min_dtod']:.3f}")
    ax.set_xlabel("cell index (row-major)")
    ax.set_ylabel("mean step per coincidence (LSB)")
    ax.set_title("4. cell-to-cell spread sets dw_min_dtod", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    # ---- 5. the extrapolation caveat ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ws = np.linspace(L["w_min"], L["w_max"], 400)
    ax.plot(ws * lsb, [dev.step(w, True) * lsb for w in ws], color=POT, lw=2,
            label="POT step (fitted, gamma=%.2f)" % L["gamma_up"])
    ax.plot(ws * lsb, [dev.step(w, False) * lsb for w in ws], color=DEP, lw=2,
            label="DEP step (fitted, gamma=%.2f)" % L["gamma_down"])
    soft_up = dev.s_up * (1 - ws / L["w_max"]) * lsb
    ax.plot(ws * lsb, soft_up, color=POT, lw=1.3, ls=":",
            label="POT step if gamma=1 (soft bounds)")
    lo = d["before"].min()
    hi = d["before"].max()
    ax.axvspan(lo, hi, color="#cfe0f5", alpha=0.55, zorder=0,
               label=f"measured range ({lo:.0f}..{hi:.0f} LSB)")
    ax.axhline(0, color=INK, lw=1)
    for wf, col in ((-dev.s_up / dev.sl_up, POT),
                    (dev.s_dn / dev.sl_dn * -1, DEP)):
        ax.axvline(wf * lsb, color=col, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(L["w_max"] * lsb, color=MUTED, lw=1.2)
    ax.axvline(L["w_min"] * lsb, color=MUTED, lw=1.2)
    ax.set_xlabel("cell state w (LSB)")
    ax.set_ylabel("step size (LSB)")
    ax.set_title("5. CAUTION: outside the shaded band this is\n"
                 "extrapolation. Fitted gamma stalls the cell at\n"
                 "+255/-260 LSB, short of the +455/-444 rails",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(color=GRID, lw=0.8)

    # ---- 6. residual vs half-select, which the device cannot express -----
    ax = fig.add_subplot(gs[1, 2])
    resid = d["delta"] - pred
    net = d["hp"] - d["hd"]
    xs, ys, es = [], [], []
    for n in sorted({int(v) for v in net}):
        s = net == n
        if s.sum() >= 8:
            xs.append(n)
            ys.append(resid[s].mean())
            es.append(resid[s].std() / np.sqrt(s.sum()))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=6, color="#7a5bb5", capsize=3,
                lw=1.8)
    sl, ic = np.polyfit(net, resid, 1)
    xx = np.array([min(xs), max(xs)])
    ax.plot(xx, sl * xx + ic, color=INK, lw=1.3, ls="--",
            label=f"{sl:+.2f} LSB per net pair")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("net half-select drive (POT pairs - DEP pairs)")
    ax.set_ylabel("residual: measured - LinearStep (LSB)")
    ax.set_title("6. what LinearStepDevice cannot represent\n"
                 "half-select leak survives in the residual", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(color=GRID, lw=0.8)

    fig.suptitle("Measured 6T1C cell vs fitted aihwkit LinearStepDevice",
                 fontsize=13)
    fig.savefig("aihwkit_fit.png", dpi=110, bbox_inches="tight")
    print("saved -> aihwkit_fit.png")
    print(f"parity r {np.corrcoef(pred, d['delta'])[0,1]:+.4f}, rms {rms:.2f} LSB")
    print(f"residual vs net half-select: {sl:+.3f} LSB/pair")


if __name__ == "__main__":
    main()

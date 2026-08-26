#!/usr/bin/env python3
"""Fit the measured 6T1C cell to an aihwkit LinearStepDevice.

WHY THIS IS THE RIGHT DEVICE MODEL
----------------------------------
aihwkit's LinearStepDevice applies, once per coincidence,

    w += slope_up   * w + scale_up      (positive pulse)
    w -= slope_down * w + scale_down    (negative pulse)

then clips to [min_bound, max_bound].  (Source: update_once_add in
src/rpucuda/rpu_linearstep_device.cpp.)  Iterating that recursion over n
pulses gives a geometric series, so the nth pulse contributes

    step(n) = (slope*w0 + scale) * (1 + slope)^(n-1)  ~  A * exp(slope * n)

which is exactly the form fitted in fit_update_model.py,

    step(n) = A * exp(-k * n)

So the measured within-burst saturation and LinearStep's linear-in-w rule are
the SAME model, with slope = -k.  That correspondence is what makes this fit
meaningful rather than a curve-shape coincidence.

WHAT THE MEASUREMENT CAN AND CANNOT PIN DOWN
--------------------------------------------
The hardware data (uv_random_signed) fixes the PRODUCT relationship
A = slope*w0 + scale near the reset state w0, and the decay rate k.  It does
not independently fix the bounds, because every trial starts from a hard reset
and no burst travels more than ~16% of the device range.  The bounds are
therefore taken from the separately measured saturation rails (+455/-444 LSB,
from the half-select attractor sweep), and `scale` is then derived so that the
model reproduces the measured first-pulse step at the reset state.

aihwkit parameterises the slope as gamma normalised by the bounds:

    slope_up   = -|gamma_up|   / max_bound
    slope_down = -|gamma_down| / |min_bound|

(see the LinearStepDevice docstring), so gamma is recovered as
gamma_up = k_pot * max_bound in normalised weight units.

UNITS
-----
Hardware is in ADC LSB; aihwkit works in normalised weight units.  Everything
is converted with  w_norm = w_LSB / LSB_PER_W  where LSB_PER_W is chosen so
the measured rails land on the requested w_max.  Both conventions are printed.

HALF-SELECT
-----------
LinearStepDevice has no notion of a half-select leak, so the measured
-1.4..+0.7 LSB/pair term cannot be expressed as a device parameter.  It is
reported here as an equivalent per-coincidence bias and folded into
`dw_min_std` only as a variability contribution, with the caveat stated.

Running this does NOT require aihwkit to be installed: the fit is closed-form
and the output is a parameter set plus a ready-to-run config snippet.  If
aihwkit IS importable, the script additionally instantiates the device and
verifies the round-trip by simulating pulse bursts.
"""
import csv
import importlib.util
import json

import numpy as np

# measured rails, from the half-select attractor sweep (seq_attractor)
LSB_MAX = 455.0
LSB_MIN = -444.0
W_MAX = 1.0          # target normalised bound


def measured_params():
    """Refit the hardware model so this script is self-contained."""
    spec = importlib.util.spec_from_file_location("m", "fit_update_model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    d = M.load()
    p = M.fit(M.m_sat_c, [14.5, .05, 15.5, .05, 1.5, -1.5, -1.5, 0.], d)
    gp, kp, gd, kd, ap, ar, ad, c = p
    # residual scatter of a single coincidence, in LSB
    pred = M.m_sat_c(p, d)
    resid = d["delta"] - pred
    return dict(A_pot=gp, k_pot=kp, A_dep=gd, k_dep=kd,
                hs_pot=ap, hs_reset=ar, hs_dep=ad, offset=c,
                resid_sd=float(resid.std()), data=d, M=M, p=p)


def cell_gains():
    """Per-cell first-pulse step, for the device-to-device spread."""
    spec = importlib.util.spec_from_file_location("m", "fit_update_model.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    d = M.load()
    gp, gd = [], []
    for cid in range(25):
        s = (d["cid"] == cid) & (d["C_pot"] > 0) & (d["C_dep"] == 0)
        if s.sum() > 3:
            gp.append((d["delta"][s] / d["C_pot"][s]).mean())
        s = (d["cid"] == cid) & (d["C_dep"] > 0) & (d["C_pot"] == 0)
        if s.sum() > 3:
            gd.append((-d["delta"][s] / d["C_dep"][s]).mean())
    return np.array(gp), np.array(gd)


def main():
    m = measured_params()
    gp_cells, gd_cells = cell_gains()

    lsb_per_w = LSB_MAX / W_MAX          # LSB per unit normalised weight
    w_min = LSB_MIN / lsb_per_w
    w0 = float(np.mean(m["data"]["before"])) / lsb_per_w

    # aihwkit's exact construction (rpu_pulsed_device.cpp lines 603-630,
    # rpu_linearstep_device.cpp lines 44-68), with all dtod/std draws at 0:
    #
    #   up_bias   = min(up_down, 0)          down_bias = max(-up_down, 0)
    #   scale_up   = (up_bias   + 1) * dw_min
    #   scale_down = (down_bias + 1) * dw_min
    #   slope_up   = -gamma_up   * scale_up   / w_max     (mean_bound_reference)
    #   slope_down = -gamma_down * scale_down / w_min
    #
    # and the update, per coincidence (update_once_add):
    #   POT: w += slope_up   * w + scale_up
    #   DEP: w -= slope_down * w + scale_down
    #
    # Note slope_down is divided by w_min < 0, so slope_down > 0, and the DEP
    # step SHRINKS as w falls toward w_min.  Both directions therefore decay
    # with pulse index n as (1 - |rate|)^n; matching that to the measured
    # A*exp(-k n) gives the two conditions below.
    #
    # Unknowns: dw_min, up_down, gamma_up, gamma_down.  Measured: the POT and
    # DEP first-pulse steps at w0, and the two decay rates k.
    dw_up = m["A_pot"] / lsb_per_w        # first-pulse POT step, w units
    dw_dn = m["A_dep"] / lsb_per_w        # first-pulse DEP step (magnitude)

    # decay rate per pulse must equal k:  |slope| = k
    #   slope_up   = -gamma_up  * scale_up  / w_max   -> |slope_up| = k_pot
    #   slope_down = -gamma_down* scale_down/ w_min   -> |slope_down| = k_dep
    # and the first step at w0:
    #   POT: slope_up*w0 + scale_up = dw_up
    #        -k_pot*w0 + scale_up   = dw_up   -> scale_up = dw_up + k_pot*w0
    #   DEP: slope_down*w0 + scale_down = dw_dn
    #        +k_dep*w0 + scale_down = dw_dn   -> scale_dn = dw_dn - k_dep*w0
    scale_up = dw_up + m["k_pot"] * w0
    scale_dn = dw_dn - m["k_dep"] * w0

    # Invert the (up_bias, down_bias) construction.  The convention was
    # probed empirically on aihwkit 1.1.0 (ConstantStepDevice, deterministic
    # single pulse): up_down only SHRINKS one side, the other keeps dw_min:
    #   up_down <= 0:  scale_up = (1+up_down)*dw_min,  scale_down = dw_min
    #   up_down >  0:  scale_up = dw_min,  scale_down = (1-up_down)*dw_min
    # (the symmetric (1+-up_down)*dw_min reading of the docs gives a DEP step
    # ~20% small).  So dw_min is the LARGER scale and up_down the ratio:
    if scale_up <= scale_dn:
        dw_min = scale_dn
        up_down = scale_up / scale_dn - 1.0      # <= 0
    else:
        dw_min = scale_up
        up_down = 1.0 - scale_dn / scale_up      # >= 0

    # gamma from |slope| = gamma * scale / |w_ref|, using the SAME scales
    # aihwkit will reconstruct, so the round trip is exact
    gamma_up = m["k_pot"] * W_MAX / scale_up
    gamma_down = m["k_dep"] * abs(w_min) / scale_dn

    # --- variability -------------------------------------------------------
    dtod = float(np.std(np.concatenate([gp_cells, gd_cells]))
                 / np.mean(np.concatenate([gp_cells, gd_cells])))
    ctoc = m["resid_sd"] / m["A_pot"]              # per-coincidence scatter
    ud_dtod = float(np.std(gp_cells / gd_cells.mean()))

    print("=" * 72)
    print("measured hardware model (fit_update_model.py)")
    print("=" * 72)
    print(f"  POT: step(n) = {m['A_pot']:.2f} * exp(-{m['k_pot']:.4f} n) LSB")
    print(f"  DEP: step(n) = {m['A_dep']:.2f} * exp(-{m['k_dep']:.4f} n) LSB")
    print(f"  half-select  pot {m['hs_pot']:+.2f}  reset {m['hs_reset']:+.2f}"
          f"  dep {m['hs_dep']:+.2f} LSB/pair")
    print(f"  residual sd {m['resid_sd']:.2f} LSB")
    print(f"  per-cell first-pulse step: POT "
          f"{gp_cells.mean():.2f}+-{gp_cells.std():.2f}, DEP "
          f"{gd_cells.mean():.2f}+-{gd_cells.std():.2f} LSB")
    print(f"  rails {LSB_MIN:+.0f} .. {LSB_MAX:+.0f} LSB")

    print("\n" + "=" * 72)
    print("mapped to aihwkit LinearStepDevice")
    print("=" * 72)
    print(f"  scale: 1 normalised weight unit = {lsb_per_w:.1f} LSB")
    print(f"  w_max {W_MAX:+.3f}   w_min {w_min:+.3f}")
    print(f"  reset state w0 = {w0:+.4f}")
    print()
    print(f"  dw_min       {dw_min:.6f}   "
          f"(= {dw_min*lsb_per_w:.2f} LSB, mean first-pulse step)")
    print(f"  up_down      {up_down:+.4f}   (P/D asymmetry)")
    print(f"  gamma_up     {gamma_up:.4f}")
    print(f"  gamma_down   {gamma_down:.4f}")
    print(f"  dw_min_dtod  {dtod:.4f}   (cell-to-cell step spread)")
    print(f"  dw_min_std   {ctoc:.4f}   (cycle-to-cycle, per coincidence)")
    print(f"  up_down_dtod {ud_dtod:.4f}")

    cfg = dict(
        construction_seed=0,
        dw_min=round(float(dw_min), 6),
        dw_min_dtod=round(float(dtod), 4),
        dw_min_std=round(float(ctoc), 4),
        up_down=round(float(up_down), 4),
        up_down_dtod=round(float(ud_dtod), 4),
        w_max=float(W_MAX), w_min=float(w_min),
        w_max_dtod=0.0, w_min_dtod=0.0,
        gamma_up=round(float(gamma_up), 4),
        gamma_down=round(float(gamma_down), 4),
        gamma_up_dtod=0.0, gamma_down_dtod=0.0,
        allow_increasing=False, mean_bound_reference=True,
        mult_noise=False,
    )
    print("\n" + "=" * 72)
    print("config snippet")
    print("=" * 72)
    print("from aihwkit.simulator.configs.devices import LinearStepDevice")
    print("device = LinearStepDevice(")
    for k, v in cfg.items():
        print(f"    {k}={v!r},")
    print(")")

    with open("aihwkit_linearstep_params.json", "w", encoding="utf-8") as f:
        json.dump({"lsb_per_w": lsb_per_w, "w0_reset": w0,
                   "measured": {k: float(m[k]) for k in
                                ("A_pot", "k_pot", "A_dep", "k_dep",
                                 "hs_pot", "hs_reset", "hs_dep",
                                 "resid_sd")},
                   "linearstep": cfg}, f, indent=2)
    print("\nsaved -> aihwkit_linearstep_params.json")

    # --- verify the mapping by replaying the recursion in LSB -------------
    print("\n" + "=" * 72)
    print("verification: LinearStep recursion vs measured burst curve")
    print("=" * 72)
    print(f"{'C':>3s} {'measured':>10s} {'LinearStep':>11s} {'err':>7s}   "
          f"| {'measured':>10s} {'LinearStep':>11s} {'err':>7s}")
    print(f"{'':>3s} {'POT (LSB)':>10s} {'POT (LSB)':>11s} {'':>7s}   "
          f"| {'DEP (LSB)':>10s} {'DEP (LSB)':>11s} {'':>7s}")
    print("-" * 72)
    # replay aihwkit's own recursion from the fitted config -- if the mapping
    # is right this reproduces the measured burst curve without ever using
    # A or k directly
    s_up = (1.0 + min(up_down, 0.0)) * dw_min
    s_dn = (1.0 - max(up_down, 0.0)) * dw_min
    sl_up = -gamma_up * s_up / W_MAX          # < 0
    sl_dn = -gamma_down * s_dn / w_min        # > 0 (w_min < 0)

    def replay(C, positive):
        w = w0
        for _ in range(C):
            if positive:
                w = min(w + sl_up * w + s_up, W_MAX)
            else:
                w = max(w - (sl_dn * w + s_dn), w_min)
        return (w - w0) * lsb_per_w

    d = m["data"]
    errs = []
    for C in range(1, 11):
        row = []
        for key, other, pos in (("C_pot", "C_dep", True),
                                ("C_dep", "C_pot", False)):
            s = (d[key] == C) & (d[other] == 0)
            meas = d["delta"][s].mean() if s.sum() >= 8 else np.nan
            row.append((meas, replay(C, pos)))
            if not np.isnan(meas):
                errs.append(meas - row[-1][1])
        (mp, pp), (md, pd) = row
        f1 = (f"{mp:10.1f} {pp:11.1f} {mp-pp:+7.1f}" if not np.isnan(mp)
              else f"{'-':>10s} {pp:11.1f} {'-':>7s}")
        f2 = (f"{md:10.1f} {pd:11.1f} {md-pd:+7.1f}" if not np.isnan(md)
              else f"{'-':>10s} {pd:11.1f} {'-':>7s}")
        print(f"{C:3d} {f1}   | {f2}")
    print(f"\nreplay of the fitted config vs measurement: "
          f"rms {np.sqrt(np.mean(np.square(errs))):.2f} LSB over "
          f"{len(errs)} points")
    print(f"round-trip first step: POT {(sl_up*w0+s_up)*lsb_per_w:.2f} LSB "
          f"(measured {m['A_pot']:.2f}), DEP "
          f"{(sl_dn*w0+s_dn)*lsb_per_w:.2f} LSB (measured {m['A_dep']:.2f})")

    # --- the extrapolation problem, stated explicitly ---------------------
    w_fix_up = -s_up / sl_up
    w_fix_dn = s_dn / sl_dn
    print("\n" + "=" * 72)
    print("CAUTION: what this fit does NOT establish")
    print("=" * 72)
    print(f"  The step vanishes at w = {w_fix_up:+.3f} "
          f"({w_fix_up*lsb_per_w:+.0f} LSB) going up and "
          f"{-w_fix_dn:+.3f} ({-w_fix_dn*lsb_per_w:+.0f} LSB) going down,")
    print(f"  i.e. at {100*w_fix_up/W_MAX:.0f}% / "
          f"{100*w_fix_dn/abs(w_min):.0f}% of the bounds.  The device would "
          f"stall there and never")
    print(f"  reach the measured rails ({LSB_MIN:+.0f}/{LSB_MAX:+.0f} LSB).")
    print()
    print("  That is an EXTRAPOLATION artefact, not a measurement.  The decay")
    print("  rate k was fitted over bursts of <=10 pulses that move the cell")
    print("  ~16% of its range, all starting from a hard reset; the data has")
    print("  no leverage on how the step behaves near the rails.  gamma > 1")
    print("  simply projects that short-range decay onto the whole range.")
    print()
    print("  Two usable options:")
    print(f"    (a) as fitted    - matches the measured burst curve, but")
    print(f"                       saturates early. Use for short bursts near")
    print(f"                       the reset state, which is the e-prop case.")
    print(f"    (b) gamma = 1.0  - 'soft bounds': the step vanishes exactly at")
    print(f"                       the rails. Use for full-range sweeps.")
    print(f"                       First-pulse step becomes "
          f"{(-1.0*s_up/W_MAX*w0+s_up)*lsb_per_w:.1f} LSB (POT).")
    print()
    print("  Resolving this properly needs the state-dependence sweep noted in")
    print("  fit_update_model.py: set the starting level on purpose, then")
    print("  measure the step at each level.")

    # --- optional: run the real device if aihwkit is available ------------
    print()
    try:
        from aihwkit.simulator.configs.devices import LinearStepDevice
        from aihwkit.simulator.tiles import AnalogTile
        from aihwkit.simulator.configs import SingleRPUConfig
        import torch
        from aihwkit.simulator.parameters.enums import PulseType
        # deterministic single-coincidence drive: zero the noise terms and set
        # lr = dw_min so ONE update fires ONE pulse (default lr 0.1 fires
        # lr*x*d/dw_min ~ 3 pulses per update and triples every step)
        cfg_det = dict(cfg, dw_min_dtod=0.0, dw_min_std=0.0, up_down_dtod=0.0)
        rpu = SingleRPUConfig(device=LinearStepDevice(**cfg_det))
        rpu.update.desired_bl = 1
        rpu.update.update_bl_management = False
        rpu.update.update_management = False
        rpu.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
        tile = AnalogTile(1, 1, rpu_config=rpu)
        tile.set_learning_rate(dw_min)
        print("aihwkit present -- simulating bursts (deterministic, 1 pulse "
              "per update)")
        print(f"{'C':>3s} {'closed POT':>11s} {'tile POT':>9s} "
              f"{'closed DEP':>11s} {'tile DEP':>9s}")
        for C in (1, 3, 5, 10):
            row = []
            for sgn, A, k in ((+1, m["A_pot"], m["k_pot"]),
                              (-1, m["A_dep"], m["k_dep"])):
                tile.set_weights(torch.tensor([[w0]]))
                for _ in range(C):
                    tile.update(torch.ones(1, 1), -sgn * torch.ones(1, 1))
                w_after = float(tile.get_weights()[0][0, 0])
                closed = sgn * A * (1 - np.exp(-k * C)) / k
                row.extend([closed, (w_after - w0) * lsb_per_w])
            print(f"{C:3d} {row[0]:11.1f} {row[1]:9.1f} "
                  f"{row[2]:11.1f} {row[3]:9.1f}")
    except ImportError:
        print("aihwkit not importable here (no Windows wheel; it is a")
        print("CMake source build for Linux/macOS).  The parameters above are")
        print("closed-form and can be pasted into any environment that has it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
9x9 (u, v) grid sweep: how far does the hardware update drift from the
update the algorithm actually asked for, as a function of drive level?

Extends the single-shot correlation cell in quick_measure.ipynb (which
measured one random (u, v) pair and reported Pearson r = 0.9001) into a
systematic sweep over the drive plane.

WHAT IS MEASURED
----------------
A stochastic-pulse update passes through two lossy stages:

    u (x) v * L      what the algorithm wants (ideal outer product)
        |  <-- stochastic sampling: Bernoulli draws of finite length L
        v
    C[i,j]           what the device is actually told to do
        |  <-- device physics: nonlinearity, saturation, drift, crosstalk
        v
    delta            what the conductance actually did

PRIMARY METRIC: r = Pearson(delta, C), per grid point, pooled over repeats.

If the device follows its pulses exactly, r = 1 regardless of how noisy the
pulse generation was -- the sampling noise is already baked into C. So r
isolates the device stage cleanly, and it needs no unit conversion between
ADC LSB and coincidence counts (Pearson is scale-invariant). Verified: a
perfect simulated device returns r = 1.000 at every drive level.

The sampling stage is then reported separately as a ratio-of-means, which is
well defined even where correlation is not:

    C_over_ideal = mean(C) / mean(ideal)     1.0 = pulses match the request

With scalar (u, v) the ideal matrix is constant across all 25 cells, so its
variance is zero and Pearson(delta, ideal) is mathematically undefined -- it
is recorded but will be nan; do not read anything into it. The sampling
error magnitude is instead predicted analytically (err_sampling_theory),
since it is pure statistics and needs no hardware.

Gain (LSB per coincidence) is still fitted and plotted, because it is the
physical conversion factor and its collapse marks saturation, but no error
metric depends on it any more.

Usage:
    python uv_grid_sweep.py --port COM4                    # BL 10,20,40
    python uv_grid_sweep.py --port COM4 --bit-lengths 10   # single BL
    python uv_grid_sweep.py --dry-run                      # no hardware
    python uv_grid_sweep.py --plot-only 2026-08-05_*.csv   # re-plot
"""

import argparse
import csv
import datetime
import glob
import os
import re
import sys
import time

import numpy as np

N = 5
LEVELS = np.round(np.arange(0.1, 0.95, 0.1), 2)  # 0.1 .. 0.9; --levels overrides

# Timing copied from the quick_measure.ipynb correlation cell so results are
# directly comparable with the recorded r = 0.9001 run.
PULSE_TIMING = {"width": 15, "pre": 100, "post": 100, "zero": 10}
READ_TIMING = {"time": 20, "delay": 10}
READ_FUNCTION = "N56"
# update_function sent to the firmware. "dno" uses the disturb-null variant:
# per slot, rows whose N1 bit is 0 get N3 raised instead (active shield), so
# unselected cells see no net voltage across the cap when a column fires.
# Parsing note: firmware routes streams by
# `is_potentiation = (update_function != "STOCHASTIC_DEPRESSION")`, so
# STOCHASTIC_DNO_POTENTIATION correctly lands in N1/N2 (but the DNO
# *DEPRESSION* opcode would mis-parse into N1/N2 too — do not use it
# without fixing the firmware condition).
MODES = {"normal": "STOCHASTIC_POTENTIATION",
         "dno": "STOCHASTIC_DNO_POTENTIATION"}
MODE = MODES["normal"]


# ---------------------------------------------------------------- pulses ---

def generate_pulse_streams(prob_vector, bit_length, rng):
    return ["".join(map(str, (rng.random(bit_length) < p).astype(int)))
            for p in prob_vector]


def coincidence_matrix(n1_streams, n2_streams):
    """C[i,j] = number of timesteps where row i and column j are both 1.

    This is the notebook's `update_count_matrix`: the number of times cell
    (i,j) was actually commanded to update.
    """
    C = np.zeros((N, N), dtype=int)
    arr1 = [np.array(list(s), dtype=int) for s in n1_streams]
    arr2 = [np.array(list(s), dtype=int) for s in n2_streams]
    for i in range(N):
        for j in range(N):
            C[i, j] = int((arr1[i] * arr2[j]).sum())
    return C


def analytic_sampling_error(u, v, bit_length, n_mc=4000, rng=None):
    """E|C - ideal| / mean(ideal) for a perfect device -- no hardware needed.

    Closed form is awkward (product of two binomials), so this is a fast
    Monte-Carlo estimate of the reference surface.
    """
    rng = rng or np.random.default_rng(0)
    p, q = float(u[0]), float(v[0])          # scalar drive
    ideal = p * q * bit_length
    if ideal <= 0:
        return float("nan")
    # C for one cell = sum over L of Bernoulli(p)*Bernoulli(q) = Binom(L, p*q)
    C = rng.binomial(bit_length, p * q, size=n_mc)
    return float(np.abs(C - ideal).mean() / ideal)


# ------------------------------------------------------------------- I/O ---

def build_command(n1, n2, bit_length):
    parts = [
        "F", "1", "1", MODE, READ_FUNCTION,
        str(bit_length), str(PULSE_TIMING["width"]), str(PULSE_TIMING["pre"]),
        str(PULSE_TIMING["post"]), str(PULSE_TIMING["zero"]),
        str(READ_TIMING["time"]), str(READ_TIMING["delay"]),
    ] + n1 + n2
    return ",".join(parts)


RESET_UPDATE_NUM = 30
RESET_READ_PERIOD = 10


def hard_reset(arduino, set_num=10, silence_timeout=1.5):
    """Drive all 25 cells back toward baseline conductance.

    Needed because this sweep sends POTENTIATION only: without it the array
    walks into saturation, and late grid points (which under shuffling are
    arbitrary drive levels) would be measured on an already-pinned device.
    Verified on hardware: 10x full potentiation pushed the mean N5-N6 from
    38 to 342 LSB, and one Reset brought it back to 33.

    Firmware behaviour (6T1C_5x5_add_stochastic.ino, Reset branch at L717 and
    Reset_update() at L1207):
      * The array runs two interleaved paths:
            Potentiation_6T : N1 (row) + N2 (col)  -> charge the storage cap
            Depression_6T   : N3 (row) + N4 (col)  -> discharge it
        Both are coincidence operations: only the cell where the selected row
        and column meet is updated.
      * Reset_update asserts N1 and N3 -- the two ROW lines -- and no column
        line at all. Turning both on ties the two sides of the storage cap
        together, collapsing the voltage across it. So this is not a strong
        directional write; it is a capacitor equalization/discharge, which is
        why it needs no column select and why it drives every cell in the
        selected rows to the same state.
      * That also explains the timing: P/D deliberately overlap their two
        lines for only `pulse_width` so the transferred charge is metered,
        whereas Reset holds N1 and N3 overlapped for pre+width+post -- when
        the goal is to fully drain the cap, longer overlap is simply better.
      * row_num=5 selects all five rows at once, so all 25 cells discharge
        together.
      * Measured: 10x full potentiation drove mean N5-N6 to 342 LSB; one
        Reset brought it to 33, i.e. collapse toward zero rather than an
        overshoot into depression -- consistent with equalization.
      * It emits '<idx>,<read>>' blocks but NO EOD marker.

    Block count: 1 startup read, then one per (k+1) % read_period == 0 within
    each set, i.e. update_num/read_period per set:

        1 + set_num * (update_num // read_period)

    With the parameters below that is 1 + 10*3 = 31, confirmed by counting
    lines off the wire. An earlier version waited for only set_num+1 = 11 and
    left 20 blocks sitting in the serial buffer, which then corrupted the
    parse of the following measurement command.
    """
    cmd = ",".join(["F", "5", "5", "Reset", "N56", str(set_num),
                    str(RESET_UPDATE_NUM), str(RESET_READ_PERIOD),
                    "F", "5", "100", "100", "10",
                    "10", "20", "10"]) + "\n"
    arduino.reset_input_buffer()
    arduino.write(cmd.encode("utf-8"))

    reads_per_set = max(1, RESET_UPDATE_NUM // RESET_READ_PERIOD)
    expected, seen = 1 + set_num * reads_per_set, 0
    last = time.time()
    while True:
        raw = arduino.readline()
        if raw:
            line = raw.decode("utf-8", "ignore").strip()
            if line and ">" in line:
                seen += 1
                last = time.time()
                if seen >= expected:
                    return True
            continue
        if time.time() - last > silence_timeout:
            # Drain anything still arriving so it cannot bleed into the next
            # command's response.
            arduino.reset_input_buffer()
            return seen > 0


def send_and_read(arduino, command, timeout_s=30.0):
    arduino.reset_input_buffer()
    arduino.write((command + "\n").encode("utf-8"))
    rows, deadline = [], time.time() + timeout_s
    while time.time() < deadline:
        line = arduino.readline().decode("utf-8", "ignore").strip()
        if not line:
            continue
        if "EOD" in line:
            break
        if "Row" in line:
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                rows.append(nums[1:11])
    if len(rows) < 2 * N:
        raise RuntimeError(f"expected {2*N} rows, got {len(rows)}")
    a = np.array(rows[:N], float)
    b = np.array(rows[N:2 * N], float)
    return a[:, :N] - a[:, N:], b[:, :N] - b[:, N:]   # pre_diff, post_diff


# --------------------------------------------------------------- metrics ---

def safe_pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_err(a, b):
    """mean|a-b| normalized by mean|b|."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(b).mean()
    return float(np.abs(a - b).mean() / d) if d > 1e-12 else float("nan")


# ------------------------------------------------------------------ main ---

def run_grid(args, bit_length, arduino, rng, cell_rows=None):
    """Sweep the grid at one bit_length.

    Returns the per-grid-point summary rows. If `cell_rows` is given, raw
    per-cell (C, delta) observations are appended to it in long format --
    one row per (grid point, repeat, cell). Aggregates alone cannot be
    re-analysed for a cell SUBSET, which is why the raw values are kept.
    """
    sim_state = np.zeros((N, N))

    def one_command(n1, n2):
        if args.dry_run:
            C = coincidence_matrix(n1, n2)
            pre = sim_state.copy()
            # toy device: saturating potentiation + constant decay
            sim_state[:] = pre + 11.0 * C * (1 - np.abs(pre) / 190.0) - 1.5
            post = sim_state.copy()
            n = lambda: rng.normal(0, 2.5, (N, N))
            return pre + n(), post + n()
        return send_and_read(arduino, build_command(n1, n2, bit_length))

    zeros = ["0" * bit_length] * N
    rows = []

    # Per-cell state right after a Reset. The cap is equalized, so any
    # residual spread here is read-path offset (integrator / ADC per-cell
    # mismatch), not stored charge -- measured spread was -29..+98 LSB. The
    # sweep's metrics use post-pre differences so it cancels, but recording
    # it lets absolute levels be interpreted later.
    baseline = {"matrix": None}

    def maybe_reset(tag):
        if args.reset_every <= 0 and not args.reset_per_repeat:
            return
        if args.dry_run:
            sim_state[:] = 0.0            # simulate return to baseline
            if args.baseline:
                baseline["matrix"] = sim_state.copy()
            return
        ok = hard_reset(arduino, set_num=args.reset_sets)
        if not ok:
            print(f"    [WARN] hard reset did not confirm at {tag}")
        time.sleep(args.settle)
        if args.baseline and not args.reset_per_repeat:
            # zero-pulse read: no coincidences, so this is a pure state read.
            # Skipped in per-repeat mode -- the drift control that immediately
            # follows each reset already reads the post-reset state
            # (pre_levels), so an extra read here would only add time.
            pre_b, _ = one_command(zeros, zeros)
            baseline["matrix"] = pre_b
            time.sleep(args.settle)

    # Visit grid points in a fixed but shuffled order. With sequential order
    # the loop runs low->high drive monotonically, so any residual cumulative
    # drift would alias perfectly onto the drive axis and masquerade as a
    # drive-dependent effect. Shuffling decorrelates the two; the seed keeps
    # it reproducible.
    points = [(gi, um, gj, vm)
              for gi, um in enumerate(LEVELS)
              for gj, vm in enumerate(LEVELS)]
    if args.only:
        # snap the requested (u, v) to the nearest grid levels so the row
        # still carries valid gi/gj indices for plotting
        tu, tv = args.only
        gi = int(np.argmin(np.abs(LEVELS - tu)))
        gj = int(np.argmin(np.abs(LEVELS - tv)))
        points = [(gi, LEVELS[gi], gj, LEVELS[gj])]
    elif args.shuffle:
        np.random.default_rng(args.seed).shuffle(points)
    total = len(points)

    for k, (gi, um, gj, vm) in enumerate(points, 1):
        u = np.full(N, float(um))
        v = np.full(N, float(vm))
        ideal = np.outer(u, v) * bit_length

        if (not args.reset_per_repeat
                and args.reset_every > 0 and (k - 1) % args.reset_every == 0):
            maybe_reset(f"point {k}")

        Cs, Ds = [], []
        drift_log = []
        pre_levels = []
        for rep in range(args.repeats):
            # Reset before EVERY trial so each repeat starts from the same
            # equalized state. Without this, repeats accumulate and the
            # update's state dependence (step ~ 0.016*C*(477 - V), measured
            # 2026-08-05) corrupts the delta-vs-C relation at high drive --
            # the first sweep's r collapse at high C was largely this
            # accumulation artifact, not per-command physics.
            if args.reset_per_repeat:
                maybe_reset(f"point {k} rep {rep}")
            # Zero-pulse control: no coincidences, so whatever moves is decay
            # / read-disturb from the command itself. Measured every repeat
            # and RECORDED, but by default NOT subtracted.
            #
            # Measured on this rig (2026-08-05): drift is about -1.4 LSB
            # against a ~17 LSB signal, and the control read carries its own
            # noise, so subtracting it injected more noise than it removed --
            # r fell from +0.859 to +0.755. Logging it keeps the option of
            # correcting in post-processing if a future rig drifts more.
            p0, q0 = one_command(zeros, zeros)
            drift = q0 - p0
            drift_log.append(drift)
            # absolute state right before the probe command (post-reset in
            # per-repeat mode) -- lets state be regressed out in analysis
            pre_levels.append(p0)
            time.sleep(args.settle)

            n1 = generate_pulse_streams(u, bit_length, rng)
            n2 = generate_pulse_streams(v, bit_length, rng)
            C = coincidence_matrix(n1, n2)
            pre, post = one_command(n1, n2)
            d_rep = post - pre - (drift if args.drift_correct else 0.0)
            Cs.append(C)
            Ds.append(d_rep)
            if cell_rows is not None:
                for ci in range(N):
                    for cj in range(N):
                        cell_rows.append(dict(
                            mode=args.mode, bit_length=bit_length,
                            u=float(um), v=float(vm), gi=gi, gj=gj, rep=rep,
                            cell_row=ci + 1, cell_col=cj + 1,
                            C=int(C[ci, cj]), delta=float(d_rep[ci, cj]),
                            pre=float(pre[ci, cj]), post=float(post[ci, cj]),
                            drift=float(drift[ci, cj]),
                        ))
            time.sleep(args.settle)

        C_all = np.stack(Cs).astype(float)
        D_all = np.stack(Ds)
        Cf, Df = C_all.ravel(), D_all.ravel()

        # PRIMARY METRIC: does the conductance change track the pulses
        # it actually received? Scale-invariant, so no LSB<->coincidence
        # conversion is needed and a perfect device scores exactly 1.
        r_delta_C = safe_pearson(Df, Cf)

        # Per-repeat r, to separate "the device is noisy" from "this one
        # grid point happened to get an unlucky pulse draw".
        r_per_rep = [safe_pearson(d.ravel(), c.ravel())
                     for d, c in zip(D_all, C_all)]
        r_rep_valid = [x for x in r_per_rep if np.isfinite(x)]

        # Local gain (LSB per coincidence). Diagnostic only -- nothing
        # depends on it now. Within one grid point C barely varies under
        # scalar drive, so this slope is noisy; its collapse toward zero
        # is still a useful saturation flag.
        if Cf.std() > 1e-6:
            gain = float(np.polyfit(Cf, Df, 1)[0])
        else:
            gain = float(Df.mean() / Cf.mean()) if Cf.mean() > 1e-9 else float("nan")

        # Sampling stage as a ratio of means: 1.0 means the realized
        # pulses delivered exactly the requested drive on average.
        # Well defined even where ideal has zero variance.
        C_over_ideal = (float(C_all.mean() / ideal.mean())
                        if ideal.mean() > 1e-12 else float("nan"))

        mask = C_all > 0
        crosstalk = (float(np.abs(D_all[~mask]).mean() /
                           (np.abs(D_all[mask]).mean() + 1e-9))
                     if (~mask).any() and mask.any() else float("nan"))

        rows.append(dict(
            mode=args.mode,
            bit_length=bit_length, u=float(um), v=float(vm),
            gi=gi, gj=gj,
            r_delta_C=r_delta_C,
            r_rep_mean=float(np.mean(r_rep_valid)) if r_rep_valid else float("nan"),
            r_rep_std=float(np.std(r_rep_valid)) if r_rep_valid else float("nan"),
            # undefined for scalar drive (ideal is constant); kept only
            # so the column exists if spread vectors are used later
            r_delta_ideal=safe_pearson(Df, np.tile(ideal.ravel(), args.repeats)),
            C_over_ideal=C_over_ideal,
            err_sampling_theory=analytic_sampling_error(u, v, bit_length, rng=rng),
            gain=gain,
            C_mean=float(C_all.mean()), C_std=float(C_all.std()),
            delta_mean=float(D_all.mean()), delta_std=float(D_all.std()),
            crosstalk=crosstalk,
            ideal_mean=float(ideal.mean()),
            # recorded so drift can be removed in post-processing; NOT
            # subtracted from delta unless --drift-correct is passed
            drift_mean=float(np.mean(drift_log)),
            drift_std=float(np.std(drift_log)),
            drift_absmax=float(np.abs(np.stack(drift_log)).max()),
            drift_corrected=int(bool(args.drift_correct)),
            # absolute state before each probe command (mean over cells);
            # in per-repeat mode these should all sit at the post-reset
            # level -- their spread verifies the reset actually landed
            pre_level_mean=float(np.mean([p.mean() for p in pre_levels])),
            pre_level_std=float(np.std([p.mean() for p in pre_levels])),
            **{f"pre_rep{i}": float(p.mean())
               for i, p in enumerate(pre_levels)},
            # post-Reset per-cell state (read-path offset, see maybe_reset)
            baseline_mean=(float(baseline["matrix"].mean())
                           if baseline["matrix"] is not None else float("nan")),
            baseline_std=(float(baseline["matrix"].std())
                          if baseline["matrix"] is not None else float("nan")),
            **({f"base{i}": float(baseline["matrix"].ravel()[i])
                for i in range(N * N)} if baseline["matrix"] is not None else {}),
        ))

        print(f"  BL={bit_length:3d} [{k:3d}/{total}] u={um:.1f} v={vm:.1f} "
              f"C={C_all.mean():5.2f} d={D_all.mean():+7.1f} "
              f"r={r_delta_C:+.3f} gain={gain:+6.2f} "
              f"C/ideal={C_over_ideal:4.2f} xtalk={crosstalk:5.2f}")

        # Single-point smoke test: dump the raw matrices so the protocol and
        # the parsing can be eyeballed before committing to a long sweep.
        if args.only:
            np.set_printoptions(precision=1, suppress=True, linewidth=120)
            for ri in range(len(Cs)):
                print(f"\n    --- repeat {ri} ---")
                print(f"    coincidence C (what the device was told):\n{Cs[ri]}")
                print(f"    delta, drift-corrected (what it did):\n"
                      f"{np.round(Ds[ri], 1)}")
                print(f"    per-repeat r = "
                      f"{safe_pearson(Ds[ri].ravel(), Cs[ri].ravel()):+.4f}")
            print(f"\n    pooled r over {len(Cs)} repeats = {r_delta_C:+.4f}")
            print(f"    ideal (u*v*L) = {ideal.mean():.2f} coincidences/cell")
    return rows


def save_csv(rows, path):
    # union of keys, first-seen order -- baseline columns are absent when
    # --no-baseline is used, and absent from the first row if a reset was
    # skipped, so keying off rows[0] alone would drop or crash on them
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rows = [{k: r.get(k, "") for k in keys} for r in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path}")


def plot(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bls = sorted({r["bit_length"] for r in rows})
    panels = [
        # primary result: 1.0 = device followed its pulses exactly
        ("r_delta_C", "Pearson r (delta vs C)   PRIMARY", "RdYlGn", (0, 1)),
        ("r_rep_std", "r spread across repeats (lower = stabler)", "magma_r", None),
        ("C_over_ideal", "mean(C) / mean(ideal)   1.0 = pulses on target",
         "coolwarm", (0, 2)),
        ("C_mean", "Mean coincidences C  (drive actually delivered)",
         "viridis", None),
        ("gain", "Gain (LSB per coincidence)  diagnostic", "viridis", None),
        ("crosstalk", "Crosstalk  |delta| off-drive / on-drive", "magma_r", None),
        ("drift_mean", "Drift, zero-pulse control (LSB, logged not removed)",
         "coolwarm", None),
    ]

    # 3 panels per row keeps each heatmap large enough to read the cell
    # annotations; one block of rows per bit_length.
    per_row = 3
    blocks = int(np.ceil(len(panels) / per_row))
    nrow, ncol = len(bls) * blocks, per_row
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow),
                             squeeze=False)

    for bi, bl in enumerate(bls):
        sub = [r for r in rows if r["bit_length"] == bl]

        for pi, (key, title, cmap, vlim) in enumerate(panels):
            ax = axes[bi * blocks + pi // per_row][pi % per_row]
            M = np.full((len(LEVELS), len(LEVELS)), np.nan)
            for r in sub:
                M[r["gi"], r["gj"]] = r[key]

            if vlim:
                vmin, vmax = vlim
            elif key == "drift_mean":
                # signed quantity: center the diverging map on zero
                fin = M[np.isfinite(M)]
                lim = float(np.percentile(np.abs(fin), 95)) if fin.size else 1.0
                vmin, vmax = -max(lim, 1e-6), max(lim, 1e-6)
            else:
                # robust range; a single outlier grid point must not flatten
                # the rest of the plane into one color
                fin = M[np.isfinite(M)]
                vmin, vmax = ((0, float(np.percentile(fin, 95)))
                              if fin.size else (None, None))

            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[0.05, 0.95, 0.05, 0.95], aspect="equal")
            ax.set_title(f"BL={bl}   {title}", fontsize=10)
            ax.set_xlabel("v  (column drive)")
            ax.set_ylabel("u  (row drive)")
            ax.set_xticks(LEVELS)
            ax.set_yticks(LEVELS)
            ax.tick_params(labelsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)

            for a in range(len(LEVELS)):
                for b in range(len(LEVELS)):
                    val = M[a, b]
                    if not np.isfinite(val):
                        ax.text(LEVELS[b], LEVELS[a], "-", ha="center",
                                va="center", fontsize=7, color="0.5")
                        continue
                    ax.text(LEVELS[b], LEVELS[a], f"{val:.2f}", ha="center",
                            va="center", fontsize=7, color="black")

        # hide unused axes in the last block row
        for pi in range(len(panels), blocks * per_row):
            axes[bi * blocks + pi // per_row][pi % per_row].axis("off")

    fig.suptitle("Stochastic update fidelity across the (u, v) drive plane "
                 "— POTENTIATION\n"
                 "primary metric: Pearson r(delta, C) — 1.0 means the device "
                 "followed its pulses exactly",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    print(f"saved -> {out_png}")


def summarize(rows):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for bl in sorted({r["bit_length"] for r in rows}):
        sub = [r for r in rows if r["bit_length"] == bl]
        f = lambda k: np.array([r[k] for r in sub], float)
        r_all, cmean = f("r_delta_C"), f("C_mean")
        fin = np.isfinite(r_all)

        print(f"\nBL={bl}")
        print(f"  r(delta,C) over all {len(sub)} grid points: "
              f"median {np.nanmedian(r_all):+.3f}  "
              f"mean {np.nanmean(r_all):+.3f}  max {np.nanmax(r_all):+.3f}")
        print(f"  points reaching r>=0.9: {(r_all[fin] >= 0.9).sum()}/{fin.sum()}"
              f"   r>=0.7: {(r_all[fin] >= 0.7).sum()}/{fin.sum()}")
        if (~fin).any():
            print(f"  undefined r (no variance in C or delta): {(~fin).sum()}")

        # Low-drive points are included in the totals above, but split out
        # here too: where C is almost always 0 or 1 the device is barely being
        # programmed, so a low r there reflects lack of drive rather than a
        # device defect. Both readings are useful, so both are shown.
        lo = cmean < 1.0
        if lo.any() and (~lo).any():
            print(f"  split by drive: C_mean<1  median r={np.nanmedian(r_all[lo]):+.3f} "
                  f"(n={lo.sum()})   |   C_mean>=1  "
                  f"median r={np.nanmedian(r_all[~lo]):+.3f} (n={(~lo).sum()})")

        good = [r for r in sub if np.isfinite(r["r_delta_C"])]
        if good:
            best = max(good, key=lambda r: r["r_delta_C"])
            print(f"  best operating point: u={best['u']:.1f} v={best['v']:.1f} "
                  f"-> r={best['r_delta_C']:+.3f}, C_mean={best['C_mean']:.1f}, "
                  f"gain={best['gain']:+.2f}")

    bls = sorted({r["bit_length"] for r in rows})
    if len(bls) > 1:
        print("\n  bit_length trend (all grid points):")
        for bl in bls:
            v = np.array([r["r_delta_C"] for r in rows
                          if r["bit_length"] == bl], float)
            if np.isfinite(v).any():
                print(f"    BL={bl:3d}: median r = {np.nanmedian(v):+.3f}")
        print("  If r rises with bit_length the limit is pulse statistics;")
        print("  if it stays flat the device itself is the bottleneck.")

    print("\nr = 1.0 would mean the conductance change tracked the pulses it")
    print("actually received. Sampling noise is already inside C, so it cannot")
    print("lower r -- whatever shortfall you see is the device.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--bit-lengths", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.15)
    ap.add_argument("--reset-every", type=int, default=1,
                    help="hard-reset the array every N grid points "
                         "(1 = before every point, 0 = never). This sweep is "
                         "POTENTIATION-only, so without resets the device "
                         "walks into saturation.")
    ap.add_argument("--reset-sets", type=int, default=10,
                    help="reset cycles per hard reset (6th command field)")
    ap.add_argument("--reset-per-repeat", action="store_true", default=True,
                    help="hard-reset before EVERY trial so each repeat "
                         "starts from the equalized state; removes the "
                         "accumulation confound that corrupted the first "
                         "sweep's high-drive points (default on)")
    ap.add_argument("--no-reset-per-repeat", dest="reset_per_repeat",
                    action="store_false")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="visit grid points in sequential order instead of a "
                         "seeded shuffle; sequential order aliases any "
                         "residual drift onto the drive axis")
    ap.add_argument("--baseline", action="store_true", default=True,
                    help="after each Reset, read and record the per-cell "
                         "state as a baseline (base0..base24 columns). The "
                         "cap is equalized at that point, so residual spread "
                         "is read-path offset rather than stored charge.")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="drive levels for the u and v axes (default 0.1..0.9 "
                         "in steps of 0.1 = 9x9). e.g. --levels 0.3 0.5 0.7 "
                         "gives a 3x3 grid, ~1/9 the runtime.")
    ap.add_argument("--per-cell", action="store_true", default=True,
                    help="also write raw per-cell (C, delta) values to a "
                         "long-format CSV so any cell subset can be "
                         "re-analysed later (default on)")
    ap.add_argument("--no-per-cell", dest="per_cell", action="store_false")
    ap.add_argument("--mode", choices=list(MODES), default="normal",
                    help="update opcode: 'normal' = STOCHASTIC_POTENTIATION, "
                         "'dno' = STOCHASTIC_DNO_POTENTIATION (half-select "
                         "shield via N3 on unselected rows)")
    ap.add_argument("--drift-correct", action="store_true",
                    help="subtract the zero-pulse control from delta. OFF by "
                         "default: measured drift on this rig is ~-1.4 LSB "
                         "against a ~17 LSB signal, and the extra control "
                         "read adds more noise than it removes (r fell "
                         "0.859 -> 0.755). Drift is always logged either way.")
    ap.add_argument("--only", nargs=2, type=float, metavar=("U", "V"),
                    default=None,
                    help="measure a single (u, v) point instead of the full "
                         "grid -- use for a smoke test before committing to "
                         "the ~25 min sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", nargs="+", default=None,
                    help="CSV file(s) to re-plot without measuring")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.plot_only:
        rows = []
        for pat in args.plot_only:
            for path in glob.glob(pat):
                with open(path, encoding="utf-8") as f:
                    for d in csv.DictReader(f):
                        rows.append({k: (float(v) if k not in ("gi", "gj", "bit_length")
                                         else int(float(v)))
                                     for k, v in d.items()})
        if not rows:
            print("no rows found")
            return 1
        plot(rows, f"{stamp}_uv_grid.png")
        summarize(rows)
        return 0

    global MODE, LEVELS
    MODE = MODES[args.mode]
    if args.levels:
        LEVELS = np.round(np.array(args.levels, dtype=float), 3)
    tag = "" if args.mode == "normal" else f"_{args.mode}"

    rng = np.random.default_rng(args.seed)
    n_cmd = len(args.bit_lengths) * len(LEVELS) ** 2 * args.repeats * 2
    print(f"grid {len(LEVELS)}x{len(LEVELS)} | BL {args.bit_lengths} | "
          f"repeats {args.repeats} | mode {MODE}")
    print(f"{n_cmd} commands total\n")

    arduino = None
    if not args.dry_run:
        import serial
        arduino = serial.Serial(args.port, args.baud, timeout=20)
        time.sleep(2)
        print(f"connected: {arduino.name}\n")

    all_rows = []
    cell_rows = [] if args.per_cell else None
    try:
        for bl in args.bit_lengths:
            rows = run_grid(args, bl, arduino, rng, cell_rows=cell_rows)
            # per-BL CSV so a long sweep survives an interruption
            save_csv(rows, f"{stamp}_uv_grid{tag}_BL{bl}.csv")
            all_rows += rows
    finally:
        if arduino is not None:
            arduino.close()

    if cell_rows:
        save_csv(cell_rows, f"{stamp}_uv_grid{tag}_cells.csv")
    if all_rows:
        save_csv(all_rows, f"{stamp}_uv_grid{tag}_all.csv")
        if args.only:
            print("\n[smoke test] single point measured -- skipping heatmap.")
            print("If C and delta above look sane, run the full sweep with:")
            print(f"  python {os.path.basename(__file__)} --port {args.port}")
        else:
            plot(all_rows, f"{stamp}_uv_grid{tag}.png")
            summarize(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

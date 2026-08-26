#!/usr/bin/env python3
"""Column-5 noise diagnosis: is the noise column-side, row-side, or per-cell?

Phase 1  read noise: 20 back-to-back READ_ROW sweeps, no updates.
         Per-cell std after linear detrend (removes read-disturb drift).
         Elevated col-5 read std -> read path (N5/N6 mux), not update devices.

Phase 2  update noise: for each cell (i,j) send one-hot POTENTIATION
         (u=e_i, v=e_j, prob 1.0 -> C=10) immediately followed by the same
         one-hot DEPRESSION; per-command delta from the built-in pre/post
         reads. REPS repetitions, hard reset between reps so every rep
         starts from the same state. Half-select from one-hot singles is
         ~0.3 LSB/command (measured 2026-08-05), so off-target cells are
         untouched.

Verdict logic on the 5x5 std map:
  whole column 5 elevated, rows fine        -> column-side (v line / driver)
  cells (i,5) elevated AND row i elevated
    at other columns too                    -> row-side
  only some (i,5) cells, their rows fine    -> individual cells
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # package root
from hardware import MemristorInterface

N_READS = 20
REPS = 4
STAMP = time.strftime("%Y-%m-%d_%H-%M")
OUT = f"{STAMP}_col5_diagnosis.npz"

hw = MemristorInterface(port="COM4", baud_rate=115200, bit_length=10,
                        pulse_width=15, pulse_pre=100, pulse_post=100,
                        pulse_zero=10, read_time=20, read_delay=10,
                        timeout=30.0)
if not hw.connect():
    print("FAIL: could not open COM4")
    sys.exit(1)


def fmt(m, f="{:7.1f}"):
    return "\n".join("  " + " ".join(f.format(v) for v in row) for row in m)


try:
    print("=== hard reset ===", flush=True)
    hw.hard_reset()

    print(f"\n=== phase 1: {N_READS} back-to-back reads ===", flush=True)
    reads = np.stack([hw.read_array() for _ in range(N_READS)])  # (N,5,5)
    t = np.arange(N_READS)
    resid = np.empty_like(reads)
    for i in range(5):
        for j in range(5):
            c = np.polyfit(t, reads[:, i, j], 1)
            resid[:, i, j] = reads[:, i, j] - np.polyval(c, t)
    read_std = resid.std(axis=0)
    drift = reads[-1] - reads[0]
    print("read std (detrended, LSB):")
    print(fmt(read_std, "{:6.2f}"))
    print("drift over 20 reads (LSB):")
    print(fmt(drift))

    print(f"\n=== phase 2: per-cell P/D update noise, {REPS} reps ===",
          flush=True)
    pot = np.full((REPS, 5, 5), np.nan)
    dep = np.full((REPS, 5, 5), np.nan)
    for rep in range(REPS):
        if rep > 0:
            print(f"--- hard reset before rep {rep} ---", flush=True)
            hw.hard_reset()
        t0 = time.time()
        for i in range(5):
            for j in range(5):
                u = np.zeros(5); u[i] = 1.0
                v = np.zeros(5); v[j] = 1.0
                for mode, store in (("POTENTIATION", pot), ("DEPRESSION", dep)):
                    pre, post = hw._send_command(mode, u, v)
                    if pre.shape != (5, 10) or post.shape != (5, 10):
                        print(f"WARN rep{rep} ({i},{j}) {mode}: bad shape")
                        continue
                    d = (post[:, :5] - post[:, 5:]) - (pre[:, :5] - pre[:, 5:])
                    store[rep, i, j] = d[i, j]
        print(f"rep {rep}: {time.time()-t0:.0f}s  "
              f"POT mean {np.nanmean(pot[rep]):+.1f}  "
              f"DEP mean {np.nanmean(dep[rep]):+.1f}", flush=True)

    np.savez(OUT, reads=reads, pot=pot, dep=dep)
    print(f"\nsaved -> {OUT}")

    for name, arr in (("POT", pot), ("DEP", dep)):
        mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
        cv = std / np.maximum(np.abs(mean), 1e-9)
        print(f"\n{name} delta mean (LSB):");  print(fmt(mean))
        print(f"{name} delta std over {REPS} reps (LSB):")
        print(fmt(std, "{:6.2f}"))
        print(f"{name} per-column median std : "
              + " ".join(f"{v:6.2f}" for v in np.median(std, axis=0)))
        print(f"{name} per-row    median std : "
              + " ".join(f"{v:6.2f}" for v in np.median(std, axis=1)))
        print(f"{name} per-column median |mean|: "
              + " ".join(f"{v:6.1f}" for v in
                         np.median(np.abs(mean), axis=0)))
    print("\nread  per-column median std : "
          + " ".join(f"{v:6.2f}" for v in np.median(read_std, axis=0)))
    print("read  per-row    median std : "
          + " ".join(f"{v:6.2f}" for v in np.median(read_std, axis=1)))
finally:
    hw.disconnect()

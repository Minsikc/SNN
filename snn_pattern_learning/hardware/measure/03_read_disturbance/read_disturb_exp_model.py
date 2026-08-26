#!/usr/bin/env python3
"""Exponential-decay characterisation of read disturbance and leakage.

The first pass reported absolute LSB, which is the wrong frame: if each read
removes a fixed FRACTION of the stored charge, the LSB lost per read must
shrink as the state decays, and the invariant to report is the fraction.
Re-analysing that data confirmed the exponential form (ln(V_n/V0) linear in
n, R^2 = 0.9993, ~0.72 %/read; exponential SSE 4.5 vs linear 24.5), so this
script measures the two rate constants properly.

Two gaps in the first design are fixed here:

  * every dose-response arm was charged to the SAME ~390 LSB, so the
    state-dependence was never actually exercised. Here the array is charged
    to several distinct levels (V0 swept) and the per-read retention factor
    is fitted at each -- if disturbance is proportional to state, f is
    constant across V0; if there is also a fixed-offset component, f drifts.
  * leakage waits only reached 30 s (5 tau) and the probe read's own cost was
    SUBTRACTED. Here waits extend to 120 s and the probe cost is DIVIDED out,
    which is the correct operation for a multiplicative process.

Model under test
    read disturbance : V_n   = V_inf + (V0 - V_inf) * f**n
    leakage          : V(t)  = V_inf + (V0 - V_inf) * exp(-t / tau)
Both allow a non-zero floor V_inf, since the cap need not discharge to 0.

Analysis uses columns 3-5 (15 good devices); column 2 is faulty.
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5
BL = 10
ONES = "1" * BL
ZEROS = "0" * BL


def _cmd(mode, n1, n2):
    return ",".join(["F", "1", "1", mode, "N56", str(BL),
                     "15", "100", "100", "10", "20", "10"] + n1 + n2)


def _drain(ard):
    rows = []
    while True:
        line = ard.readline().decode("utf-8", "ignore").strip()
        if not line or "EOD" in line:
            break
        if "Row" in line:
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                rows.append(nums[1:11])
    return rows


def read_state(ard):
    """One command = 2 array reads; returns both so the per-read step is
    visible without an extra command."""
    ard.reset_input_buffer()
    ard.write((_cmd("STOCHASTIC_POTENTIATION", [ZEROS] * N, [ZEROS] * N)
               + "\n").encode())
    rows = _drain(ard)
    a = np.array(rows[:N], float)
    b = np.array(rows[N:2 * N], float)
    return a[:, :N] - a[:, N:], b[:, :N] - b[:, N:]


def program(ard, times=1):
    for _ in range(times):
        ard.reset_input_buffer()
        ard.write((_cmd("STOCHASTIC_POTENTIATION", [ONES] * N, [ONES] * N)
                   + "\n").encode())
        _drain(ard)


def hard_reset(ard, set_num=10, silence=1.5):
    cmd = ",".join(["F", "5", "5", "Reset", "N56", str(set_num), "30", "10",
                    "F", "5", "100", "100", "10", "10", "20", "10"]) + "\n"
    ard.reset_input_buffer()
    ard.write(cmd.encode())
    expected, seen, last = 1 + set_num * 3, 0, time.time()
    while True:
        raw = ard.readline()
        if raw:
            if ">" in raw.decode("utf-8", "ignore"):
                seen += 1
                last = time.time()
                if seen >= expected:
                    return
            continue
        if time.time() - last > silence:
            ard.reset_input_buffer()
            return


def g15(M):
    return float(M[:, 2:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-reads", type=int, default=30,
                    help="read commands in the decay trace (x2 array reads)")
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    recs = []

    # distinct charge levels, so V0 itself is a swept variable
    CHARGE = [1, 2, 4, 8]
    WAITS = [0, 1, 2, 5, 10, 20, 45, 90]

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")

        # ---- Exp D: full decay TRACE per charge level -------------------
        # One trace = charge once, then read repeatedly, recording state at
        # every step. A trace gives the whole curve from a single charge, so
        # the fit is not assembled from independent arms.
        print("=== Exp D: read-disturbance decay traces vs initial state ===")
        for rep in range(args.reps):
            for npg in CHARGE:
                hard_reset(ard)
                time.sleep(0.2)
                program(ard, npg)
                for k in range(args.n_reads):
                    r1, r2 = read_state(ard)
                    # both halves of the command are genuine array reads
                    recs.append(dict(exp="D", rep=rep, n_prog=npg,
                                     read_idx=2 * k + 1, wait=0.0,
                                     level=g15(r1)))
                    recs.append(dict(exp="D", rep=rep, n_prog=npg,
                                     read_idx=2 * k + 2, wait=0.0,
                                     level=g15(r2)))
                tr = [x for x in recs if x["exp"] == "D" and x["rep"] == rep
                      and x["n_prog"] == npg]
                v0, vn = tr[0]["level"], tr[-1]["level"]
                print(f"  rep{rep} charge x{npg}: V0={v0:6.1f} -> "
                      f"V[{len(tr)}]={vn:6.1f}   retained {vn/v0:.3f}")

        # ---- Exp E: leakage out to 90 s, single probe read --------------
        print("\n=== Exp E: leakage, long waits, single probe read ===")
        for rep in range(args.reps):
            for w in WAITS:
                hard_reset(ard)
                time.sleep(0.2)
                program(ard, 6)
                r1, _ = read_state(ard)          # reference (2 reads)
                time.sleep(w)
                p1, _ = read_state(ard)          # probe (2 reads)
                recs.append(dict(exp="E", rep=rep, n_prog=6, read_idx=2,
                                 wait=float(w), level=g15(p1),
                                 ref=g15(r1)))
                print(f"  rep{rep} wait={w:5.1f}s  {g15(r1):6.1f} -> "
                      f"{g15(p1):6.1f}   ratio {g15(p1)/g15(r1):.4f}")

    out = f"{stamp}_disturb_exp_model.csv"
    keys = sorted({k for r in recs for k in r})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in recs])
    print(f"\nsaved -> {out}  ({len(recs)} rows)")
    print("run  python plot_disturb_exp.py  to fit and plot")


if __name__ == "__main__":
    main()

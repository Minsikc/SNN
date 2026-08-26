#!/usr/bin/env python3
"""Read disturbance from POSITIVE and NEGATIVE starting levels, same session.

WHY
---
Model B, V_k = V_inf + (V_0 - V_inf) exp(-k/tau), fitted the positive-start
data well (rms 4.5 LSB) and gave V_inf = +67 LSB.  But every trace started
positive, so two incompatible readings fitted equally well:

    absolute   the cell relaxes toward a fixed attractor near +67 LSB
    fractional the cell keeps a fixed fraction (~0.17) of wherever it started

They differ sharply from a NEGATIVE start:

    absolute   -> a cell at -400 LSB must RISE toward +67
    fractional -> it must fall toward 0, i.e. rise only to about -68

One negative-start run settles it.

Both arms run in the SAME session and are interleaved rep by rep, because this
device has degraded measurably within a day before (update fidelity r fell
0.91 -> 0.48 in one session).  Comparing a negative arm measured now against
the positive data from 2026-08-06 would confound sign with degradation, so the
positive arm is re-measured here too and only the two arms from this file are
compared.  Interleaving rather than running all-positive-then-all-negative
means any drift during the session hits both arms equally.

The depression arm uses STOCHASTIC_DEPRESSION with all-ones streams, the
mirror of the potentiation programming, so both arms are driven to their
respective rails by the same mechanism.
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


def _readlines(ard, timeout_s=20.0):
    rows, deadline = [], time.time() + timeout_s
    while time.time() < deadline:
        line = ard.readline().decode("utf-8", "ignore").strip()
        if not line:
            break
        if "EOD" in line:
            break
        if "Row" in line:
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                rows.append((nums[0], nums[1:11]))
    return rows


def read_row(ard, row):
    cmd = f"F,{row},5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    got = {}
    for idx, vals in _readlines(ard):
        v = np.array(vals, float)
        got[idx] = v[:N] - v[N:]
    return got


def program_all(ard, times, direction):
    """Drive the whole array toward a rail. direction in {'pot','dep'}."""
    op = ("STOCHASTIC_POTENTIATION" if direction == "pot"
          else "STOCHASTIC_DEPRESSION")
    for _ in range(times):
        cmd = ",".join(["F", "1", "1", op, "N56",
                        str(BL), "15", "100", "100", "10", "20", "10"]
                       + [ONES] * N + [ONES] * N)
        ard.reset_input_buffer()
        ard.write((cmd + "\n").encode())
        _readlines(ard)


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
                    return True
            continue
        if time.time() - last > silence:
            ard.reset_input_buffer()
            return seen > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--n-reads", type=int, default=1000)
    ap.add_argument("--n-prog", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    pts = np.unique(np.round(np.logspace(
        0, np.log10(args.n_reads), 35)).astype(int))
    sample_at = set(int(x) for x in pts) | {args.n_reads}

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    recs = []

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        chk = read_row(ard, 0)
        if list(chk) != [0]:
            print(f"[WARN] READ_ROW returned {list(chk)}; expected [0]")
        print(f"{args.reps} reps x 2 arms x 5 rows x {args.n_reads} reads")
        print("arms interleaved so session drift hits both equally\n")

        t0 = time.time()
        for rep in range(args.reps):
            for arm in ("pot", "dep"):
                for P in range(N):
                    hard_reset(ard)
                    time.sleep(0.2)
                    program_all(ard, args.n_prog, arm)
                    for k in range(1, args.n_reads + 1):
                        got = read_row(ard, P)
                        if P not in got:
                            continue
                        if k in sample_at:
                            for j in range(N):
                                recs.append(dict(rep=rep, arm=arm, row=P,
                                                 col=j + 1, reads=k,
                                                 level=float(got[P][j])))
                    sub = [r for r in recs if r["rep"] == rep
                           and r["arm"] == arm and r["row"] == P]
                    ks = sorted({r["reads"] for r in sub})
                    f = np.mean([r["level"] for r in sub if r["reads"] == 1])
                    l = np.mean([r["level"] for r in sub
                                 if r["reads"] == args.n_reads])
                    print(f"  rep{rep} {arm} row{P}: {f:7.1f} -> {l:7.1f} LSB"
                          f"   [{time.strftime('%H:%M:%S')}, "
                          f"{(time.time()-t0)/60:.1f} min]", flush=True)

    out = f"{stamp}_disturb_signed.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "arm", "row", "col",
                                          "reads", "level"])
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    for arm in ("pot", "dep"):
        s = [r for r in recs if r["arm"] == arm]
        f = np.mean([r["level"] for r in s if r["reads"] == 1])
        l = np.mean([r["level"] for r in s if r["reads"] == args.n_reads])
        print(f"  {arm}: V(1) {f:+7.1f} -> V({args.n_reads}) {l:+7.1f} LSB")
    print("\nabsolute-attractor prediction: both arms converge to the SAME "
          "level")
    print("fractional prediction:         dep arm ends near "
          "-(0.17 x |start|), i.e. still negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

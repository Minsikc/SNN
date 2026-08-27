#!/usr/bin/env python3
"""Per-cell read-disturbance decay for all 25 cells.

Each row is exposed in its own run: charge the array, then hammer ONE row
with READ_ROW while recording that row's five cells at every checkpoint.
Repeating for rows 0..4 yields a decay curve for every cell, with each cell
disturbed only by reads of its own row (verified 2026-08-06: the non-local
share of disturbance is 2.8%).

Why per-cell rather than the row/array average: averaging cells with slightly
different rates manufactures a fake double exponential. A simulated mixture
with a 0.4% spread in rate fits a double exponential 11000x better than a
single one, even though every constituent is a pure single exponential. The
per-cell curves here are the honest way to get the rate and its spread.

Also note the fitted "floor" is not physical -- per-column floors ranged from
-41 to +49 LSB, and a negative asymptote cannot be stored charge. It is a
read-path offset, so it is fitted per cell and reported, not interpreted.
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


def program_all(ard, times=1):
    for _ in range(times):
        cmd = ",".join(["F", "1", "1", "STOCHASTIC_POTENTIATION", "N56",
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
    ap.add_argument("--n-reads", type=int, default=300)
    ap.add_argument("--n-prog", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--every", type=int, default=10,
                    help="linear sampling interval; ignored when --log-sample "
                         "is set (the READ_ROW itself returns the value, so "
                         "sampling costs nothing extra either way)")
    ap.add_argument("--log-sample", action="store_true", default=True,
                    help="sample on a log-spaced grid. For an asymptote the "
                         "late reads carry the information, but linear "
                         "sampling spends most points on the early decay "
                         "where the curve is already well determined.")
    ap.add_argument("--no-log-sample", dest="log_sample", action="store_false")
    args = ap.parse_args()

    if args.log_sample:
        pts = np.unique(np.round(np.logspace(
            0, np.log10(args.n_reads), 40)).astype(int))
        sample_at = set(int(x) for x in pts) | {args.n_reads}
    else:
        sample_at = None

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    recs = []

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        chk = read_row(ard, 0)
        if list(chk) != [0]:
            print(f"[WARN] READ_ROW returned {list(chk)}; expected [0]")
        print(f"{args.reps} reps x 5 rows x {args.n_reads} reads\n")

        for rep in range(args.reps):
            for P in range(N):
                hard_reset(ard)
                time.sleep(0.2)
                program_all(ard, args.n_prog)
                for k in range(1, args.n_reads + 1):
                    got = read_row(ard, P)
                    if P not in got:
                        continue
                    take = (k in sample_at) if sample_at is not None \
                        else (k == 1 or k % args.every == 0)
                    if take:
                        for j in range(N):
                            recs.append(dict(rep=rep, row=P, col=j + 1,
                                             reads=k,
                                             level=float(got[P][j])))
                sub = [r for r in recs if r["rep"] == rep and r["row"] == P]
                first = [r for r in sub if r["reads"] == 1]
                last = [r for r in sub if r["reads"] == args.n_reads]
                # nearest sampled point to a third of the way in (the log
                # grid will not contain n_reads//3 exactly)
                ks = sorted({r["reads"] for r in sub})
                kmid = min(ks, key=lambda k: abs(k - args.n_reads / 3))
                mid = [r for r in sub if r["reads"] == kmid]
                print(f"  rep{rep} row{P}: "
                      f"{np.mean([r['level'] for r in first]):6.1f} -> "
                      f"{np.mean([r['level'] for r in mid]):6.1f} (1/3) -> "
                      f"{np.mean([r['level'] for r in last]):6.1f} LSB "
                      f"[{time.strftime('%H:%M:%S')}]", flush=True)

    out = f"{stamp}_disturb_all25.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "row", "col", "reads",
                                          "level"])
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")
    print("run  python plot_all25_disturb.py  to fit and plot")


if __name__ == "__main__":
    main()

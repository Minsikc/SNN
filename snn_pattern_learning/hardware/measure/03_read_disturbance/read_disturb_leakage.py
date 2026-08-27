#!/usr/bin/env python3
"""Separate read disturbance from leakage on the 6T1C array.

Both shrink the stored state at once, so repeated reading alone cannot tell
them apart: read count and elapsed time rise together. The design here breaks
that collinearity using the measured cost of one array read (~0.21 s) -- by
inserting waits, elapsed time can be varied while the read count is held
fixed, and vice versa.

  Exp A  2x2: many-reads/no-wait vs few-reads/padded-wait at matched total
              time. Difference = read disturbance; the few-read arm's decay
              approximates leakage.
  Exp B  dose-response in read count (recharge before each arm).
  Exp C  leakage time constant: wait, then read exactly ONCE, so the probe
         itself cannot contaminate the decay curve.

Read granularity note (firmware): Read_all_rows_sequentially walks row_num
0..4 and each row read drives that row's N1/N3 plus its word line, with all
5 columns sampled in parallel by the 5 ADCs. So one "read" = 5 row reads =
every cell touched once; there is no single-cell read, and column selection
does not exist. Rows measured to decay near-uniformly (87-107 LSB over 40
reads), consistent with each row being disturbed during its own read.

Caveat kept in mind when reading Exp A: even the few-read arm still performs
some reads, so its drop is leakage PLUS a little disturbance -- it is an
upper bound on leakage, which makes the disturbance estimate a lower bound.
Exp C is the clean leakage measurement.

Analysis uses columns 3-5 only (15 devices) -- column 2 is faulty.
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
SEC_PER_CMD = 0.43          # measured: one zero-pulse command = 2 array reads


def _cmd(mode, n1, n2, read_fn="N56"):
    return ",".join(["F", "1", "1", mode, read_fn, str(BL),
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
    """One zero-pulse command = 2 array reads. Returns (first, second)."""
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


def cells15(M):
    return M[:, 2:]          # columns 3-5


def charge(ard, n_prog):
    hard_reset(ard)
    time.sleep(0.2)
    program(ard, n_prog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--n-prog", type=int, default=6,
                    help="potentiation commands used to charge before a run")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    recs = []
    MANY, FEW = 20, 4
    pad = (MANY - FEW) * SEC_PER_CMD / FEW    # keeps both arms equal in time

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")

        print("=== Exp A: reads vs time (matched total duration) ===")
        for rep in range(args.reps):
            for arm, n_cmd, gap in [("many_reads", MANY, 0.0),
                                    ("few_reads", FEW, pad)]:
                charge(ard, args.n_prog)
                s0, _ = read_state(ard)
                t0 = time.time()
                last = s0
                for _ in range(n_cmd):
                    _, last = read_state(ard)
                    if gap:
                        time.sleep(gap)
                el = time.time() - t0
                drop = float(cells15(s0).mean() - cells15(last).mean())
                recs.append(dict(exp="A", rep=rep, arm=arm, n_cmd=n_cmd,
                                 reads=n_cmd * 2, elapsed=el,
                                 start=float(cells15(s0).mean()),
                                 end=float(cells15(last).mean()), drop=drop))
                print(f"  rep{rep} {arm:11s} reads={n_cmd*2:3d} "
                      f"t={el:5.1f}s  {cells15(s0).mean():6.1f} -> "
                      f"{cells15(last).mean():6.1f}  drop {drop:6.1f}")

        print("\n=== Exp B: dose-response (recharge each arm) ===")
        for rep in range(args.reps):
            for n_cmd in [1, 2, 5, 10, 20]:
                charge(ard, args.n_prog)
                s0, _ = read_state(ard)
                last = s0
                for _ in range(n_cmd):
                    _, last = read_state(ard)
                drop = float(cells15(s0).mean() - cells15(last).mean())
                recs.append(dict(exp="B", rep=rep, arm="dose", n_cmd=n_cmd,
                                 reads=n_cmd * 2, elapsed=float("nan"),
                                 start=float(cells15(s0).mean()),
                                 end=float(cells15(last).mean()), drop=drop))
                print(f"  rep{rep} reads={n_cmd*2:3d}  drop {drop:6.1f}")

        print("\n=== Exp C: leakage (wait, then ONE read) ===")
        for rep in range(args.reps):
            for wait in [0.5, 2, 5, 10, 30]:
                charge(ard, args.n_prog)
                s0, _ = read_state(ard)
                time.sleep(wait)
                probe, _ = read_state(ard)
                drop = float(cells15(s0).mean() - cells15(probe).mean())
                recs.append(dict(exp="C", rep=rep, arm="leak", n_cmd=1,
                                 reads=2, elapsed=wait,
                                 start=float(cells15(s0).mean()),
                                 end=float(cells15(probe).mean()), drop=drop))
                print(f"  rep{rep} wait={wait:5.1f}s  "
                      f"{cells15(s0).mean():6.1f} -> "
                      f"{cells15(probe).mean():6.1f}  drop {drop:6.1f}")

    out = f"{stamp}_read_disturb_leakage.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}")

    A = [r for r in recs if r["exp"] == "A"]
    dm = np.mean([r["drop"] for r in A if r["arm"] == "many_reads"])
    df = np.mean([r["drop"] for r in A if r["arm"] == "few_reads"])
    tm = np.mean([r["elapsed"] for r in A if r["arm"] == "many_reads"])
    tf = np.mean([r["elapsed"] for r in A if r["arm"] == "few_reads"])
    print(f"\nExp A summary (matched time {tm:.1f}s vs {tf:.1f}s):")
    print(f"  many reads ({MANY*2}): drop {dm:6.2f} LSB")
    print(f"  few reads  ({FEW*2}): drop {df:6.2f} LSB   <- leakage upper bound")
    print(f"  disturbance = difference: {dm - df:6.2f} LSB over "
          f"{(MANY - FEW) * 2} extra reads "
          f"({(dm - df) / ((MANY - FEW) * 2):.2f} LSB/read)")

    C = [r for r in recs if r["exp"] == "C"]
    print("\nExp C leakage curve (clean, single probe read):")
    for w in sorted({r["elapsed"] for r in C}):
        d = np.mean([r["drop"] for r in C if r["elapsed"] == w])
        print(f"  wait {w:5.1f}s -> drop {d:6.2f} LSB")


if __name__ == "__main__":
    main()

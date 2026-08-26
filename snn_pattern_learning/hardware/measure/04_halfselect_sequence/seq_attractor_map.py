#!/usr/bin/env python3
"""Where does an alternating half-select pair drive the cell, from any start?

Earlier runs showed each pair converging to the same endpoint whether it
began at ~205 or ~23 LSB, which suggests the pairs act as attractors rather
than as fixed increments. This maps that directly: five starting states
spanning the full range, a cycle sweep up to 300, and the ADC level reached
at each point.

Starting states (calibrated 2026-08-07):
    full potentiation   P x12  ->  +443 LSB
    mid potentiation    P x3   ->  +228
    reset               none   ->   +14
    mid depression      D x3   ->  -193
    full depression     D x12  ->  -425

If a pair is an attractor, curves from all five starts converge on one level
and the sign of the movement flips depending on which side of it you began.
If instead the pair applies a fixed increment per cycle, the five curves stay
parallel and never meet.

Read cost is 3 array reads per trial regardless of cycle count (before,
HS_SEQ entry, after), so the cycle axis is not contaminated by reading.
"""
import argparse
import csv
import datetime
import itertools
import re
import time

import numpy as np

N = 5
ONES = "1" * 10
LINES = ["N1", "N2", "N3", "N4"]
CODE = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}

# label -> (potentiation commands, depression commands)
STARTS = {
    "full_pot": (12, 0),
    "mid_pot": (3, 0),
    "reset": (0, 0),
    "mid_dep": (0, 3),
    "full_dep": (0, 12),
}


def prog_command(direction):
    op = ("STOCHASTIC_POTENTIATION" if direction == "P"
          else "STOCHASTIC_DEPRESSION")
    return ",".join(["F", "1", "1", op, "N56", "10",
                     "15", "100", "100", "10", "20", "10"]
                    + [ONES] * N + [ONES] * N)


def seq_command(first, second, cycles):
    pair = (CODE[first] * 10 + CODE[second]) if first else 0
    return ",".join(["F", str(pair), "5", "HS_SEQ", "N56", "1", str(cycles),
                     "99999", "F", "15", "100", "100", "10", "20", "0", "10"])


def read_command():
    return "F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"


def reset_command():
    return ",".join(["F", "5", "5", "Reset", "N56", "10", "30", "10",
                     "F", "5", "100", "100", "10", "10", "20", "10"])


def _drain(ard, timeout_s, expect=None, quiet=1.2):
    blocks, last = [], time.time()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ard.readline().decode("utf-8", "ignore").strip()
        if line:
            last = time.time()
            if "EOD" in line or "operation end" in line:
                break
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                blocks.append(nums[-10:])
                if expect and len(blocks) >= expect:
                    break
            continue
        if time.time() - last > quiet:
            break
    return blocks


def send(ard, cmd, timeout_s=60.0, expect=None, settle=0.1):
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    b = _drain(ard, timeout_s, expect)
    time.sleep(settle)
    ard.reset_input_buffer()
    return b


def to_diff(blocks):
    a = np.array(blocks[:N], float)
    return a[:, :N] - a[:, N:]


def set_start(ard, label):
    nP, nD = STARTS[label]
    send(ard, reset_command(), timeout_s=40, expect=31)
    time.sleep(0.15)
    for _ in range(nP):
        send(ard, prog_command("P"))
    for _ in range(nD):
        send(ard, prog_command("D"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--pairs", default="N1+N2,N3+N4,N1+N3,N2+N4")
    ap.add_argument("--sweep", default="1,3,10,30,100,300")
    ap.add_argument("--starts", default=",".join(STARTS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    cycles = [int(x) for x in args.sweep.split(",")]
    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    want = {p.strip() for p in args.pairs.split(",") if p.strip()}
    pairs = [(a, b) for a, b in itertools.product(LINES, LINES)
             if f"{a}+{b}" in want]
    pairs.append((None, None))          # control: same reads, no pulses

    recs = []
    trials = [(p, s, c) for p in pairs for s in starts for c in cycles]
    print(f"{len(pairs)} pairs x {len(starts)} starts x {len(cycles)} cycles "
          f"x {args.reps} reps = {len(trials)*args.reps} trials")

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")
        for rep in range(args.reps):
            order = list(trials)
            rng.shuffle(order)
            # stable-sort by start so identical starts run back to back, but
            # the shuffled order within each start block is preserved
            order.sort(key=lambda t: t[1])
            for (a, b), st, cyc in order:
                set_start(ard, st)

                bb = send(ard, read_command())
                if len(bb) < N:
                    print(f"  [SKIP] before-read {len(bb)}/{N}")
                    continue
                before = to_diff(bb)

                send(ard, seq_command(a, b, cyc), timeout_s=120)

                ab = send(ard, read_command())
                if len(ab) < N:
                    print(f"  [SKIP] after-read {len(ab)}/{N}")
                    continue
                after = to_diff(ab)

                label = f"{a}+{b}" if a else "control"
                for i in range(N):
                    for j in range(N):
                        recs.append(dict(rep=rep, combo=label, start=st,
                                         cycles=cyc, cell_row=i + 1,
                                         cell_col=j + 1,
                                         before=float(before[i, j]),
                                         after=float(after[i, j]),
                                         delta=float(after[i, j] - before[i, j])))
                print(f"  rep{rep} {label:>8} {st:>9} cyc={cyc:4d}: "
                      f"{before.mean():7.1f} -> {after.mean():7.1f}",
                      flush=True)

    out = f"{stamp}_seq_attractor.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    for label in [f"{a}+{b}" for a, b in pairs if a] + ["control"]:
        sub = [r for r in recs if r["combo"] == label]
        if not sub:
            continue
        print(f"\n{label}: level reached (LSB)")
        print(f"{'start':>10} {'before':>8} " +
              "".join(f"{c:>8}" for c in cycles))
        for st in starts:
            b0 = np.mean([r["before"] for r in sub if r["start"] == st])
            row = []
            for c in cycles:
                v = [r["after"] for r in sub
                     if r["start"] == st and r["cycles"] == c]
                row.append(np.mean(v) if v else float("nan"))
            print(f"{st:>10} {b0:>8.1f} " +
                  "".join(f"{v:>8.1f}" for v in row))
    print("\nrun  python plot_seq_attractor.py  to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

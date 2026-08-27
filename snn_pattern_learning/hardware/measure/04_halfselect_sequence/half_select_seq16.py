#!/usr/bin/env python3
"""All 16 ordered pairs of half-selects, applied ALTERNATING, vs cycle count.

Uses the HS_SEQ opcode, which takes the pair encoded in row_num as two
digits (tens = first line, units = second, 1..4 = N1..N4, 0 = skip) and
alternates them for `update_num` cycles: first, second, first, second, ...

Why a dedicated opcode rather than two HS_<line> commands: each command
carries its own entry read, so delivering a pair as two commands would both
double the read cost and force block-wise (AAA...BBB) rather than
alternating order. HS_SEQ emits exactly ONE read and interleaves properly.

Read accounting -- identical for all 17 conditions:
    READ_ROW (before)  +  HS_SEQ entry read  +  READ_ROW (after)  =  3
and it does not grow with cycle count, so the cycle axis isolates the
half-select. The control (row_num=0) runs the same command with no pulses,
so subtracting it removes the read toll exactly.

Context: single lines were measured first (2026-08-07,
half_select_cycles.py). N1 and N3 do nothing; N2 and N4 disturb at only
~0.015 LSB/cycle. Against a per-cell noise of ~3 LSB that needs hundreds of
cycles to resolve, hence the large cycle counts here. This run asks whether
pairs interact, and whether order matters.
"""
import argparse
import csv
import datetime
import itertools
import re
import time

import numpy as np

N = 5
LINES = ["N1", "N2", "N3", "N4"]
CODE = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}
READ_PERIOD = 99999


def seq_command(first, second, cycles):
    """HS_SEQ with the pair encoded in row_num (tens=first, units=second)."""
    pair = (CODE[first] * 10 + CODE[second]) if first else 0
    return ",".join([
        "F", str(pair), "5", "HS_SEQ", "N56",
        "1",                     # set_num
        str(cycles),             # update_num = cycles
        str(READ_PERIOD),
        "F",
        "15", "100", "100", "10",
        "20", "0", "10",
    ])


def read_command():
    return "F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"


def program_command():
    ones = "1" * 10
    return ",".join(["F", "1", "1", "STOCHASTIC_POTENTIATION", "N56", "10",
                     "15", "100", "100", "10", "20", "10"]
                    + [ones] * N + [ones] * N)


def reset_command():
    return ",".join(["F", "5", "5", "Reset", "N56", "10", "30", "10",
                     "F", "5", "100", "100", "10", "10", "20", "10"])


def _drain(ard, timeout_s, expect=None, quiet=1.0):
    """Read until EOD, or until `expect` blocks / a silence window.

    READ_ROW, STOCHASTIC_* and HS_SEQ all terminate with EOD>, so they are
    read to the marker -- stopping early would leave EOD> in the buffer and
    it would be consumed as the first line of the next reply. Only Reset
    lacks a terminator and needs the block-count fallback.
    """
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


def send(ard, cmd, timeout_s=30.0, expect=None, settle=0.1):
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    blocks = _drain(ard, timeout_s, expect)
    time.sleep(settle)
    ard.reset_input_buffer()
    return blocks


def to_diff(blocks):
    a = np.array(blocks[:N], float)
    return a[:, :N] - a[:, N:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--sweep", default="100,500,2000")
    ap.add_argument("--n-prog", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pairs", default=None,
                    help="comma-separated pairs to run, e.g. N1+N2,N3+N4. "
                         "Default: all 16.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    cycles = [int(x) for x in args.sweep.split(",")]
    if args.pairs:
        want = {p.strip() for p in args.pairs.split(",") if p.strip()}
        pairs = [(a, b) for a, b in itertools.product(LINES, LINES)
                 if f"{a}+{b}" in want]
    else:
        pairs = [(a, b) for a, b in itertools.product(LINES, LINES)]
    pairs.append((None, None))
    recs = []

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        print(f"{len(pairs)} conditions x {len(cycles)} cycle counts "
              f"x {args.reps} reps = {len(pairs)*len(cycles)*args.reps} trials\n")

        trials = [(p, c) for p in pairs for c in cycles]
        for rep in range(args.reps):
            order = list(trials)
            rng.shuffle(order)
            for (a, b), cyc in order:
                send(ard, reset_command(), timeout_s=40, expect=31)
                time.sleep(0.15)
                for _ in range(args.n_prog):
                    send(ard, program_command())

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

                delta = after - before
                label = f"{a}+{b}" if a else "control"
                for i in range(N):
                    for j in range(N):
                        recs.append(dict(rep=rep, first=a or "",
                                         second=b or "", combo=label,
                                         cycles=cyc, cell_row=i + 1,
                                         cell_col=j + 1,
                                         before=float(before[i, j]),
                                         after=float(after[i, j]),
                                         delta=float(delta[i, j])))
                print(f"  rep{rep} {label:>8} cyc={cyc:5d}: "
                      f"{before.mean():6.1f} -> {after.mean():6.1f}  "
                      f"delta {delta.mean():+7.2f}", flush=True)

    out = f"{stamp}_half_select_seq16.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    ctrl = {c: np.mean([r["delta"] for r in recs
                        if r["combo"] == "control" and r["cycles"] == c])
            for c in cycles}
    # The level reached matters more than the delta: these pairs converge to
    # an attractor (a rail, or zero) rather than removing a fixed amount, so
    # the same endpoint is hit from any starting level.
    print("\nLEVEL REACHED (LSB)")
    print(f"{'combo':>8} " + "".join(f"{c:>9}" for c in cycles))
    for a in LINES:
        for b in LINES:
            row = []
            for c in cycles:
                d = [r["after"] for r in recs
                     if r["combo"] == f"{a}+{b}" and r["cycles"] == c]
                row.append(np.mean(d) if d else float("nan"))
            if any(np.isfinite(v) for v in row):
                print(f"{a+'+'+b:>8} " + "".join(f"{v:>9.1f}" for v in row))
    print(f"(start level {np.mean([r['before'] for r in recs]):.1f} LSB)")

    print("\nnet disturbance (control subtracted), LSB")
    print(f"{'combo':>8} " + "".join(f"{c:>9}" for c in cycles))
    table = {}
    for a in LINES:
        for b in LINES:
            row = []
            for c in cycles:
                d = [r["delta"] for r in recs
                     if r["combo"] == f"{a}+{b}" and r["cycles"] == c]
                v = np.mean(d) - ctrl[c] if d else float("nan")
                row.append(v)
                table[(a, b, c)] = v
            print(f"{a+'+'+b:>8} " + "".join(f"{v:>9.2f}" for v in row))

    cmax = cycles[-1]
    print(f"\nmatrix at {cmax} cycles (rows = first, cols = second):")
    print("        " + "".join(f"{b:>9}" for b in LINES))
    for a in LINES:
        print(f"{a:>6}  " + "".join(f"{table.get((a,b,cmax), float('nan')):>9.2f}"
                                    for b in LINES))
    print("\norder effect at max cycles:")
    for a, b in itertools.combinations(LINES, 2):
        x, y = table.get((a, b, cmax)), table.get((b, a, cmax))
        if x is not None and y is not None:
            print(f"  {a}+{b} {x:+7.2f}  vs  {b}+{a} {y:+7.2f}  "
                  f"-> {x-y:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

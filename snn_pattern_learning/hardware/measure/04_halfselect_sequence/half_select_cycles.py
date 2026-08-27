#!/usr/bin/env python3
"""Half-select disturbance vs cycle count, with exactly TWO reads per trial.

Replaces half_select_pairs.py, which had two defects:
  * it delivered each half-select with a STOCHASTIC_* command, and those
    perform a pre-read AND a post-read. A 20-cycle trial therefore carried
    80 array reads, and at the measured 0.79 %/read that alone accounts for
    ~-113 LSB -- the whole of the "-86 LSB" attributed to N1/N2/N3.
  * its control applied no commands at all (a=b=None), so it measured a
    0-read baseline and could not cancel that toll.

Here the HS_* opcodes deliver the pulses, with read_period > update_num so
no read fires inside the pulse loop. Both measurement reads are separate
READ_ROW commands:

    reset -> program -> READ_ROW   (read #1, before)
                     -> HS_<line>  (N cycles of pulses)
                     -> READ_ROW   (read #2, after)

The HS opcode does emit one entry read of its own, but with row_num=5 that
is a single all-rows-at-once line rather than the five per-row lines needed
for a 5x5 matrix, so it is discarded and READ_ROW is used instead. The read
toll per trial is therefore fixed and independent of cycle count -- a cycle
sweep measures the half-select, not the reading. (The discarded entry read
still costs the array one exposure, but it costs every condition equally,
including the control.)

The control (--line none) runs the same two reads with zero pulses.
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5
LINES = ["N1", "N2", "N3", "N4"]
OPCODE = {"N1": "HS_N1", "N2": "HS_N2", "N3": "HS_N3", "N4": "HS_N4"}

# read_period must exceed update_num so `(k+1) % read_period == j` never
# fires inside the pulse loop
READ_PERIOD = 99999


def hs_command(line, cycles):
    """HS_<line>: one entry read, then `cycles` half-select pulses, no more reads.

    row_num=5 selects all five rows (used by N1/N3); col_num=5 selects all
    five columns (used by N2/N4). Both are set so the same string works for
    every line.
    """
    return ",".join([
        "F", "5", "5", OPCODE[line], "N56",
        "1",                      # set_num
        str(cycles),              # update_num  = cycles
        str(READ_PERIOD),         # read_period = suppress mid-loop reads
        "F",                      # remainder_string -> cycle_num = 1
        "15", "100", "100", "10",  # pulse_width, pre, post, zero
        "20", "0", "10",          # read_time, read_set_time, read_delay
    ])


def read_command():
    """READ_ROW with row_num=5: the whole array in exactly one array read."""
    return "F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"


def program_command(n_prog_pulses):
    ones = "1" * 10
    return ",".join(["F", "1", "1", "STOCHASTIC_POTENTIATION", "N56", "10",
                     "15", "100", "100", "10", "20", "10"]
                    + [ones] * N + [ones] * N)


def reset_command():
    return ",".join(["F", "5", "5", "Reset", "N56", "10", "30", "10",
                     "F", "5", "100", "100", "10", "10", "20", "10"])


# --------------------------------------------------------------- serial ---

def _drain(ard, timeout_s=30.0, expect=None, quiet=1.0):
    """Collect Row blocks until the command is done.

    Termination is awkward on this firmware: only some opcodes end with
    "EOD>". HS_* finishes with a bare Serial.print("operation end") -- no
    newline -- so readline() never returns it and a naive reader waits out
    the whole serial timeout. Measured: 30 s per HS command and per Reset,
    which is what made a 15-trial run take minutes instead of seconds.

    So stop as soon as `expect` blocks have arrived, and otherwise fall back
    to a short silence window instead of the serial timeout.
    """
    blocks = []
    last = time.time()
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


def send(ard, cmd, timeout_s=30.0, expect=None):
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    return _drain(ard, timeout_s, expect)


def to_diff(blocks):
    a = np.array(blocks[:N], float)
    return a[:, :N] - a[:, N:]


# ------------------------------------------------------------------ dry ---

def dry_run(args):
    """Validate command construction without any hardware."""
    print("DRY RUN -- no serial port opened\n")
    ok = True

    print("commands that a single trial issues:")
    print(f"  1. {reset_command()}")
    print(f"  2. {program_command(1)[:70]}...   x{args.n_prog}")
    print(f"  3. {hs_command('N4', args.cycles)}")
    print(f"  4. {read_command()}")

    print("\nfield check on the HS command:")
    f = hs_command("N4", args.cycles).split(",")
    named = ["Direction", "row_num", "col_num", "update_function",
             "read_function", "set_num", "update_num", "read_period",
             "remainder", "pulse_width", "pre", "post", "zero",
             "read_time", "read_set_time", "read_delay"]
    for k, v in zip(named, f):
        print(f"    {k:16s} = {v}")
    if len(f) != len(named):
        print(f"    [FAIL] field count {len(f)}, expected {len(named)}")
        ok = False

    print("\nmid-loop read suppression:")
    for c in [1, 2, 5, 20, 100]:
        fires = sum(1 for k in range(c) if (k + 1) % READ_PERIOD == 0)
        status = "OK" if fires == 0 else "FAIL"
        print(f"    cycles={c:4d} -> {fires} mid-loop read(s)   [{status}]")
        if fires:
            ok = False

    print("\nread accounting per trial (must be 2, independent of cycles):")
    for c in [1, 5, 20, 100]:
        reads = 1 + 1          # HS entry read + READ_ROW
        print(f"    cycles={c:4d} -> {reads} array reads")

    print("\nopcode per line:")
    for ln in LINES:
        cmd = hs_command(ln, args.cycles)
        sel = "row_num" if ln in ("N1", "N3") else "col_num"
        print(f"    {ln} -> {OPCODE[ln]:8s} (selector {sel}=5)  {cmd[:38]}...")

    print("\nparsing a synthetic device response:")
    fake = ["0,120,121,122,123,124,20,21,22,23,24>"]
    fake += [f"Row_{i},1{i}0,1{i}1,1{i}2,1{i}3,1{i}4,2{i}0,2{i}1,2{i}2,2{i}3,2{i}4>"
             for i in range(N)]
    blocks = []
    for line in fake:
        nums = [int(x) for x in re.findall(r"-?\d+", line)]
        if len(nums) >= 11:
            blocks.append(nums[-10:])
    print(f"    parsed {len(blocks)} blocks from {len(fake)} lines")
    if len(blocks) >= N:
        d = to_diff(blocks[-N:])
        print(f"    differential matrix shape {d.shape}, mean {d.mean():.1f}")
    else:
        print("    [FAIL] fewer than 5 blocks parsed")
        ok = False

    print("\nsweep plan:")
    cycles = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.cycles]
    conds = LINES + ["none"]
    print(f"    lines {conds} x cycles {cycles} x {args.reps} reps "
          f"= {len(conds)*len(cycles)*args.reps} trials")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# ----------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated cycle counts, e.g. 1,2,5,10,20,50")
    ap.add_argument("--lines", default="N1,N2,N3,N4")
    ap.add_argument("--n-prog", type=int, default=3,
                    help="potentiation commands before the half-selects; "
                         "0 starts from the reset state")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    cycles = [int(x) for x in args.sweep.split(",")] if args.sweep \
        else [args.cycles]
    conds = [l for l in args.lines.split(",") if l] + ["none"]
    recs = []

    with serial.Serial(args.port, 115200, timeout=30) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        print(f"{len(conds)} conditions x {len(cycles)} cycle counts "
              f"x {args.reps} reps\n")

        trials = [(l, c) for l in conds for c in cycles]
        for rep in range(args.reps):
            order = list(trials)
            rng.shuffle(order)
            for line, cyc in order:
                send(ard, reset_command(), timeout_s=40, expect=31)
                time.sleep(0.15)
                for _ in range(args.n_prog):
                    send(ard, program_command(1))

                b_blocks = send(ard, read_command())
                if len(b_blocks) < N:
                    print(f"  [SKIP] {line} cyc={cyc}: before-read returned "
                          f"{len(b_blocks)}/{N} rows")
                    continue
                before = to_diff(b_blocks)

                if line != "none":
                    send(ard, hs_command(line, cyc), timeout_s=180, expect=1)

                a_blocks = send(ard, read_command())
                if len(a_blocks) < N:
                    print(f"  [SKIP] {line} cyc={cyc}: after-read returned "
                          f"{len(a_blocks)}/{N} rows")
                    continue
                after = to_diff(a_blocks)
                delta = after - before
                for i in range(N):
                    for j in range(N):
                        recs.append(dict(rep=rep, line=line, cycles=cyc,
                                         cell_row=i + 1, cell_col=j + 1,
                                         before=float(before[i, j]),
                                         after=float(after[i, j]),
                                         delta=float(delta[i, j])))
                print(f"  rep{rep} {line:>5} cyc={cyc:4d}: "
                      f"{before.mean():6.1f} -> {after.mean():6.1f}  "
                      f"delta {delta.mean():+7.2f}", flush=True)

    tag = "reset" if args.n_prog == 0 else f"prog{args.n_prog}"
    out = f"{stamp}_half_select_cycles_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    ctrl = {c: np.mean([r["delta"] for r in recs
                        if r["line"] == "none" and r["cycles"] == c])
            for c in cycles}
    print(f"\n{'line':>5} " + "".join(f"{c:>10}" for c in cycles))
    for line in conds:
        if line == "none":
            continue
        row = []
        for c in cycles:
            d = [r["delta"] for r in recs
                 if r["line"] == line and r["cycles"] == c]
            row.append(np.mean(d) - ctrl[c] if d else float("nan"))
        print(f"{line:>5} " + "".join(f"{v:>10.2f}" for v in row))
    print("\n(net of the read-only control; both reads are already "
          "included in every trial)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

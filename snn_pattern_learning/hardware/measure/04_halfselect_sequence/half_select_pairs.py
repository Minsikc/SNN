#!/usr/bin/env python3
"""Disturbance from ORDERED PAIRS of half-select pulses: 4 x 4 = 16 combos.

A half-select drives one line only (N1, N2, N3 or N4) with no coincidence, so
in principle it should not program anything. This measures how much a
programmed state actually moves when two half-selects are applied back to
back, for every ordered pair -- (N1,N2) is a different sequence from (N2,N1),
so order is part of the condition.

Line primitives (no firmware change needed; each is one existing command with
one side of the stream pair zeroed):
    N1 -> STOCHASTIC_POTENTIATION, row=1s, col=0s
    N2 -> STOCHASTIC_POTENTIATION, row=0s, col=1s
    N3 -> STOCHASTIC_DEPRESSION,   row=1s, col=0s
    N4 -> STOCHASTIC_DEPRESSION,   row=0s, col=1s

Design points
-------------
* One read BEFORE and one read AFTER the half-select sequence; the reported
  quantity is their difference. Exactly two reads per trial, no more --
  reading is itself the dominant disturbance (~0.79 %/read of the
  above-floor state), so any extra read would be charged to the
  half-select's account.
* Every combo is exactly 2 half-select commands, so both the half-select
  dose and the read toll are identical across all 16 conditions; only the
  identity and order of the two lines differ. A (none,none) control repeats
  the same read-apply-read structure with zero half-selects, which measures
  the read toll directly so it can be subtracted instead of assumed.
* Starting level matters and is a parameter (--n-prog). At a MID level
  (n-prog 3, ~235 LSB) every combo comes out negative, but that is ambiguous:
  the step scales with headroom (~477-V), so a charged cell falls whatever is
  done to it. --n-prog 0 starts from the reset state (~0 LSB) where there is
  nothing to lose, so any movement is the half-select's own direction. Run
  both: mid level says how badly a stored value is corrupted, reset level
  says which way each line actually pushes.
* Combos are visited in shuffled order each rep, so slow drift cannot alias
  onto the combo axis.
"""
import argparse
import csv
import datetime
import itertools
import re
import time

import numpy as np

N = 5
BL = 10
ONES = "1" * BL
ZEROS = "0" * BL
LINES = ["N1", "N2", "N3", "N4"]

# opcode and which stream side carries the ones
PRIMITIVE = {
    "N1": ("STOCHASTIC_POTENTIATION", "row"),
    "N2": ("STOCHASTIC_POTENTIATION", "col"),
    "N3": ("STOCHASTIC_DEPRESSION", "row"),
    "N4": ("STOCHASTIC_DEPRESSION", "col"),
}


def _drain(ard, timeout_s=20.0):
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
                rows.append(nums[1:11])
    return rows


def _send(ard, opcode, row_streams, col_streams):
    cmd = ",".join(["F", "1", "1", opcode, "N56", str(BL),
                    "15", "100", "100", "10", "20", "10"]
                   + row_streams + col_streams)
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    return _drain(ard)


def half_select(ard, line):
    """Drive exactly one line; the other side of the pair stays at zero."""
    opcode, side = PRIMITIVE[line]
    if side == "row":
        _send(ard, opcode, [ONES] * N, [ZEROS] * N)
    else:
        _send(ard, opcode, [ZEROS] * N, [ONES] * N)


def read_state(ard):
    """Read the whole array with exactly ONE array read.

    READ_ROW with row_num=5 reads all five rows once. A zero-pulse
    STOCHASTIC command would also return the state, but it performs a
    pre-read AND a post-read, i.e. two array reads -- double the read
    disturbance charged to every trial.
    """
    cmd = "F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    rows = _drain(ard)
    a = np.array(rows[:N], float)
    return a[:, :N] - a[:, N:]


def program_mid(ard, times):
    for _ in range(times):
        _send(ard, "STOCHASTIC_POTENTIATION", [ONES] * N, [ONES] * N)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--n-prog", type=int, default=3,
                    help="potentiation commands before the pair is applied. "
                         "3 = mid level (~235 LSB); 0 = start from the reset "
                         "state so the sign of each line's push is visible")
    ap.add_argument("--cycles", type=int, default=20,
                    help="how many times the pair is repeated before re-reading")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)

    combos = [(a, b) for a, b in itertools.product(LINES, LINES)]
    combos.append((None, None))            # read-only control
    recs = []

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        print(f"{len(combos)} conditions (16 ordered pairs + control) "
              f"x {args.reps} reps, {args.cycles} cycles each\n")

        for rep in range(args.reps):
            order = list(combos)
            rng.shuffle(order)
            for (a, b) in order:
                hard_reset(ard)
                time.sleep(0.15)
                program_mid(ard, args.n_prog)
                before = read_state(ard)

                for _ in range(args.cycles):
                    if a is not None:
                        half_select(ard, a)
                    if b is not None:
                        half_select(ard, b)

                after = read_state(ard)
                delta = after - before
                label = f"{a}+{b}" if a else "control"
                for i in range(N):
                    for j in range(N):
                        recs.append(dict(
                            rep=rep, first=a or "", second=b or "",
                            combo=label, cell_row=i + 1, cell_col=j + 1,
                            before=float(before[i, j]),
                            after=float(after[i, j]),
                            delta=float(delta[i, j])))
                print(f"  rep{rep} {label:>10}: "
                      f"{before.mean():6.1f} -> {after.mean():6.1f}  "
                      f"delta {delta.mean():+6.2f} LSB", flush=True)

    tag = "reset" if args.n_prog == 0 else f"prog{args.n_prog}"
    out = f"{stamp}_half_select_pairs_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    # ---- summary: subtract the read-only control ----
    ctrl = np.mean([r["delta"] for r in recs if r["combo"] == "control"])
    print(f"\ncontrol (reads only, no half-select): {ctrl:+.2f} LSB")
    print("\nnet disturbance after removing the read toll:")
    print(f"{'first':>6} {'second':>7} {'raw':>9} {'net':>9} {'sd':>7}")
    table = {}
    for a in LINES:
        for b in LINES:
            d = [r["delta"] for r in recs if r["combo"] == f"{a}+{b}"]
            if not d:
                continue
            raw = float(np.mean(d))
            table[(a, b)] = raw - ctrl
            print(f"{a:>6} {b:>7} {raw:>9.2f} {raw-ctrl:>9.2f} "
                  f"{np.std(d):>7.2f}")

    if table:
        print("\nnet disturbance matrix (rows = first pulse, cols = second):")
        print("        " + "".join(f"{b:>9}" for b in LINES))
        for a in LINES:
            print(f"{a:>6}  " +
                  "".join(f"{table.get((a,b), float('nan')):>9.2f}"
                          for b in LINES))
        print("\norder effect (upper minus lower triangle):")
        for a, b in itertools.combinations(LINES, 2):
            ab, ba = table.get((a, b)), table.get((b, a))
            if ab is not None and ba is not None:
                print(f"  {a}+{b} vs {b}+{a}: {ab:+.2f} vs {ba:+.2f} "
                      f"-> difference {ab-ba:+.2f} LSB")


if __name__ == "__main__":
    main()

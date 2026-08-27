#!/usr/bin/env python3
"""Ordered OVERLAP half-select pairs: N1->N4, N4->N1, N3->N2, N2->N3.

Why this exists
---------------
`seq_attractor_map.py` drives pairs with HS_SEQ, which ALTERNATES them:
N4 N1 N4 N1 ...  Such a run contains equal numbers of N4->N1 and N1->N4
transitions, so if the two orderings disturb in opposite directions they
cancel and the pair reads as inert. That is exactly what the four diagonal
pairs did (-4..-6 LSB, i.e. the control level) at every cycle count.

The HS_N14_Only / HS_N23_Only opcodes use a different primitive: an OVERLAP,
where one line is held while the other pulses inside it --

    N1 set -> [pre] -> N4 set -> [pulse_width] -> N4 clear -> [post] -> N1 clear

Every repetition has the same rise/fall order, so nothing cancels. The
reverse-order forms (HS_N41_Only / HS_N32_Only) were added to the firmware
for this run; they flip only the SODR/CODR order.

pre_enable_time and post_enable_time are the knobs that separate the two
lines in time. At 0 the two SODR writes land a few clocks apart and the
ordering vanishes, so they are kept at 100 us as in the earlier runs.

Read accounting -- identical for all conditions:
    READ_ROW (before) + opcode entry read + READ_ROW (after) = 3
and it does not grow with cycle count, so the cycle axis isolates the
half-select. The control runs HS_N14_Only with update_num=0: same command,
same entry read, zero pulses.

Starting states are the same five as seq_attractor_map.py so the two figures
can be read side by side.
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5
ONES = "1" * 10

# label -> opcode. The label names the rise order: N1+N4 means N1 rises first.
OPS = {
    "N1+N4": "HS_N14_Only",
    "N4+N1": "HS_N41_Only",
    "N3+N2": "HS_N23_Only",
    "N2+N3": "HS_N32_Only",
}

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


def hs_command(opcode, cycles):
    """row_num=col_num=5 selects every line, exposing all 25 cells.

    Side effect, harmless here: the opcode's entry read calls Read_scaling
    once with row_num=5, which raises all five WLs at once and rails the ADC
    at 1023. That reading is discarded -- before/after come from separate
    READ_ROW calls, which drive one row at a time and return normal values.
    Verified on the board: row_num=0 entry read gives ~370 LSB, row_num=5
    gives 1023. It still costs one read, but it costs the same one in every
    condition including the control, so subtracting the control removes it.
    """
    return ",".join(["F", "5", "5", opcode, "N56", "1", str(cycles),
                     "99999", "F", "15", "100", "100", "10", "20", "0", "10"])


def read_command():
    return "F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"


def reset_command():
    return ",".join(["F", "5", "5", "Reset", "N56", "10", "30", "10",
                     "F", "5", "100", "100", "10", "10", "20", "10"])


def _drain(ard, timeout_s, expect=None, quiet=1.2):
    """Read until EOD / 'operation end', or until `expect` blocks / silence.

    HS_N14_Only and HS_N23_Only end with a bare Serial.print("operation end")
    -- no newline -- so readline() cannot see it until something else arrives.
    The quiet window is what actually terminates those. HS_N41_Only and
    HS_N32_Only were added with println, so they return immediately.
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
    ap.add_argument("--pairs", default=",".join(OPS))
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
    labels = [p.strip() for p in args.pairs.split(",") if p.strip()]
    for lb in labels:
        if lb not in OPS:
            raise SystemExit(f"unknown pair {lb}; choose from {list(OPS)}")
    labels.append("control")

    recs = []
    trials = [(lb, s, c) for lb in labels for s in starts for c in cycles]
    print(f"{len(labels)} conditions x {len(starts)} starts x {len(cycles)} "
          f"cycles x {args.reps} reps = {len(trials)*args.reps} trials")

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")
        for rep in range(args.reps):
            order = list(trials)
            rng.shuffle(order)
            # stable-sort by start so identical starts run back to back, but
            # the shuffled order within each start block is preserved
            order.sort(key=lambda t: t[1])
            for label, st, cyc in order:
                set_start(ard, st)

                bb = send(ard, read_command())
                if len(bb) < N:
                    print(f"  [SKIP] before-read {len(bb)}/{N}")
                    continue
                before = to_diff(bb)

                if label == "control":
                    cmd = hs_command(OPS["N1+N4"], 0)
                else:
                    cmd = hs_command(OPS[label], cyc)
                send(ard, cmd, timeout_s=120)

                ab = send(ard, read_command())
                if len(ab) < N:
                    print(f"  [SKIP] after-read {len(ab)}/{N}")
                    continue
                after = to_diff(ab)

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

    out = f"{stamp}_overlap_seq_attractor.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    for label in labels:
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

    # The question this run exists to answer: do the two orderings of one
    # pair move charge in OPPOSITE directions? Compare them at max cycles,
    # control subtracted, averaged over starts.
    cmax = cycles[-1]
    ctrl = np.mean([r["delta"] for r in recs
                    if r["combo"] == "control" and r["cycles"] == cmax])
    print(f"\nordering asymmetry at {cmax} cycles "
          f"(control {ctrl:+.2f} LSB subtracted)")
    for a, b in (("N1+N4", "N4+N1"), ("N3+N2", "N2+N3")):
        va = [r["delta"] for r in recs
              if r["combo"] == a and r["cycles"] == cmax]
        vb = [r["delta"] for r in recs
              if r["combo"] == b and r["cycles"] == cmax]
        if not va or not vb:
            continue
        ma, mb = np.mean(va) - ctrl, np.mean(vb) - ctrl
        tag = "OPPOSITE SIGN" if ma * mb < 0 else "same sign"
        print(f"  {a} {ma:+8.2f}   {b} {mb:+8.2f}   "
              f"diff {ma-mb:+8.2f}   {tag}")
    print("\nrun  python plot_overlap_seq.py  to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

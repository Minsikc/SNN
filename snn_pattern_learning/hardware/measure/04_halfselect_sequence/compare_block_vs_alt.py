#!/usr/bin/env python3
"""Block-wise vs alternating half-select pairs, same pair and same pulse count.

HS_SEQ (added for this study) alternates the pair -- N2 N4 N2 N4 ... -- and
produced huge effects (N1+N2 -> +468, N3+N4 -> -428 LSB). The legacy
HS_N24/HS_N13 opcodes apply the same two lines BLOCK-wise -- N2 x update_num,
then N4 x update_num -- so with update_num=500 the alternating form has 999
line-to-line transitions while the block form has exactly 1. The legacy
opcodes also carry a delay(2) after every pulse, spacing them ~2 ms apart
versus ~120 us in HS_SEQ.

If the effect comes from one line's residue still being present when the
other fires, block-wise should show nothing. This runs both with a matched
pulse budget and reads only before and after.

Pulse budget: HS_N24 with update_num=U fires U pulses of N2 and U of N4.
HS_SEQ with update_num=U fires U of each as well, so passing the same U to
both gives an equal number of pulses per line -- only their ORDER and
spacing differ.
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5

# legacy block-wise opcodes and the (first, second) pair they encode
# Legacy opcodes fall into two families:
#   HS_N24 / HS_N42 / HS_N13  -- BLOCK-wise (line A x U, then line B x U),
#                                with delay(2) after every pulse
#   N12_cross / N34_cross     -- ALTERNATING inside one loop, no delay(2):
#                                structurally identical to HS_SEQ
# Only the second family covers the pairs that drive to a rail, so those are
# the ones that can actually discriminate the two schemes.
BLOCK = {
    "N2+N4": ("HS_N24", "N2", "N4"),
    "N4+N2": ("HS_N42", "N4", "N2"),
    "N1+N3": ("HS_N13", "N1", "N3"),
}
LEGACY_ALT = {
    "N1+N2": ("N12_cross", "N1", "N2"),
    "N3+N4": ("N34_cross", "N3", "N4"),
}
CODE = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}


def block_command(opcode, u):
    return ",".join(["F", "5", "5", opcode, "N56", "1", str(u), "99999", "F",
                     "15", "100", "100", "10", "20", "0", "10"])


def alt_command(first, second, u):
    pair = CODE[first] * 10 + CODE[second]
    return ",".join(["F", str(pair), "5", "HS_SEQ", "N56", "1", str(u),
                     "99999", "F", "15", "100", "100", "10", "20", "0", "10"])


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


def _drain(ard, timeout_s, expect=None, quiet=1.5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--update-num", type=int, default=500)
    ap.add_argument("--n-prog", type=int, default=0,
                    help="0 = start from reset, where the alternating form "
                         "drove the cell to a rail")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    U = args.update_num

    conds = []
    for label, (op, a, b) in BLOCK.items():
        conds.append((f"{label} block", block_command(op, U)))
        conds.append((f"{label} alt", alt_command(a, b, U)))
    for label, (op, a, b) in LEGACY_ALT.items():
        conds.append((f"{label} legacy", block_command(op, U)))
        conds.append((f"{label} alt", alt_command(a, b, U)))
    conds.append(("control", alt_command("N1", "N1", 0)))

    recs = []
    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")
        print(f"update_num={U} per line, {args.reps} reps, "
              f"start={'reset' if args.n_prog == 0 else 'mid'}\n")

        for rep in range(args.reps):
            order = list(conds)
            rng.shuffle(order)
            for label, cmd in order:
                send(ard, reset_command(), timeout_s=40, expect=31)
                time.sleep(0.15)
                for _ in range(args.n_prog):
                    send(ard, program_command())

                bb = send(ard, read_command())
                if len(bb) < N:
                    print(f"  [SKIP] {label}: before-read {len(bb)}/{N}")
                    continue
                before = to_diff(bb)

                t0 = time.time()
                send(ard, cmd, timeout_s=120)
                dt = time.time() - t0

                ab = send(ard, read_command())
                if len(ab) < N:
                    print(f"  [SKIP] {label}: after-read {len(ab)}/{N}")
                    continue
                after = to_diff(ab)

                delta = after - before
                for i in range(N):
                    for j in range(N):
                        recs.append(dict(rep=rep, cond=label, update_num=U,
                                         cell_row=i + 1, cell_col=j + 1,
                                         before=float(before[i, j]),
                                         after=float(after[i, j]),
                                         delta=float(delta[i, j])))
                print(f"  rep{rep} {label:>14}: {before.mean():7.1f} -> "
                      f"{after.mean():7.1f}  delta {delta.mean():+8.1f}  "
                      f"({dt:.1f}s)", flush=True)

    out = f"{stamp}_block_vs_alt.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    ctrl = np.mean([r["delta"] for r in recs if r["cond"] == "control"])
    print(f"\ncontrol {ctrl:+.2f} LSB")
    print(f"\n{'pair':>8} {'block':>12} {'alternating':>13} {'ratio':>9}")
    for label in list(BLOCK) + list(LEGACY_ALT):
        suffix = "block" if label in BLOCK else "legacy"
        bl = [r["delta"] for r in recs if r["cond"] == f"{label} {suffix}"]
        al = [r["delta"] for r in recs if r["cond"] == f"{label} alt"]
        if not bl or not al:
            continue
        b, a = np.mean(bl) - ctrl, np.mean(al) - ctrl
        ratio = f"{a/b:.1f}x" if abs(b) > 1 else "n/a"
        print(f"{label:>8} {b:>12.1f} {a:>13.1f} {ratio:>9}")
    print("\nBlock fires the same number of pulses per line but only ONE "
          "line-to-line\ntransition, versus 2U-1 when alternating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

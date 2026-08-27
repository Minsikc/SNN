#!/usr/bin/env python3
"""Order-controlled ABAB comparison of NORMAL vs DNO stochastic updates.

Why this exists: the sequential 3x3 sweeps ran DNO first and NORMAL second,
and the device drifts downward over a session, so order alone could inflate
the DNO advantage. Here the two modes alternate which one goes first on every
repeat, so any monotonic drift cancels in the paired difference.

Controls applied per trial:
  * identical pulse streams for both modes within a (grid point, repeat) pair
  * hard reset before every single trial (equalized start state)
  * ABAB order alternation keyed on the repeat index
  * raw per-cell (C, delta) written out for arbitrary re-analysis
"""
import argparse
import csv
import datetime
import re
import sys
import time

import numpy as np

N = 5
PULSE = {"width": 15, "pre": 100, "post": 100, "zero": 10}
READ = {"time": 20, "delay": 10}
MODES = {"normal": "STOCHASTIC_POTENTIATION",
         "dno": "STOCHASTIC_DNO_POTENTIATION"}


def streams(probs, bl, rng):
    return ["".join(map(str, (rng.random(bl) < p).astype(int))) for p in probs]


def coincidence(n1, n2):
    a = [np.array(list(s), int) for s in n1]
    b = [np.array(list(s), int) for s in n2]
    return np.array([[int((a[i] * b[j]).sum()) for j in range(N)]
                     for i in range(N)], float)


def send(ard, mode, n1, n2, bl):
    cmd = ",".join(["F", "1", "1", MODES[mode], "N56", str(bl),
                    str(PULSE["width"]), str(PULSE["pre"]), str(PULSE["post"]),
                    str(PULSE["zero"]), str(READ["time"]), str(READ["delay"])]
                   + n1 + n2)
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    rows, deadline = [], time.time() + 30
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
    if len(rows) < 2 * N:
        raise RuntimeError(f"expected {2*N} rows, got {len(rows)}")
    pre = np.array(rows[:N], float)
    post = np.array(rows[N:2 * N], float)
    return (pre[:, :N] - pre[:, N:]), (post[:, :N] - post[:, N:])


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
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bit-length", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.15)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    bl = args.bit_length
    cells = []

    pts = [(u, v) for u in args.levels for v in args.levels]
    print(f"ABAB: {len(pts)} grid points x {args.repeats} repeats x 2 modes "
          f"= {len(pts)*args.repeats*2} trials (+ resets)")

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")
        for gi, u in enumerate(args.levels):
            for gj, v in enumerate(args.levels):
                for rep in range(args.repeats):
                    n1 = streams(np.full(N, u), bl, rng)
                    n2 = streams(np.full(N, v), bl, rng)
                    C = coincidence(n1, n2)
                    # alternate which mode is measured first
                    order = ["normal", "dno"] if rep % 2 == 0 else ["dno", "normal"]
                    for mode in order:
                        hard_reset(ard)
                        time.sleep(args.settle)
                        pre, post = send(ard, mode, n1, n2, bl)
                        d = post - pre
                        for i in range(N):
                            for j in range(N):
                                cells.append(dict(
                                    mode=mode, u=u, v=v, gi=gi, gj=gj, rep=rep,
                                    order_pos=order.index(mode),
                                    cell_row=i + 1, cell_col=j + 1,
                                    C=int(C[i, j]), delta=float(d[i, j]),
                                    pre=float(pre[i, j]), post=float(post[i, j]),
                                ))
                        time.sleep(args.settle)
                    sub = [c for c in cells if c["u"] == u and c["v"] == v
                           and c["rep"] == rep and c["cell_col"] >= 3]
                    def rr(m):
                        cc = np.array([c["C"] for c in sub if c["mode"] == m])
                        dd = np.array([c["delta"] for c in sub if c["mode"] == m])
                        return (np.corrcoef(cc, dd)[0, 1]
                                if cc.std() > 1e-9 and dd.std() > 1e-9 else np.nan)
                    print(f"  u={u:.1f} v={v:.1f} rep{rep} "
                          f"[{order[0][:3]} first]  normal r={rr('normal'):+.3f}  "
                          f"dno r={rr('dno'):+.3f}")

    out = f"{stamp}_abab_dno_cells.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    print(f"\nsaved -> {out}  ({len(cells)} rows)")


if __name__ == "__main__":
    main()

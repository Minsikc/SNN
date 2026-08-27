#!/usr/bin/env python3
"""Does correcting for half-select exposure improve desired-vs-actual r?

The stochastic update correlates the measured delta against C, the number of
slots where a cell got BOTH its row and column pulse. But in the other slots
the cell still sees one line alone, and the half-select study
(2026-08-07) showed those single-line exposures are not inert once they
interleave: alternating N1/N2 drives a cell to +457 LSB and N3/N4 to -417.
Inside a stochastic command the lines DO overlap (SODR n1 -> pre -> SODR n2),
so the half-select contribution is plausibly even larger than measured there.

Per slot i, cell (r,c) falls in one of four cases; summing over bit_length:

    C  = sum_i  n1[r,i] * n2[c,i]         intended coincidence
    H1 = sum_i  n1[r,i] * (1 - n2[c,i])   row line alone
    H2 = sum_i  (1 - n1[r,i]) * n2[c,i]   column line alone

H1 and H2 are strongly anti-correlated with C by construction (a slot spent
half-selecting is a slot not spent coinciding), so if they carry any real
weight they bias the plain r(delta, C). The model fitted here is

    delta ~ g*C + a1*H1 + a2*H2 + b

and the question is whether the residual correlation improves over the plain
one-term fit. Existing sweep data cannot answer this -- it stored C but not
the streams -- so the streams are recorded here.
"""
import argparse
import csv
import datetime
import itertools
import re
import time

import numpy as np

N = 5


def streams(prob, bit_length, rng):
    return [(rng.random(bit_length) < prob).astype(int) for _ in range(N)]


def exposure(n1, n2):
    """Return (C, H1, H2) matrices for one command's pulse streams."""
    C = np.zeros((N, N))
    H1 = np.zeros((N, N))
    H2 = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            C[r, c] = np.sum(n1[r] * n2[c])
            H1[r, c] = np.sum(n1[r] * (1 - n2[c]))
            H2[r, c] = np.sum((1 - n1[r]) * n2[c])
    return C, H1, H2


def stoch_command(n1, n2, bit_length, mode="STOCHASTIC_POTENTIATION"):
    s1 = ["".join(map(str, v)) for v in n1]
    s2 = ["".join(map(str, v)) for v in n2]
    return ",".join(["F", "1", "1", mode, "N56", str(bit_length),
                     "15", "100", "100", "10", "20", "10"] + s1 + s2)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--bit-length", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    BL = args.bit_length
    recs = []

    pts = [(u, v) for u, v in itertools.product(args.levels, args.levels)]
    print(f"{len(pts)} grid points x {args.reps} reps = "
          f"{len(pts)*args.reps} trials")

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")
        for rep in range(args.reps):
            for (u, v) in pts:
                # reset per trial: the update is state-dependent, so every
                # trial must start from the same place
                send(ard, reset_command(), timeout_s=40, expect=31)
                time.sleep(0.15)

                n1 = streams(u, BL, rng)
                n2 = streams(v, BL, rng)
                C, H1, H2 = exposure(n1, n2)

                before = to_diff(send(ard, read_command()))
                if before.shape != (N, N):
                    continue
                send(ard, stoch_command(n1, n2, BL))
                after = to_diff(send(ard, read_command()))
                if after.shape != (N, N):
                    continue
                delta = after - before

                for r in range(N):
                    for c in range(N):
                        recs.append(dict(
                            rep=rep, u=u, v=v, cell_row=r + 1, cell_col=c + 1,
                            C=float(C[r, c]), H1=float(H1[r, c]),
                            H2=float(H2[r, c]),
                            before=float(before[r, c]),
                            after=float(after[r, c]),
                            delta=float(delta[r, c]),
                            n1="".join(map(str, n1[r])),
                            n2="".join(map(str, n2[c]))))
                print(f"  rep{rep} u={u:.1f} v={v:.1f}: C={C.mean():.2f} "
                      f"H1={H1.mean():.2f} H2={H2.mean():.2f} "
                      f"delta={delta.mean():+.1f}", flush=True)

    out = f"{stamp}_uv_halfselect_corrected.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    C = np.array([r["C"] for r in recs])
    H1 = np.array([r["H1"] for r in recs])
    H2 = np.array([r["H2"] for r in recs])
    d = np.array([r["delta"] for r in recs])

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    print(f"\npooled over {len(recs)} cell-observations")
    print(f"  plain      r(delta, C)          = {pearson(d, C):+.4f}")
    print(f"  exposure   corr(C, H1)          = {pearson(C, H1):+.4f}")
    print(f"             corr(C, H2)          = {pearson(C, H2):+.4f}")

    # least squares delta ~ g*C + a1*H1 + a2*H2 + b
    A = np.column_stack([C, H1, H2, np.ones_like(C)])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    print(f"\n  fit: delta = {coef[0]:+.3f}*C {coef[1]:+.3f}*H1 "
          f"{coef[2]:+.3f}*H2 {coef[3]:+.2f}")
    print(f"  r(delta, full model)             = {pearson(d, pred):+.4f}")

    # correct the measurement by removing the fitted half-select terms,
    # then re-correlate against C alone
    d_corr = d - coef[1] * H1 - coef[2] * H2
    print(f"  r(delta - halfselect, C)         = {pearson(d_corr, C):+.4f}")

    print("\nper grid point:")
    print(f"{'u':>5} {'v':>5} {'plain r':>9} {'corrected r':>12} {'gain':>8}")
    for (u, v) in pts:
        m = np.array([(r["u"] == u and r["v"] == v) for r in recs])
        if m.sum() < 5 or np.std(C[m]) < 1e-9:
            continue
        p0 = pearson(d[m], C[m])
        p1 = pearson(d_corr[m], C[m])
        print(f"{u:>5.1f} {v:>5.1f} {p0:>9.4f} {p1:>12.4f} {p1-p0:>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

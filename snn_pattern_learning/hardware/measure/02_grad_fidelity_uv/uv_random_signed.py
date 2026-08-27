#!/usr/bin/env python3
"""Desired vs actual update for fully random signed u, v in [-1, 1].

Earlier passes constrained the vectors in ways that made them easier than
what e-prop actually sends:

  uv_halfselect_corrected.py  every element the same POSITIVE magnitude
                              -> only the P quadrant, N3/N4 never fire
  signed_uv_exposure.py       signs mixed but magnitudes equal and the
                              negative COUNT fixed, so all four quadrants
                              were the same size on every trial

Here each element of u and v is drawn independently from U(-1, 1), so sign
AND magnitude vary per row/column. That means the four quadrants come out
different sizes on every trial, each row/column has its own pulse
probability, and the 25 desired updates span a continuous range rather than
a handful of levels -- the realistic case.

Quadrant decomposition matches hw_interface.accumulate_outer_product:
    P(err+, trc+)  P(err-, trc-)  D(err+, trc-)  D(err-, trc+)
with |u|, |v| used as the Bernoulli probabilities within each quadrant.

Ground truth is the signed realized coincidence count, so the comparison is
against what the device was actually told to do, not against the idealised
outer product (which would also fold in the stochastic sampling error).
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5


def streams(mask, probs, bit_length, rng):
    """One Bernoulli stream per row/col; masked-out lines stay all-zero.

    probs is per-element, so each line gets its own rate -- this is what
    makes the random-magnitude case different from the fixed-magnitude one.
    """
    out = []
    for k in range(N):
        if mask[k]:
            out.append((rng.random(bit_length) < probs[k]).astype(int))
        else:
            out.append(np.zeros(bit_length, int))
    return out


def stoch_command(n1, n2, bit_length, mode):
    s1 = ["".join(map(str, v)) for v in n1]
    s2 = ["".join(map(str, v)) for v in n2]
    return ",".join(["F", "1", "1", f"STOCHASTIC_{mode}", "N56",
                     str(bit_length), "15", "100", "100", "10", "20", "10"]
                    + s1 + s2)


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
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--bit-length", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplies |u|,|v| before use as pulse probability")
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    BL = args.bit_length
    recs = []

    print(f"{args.trials} trials, u,v ~ U(-1,1), bit_length {BL}, "
          f"scale {args.scale}")

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")

        for t in range(args.trials):
            u = rng.uniform(-1, 1, N)
            v = rng.uniform(-1, 1, N)
            pu = np.clip(np.abs(u) * args.scale, 0, 1)
            pv = np.clip(np.abs(v) * args.scale, 0, 1)

            send(ard, reset_command(), timeout_s=40, expect=31)
            time.sleep(0.15)
            bb = send(ard, read_command())
            if len(bb) < N:
                print("  [SKIP] before-read")
                continue
            before = to_diff(bb)

            Cp = np.zeros((N, N))
            Cd = np.zeros((N, N))
            H = {k: np.zeros((N, N)) for k in ("N1", "N2", "N3", "N4")}

            quads = [("POTENTIATION", u > 0, v > 0),
                     ("POTENTIATION", u < 0, v < 0),
                     ("DEPRESSION", u > 0, v < 0),
                     ("DEPRESSION", u < 0, v > 0)]
            for mode, rmask, cmask in quads:
                if not (rmask.any() and cmask.any()):
                    continue
                n1 = streams(rmask, pu, BL, rng)
                n2 = streams(cmask, pv, BL, rng)
                send(ard, stoch_command(n1, n2, BL, mode))
                rl = "N1" if mode == "POTENTIATION" else "N3"
                cl = "N2" if mode == "POTENTIATION" else "N4"
                for r in range(N):
                    for c in range(N):
                        co = int(np.sum(n1[r] * n2[c]))
                        if mode == "POTENTIATION":
                            Cp[r, c] += co
                        else:
                            Cd[r, c] += co
                        H[rl][r, c] += int(np.sum(n1[r] * (1 - n2[c])))
                        H[cl][r, c] += int(np.sum((1 - n1[r]) * n2[c]))

            ab = send(ard, read_command())
            if len(ab) < N:
                print("  [SKIP] after-read")
                continue
            after = to_diff(ab)
            delta = after - before

            sign = np.sign(np.outer(u, v))
            desired = sign * (Cp + Cd)
            ideal = np.outer(u, v) * BL      # what was asked for, pre-sampling

            for r in range(N):
                for c in range(N):
                    recs.append(dict(
                        trial=t, cell_row=r + 1, cell_col=c + 1,
                        u=float(u[r]), v=float(v[c]),
                        C_pot=float(Cp[r, c]), C_dep=float(Cd[r, c]),
                        H_N1=float(H["N1"][r, c]), H_N2=float(H["N2"][r, c]),
                        H_N3=float(H["N3"][r, c]), H_N4=float(H["N4"][r, c]),
                        desired=float(desired[r, c]),
                        ideal=float(ideal[r, c]),
                        before=float(before[r, c]), after=float(after[r, c]),
                        delta=float(delta[r, c])))

            rr = (np.corrcoef(desired.ravel(), delta.ravel())[0, 1]
                  if desired.std() > 1e-9 else float("nan"))
            print(f"  trial {t:3d}: quadrants "
                  f"{sum(1 for m,a,b in quads if a.any() and b.any())}  "
                  f"Cp={Cp.mean():5.2f} Cd={Cd.mean():5.2f}  "
                  f"delta={delta.mean():+7.1f}  r={rr:+.3f}", flush=True)

    out = f"{stamp}_uv_random_signed.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    col = lambda k: np.array([r[k] for r in recs])
    d, des, idl = col("delta"), col("desired"), col("ideal")
    print(f"\npooled over {len(recs)} cell-observations")
    print(f"  r(delta, desired realized) = {np.corrcoef(des, d)[0,1]:+.4f}")
    print(f"  r(delta, ideal u*v*L)      = {np.corrcoef(idl, d)[0,1]:+.4f}")

    g, b = np.polyfit(des, d, 1)
    res = d - (g * des + b)
    print(f"  gain {g:.2f} LSB/coincidence, residual sd {res.std():.2f} LSB "
          f"= {res.std()/abs(g):.2f} coincidences")

    A = np.column_stack([col("C_pot"), col("C_dep"), col("H_N1"),
                         col("H_N2"), col("H_N3"), col("H_N4"),
                         np.ones(len(recs))])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    print("\nper-exposure contribution (LSB):")
    for n_, c_ in zip(["C_pot", "C_dep", "H_N1", "H_N2", "H_N3", "H_N4",
                       "const"], coef):
        print(f"  {n_:>6} {c_:+9.3f}")
    print(f"  r(delta, 6-term) = {np.corrcoef(A @ coef, d)[0,1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Signed u,v updates: does half-select matter once D-path lines appear?

The P-only measurement found half-select worth only -23 LSB against +1101
of coincidence (2%), but that test could never produce N3 or N4 -- with
positive u and v the four-quadrant decomposition collapses to a single
POTENTIATION command. Signed vectors light up all four quadrants, and the
command boundaries then put N1 next to N3 and N2 next to N4, which is
exactly the adjacency that collapsed a cell to ~0 in the attractor study.

Four-quadrant decomposition (mirrors hw_interface.accumulate_outer_product):
    cmd1  POTENTIATION(err+, trc+)   N1/N2
    cmd2  POTENTIATION(err-, trc-)   N1/N2
    cmd3  DEPRESSION (err+, trc-)    N3/N4
    cmd4  DEPRESSION (err-, trc+)    N3/N4

Sign patterns use a FIXED number of negatives rather than random signs, so
every condition has the same quadrant structure and the conditions are
comparable; random signs would resize the quadrants on every trial.

    all_plus   err +++++  trc +++++   quadrants 25/0/0/0   (P only)
    err_neg    err +++--  trc +++++   quadrants 15/0/0/10
    trc_neg    err +++++  trc +++--   quadrants 15/0/10/0
    both_neg   err +++--  trc +++--   quadrants  9/4/6/6   (all four)

Ground truth is the SIGNED outer product: cells in P++/P-- should rise,
cells in D+-/D-+ should fall. Correlation is measured against that, not
against a bare coincidence count.
"""
import argparse
import csv
import datetime
import itertools
import re
import time

import numpy as np

N = 5

SIGN_PATTERNS = {
    "all_plus": (np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])),
    "err_neg": (np.array([1, 1, 1, -1, -1]), np.array([1, 1, 1, 1, 1])),
    "trc_neg": (np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, -1, -1])),
    "both_neg": (np.array([1, 1, 1, -1, -1]), np.array([1, 1, 1, -1, -1])),
}


def streams(mask, prob, bit_length, rng):
    """Bernoulli streams; rows/cols outside the mask stay all-zero."""
    out = []
    for k in range(N):
        if mask[k]:
            out.append((rng.random(bit_length) < prob).astype(int))
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


def quadrant_streams(es, ts, mag, bit_length, rng):
    """Build the four quadrant commands for one signed update.

    Returns a list of (mode, n1_streams, n2_streams, row_mask, col_mask).
    """
    quads = [
        ("POTENTIATION", es > 0, ts > 0),
        ("POTENTIATION", es < 0, ts < 0),
        ("DEPRESSION", es > 0, ts < 0),
        ("DEPRESSION", es < 0, ts > 0),
    ]
    out = []
    for mode, rmask, cmask in quads:
        if not (rmask.any() and cmask.any()):
            continue                     # empty quadrant: no command sent
        n1 = streams(rmask, mag, bit_length, rng)
        n2 = streams(cmask, mag, bit_length, rng)
        out.append((mode, n1, n2, rmask, cmask))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--mags", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--patterns", default=",".join(SIGN_PATTERNS))
    ap.add_argument("--bit-length", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    rng = np.random.default_rng(args.seed)
    BL = args.bit_length
    pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
    recs = []

    trials = [(p, m) for p in pats for m in args.mags]
    print(f"{len(pats)} sign patterns x {len(args.mags)} magnitudes x "
          f"{args.reps} reps = {len(trials)*args.reps} trials")

    with serial.Serial(args.port, 115200, timeout=5) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}\n")
        for rep in range(args.reps):
            order = list(trials)
            rng.shuffle(order)
            for pat, mag in order:
                es, ts = SIGN_PATTERNS[pat]
                send(ard, reset_command(), timeout_s=40, expect=31)
                time.sleep(0.15)

                bb = send(ard, read_command())
                if len(bb) < N:
                    print("  [SKIP] before-read")
                    continue
                before = to_diff(bb)

                # exposure counters, per cell
                Cp = np.zeros((N, N))    # coincidences on the P path
                Cd = np.zeros((N, N))    # coincidences on the D path
                H = {k: np.zeros((N, N)) for k in ("N1", "N2", "N3", "N4")}

                for mode, n1, n2, rmask, cmask in quadrant_streams(
                        es, ts, mag, BL, rng):
                    send(ard, stoch_command(n1, n2, BL, mode))
                    rowline = "N1" if mode == "POTENTIATION" else "N3"
                    colline = "N2" if mode == "POTENTIATION" else "N4"
                    for r in range(N):
                        for c in range(N):
                            co = int(np.sum(n1[r] * n2[c]))
                            h_r = int(np.sum(n1[r] * (1 - n2[c])))
                            h_c = int(np.sum((1 - n1[r]) * n2[c]))
                            if mode == "POTENTIATION":
                                Cp[r, c] += co
                            else:
                                Cd[r, c] += co
                            H[rowline][r, c] += h_r
                            H[colline][r, c] += h_c

                ab = send(ard, read_command())
                if len(ab) < N:
                    print("  [SKIP] after-read")
                    continue
                after = to_diff(ab)
                delta = after - before

                # signed ground truth: + where the quadrant potentiates
                desired = np.outer(es, ts) * (Cp + Cd)

                for r in range(N):
                    for c in range(N):
                        recs.append(dict(
                            rep=rep, pattern=pat, mag=mag,
                            cell_row=r + 1, cell_col=c + 1,
                            err_sign=int(es[r]), trc_sign=int(ts[c]),
                            C_pot=float(Cp[r, c]), C_dep=float(Cd[r, c]),
                            H_N1=float(H["N1"][r, c]),
                            H_N2=float(H["N2"][r, c]),
                            H_N3=float(H["N3"][r, c]),
                            H_N4=float(H["N4"][r, c]),
                            desired=float(desired[r, c]),
                            before=float(before[r, c]),
                            after=float(after[r, c]),
                            delta=float(delta[r, c])))

                rr = (np.corrcoef(desired.ravel(), delta.ravel())[0, 1]
                      if desired.std() > 1e-9 else float("nan"))
                print(f"  rep{rep} {pat:>9} mag={mag:.1f}: "
                      f"Cp={Cp.mean():5.2f} Cd={Cd.mean():5.2f}  "
                      f"delta={delta.mean():+7.1f}  r={rr:+.3f}", flush=True)

    out = f"{stamp}_signed_uv_exposure.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nsaved -> {out}  ({len(recs)} rows)")

    def col(k, mask=None):
        v = np.array([r[k] for r in recs])
        return v[mask] if mask is not None else v

    print(f"\n{'pattern':>9} {'mag':>5} {'r(delta,desired)':>18} "
          f"{'Cpot':>7} {'Cdep':>7} {'halfsel':>9}")
    for pat in pats:
        for mag in args.mags:
            m = np.array([(r["pattern"] == pat and r["mag"] == mag)
                          for r in recs])
            if m.sum() < 5:
                continue
            d, des = col("delta", m), col("desired", m)
            hs = sum(col(f"H_{k}", m).mean() for k in ("N1", "N2", "N3", "N4"))
            rr = (np.corrcoef(des, d)[0, 1] if des.std() > 1e-9
                  else float("nan"))
            print(f"{pat:>9} {mag:>5.1f} {rr:>18.4f} "
                  f"{col('C_pot', m).mean():>7.2f} "
                  f"{col('C_dep', m).mean():>7.2f} {hs:>9.2f}")

    # six-term model: do the D-path half-selects carry real weight?
    A = np.column_stack([col("C_pot"), col("C_dep"),
                         col("H_N1"), col("H_N2"), col("H_N3"), col("H_N4"),
                         np.ones(len(recs))])
    coef, *_ = np.linalg.lstsq(A, col("delta"), rcond=None)
    names = ["C_pot", "C_dep", "H_N1", "H_N2", "H_N3", "H_N4", "const"]
    print("\nfitted contribution per exposure (LSB):")
    for n_, c_ in zip(names, coef):
        print(f"  {n_:>6} {c_:+9.3f}")
    pred = A @ coef
    print(f"\n  r(delta, desired)     = "
          f"{np.corrcoef(col('desired'), col('delta'))[0,1]:+.4f}")
    print(f"  r(delta, 6-term model)= "
          f"{np.corrcoef(pred, col('delta'))[0,1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Sweep many (u, v) probability-vector combinations and measure how well the
physical conductance change tracks the intended outer product.

Extends the single-shot correlation cell in quick_measure.ipynb, which measured
one random (u, v) draw and reported Pearson r = 0.9001. That single draw is a
weak test: its coincidence matrix was 99.2% rank-1 (three of five columns were
all-zero), so most of the correlation came from "did this cell get any pulse at
all" rather than from true proportionality. Every unpulsed cell also drifted
negative (mean -26.5 LSB), so read-disturb was folded into the score.

This script fixes both problems:
  * structured (u, v) families that span rank-1 -> full-rank coincidence matrices
  * a zero-pulse control trial that measures drift so it can be subtracted
  * repeats per condition, so stochastic-sampling noise is separated from
    device error
  * reports r against the realized coincidence count C (what the device was
    actually told to do) and against the ideal outer product u (x) v
    (what the algorithm wanted), which are NOT the same thing

Usage:
    python uv_correlation_sweep.py --port COM4                  # full sweep
    python uv_correlation_sweep.py --port COM4 --family rank1   # one family
    python uv_correlation_sweep.py --dry-run                    # no hardware
"""

import argparse
import csv
import datetime
import itertools
import re
import sys
import time

import numpy as np

N = 5  # 5x5 crossbar

# Pulse/read timing taken from the quick_measure.ipynb correlation cell so the
# results are directly comparable to the recorded r = 0.9001 run.
PULSE_TIMING = {"width": 15, "pre": 100, "post": 100, "zero": 10}
READ_TIMING = {"time": 20, "delay": 10}
READ_FUNCTION = "N56"


# ----------------------------------------------------------------------------
# (u, v) families
# ----------------------------------------------------------------------------

def uv_families(rng, bit_length):
    """
    Yield (family_name, u, v) test vectors.

    The families are ordered from easiest to hardest for the device, and are
    chosen so the resulting coincidence matrices C = n1 (x) n2 span a range of
    effective ranks and dynamic ranges.
    """
    lo, hi = 0.15, 0.85  # keep away from saturation at 0 / 1

    # 1. rank1_sparse - reproduces the notebook's accidental regime:
    #    a few dominant entries, most columns near zero. Baseline for
    #    comparison against the recorded r = 0.9001.
    for k in range(3):
        u = rng.uniform(0.0, 1.0, N)
        v = rng.uniform(0.0, 1.0, N)
        v[rng.choice(N, 3, replace=False)] = 0.05  # force near-empty columns
        yield "rank1_sparse", u, v

    # 2. uniform_mid - all cells get comparable drive. Removes the
    #    "on vs off" shortcut entirely, so r here reflects true proportionality.
    for p in (0.3, 0.5, 0.7):
        yield "uniform_mid", np.full(N, p), np.full(N, p)

    # 3. graded - u ramps across rows, v ramps across columns. C becomes a
    #    smooth gradient covering the full overlap range; the single most
    #    informative family for proportionality.
    ramp = np.linspace(lo, hi, N)
    yield "graded", ramp.copy(), ramp.copy()
    yield "graded", ramp.copy(), ramp[::-1].copy()

    # 4. one_hot - isolates a single row/column pair. Directly exposes
    #    crosstalk: every cell except the target should stay flat.
    for i in (0, 2, 4):
        u = np.full(N, 0.02)
        v = np.full(N, 0.02)
        u[i] = 0.9
        v[i] = 0.9
        yield "one_hot", u, v

    # 5. full_random - unbiased sample of the space, the honest average case.
    for k in range(6):
        yield "full_random", rng.uniform(lo, hi, N), rng.uniform(lo, hi, N)

    # 6. extreme - probes saturation behaviour at both rails.
    yield "extreme", np.full(N, 0.95), np.full(N, 0.95)
    yield "extreme", np.full(N, 0.1), np.full(N, 0.1)


def generate_pulse_streams(prob_vector, bit_length, rng):
    """Convert a probability vector into stochastic bit-stream strings."""
    streams = []
    for p in prob_vector:
        bits = (rng.random(bit_length) < p).astype(int)
        streams.append("".join(map(str, bits)))
    return streams


def coincidence_matrix(n1_streams, n2_streams):
    """C[i, j] = number of timesteps where both streams are 1 (realized drive)."""
    C = np.zeros((N, N), dtype=int)
    for i in range(N):
        a = np.array(list(n1_streams[i]), dtype=int)
        for j in range(N):
            b = np.array(list(n2_streams[j]), dtype=int)
            C[i, j] = int((a * b).sum())
    return C


# ----------------------------------------------------------------------------
# Hardware I/O
# ----------------------------------------------------------------------------

def build_command(n1_streams, n2_streams, bit_length, mode="STOCHASTIC_POTENTIATION"):
    parts = [
        "F", "1", "1", mode,
        READ_FUNCTION,
        str(bit_length), str(PULSE_TIMING["width"]), str(PULSE_TIMING["pre"]),
        str(PULSE_TIMING["post"]), str(PULSE_TIMING["zero"]),
        str(READ_TIMING["time"]), str(READ_TIMING["delay"]),
    ] + n1_streams + n2_streams
    return ",".join(parts)


def send_and_read(arduino, command, timeout_s=30.0):
    """Send one command; return (pre 5x10, post 5x10) ADC arrays."""
    arduino.reset_input_buffer()
    arduino.write((command + "\n").encode("utf-8"))

    rows = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = arduino.readline().decode("utf-8", "ignore").strip()
        if not line:
            continue
        if "EOD" in line:
            break
        if "Row" in line:
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                rows.append(nums[1:11])
    if len(rows) < 2 * N:
        raise RuntimeError(f"expected {2*N} rows, got {len(rows)}")
    pre = np.array(rows[:N])
    post = np.array(rows[N:2 * N])
    return pre, post


def differential(mat):
    """N5 - N6 for a 5x10 ADC block."""
    return mat[:, :N].astype(float) - mat[:, N:].astype(float)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def safe_pearson(a, b):
    a = np.asarray(a, float).flatten()
    b = np.asarray(b, float).flatten()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def effective_rank(C):
    """Participation ratio of singular values: 1.0 = pure rank-1."""
    s = np.linalg.svd(np.asarray(C, float), compute_uv=False)
    if s.sum() < 1e-12:
        return 0.0
    p = s ** 2 / (s ** 2).sum()
    return float(1.0 / (p ** 2).sum())


def analyze(delta, C, uv_ideal, drift=None):
    """Compute the metric set for one trial."""
    d = delta.copy()
    out = {}
    out["r_raw_vs_C"] = safe_pearson(d, C)
    out["r_raw_vs_ideal"] = safe_pearson(d, uv_ideal)

    if drift is not None:
        d = d - drift
    out["r_corr_vs_C"] = safe_pearson(d, C)
    out["r_corr_vs_ideal"] = safe_pearson(d, uv_ideal)

    # Proportionality restricted to cells that actually received drive.
    # Guards against the "on vs off" shortcut inflating r.
    m = np.asarray(C) > 0
    out["n_active"] = int(m.sum())
    out["r_active_only"] = safe_pearson(d[m], np.asarray(C)[m]) if m.sum() >= 3 else float("nan")

    # Crosstalk: how much do untouched cells move relative to driven ones?
    if (~m).any() and m.any():
        out["crosstalk"] = float(np.abs(d[~m]).mean() / (np.abs(d[m]).mean() + 1e-9))
    else:
        out["crosstalk"] = float("nan")

    # Linear fit delta = gain * C + offset
    Cf = np.asarray(C, float).flatten()
    if Cf.std() > 1e-12:
        gain, offset = np.polyfit(Cf, d.flatten(), 1)
        out["gain_lsb_per_coincidence"] = float(gain)
        out["offset"] = float(offset)
    else:
        out["gain_lsb_per_coincidence"] = float("nan")
        out["offset"] = float("nan")

    out["eff_rank_C"] = effective_rank(C)
    out["delta_max_abs"] = float(np.abs(delta).max())
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--bit-length", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats per (u,v) condition, with fresh pulse sampling")
    ap.add_argument("--family", default=None, help="run only this family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.3,
                    help="seconds between trials")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate the device instead of using hardware")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = args.out or f"{stamp}_uv_corr_sweep.csv"

    conditions = [c for c in uv_families(rng, args.bit_length)
                  if args.family is None or c[0] == args.family]
    print(f"{len(conditions)} conditions x {args.repeats} repeats "
          f"= {len(conditions) * args.repeats} trials (+ drift controls)")

    arduino = None
    if not args.dry_run:
        import serial
        arduino = serial.Serial(args.port, args.baud, timeout=20)
        time.sleep(2)
        print(f"connected: {arduino.name}")

    sim_state = np.zeros((N, N))  # dry-run device state

    def run_trial(n1, n2):
        """Return (pre_diff, post_diff)."""
        if args.dry_run:
            C = coincidence_matrix(n1, n2)
            pre = sim_state.copy()
            # crude device model: saturating potentiation + drift + read noise
            gain = 12.0
            sim_state[:] = pre + gain * C * (1 - np.abs(pre) / 200.0) - 2.0
            post = sim_state.copy()
            noise = rng.normal(0, 3.0, (N, N))
            return pre + noise, post + rng.normal(0, 3.0, (N, N))
        cmd = build_command(n1, n2, args.bit_length)
        pre, post = send_and_read(arduino, cmd)
        return differential(pre), differential(post)

    rows = []
    pooled = []  # (C, drift-corrected delta) per trial, for cross-trial analysis
    try:
        for (family, u, v) in conditions:
            for rep in range(args.repeats):
                # --- drift control: all-zero streams, no coincidences ---
                # Measures baseline decay/read-disturb over one command so it
                # can be subtracted from the real trial.
                zeros = ["0" * args.bit_length] * N
                pre0, post0 = run_trial(zeros, zeros)
                drift = post0 - pre0
                time.sleep(args.settle)

                # --- actual trial ---
                n1 = generate_pulse_streams(u, args.bit_length, rng)
                n2 = generate_pulse_streams(v, args.bit_length, rng)
                C = coincidence_matrix(n1, n2)
                uv_ideal = np.outer(u, v) * args.bit_length

                pre, post = run_trial(n1, n2)
                delta = post - pre

                m = analyze(delta, C, uv_ideal, drift=drift)
                m.update(dict(
                    family=family, rep=rep,
                    u=";".join(f"{x:.3f}" for x in u),
                    v=";".join(f"{x:.3f}" for x in v),
                    drift_mean=float(drift.mean()),
                ))
                rows.append(m)
                pooled.append((C, delta - drift))

                print(f"{family:14s} rep{rep}  "
                      f"r(C)={m['r_raw_vs_C']:+.3f}  "
                      f"r_corr={m['r_corr_vs_C']:+.3f}  "
                      f"r_active={m['r_active_only']:+.3f}  "
                      f"rank={m['eff_rank_C']:.2f}  "
                      f"xtalk={m['crosstalk']:.2f}  "
                      f"gain={m['gain_lsb_per_coincidence']:+.1f}")

                time.sleep(args.settle)
    finally:
        if arduino is not None:
            arduino.close()

    if rows:
        keys = ["family", "rep", "r_raw_vs_C", "r_corr_vs_C", "r_raw_vs_ideal",
                "r_corr_vs_ideal", "r_active_only", "n_active", "crosstalk",
                "gain_lsb_per_coincidence", "offset", "eff_rank_C",
                "delta_max_abs", "drift_mean", "u", "v"]
        # keep the raw per-cell arrays for the pooled analysis
        for r, (C, d) in zip(rows, pooled):
            for idx in range(N * N):
                r[f"C{idx}"] = int(C.flatten()[idx])
                r[f"d{idx}"] = float(d.flatten()[idx])
        keys += [f"C{i}" for i in range(N * N)] + [f"d{i}" for i in range(N * N)]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved -> {out_path}")

        print("\n--- per-family summary (mean +/- std) ---")
        fams = sorted({r["family"] for r in rows})
        print(f"{'family':14s} {'r_corr_vs_C':>18s} {'r_active':>16s} "
              f"{'xtalk':>10s} {'eff_rank':>9s}")
        for fam in fams:
            sub = [r for r in rows if r["family"] == fam]
            def ms(k):
                a = np.array([r[k] for r in sub], float)
                a = a[~np.isnan(a)]
                return (a.mean(), a.std()) if a.size else (float("nan"), float("nan"))
            rc = ms("r_corr_vs_C"); ra = ms("r_active_only")
            xt = ms("crosstalk"); er = ms("eff_rank_C")
            print(f"{fam:14s} {rc[0]:+8.3f}+/-{rc[1]:5.3f} "
                  f"{ra[0]:+8.3f}+/-{ra[1]:5.3f} "
                  f"{xt[0]:9.2f} {er[0]:8.2f}")

        pooled_report(pooled)

    return 0


def pooled_report(pooled):
    """
    Per-cell regression across all trials.

    A single trial's coincidence matrix C = n1 (x) n2 is mathematically rank-1,
    so no one trial can test all 25 cells independently -- per-trial Pearson r
    mostly measures whether the row/column *pattern* came through. Pooling many
    trials gives each cell its own (C, delta) scatter, which is what actually
    answers "is every one of the 25 weights individually programmable?".
    """
    if len(pooled) < 5:
        print("\n(too few trials for pooled per-cell analysis)")
        return

    Cs = np.array([c.flatten() for c, _ in pooled], float)   # (trials, 25)
    Ds = np.array([d.flatten() for _, d in pooled], float)

    print(f"\n--- pooled per-cell regression over {len(pooled)} trials ---")
    print("cell    r      gain   n_used   verdict")
    gains, rr = [], []
    for k in range(N * N):
        c, d = Cs[:, k], Ds[:, k]
        if c.std() < 1e-9:
            print(f"({k//N+1},{k%N+1})   ---     ---       -    no drive variation")
            continue
        r = safe_pearson(c, d)
        g = float(np.polyfit(c, d, 1)[0])
        gains.append(g)
        rr.append(r)
        verdict = "ok" if (r > 0.7 and g > 0) else ("WEAK" if r > 0.4 else "BAD")
        print(f"({k//N+1},{k%N+1})  {r:+.3f}  {g:+7.2f}   {len(c):4d}    {verdict}")

    if gains:
        g = np.array(gains); r = np.array(rr)
        print(f"\nper-cell r    : mean {r.mean():+.3f}  min {r.min():+.3f}  "
              f"cells<0.7: {(r < 0.7).sum()}/{len(r)}")
        print(f"per-cell gain : mean {g.mean():+.2f}  spread(CV) "
              f"{g.std()/ (abs(g.mean()) + 1e-9):.2f}")
        print("\nGain spread is the number that matters for e-prop: it is the "
              "cell-to-cell\nmultiplicative mismatch the learning rule sees as "
              "a distorted gradient.")


if __name__ == "__main__":
    sys.exit(main())

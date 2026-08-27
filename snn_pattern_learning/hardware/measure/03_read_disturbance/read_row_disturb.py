#!/usr/bin/env python3
"""Isolate read disturbance to the row that is actually being read.

Every earlier measurement used the full-array read, which walks row_num 0..4
and therefore touches all 25 cells on every "read". That cannot answer
whether a cell decays because IT was read or because its four neighbours
were: the two are perfectly confounded. The new READ_ROW opcode reads one
row and nothing else, which breaks the confound.

Design
------
  charge all 25 cells
  probe row P once            -> baseline for every row
  repeat N times:  READ_ROW P    (only row P is exposed)
  probe all rows once         -> compare exposed vs unexposed

Row P accumulates N+2 reads; the other four accumulate exactly the 2 probe
reads. If disturbance is purely local, the unexposed rows should be flat
apart from leakage (~1.2 %, tau ~ 4 s) and the two probes.

The final probe is a full-array read, so it costs every row one read: that
is why the baseline probe is also a full read, keeping the probe cost equal
across rows and cancelling in the comparison.

Command formats
    single row : F,<row>,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10
    all rows   : F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10
"""
import argparse
import csv
import datetime
import re
import time

import numpy as np

N = 5
BL = 10
ONES = "1" * BL
# All 5 columns are included: column 2 was faulty in the 2026-08-06 sweeps
# but reads correctly now, so restricting to columns 3-5 is no longer
# warranted. Per-cell values are also written out, so any subset can be
# re-analysed without re-measuring.
GOOD_COLS = slice(0, 5)


def _readlines(ard, timeout_s=20.0):
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
                rows.append((nums[0], nums[1:11]))
    return rows


def read_row(ard, row):
    """READ_ROW: row 0-4 reads one row; row 5 reads all five."""
    cmd = f"F,{row},5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10"
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    out = _readlines(ard)
    got = {}
    for idx, vals in out:
        v = np.array(vals, float)
        got[idx] = v[:N] - v[N:]     # N5 - N6 for that row
    return got


def program_all(ard, times=1):
    """Full-array potentiation (also performs 2 full reads per command)."""
    for _ in range(times):
        cmd = ",".join(["F", "1", "1", "STOCHASTIC_POTENTIATION", "N56",
                        str(BL), "15", "100", "100", "10", "20", "10"]
                       + [ONES] * N + [ONES] * N)
        ard.reset_input_buffer()
        ard.write((cmd + "\n").encode())
        _readlines(ard)


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


def probe_all(ard):
    """One full-array read.

    Returns ({row: mean over columns}, {row: per-cell array}) so the summary
    stays readable while the raw per-cell values are still recorded.
    """
    got = read_row(ard, 5)
    return ({r: float(v[GOOD_COLS].mean()) for r, v in got.items()},
            {r: v.copy() for r, v in got.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--target-row", type=int, default=0)
    ap.add_argument("--n-reads", type=int, default=300,
                    help="READ_ROW repetitions on the target row")
    ap.add_argument("--n-prog", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--checkpoints", type=int, default=10,
                    help="how many times to sample the target row mid-run "
                         "(these cost the target row nothing extra -- the "
                         "READ_ROW itself returns the value)")
    args = ap.parse_args()

    import serial
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    recs = []
    P = args.target_row

    with serial.Serial(args.port, 115200, timeout=20) as ard:
        time.sleep(2)
        print(f"connected: {ard.name}")

        # sanity: confirm READ_ROW returns exactly one row
        chk = read_row(ard, P)
        if list(chk) != [P]:
            print(f"[WARN] READ_ROW {P} returned rows {list(chk)} "
                  f"(expected [{P}]) -- check the firmware branch")
        else:
            print(f"READ_ROW verified: single row {P} returned\n")

        for rep in range(args.reps):
            hard_reset(ard)
            time.sleep(0.2)
            program_all(ard, args.n_prog)

            base, base_cells = probe_all(ard)
            print(f"rep{rep} baseline: " +
                  "  ".join(f"r{r}={base[r]:6.1f}" for r in sorted(base)))
            for r, v in base.items():
                rec = dict(rep=rep, phase="baseline", target_row=P,
                           n_target_reads=0, row=r, level=v)
                rec.update({f"c{j+1}": float(base_cells[r][j])
                            for j in range(N)})
                recs.append(rec)

            every = max(1, args.n_reads // args.checkpoints)
            for k in range(1, args.n_reads + 1):
                got = read_row(ard, P)
                if P in got and (k % every == 0 or k == 1):
                    lv = float(got[P][GOOD_COLS].mean())
                    rec = dict(rep=rep, phase="during", target_row=P,
                               n_target_reads=k, row=P, level=lv)
                    rec.update({f"c{j+1}": float(got[P][j]) for j in range(N)})
                    recs.append(rec)

            final, final_cells = probe_all(ard)
            print(f"rep{rep} after {args.n_reads} reads of row {P}: " +
                  "  ".join(f"r{r}={final[r]:6.1f}" for r in sorted(final)))
            drops = {r: base[r] - final[r] for r in final}
            print(f"        drop:      " +
                  "  ".join(f"r{r}={drops[r]:6.1f}" for r in sorted(drops)))
            others = [drops[r] for r in drops if r != P]
            print(f"        target row {P}: {drops[P]:.1f} LSB | "
                  f"other rows mean: {np.mean(others):.1f} LSB\n")
            for r, v in final.items():
                rec = dict(rep=rep, phase="final", target_row=P,
                           n_target_reads=args.n_reads, row=r, level=v)
                rec.update({f"c{j+1}": float(final_cells[r][j])
                            for j in range(N)})
                recs.append(rec)

    out = f"{stamp}_read_row_disturb.csv"
    keys = []
    for r in recs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in recs])
    print(f"saved -> {out}  ({len(recs)} rows)")

    b = {}
    for r in recs:
        if r["phase"] in ("baseline", "final"):
            b.setdefault((r["phase"], r["row"]), []).append(r["level"])
    print("\n=== summary over reps ===")
    print(f"{'row':>4} {'baseline':>10} {'final':>10} {'drop':>8} {'exposed':>9}")
    tgt_drop, oth_drop = None, []
    for row in sorted({r["row"] for r in recs}):
        bl = np.mean(b[("baseline", row)])
        fl = np.mean(b[("final", row)])
        d = bl - fl
        exp = "YES" if row == P else "no"
        if row == P:
            tgt_drop = d
        else:
            oth_drop.append(d)
        print(f"{row:>4} {bl:>10.1f} {fl:>10.1f} {d:>8.1f} {exp:>9}")
    if tgt_drop is not None and oth_drop:
        print(f"\nexposed row lost {tgt_drop:.1f} LSB after "
              f"{args.n_reads} reads;")
        print(f"unexposed rows lost {np.mean(oth_drop):.1f} LSB "
              f"(2 probe reads + elapsed time only)")
        frac = np.mean(oth_drop) / tgt_drop if tgt_drop else float("nan")
        print(f"non-local share: {frac*100:.1f}%")
        print("  near 0% -> disturbance is local to the row being read")
        print("  large   -> reading any row disturbs the whole array")


if __name__ == "__main__":
    main()

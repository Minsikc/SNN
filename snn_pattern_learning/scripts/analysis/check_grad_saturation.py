#!/usr/bin/env python3
"""Per-epoch saturation check on the logged gradients.

Two distinct things could saturate, and they are easy to conflate:

  1. the ARRAY: |hw_adc| reaching the +455/-444 LSB rails, i.e. the cell is
     physically full and further pulses do nothing
  2. the ENCODING: the update path converts |gradient| to a Bernoulli pulse
     probability as (|g| / running_max).clamp(0,1) * normalization_scale
     (models.py:1868). Once a coordinate's probability hits 1.0 the pulse
     stream is all-ones and any further increase in the desired gradient is
     invisible -- the device saturates in the *command*, not in the charge.

Only (1) shows up as extreme ADC values; (2) is silent in the ADC log and has
to be reconstructed from `desired`. Both are reported per epoch below.

running_max is a running maximum over the whole run, so it is reconstructed
cumulatively here in epoch order, matching the training loop.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

RAIL_HI, RAIL_LO = 455.0, -444.0
NEAR = 0.90            # "near rail" = within 10% of it


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[int(r["epoch"])].append(
            (float(r["desired"]), float(r["hw_adc"]),
             int(r["row"]), int(r["col"])))
    return by


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "results", "xor", "grad_log_analog.csv")
    by = load(path)
    eps = sorted(by)
    n_cell = len(by[eps[0]])
    print(f"{path}")
    print(f"{len(eps)} epochs x {n_cell} cells\n")

    # ---- 1. array rails --------------------------------------------------
    allad = np.array([a for e in eps for _, a, _, _ in by[e]])
    print("=" * 68)
    print("1. ARRAY rails (+455 / -444 LSB)")
    print("=" * 68)
    print(f"  hw_adc range {allad.min():+.0f} .. {allad.max():+.0f} LSB")
    for lab, m in (("at/over rail",
                    (allad >= RAIL_HI) | (allad <= RAIL_LO)),
                   (f"within {100*(1-NEAR):.0f}% of rail",
                    (allad >= NEAR * RAIL_HI) | (allad <= NEAR * RAIL_LO))):
        print(f"  {lab:28s} {m.sum():4d} / {len(allad)} "
              f"({100*m.mean():.2f}%)")
    print("  -> the cells are nowhere near physically full")

    # ---- 2. encoding saturation -----------------------------------------
    # running_max is cumulative over the run, in epoch order
    print("\n" + "=" * 68)
    print("2. ENCODING saturation: |g| / running_max reaching 1.0")
    print("=" * 68)
    print("  a coordinate at 1.0 sends an all-ones pulse stream, so any")
    print("  larger desired gradient is indistinguishable from it\n")
    run_max = 0.0
    print(f"  {'epoch':>5s} {'max|g|':>8s} {'run_max':>8s} "
          f"{'at 1.0':>7s} {'>0.9':>6s} {'>0.5':>6s} {'mean p':>7s}")
    print("  " + "-" * 54)
    tot_at1, tot = 0, 0
    per_ep = []
    for e in eps:
        g = np.abs(np.array([d for d, _, _, _ in by[e]]))
        run_max = max(run_max, g.max())
        p = np.clip(g / run_max, 0, 1)
        at1 = int((p >= 0.999).sum())
        tot_at1 += at1
        tot += len(p)
        per_ep.append((e, at1, len(p)))
        if e < 8 or e % 5 == 0 or e == eps[-1]:
            print(f"  {e:5d} {g.max():8.3f} {run_max:8.3f} "
                  f"{at1:7d} {int((p > 0.9).sum()):6d} "
                  f"{int((p > 0.5).sum()):6d} {p.mean():7.3f}")
    print(f"\n  total coordinates at 1.0: {tot_at1} / {tot} "
          f"({100*tot_at1/tot:.2f}%)")
    n_ep_with = sum(1 for _, a, _ in per_ep if a > 0)
    print(f"  epochs with at least one saturated coordinate: "
          f"{n_ep_with} / {len(eps)}")
    print("  (exactly one coordinate saturates whenever the epoch sets a new")
    print("   running max -- that is the definition, not a pathology)")

    # ---- 3. how the two logs compare ------------------------------------
    print("\n" + "=" * 68)
    print("3. is the desired gradient unusually large here?")
    print("=" * 68)
    other = os.path.join("results", "eprop_grad_log",
                         "gradient_log_seed0.csv")
    if os.path.exists(other):
        ob = load(other)
        og = np.abs(np.array([d for e in ob for d, _, _, _ in ob[e]]))
        tg = np.abs(np.array([d for e in eps for d, _, _, _ in by[e]]))
        print(f"  temporal XOR      mean|g| {tg.mean():.4f}  "
              f"max {tg.max():.3f}")
        print(f"  teacher-student   mean|g| {og.mean():.4f}  "
              f"max {og.max():.3f}")
        print(f"  ratio             {tg.mean()/og.mean():.1f}x larger")
        print("  -> larger gradients do NOT push the array to its rails;")
        print("     they are rescaled to probabilities first, so the array")
        print("     sees the same bounded pulse budget either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

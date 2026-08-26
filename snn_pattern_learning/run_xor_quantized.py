#!/usr/bin/env python3
"""Digital e-prop on temporal XOR, with the update quantised to BL pulses.

The analog XOR run plateaued at 0.75 while digital reached 1.00. Two very
different explanations were still open:

  (a) the DEVICE -- nonlinearity, half-select, read disturbance, cell spread
  (b) the ENCODING -- gradients become Bernoulli pulse streams of length BL,
      so an update is quantised to multiples of 1/BL and any coordinate whose
      probability is far below 1/BL rounds to zero on most epochs

This run isolates (b). Everything stays in software -- the exact same digital
pipeline that reaches 1.00 -- except the mock interface now draws the same
bit streams the device would. If BL=10 alone drops accuracy to ~0.75, the
encoding is sufficient to explain the analog result and the device physics
need not be invoked. If it stays at 1.00, the device is implicated.

BL is swept so the trend is visible rather than resting on one point, and
several seeds are run because a single XOR run is noisy (accuracy moves in
steps of 0.25 over 4 patterns).
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)

from run_xor import run  # noqa: E402

# the operating point used by the existing digital / analog XOR runs
BASE = dict(epochs=60, lr=0.15, freeze_hidden=True, repeats=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+",
                    default=[0, 10, 20, 50, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/xor/quantized_sweep.json")
    args = ap.parse_args()

    rows = []
    print(f"digital e-prop, temporal XOR, {args.epochs} epochs, "
          f"lr {BASE['lr']}")
    print("BL=0 means the exact real-valued product (no quantisation)\n")
    print(f"{'BL':>5s} {'seed':>5s} {'best_acc':>9s} {'final_acc':>10s} "
          f"{'best_loss':>10s} {'ep@1.0':>7s}")
    print("-" * 52)
    for bl in args.bits:
        for sd in args.seeds:
            r = run("digital", args.epochs, BASE["lr"], seed=sd,
                    verbose=False, freeze_hidden=BASE["freeze_hidden"],
                    repeats=BASE["repeats"], quantize_bits=bl)
            n1 = int(sum(1 for a in r["accs"] if a == 1.0))
            rows.append(dict(bl=bl, seed=sd, best_acc=r["best_acc"],
                             final_acc=r["final_acc"],
                             best_loss=r["best_loss"], n_epochs_at_1=n1,
                             accs=r["accs"]))
            print(f"{bl:5d} {sd:5d} {r['best_acc']:9.2f} "
                  f"{r['final_acc']:10.2f} {r['best_loss']:10.3f} "
                  f"{n1:7d}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"))
    print(f"\nsaved -> {args.out}")

    print(f"\n{'BL':>5s} {'best_acc mean':>14s} {'final_acc mean':>15s} "
          f"{'n seeds at 1.0':>15s}")
    print("-" * 52)
    for bl in args.bits:
        s = [r for r in rows if r["bl"] == bl]
        ba = np.mean([r["best_acc"] for r in s])
        fa = np.mean([r["final_acc"] for r in s])
        n1 = sum(1 for r in s if r["best_acc"] == 1.0)
        print(f"{bl:5d} {ba:14.3f} {fa:15.3f} {n1:11d}/{len(s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

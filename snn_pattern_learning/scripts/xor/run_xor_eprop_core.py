#!/usr/bin/env python3
"""Temporal XOR on the e-prop core, one row per (algorithm, neuron, chain, seed).

    python scripts/xor/run_xor_eprop_core.py                       # default matrix, seeds 0-4
    python scripts/xor/run_xor_eprop_core.py --conditions analog --port COM4 --seeds 0

Rows (default): bptt | digital | digital_mock | frozen  x  LIF / ALIF(beta 0.5)
x  eligibility full / truncated (ALIF only)  x  reservoir / joint.
Writes results/xor_core/xor_core.json and prints a summary table
(best acc, first epoch with acc 1.00, #perfect epochs, mean acc, best loss).
"""
import argparse
import itertools
import json
import logging
import os
import sys
import time

logging.disable(logging.INFO)          # silence the per-epoch mock-array INFO lines

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root

from eprop import GradChainConfig, HardwareReadoutConfig      # noqa: E402
from eprop.xor import run_xor, xor_neuron                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["bptt", "digital", "digital_mock", "frozen"])
    ap.add_argument("--neurons", nargs="+", default=["lif", "alif"])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--eligibility", nargs="+", default=["full", "truncated"])
    ap.add_argument("--modes", nargs="+", default=["reservoir", "joint"], choices=["reservoir", "joint"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.15, help="e-prop conditions")
    ap.add_argument("--lr-bptt", type=float, default=0.05)
    ap.add_argument("--bptt-halfwidth", type=float, default=0.5)
    ap.add_argument("--rec-orientation", default="legacy", choices=["legacy", "corrected"])
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--out", default="results/xor_core/xor_core.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = []
    combos = list(itertools.product(args.conditions, args.neurons, args.eligibility, args.modes, args.seeds))
    t0 = time.time()
    for k, (cond, kind, elig, mode, seed) in enumerate(combos, 1):
        if kind == "lif" and elig != args.eligibility[0]:
            continue                                      # eligibility only matters for ALIF
        if cond == "bptt" and (elig != args.eligibility[0] or mode != args.modes[0]):
            continue                                      # bptt: no eligibility / reservoir concept
        if cond == "frozen" and elig != args.eligibility[0]:
            continue
        neuron = xor_neuron(kind, beta=args.beta, rho=args.rho)
        chain = GradChainConfig(eligibility=elig, rec_grad_orientation=args.rec_orientation)
        hw = HardwareReadoutConfig(enabled=True, use_mock=(cond != "analog"), serial_port=args.port,
                                   normalization_scale=0.7, no_read_updates=True)
        lr = args.lr_bptt if cond == "bptt" else args.lr
        r = run_xor(cond, args.epochs, lr, seed=seed, neuron=neuron, chain=chain, hw=hw,
                    reservoir=(mode == "reservoir"), bptt_halfwidth=args.bptt_halfwidth,
                    verbose=args.verbose, record=True, curves_path=args.out, note="run_xor_eprop_core")
        r.update(kind=kind, eligibility=elig if kind == "alif" else "-", mode=mode if cond != "bptt" else "-")
        rows.append(r)
        fp = r["first_perfect_epoch"]
        print(f"[{k:3d}/{len(combos)}] {cond:>12} {kind:>4} {r['eligibility']:>9} {r['mode']:>9} seed={seed}"
              f"  best_acc={r['best_acc']:.2f} first1.00={fp if fp else '-':>3} perfect={r['n_perfect_epochs']:2d}"
              f" mean_acc={r['mean_acc']:.2f} best_loss={r['best_loss']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=rows), open(args.out, "w"))
    print(f"\nsaved -> {args.out}\n")

    print(f"{'condition':>12} {'neuron':>6} {'elig':>9} {'mode':>9} {'n':>2}  solved  mean_best_acc  mean_first1.00  mean_perfect_ep  mean_acc  mean_best_loss")
    keys = sorted({(r["condition"], r["kind"], r["eligibility"], r["mode"]) for r in rows},
                  key=lambda k: (args.conditions.index(k[0]), k[1], k[2], k[3]))
    for key in keys:
        g = [r for r in rows if (r["condition"], r["kind"], r["eligibility"], r["mode"]) == key]
        solved = sum(r["best_acc"] >= 1.0 for r in g)
        fp = [r["first_perfect_epoch"] for r in g if r["first_perfect_epoch"]]
        print(f"{key[0]:>12} {key[1]:>6} {key[2]:>9} {key[3]:>9} {len(g):>2}   {solved}/{len(g)}"
              f"   {sum(r['best_acc'] for r in g)/len(g):12.2f}"
              f"   {(sum(fp)/len(fp)) if fp else float('nan'):13.1f}"
              f"   {sum(r['n_perfect_epochs'] for r in g)/len(g):14.1f}"
              f"   {sum(r['mean_acc'] for r in g)/len(g):7.2f}"
              f"   {sum(r['best_loss'] for r in g)/len(g):13.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extension experiment: ALIF neurons in e-prop, full vs truncated eligibility chain.

Runs the teacher--student task with an ALIF teacher/student for every
combination of condition x eligibility x beta x seed and writes one JSON with
the per-epoch curves, so the analysis is reproducible from the file alone.

    python scripts/teacher_student/run_alif_conditions.py --betas 0 0.5 1.0 --seeds 0 1 2 3 4
    python scripts/teacher_student/run_alif_conditions.py --conditions analog --mock       # mock array
    python scripts/teacher_student/run_alif_conditions.py --conditions analog --port COM4  # real array

Conditions: bptt (pure autograd, surrogate selectable), digital, frozen_wout,
analog (readout gradient on the 6T1C array / mock). ``--legacy-bptt`` reproduces
the paper-v0.1 "bptt" behaviour (e-prop + autograd gradients summed; see
tests/test_eprop_equivalence.py).
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root

from eprop import (GradChainConfig, HardwareReadoutConfig, NeuronConfig, TaskConfig,  # noqa: E402
                   run_condition)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["bptt", "digital", "frozen_wout"])
    ap.add_argument("--eligibility", nargs="+", default=["full", "truncated"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-in", type=int, default=10)
    ap.add_argument("--n-hidden", type=int, default=4)
    ap.add_argument("--n-out", type=int, default=4)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--w-scale", type=float, default=1.5)
    ap.add_argument("--ds-seed", type=int, default=25)
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--pd-gamma", type=float, default=None)
    ap.add_argument("--bptt-surrogate", default="boxcar", choices=["boxcar", "triangle"])
    ap.add_argument("--legacy-bptt", action="store_true", help="bptt = autograd + e-prop (paper v0.1)")
    ap.add_argument("--rec-orientation", default="legacy", choices=["legacy", "corrected"])
    ap.add_argument("--mock", action="store_true", help="analog condition on the mock array")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--no-nr", action="store_true", help="use read-wrapped update opcodes")
    ap.add_argument("--dno", action="store_true")
    ap.add_argument("--out", default="results/eprop_alif/alif_conditions.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    task = TaskConfig(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out, T=args.T,
                      w_scale=args.w_scale, ds_seed=args.ds_seed)
    hw = HardwareReadoutConfig(enabled=True, use_mock=args.mock, serial_port=args.port,
                               no_read_updates=not args.no_nr, dno=args.dno,
                               grad_log_path=None)
    results = []
    t0 = time.time()
    combos = list(itertools.product(args.betas, args.eligibility, args.conditions, args.seeds))
    for i, (beta, elig, cond, seed) in enumerate(combos, 1):
        if beta == 0.0 and elig == "truncated":
            continue                                  # identical to full for LIF
        neuron = NeuronConfig(kind="alif" if beta > 0 else "lif", beta=beta, rho=args.rho,
                              tau=args.tau, thresh=args.thresh, pd_gamma=args.pd_gamma)
        chain = GradChainConfig(eligibility=elig, rec_grad_orientation=args.rec_orientation,
                                bptt_surrogate=args.bptt_surrogate, bptt_add_eprop=args.legacy_bptt)
        if cond == "analog":
            hw.grad_log_path = os.path.splitext(args.out)[0] + f"_grad_b{beta}_{elig}_s{seed}.csv"
        r = run_condition(cond, args.epochs, args.lr, seed=seed, task=task, neuron=neuron,
                          chain=chain, hw=hw if cond == "analog" else None, verbose=args.verbose,
                          record=True, curves_path=args.out, note="run_alif_conditions")
        r.update(beta=beta, eligibility=elig, seed=seed)
        results.append(r)
        print(f"[{i:3d}/{len(combos)}] beta={beta:<4} {elig:<9} {cond:>11} seed={seed}"
              f"  best={r['best_loss']:.4f}@{r['best_epoch']+1:<3d} vrd={r['vrds'][-1]:.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=results), open(args.out, "w"))
    print(f"\nsaved -> {args.out}")

    print(f"\n{'beta':>5} {'elig':>9} {'condition':>11}  mean_best  median   n")
    for beta in args.betas:
        for elig in args.eligibility:
            for cond in args.conditions:
                v = sorted(r["best_loss"] for r in results
                           if r["beta"] == beta and r["eligibility"] == elig and r["condition"] == cond)
                if v:
                    print(f"{beta:>5} {elig:>9} {cond:>11}  {sum(v)/len(v):9.4f}  {v[len(v)//2]:.4f}  {len(v)}")


if __name__ == "__main__":
    main()

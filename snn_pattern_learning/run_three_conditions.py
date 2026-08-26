#!/usr/bin/env python3
"""Digital and frozen-W_out runs to compare against the analog hardware run.

Three conditions on the same single-sequence teacher task:

  bptt          exact gradients through time -- the achievable ceiling
  digital       e-prop rule, gradients computed in software
  frozen_wout   fc1/recurrent learn; W_out is held at its init values
  analog        W_out gradient comes from the memristor crossbar
                (run separately by main_unified.py on the hardware)

BPTT bounds what the architecture can do at all: on this single-sequence
task it reaches loss 0, so any shortfall in the other three is the learning
rule or the hardware, not the task.

The frozen condition is the control that says how much of any learning is
attributable to the output layer at all -- if it matches the others, the
hidden layers alone explain the result and the crossbar is not doing the
work.

Loss curves and the final output spikes are saved so the analog run can be
overlaid on identical axes.
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from datasets.customdatasets import CustomSpikeDataset_Teacher
from models.models import Basic_RSNN_eprop_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance


def build_task(n_seq=1, T=20, thresh=0.2, tau=0.6,
               n_in=10, n_hidden=5, n_out=5, w_scale=1.5, ds_seed=10):
    # w_scale WARNING (found 2026-08-25): this script has always used 1.5,
    # but the hardware pipeline (base_experiment) leaves the dataset default
    # of 1.0 -> DIFFERENT targets. To compare against a main_unified run,
    # pass w_scale=1.0.
    # teacher_thresh MUST equal the student's init_thresh. They were hardcoded
    # to 0.3 and 0.2 here, which makes the target unrealizable: planting the
    # teacher's own weights into the student gives loss 1.102 instead of 0, so
    # every condition sat on that floor and the numbers meant nothing.
    # Verified after matching: planted teacher weights give loss 0.000000 at
    # thresh 0.2 / 0.3 / 0.4.
    # Teacher and student share the same architecture, so the task stays
    # realizable at any size -- the teacher IS a student-shaped network.
    ds = CustomSpikeDataset_Teacher(
        num_samples=n_seq, sequence_length=T, input_size=n_in,
        output_size=n_out, hidden_size=n_hidden, spike_prob=0.2,
        teacher_thresh=thresh, teacher_tau=tau, w_scale=w_scale, seed=ds_seed)
    return ds.data, ds.targets


def run(condition, epochs, lr, seed=0, n_seq=1, T=20, thresh=0.2, tau=0.6,
        n_in=10, n_hidden=5, n_out=5, w_scale=1.5, ds_seed=10):
    x, tgt = build_task(n_seq, T, thresh, tau, n_in, n_hidden, n_out, w_scale,
                        ds_seed)
    torch.manual_seed(seed)
    m = Basic_RSNN_eprop_forward(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                 recurrent=True, init_thresh=thresh,
                                 init_tau=tau)
    kernel = create_exponential_kernel(3, 2.0)

    if condition == "bptt":
        # e-prop fills .grad during forward(); switching these off restores
        # ordinary autograd so loss.backward() gives the exact gradient
        m.custom_grad = False
        m.custom_grad_forward = False

    if condition == "frozen_wout":
        # Freeze by excluding W_out from the optimizer AND zeroing whatever
        # the forward pass writes into its .grad, so neither path can move it.
        params = [p for n, p in m.named_parameters() if not n.startswith("out.")]
    else:
        params = list(m.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    losses, vrds = [], []
    best = (float("inf"), None)
    for ep in range(epochs):
        opt.zero_grad()
        out = m(x, tgt, training=True)
        co = apply_convolution(out, kernel, 3)
        ct = apply_convolution(tgt, kernel, 3)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1])
        if condition == "bptt":
            loss.backward()
        if condition == "frozen_wout" and m.out.weight.grad is not None:
            m.out.weight.grad.zero_()
        opt.step()

        with torch.no_grad():
            o = m(x, tgt, training=False)
            l = mse_acc_loss_over_time(apply_convolution(o, kernel, 3), ct,
                                       o.shape[1]).item()
            v = float(van_rossum_distance(o, tgt, tau=5.0).mean())
        losses.append(l)
        vrds.append(v)
        if l < best[0]:
            best = (l, o.detach().clone())

    return dict(condition=condition, losses=losses, vrds=vrds,
                best_loss=best[0],
                final_spikes=best[1][0].numpy().tolist(),
                target_spikes=tgt[0].numpy().tolist(),
                input_spikes=x[0].numpy().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--thresh", type=float, default=0.2,
                    help="shared teacher/student threshold; the hardware "
                         "config (eprop_grad_log.yaml) uses 0.2")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-in", type=int, default=10)
    ap.add_argument("--n-hidden", type=int, default=5)
    ap.add_argument("--n-out", type=int, default=5)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--conditions", default="bptt,digital,frozen_wout")
    ap.add_argument("--out", default="results/eprop_grad_log/three_conditions.json")
    args = ap.parse_args()

    results = {}
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        r = run(cond, args.epochs, args.lr, seed=args.seed,
                thresh=args.thresh, tau=args.tau, T=args.T,
                n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
        results[cond] = r
        print(f"{cond:>12}: best loss {r['best_loss']:.4f}  "
              f"final VRD {r['vrds'][-1]:.3f}")

    json.dump(results, open(args.out, "w"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

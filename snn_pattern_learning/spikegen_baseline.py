#!/usr/bin/env python3
"""Software baselines for the spike-generation task on the 5x5-constrained model.

Establishes what loss/VRD is ACHIEVABLE before any analog run, so hardware
results can be read against a real ceiling instead of an unknown one.

Two things were wrong with the earlier setup and both are fixed here:
  * targets came from torch.randperm -- statistically independent of the
    input, so no weight setting could reproduce them (BPTT floored at 0.579
    vs 0.013 on teacher targets, measured 2026-08-06)
  * the dataset was unseeded, so HW and SW runs saw different task instances

Conditions swept: number of sequences (the factor that decided whether the
reference repo reached VRD=0) and learning rule (BPTT upper bound vs the
e-prop rule the hardware actually implements).
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


def evaluate(out, tgt, kernel, ksize):
    co = apply_convolution(out, kernel, ksize)
    ct = apply_convolution(tgt, kernel, ksize)
    loss = mse_acc_loss_over_time(co, ct, out.shape[1]).item()
    vrd = float(van_rossum_distance(out, tgt, tau=5.0).mean())
    match = float(((out > 0.5) == (tgt > 0.5)).float().mean())
    return loss, vrd, match


def run(n_seq, rule, epochs, lr, seed=0, T=20, n_in=10, H=5, n_out=5):
    ds = CustomSpikeDataset_Teacher(
        num_samples=n_seq, sequence_length=T, input_size=n_in,
        output_size=n_out, hidden_size=H, spike_prob=0.2,
        teacher_thresh=0.3, w_scale=1.5, seed=10)
    x, tgt = ds.data, ds.targets

    torch.manual_seed(seed)
    m = Basic_RSNN_eprop_forward(n_in=n_in, n_hidden=H, n_out=n_out,
                                 recurrent=True, init_thresh=0.2)
    kernel = create_exponential_kernel(3, 2.0)

    if rule == "bptt":
        m.custom_grad = False
        m.custom_grad_forward = False
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    best = None
    for ep in range(epochs):
        opt.zero_grad()
        out = m(x, tgt, training=True)
        co = apply_convolution(out, kernel, 3)
        ct = apply_convolution(tgt, kernel, 3)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1])
        if rule == "bptt":
            loss.backward()
        opt.step()          # e-prop path: grads already filled by forward()
        with torch.no_grad():
            stats = evaluate(m(x, tgt, training=False), tgt, kernel, 3)
        if best is None or stats[0] < best[0]:
            best = stats
    return dict(n_seq=n_seq, rule=rule, epochs=epochs,
                best_loss=best[0], best_vrd=best[1], best_match=best[2],
                tgt_density=float(tgt.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="spikegen_baseline.json")
    args = ap.parse_args()

    results = []
    print(f"{'n_seq':>6} {'rule':>6} {'best_loss':>10} {'VRD':>8} "
          f"{'match':>7} {'tgt_dens':>9}")
    for n_seq in [1, 2, 5]:
        for rule in ["bptt", "eprop"]:
            r = run(n_seq, rule, args.epochs, args.lr)
            results.append(r)
            print(f"{r['n_seq']:>6} {r['rule']:>6} {r['best_loss']:>10.4f} "
                  f"{r['best_vrd']:>8.3f} {r['best_match']:>7.3f} "
                  f"{r['tgt_density']:>9.3f}", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nBPTT = achievable ceiling; eprop = the rule the hardware runs.")
    print("The analog demo should be judged against the eprop row at the "
          "same n_seq.")


if __name__ == "__main__":
    main()

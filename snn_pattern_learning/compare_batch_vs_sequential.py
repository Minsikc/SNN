"""Full-batch vs per-sample e-prop on multi-sequence teacher tasks.

The HW pipeline updates fc1/recurrent once PER SAMPLE (DataLoader
batch_size=1 + opt.step() per sequence), while the SW baselines train
FULL-BATCH (all sequences in one forward; e-prop sums gradients over the
batch dim inside forward). This experiment isolates that difference.

Conditions (identical student init per seed, e-prop + Adamax lr 0.01,
thresh 0.5 aligned everywhere, 500 epochs):
    batch : one forward over all n_seq sequences, one step per epoch
    seq   : loop sequences, forward + step per sequence (HW-pipeline style)

Metrics: mean VRD over sequences, #sequences with exact raster match,
first epoch where ALL sequences match exactly.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.customdatasets import CustomSpikeDataset_Teacher
from verify_vrd0 import build_student, raster_err, vrd, THRESH, TAU

EPOCHS = 500
LR = 0.01


def build_task_multi(n_in, H, n_out, T, n_seq, ds_seed, density=0.1):
    for w_scale in [1.0, 1.5, 2.0, 2.5]:
        ds = CustomSpikeDataset_Teacher(
            num_samples=n_seq, sequence_length=T, input_size=n_in,
            output_size=n_out, hidden_size=H, spike_prob=density,
            teacher_thresh=THRESH, teacher_tau=TAU, w_scale=w_scale,
            seed=ds_seed)
        d = float(ds.targets.mean())
        if 0.03 <= d <= 0.4:
            return ds, w_scale, d
    return ds, w_scale, d


def eval_all(model, x, y):
    with torch.no_grad():
        o = model(x, y, training=False)
    n_exact = sum(int(raster_err(o[i:i + 1], y[i:i + 1]) == 0)
                  for i in range(x.shape[0]))
    return vrd(o, y), n_exact


def run(mode, dims, T, n_seq, ds_seed, epochs=EPOCHS):
    ds, w_scale, dens = build_task_multi(T=T, n_seq=n_seq, ds_seed=ds_seed,
                                         **dims)
    x, y = ds.data, ds.targets
    model = build_student(student_seed=ds_seed + 100, **dims)
    opt = torch.optim.Adamax(model.parameters(), lr=LR)

    first_all, best = -1, (float("inf"), -1)
    for ep in range(epochs):
        if mode == "batch":
            opt.zero_grad()
            model(x, y, training=True)   # e-prop grads sum over the batch dim
            opt.step()
        else:                            # per-sample, HW-pipeline style
            for i in range(n_seq):
                opt.zero_grad()
                model(x[i:i + 1], y[i:i + 1], training=True)
                opt.step()
        v, n_exact = eval_all(model, x, y)
        if n_exact == n_seq and first_all < 0:
            first_all = ep + 1
        if (v, -n_exact) < (best[0], -best[1]):
            best = (v, n_exact)
    v_fin, n_fin = eval_all(model, x, y)
    return dict(mode=mode, T=T, n_seq=n_seq, ds_seed=ds_seed,
                w_scale=w_scale, density=dens,
                best_vrd=best[0], best_exact=best[1],
                final_vrd=v_fin, final_exact=n_fin,
                first_all_zero=first_all)


def main():
    results = []
    print(f"{'scale':>10} {'T':>4} {'nseq':>4} {'mode':>6} {'seed':>4} | "
          f"{'bestVRD':>8} {'exact':>7} | {'finalVRD':>8} {'exact':>7} "
          f"{'all0@':>6}")
    grid = [
        ("B(HW 5x5)", dict(n_in=10, H=5, n_out=5), 25, [2, 5]),
        ("A(theirs)", dict(n_in=100, H=40, n_out=10), 50, [5]),
    ]
    for scale_name, dims, T, nseqs in grid:
        for n_seq in nseqs:
            for ds_seed in [21, 22, 23]:
                for mode in ["batch", "seq"]:
                    r = run(mode, dims, T, n_seq, ds_seed)
                    r["scale"] = scale_name
                    results.append(r)
                    print(f"{scale_name:>10} {T:>4} {n_seq:>4} {mode:>6} "
                          f"{ds_seed:>4} | {r['best_vrd']:>8.3f} "
                          f"{r['best_exact']:>4}/{n_seq:<2} | "
                          f"{r['final_vrd']:>8.3f} "
                          f"{r['final_exact']:>4}/{n_seq:<2} "
                          f"{r['first_all_zero']:>6}", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/batch_vs_seq.json", "w"), indent=1)
    print("\nsaved -> results/batch_vs_seq.json")


if __name__ == "__main__":
    main()

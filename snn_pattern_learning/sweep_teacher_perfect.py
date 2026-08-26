"""Find single-sequence teacher conditions where BPTT AND digital e-prop
reproduce the target raster (near-)exactly.

Raster match metric per epoch: output spikes vs target spikes over the full
(T x 5) grid -> (missed, extra). Perfect = missed + extra == 0.
For each config we report, per learning rule:
    first   first epoch with perfect raster (-1 if never)
    hold    fraction of the last 30 epochs that are perfect
    besterr min(missed+extra) over the run
Conditions must be reachable on hardware, so digital first-perfect should be
well under ~100 epochs.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.customdatasets import CustomSpikeDataset_Teacher
from models.models import Basic_RSNN_eprop_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time

EPOCHS = 300


def build_task(T, ds_seed, spike_prob=0.2, w_scale=1.5):
    ds = CustomSpikeDataset_Teacher(
        num_samples=1, sequence_length=T, input_size=10, output_size=5,
        hidden_size=5, spike_prob=spike_prob, teacher_thresh=0.3,
        w_scale=w_scale, seed=ds_seed)
    return ds.data, ds.targets


def raster_err(out, tgt):
    o = (out > 0.5).float()
    missed = int(((tgt == 1) & (o == 0)).sum())
    extra = int(((tgt == 0) & (o == 1)).sum())
    return missed, extra


def run_one(rule, lr, T, ds_seed, model_seed=0, epochs=EPOCHS):
    x, tgt = build_task(T, ds_seed)
    n_target = int(tgt.sum())
    if n_target < 3:            # trivial/degenerate target, skip
        return None
    torch.manual_seed(model_seed)
    m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=5, n_out=5,
                                 recurrent=True, init_thresh=0.2)
    if rule == "bptt":
        m.custom_grad = False
        m.custom_grad_forward = False
        # widened surrogate that fixed BPTT on the XOR task
        m.LIF0.surrogate_function.subthresh = torch.tensor(0.5)
        m.out_node.surrogate_function.subthresh = torch.tensor(0.5)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    kernel = create_exponential_kernel(3, 2.0)

    first, errs, perfect_flags = -1, [], []
    for ep in range(epochs):
        opt.zero_grad()
        out = m(x, tgt, training=True)
        co = apply_convolution(out, kernel, 3)
        ct = apply_convolution(tgt, kernel, 3)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1])
        if rule == "bptt":
            m.init_net()
            loss.backward()
        opt.step()

        with torch.no_grad():
            o = m(x, tgt, training=False)
        missed, extra = raster_err(o, tgt)
        err = missed + extra
        errs.append(err)
        perfect_flags.append(err == 0)
        if err == 0 and first < 0:
            first = ep + 1

    hold = float(np.mean(perfect_flags[-30:]))
    return dict(first=first, hold=hold, besterr=int(min(errs)),
                n_target=n_target)


def main():
    grid = dict(
        T=[12, 16, 20],
        ds_seed=[10, 11, 12],
        lr=[0.05, 0.1, 0.2],
    )
    print(f"{'T':>3} {'seed':>4} {'lr':>5} {'ntgt':>4} | "
          f"{'bptt 1st':>8} {'hold':>5} {'best':>4} | "
          f"{'eprop 1st':>9} {'hold':>5} {'best':>4}")
    results = []
    for T, ds_seed, lr in itertools.product(*grid.values()):
        rb = run_one("bptt", lr, T, ds_seed)
        re_ = run_one("digital", lr, T, ds_seed)
        if rb is None or re_ is None:
            print(f"{T:>3} {ds_seed:>4} {lr:>5} skipped (degenerate target)")
            continue
        results.append((T, ds_seed, lr, rb, re_))
        print(f"{T:>3} {ds_seed:>4} {lr:>5} {rb['n_target']:>4} | "
              f"{rb['first']:>8} {rb['hold']:>5.2f} {rb['besterr']:>4} | "
              f"{re_['first']:>9} {re_['hold']:>5.2f} {re_['besterr']:>4}",
              flush=True)

    # rank: both rules perfect, e-prop early and stable
    good = [r for r in results
            if r[3]["first"] > 0 and r[4]["first"] > 0]
    good.sort(key=lambda r: (-r[4]["hold"], r[4]["first"]))
    print("\n=== candidates (both perfect; sorted by e-prop hold, first) ===")
    for T, s, lr, rb, re_ in good[:8]:
        print(f"T={T} seed={s} lr={lr}: eprop first@{re_['first']} "
              f"hold {re_['hold']:.2f} | bptt first@{rb['first']} "
              f"hold {rb['hold']:.2f} | targets {rb['n_target']}")


if __name__ == "__main__":
    main()

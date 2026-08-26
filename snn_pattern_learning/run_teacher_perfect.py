#!/usr/bin/env python3
"""Single-sequence teacher demo at the perfect-raster operating point.

Condition (found 2026-08-13 by sweep_teacher_perfect / check_intersection):
    task : CustomSpikeDataset_Teacher, T=12, input 10, ds seed 11,
           teacher_thresh 0.3, w_scale 1.5  (13? no: 11 target spikes)
    init : model seed 2 (identical weights for every condition)
    SW   : e-prop Adam lr 0.15 -> perfect raster @ep107, holds
           BPTT subthr 0.5 lr 0.01 -> perfect @ep~187, holds
Perfect raster = output spikes == target spikes on the whole T x 5 grid.

Conditions here (HW pipeline: W_out per-epoch from the accumulator):
    digital  Mock interface (exact outer products)
    analog   real 5x5 crossbar on COM4

Usage: python run_teacher_perfect.py --condition digital|analog
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.customdatasets import CustomSpikeDataset_Teacher
from models.models import Basic_RSNN_eprop_forward, Basic_RSNN_eprop_HW_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time
from run_xor import HW_KW

T, DS_SEED, MODEL_SEED = 12, 11, 2
MODEL_KW = dict(n_in=10, n_hidden=5, n_out=5, recurrent=True,
                init_tau=0.6, init_thresh=0.2, init_tau_o=0.6, gamma=0.3)


def build_task():
    ds = CustomSpikeDataset_Teacher(
        num_samples=1, sequence_length=T, input_size=10, output_size=5,
        hidden_size=5, spike_prob=0.2, teacher_thresh=0.3, w_scale=1.5,
        seed=DS_SEED)
    return ds.data, ds.targets


def raster_err(out, tgt):
    o = (out > 0.5).float()
    missed = int(((tgt == 1) & (o == 0)).sum())
    extra = int(((tgt == 0) & (o == 1)).sum())
    return missed, extra


def run(condition, epochs, lr, grad_log_path=None, batch_quadrants=True,
        repeats=1, verbose=True, wout_adam=True):
    x, tgt = build_task()

    torch.manual_seed(MODEL_SEED)
    ref = Basic_RSNN_eprop_forward(**MODEL_KW)
    model = Basic_RSNN_eprop_HW_forward(
        **MODEL_KW, hw_enabled=True,
        use_mock_hw=(condition == "digital"), **HW_KW)
    with torch.no_grad():
        model.fc1.weight.copy_(ref.fc1.weight)
        model.recurrent.copy_(ref.recurrent)
        model.out.weight.copy_(ref.out.weight)
    model.hw_batch_quadrants = batch_quadrants
    # exact-accumulation normalization: err is +-1 (batch=1 spike diff),
    # trace max = sum tau_o^k = 1/(1-0.6) = 2.5
    model.fixed_norm = (1.0, 2.5)

    if grad_log_path:
        os.makedirs(os.path.dirname(grad_log_path), exist_ok=True)
        if os.path.exists(grad_log_path):
            os.remove(grad_log_path)
        model.grad_log_path = grad_log_path
    if not model.connect_hardware() and condition == "analog":
        raise RuntimeError("hardware connection failed")

    params = [p for n, p in model.named_parameters()
              if not n.startswith("out.")]
    opt = torch.optim.Adam(params, lr=lr)
    # W_out optimizer fed by the hardware gradient. The plain per-epoch SGD
    # write of apply_hw_gradient does not reach a perfect raster even with
    # the exact (mock) gradient; Adam with the same lr as the converged SW
    # run does. The hardware's role (gradient computation) is unchanged.
    out_opt = (torch.optim.Adam([model.out.weight], lr=lr)
               if wout_adam else None)
    kernel = create_exponential_kernel(3, 2.0)

    losses, errs, epoch_secs = [], [], []
    first_perfect, best = -1, (99, None)
    for ep in range(epochs):
        t0 = time.time()
        model.reset_hardware(hard_reset=True)
        model.train()
        for _ in range(repeats):
            opt.zero_grad()
            out = model(x, tgt, training=True)
            co = apply_convolution(out, kernel, 3)
            ct = apply_convolution(tgt, kernel, 3)
            mse_acc_loss_over_time(co, ct, out.shape[1])
            opt.step()
        if out_opt is not None:
            g = model.apply_hw_gradient(learning_rate=0.0)
            if g is not None:
                out_opt.zero_grad()
                model.out.weight.grad = g / repeats
                out_opt.step()
        else:
            model.apply_hw_gradient(learning_rate=lr / repeats)

        with torch.no_grad():
            o = model(x, tgt, training=False)
            co = apply_convolution(o, kernel, 3)
            ct = apply_convolution(tgt, kernel, 3)
            loss = mse_acc_loss_over_time(co, ct, o.shape[1]).item()
        missed, extra = raster_err(o, tgt)
        err = missed + extra
        losses.append(loss)
        errs.append(err)
        epoch_secs.append(time.time() - t0)
        if err == 0 and first_perfect < 0:
            first_perfect = ep + 1
        if err < best[0]:
            best = (err, o.detach().clone())
        if verbose:
            print(f"[{condition}] epoch {ep + 1:3d}/{epochs}  loss {loss:.4f}"
                  f"  raster_err {err} (miss {missed}/extra {extra})"
                  f"  ({epoch_secs[-1]:.1f}s)", flush=True)

    model.disconnect_hardware()
    tail = errs[-30:]
    return dict(condition=condition, epochs=epochs, lr=lr, repeats=repeats,
                T=T, ds_seed=DS_SEED, model_seed=MODEL_SEED,
                losses=losses, raster_errs=errs, epoch_secs=epoch_secs,
                first_perfect=first_perfect,
                hold=float(np.mean([e == 0 for e in tail])),
                best_err=best[0],
                best_outputs=best[1][0].numpy().tolist(),
                targets=tgt[0].numpy().tolist(),
                inputs=x[0].numpy().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["digital", "analog"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-batch-quadrants", action="store_true")
    ap.add_argument("--sgd-wout", action="store_true",
                    help="use the original per-epoch SGD W_out write instead "
                         "of Adam fed by the hardware gradient")
    ap.add_argument("--outdir", default="results/teacher_perfect")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    glog = (os.path.join(args.outdir, "grad_log_analog.csv")
            if args.condition == "analog" else None)
    r = run(args.condition, args.epochs, args.lr, grad_log_path=glog,
            batch_quadrants=not args.no_batch_quadrants,
            repeats=args.repeats, wout_adam=not args.sgd_wout)
    p = os.path.join(args.outdir, f"teacher_{args.condition}.json")
    json.dump(r, open(p, "w"))
    print(f"\n[{args.condition}] first perfect @ep{r['first_perfect']}, "
          f"hold(last30) {r['hold']:.2f}, best_err {r['best_err']}")
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()

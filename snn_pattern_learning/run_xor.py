#!/usr/bin/env python3
"""Temporal XOR on the 5x5 memristor crossbar -- 4-condition runner.

Task (TemporalXORDataset): bit A (steps 1-3) -> 5 silent steps -> bit B
(steps 9-11) -> go cue + response window (steps 14-19). The network must hold
bit A in recurrent activity across the gap and answer XOR(A,B) by firing
output group {0,1} (class 0) or {3,4} (class 1) during the response window.
XOR is not linearly separable from the input rates, and the gap defeats pure
membrane decay (tau=0.6 -> ~0.08 over 5 steps), so both the hidden
nonlinearity and recurrence are required.

Conditions (identical initial weights, identical sample order):
  bptt     exact gradients through time -- ceiling for this architecture
  digital  e-prop, W_out gradient through the Mock interface (exact software
           outer product, but the same normalize -> accumulate -> per-epoch
           apply pipeline the hardware uses)
  frozen   like digital but W_out never updated -- control isolating the
           output layer's contribution
  analog   W_out gradient physically accumulated on the 5x5 crossbar
           (Basic_RSNN_eprop_HW_forward + MemristorInterface on COM4)

Usage:
  python run_xor.py --condition bptt
  python run_xor.py --condition digital
  python run_xor.py --condition frozen
  python run_xor.py --condition analog          # real hardware!
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.customdatasets import TemporalXORDataset
from models.models import Basic_RSNN_eprop_forward, Basic_RSNN_eprop_HW_forward
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from models.loss import mse_acc_loss_over_time

# tau=0.8 / tau_o=0.9 chosen by the reservoir separability test
# (debug_xor_separability.py): with the 5-step A->B gap, the response-window
# traces of a frozen random reservoir are perfectly linearly separable for
# 7/8 seeds at this operating point, so an ideal W_out exists.
#
# init_thresh: 0.5, not the historical 0.2. Until 2026-08-19 the spiking
# nodes ignored init_thresh and fired at LIF_Node's default 0.5, so every
# pre-fix result (and the separability test that chose this operating point)
# actually ran at 0.5. Re-sweeping thresh x tau x tau_o with the fixed wiring
# (2026-08-20) reproduces 7/8 separable ONLY at thresh=0.5 (thresh=0.2 gives
# 2/8, seed 0 not separable); 0.5 keeps the same reservoir dynamics and makes
# the e-prop pseudo-derivative (self.thr) consistent with the firing
# threshold for the first time.
MODEL_KW = dict(n_in=10, n_hidden=5, n_out=5, recurrent=True,
                init_tau=0.8, init_thresh=0.5, init_tau_o=0.9, gamma=0.3)

# Operating point validated by the 2026-08-05 uv_grid_sweep (r ~ 0.92)
HW_KW = dict(serial_port="COM4", baud_rate=115200, bit_length=10,
             pulse_width=15, pulse_pre=100, pulse_post=100, pulse_zero=10,
             read_time=20, read_delay=10,
             normalization_scale=0.7, adc_to_grad_scale=0.001,
             auto_calibrate_scale=True,
             # STOCHASTIC_*_NR opcodes: no pre/post read per update command.
             # run3 (2026-08-20) measured ~172 array reads/epoch from the
             # built-in reads; POT flushes before DEP, so POT charge decayed
             # through ~120 reads (0.79%/read) -> delivered 0.35x vs DEP
             # 1.84x. NR leaves 2 reads/epoch (reference + final gradient).
             no_read_updates=True)


def build_model(condition, seed, model_kw, bptt_subthresh=None,
                quantize_bits=0):
    """Build the model for a condition, with weights copied from the SW
    reference model built under `seed`, so all conditions start identical."""
    torch.manual_seed(seed)
    ref = Basic_RSNN_eprop_forward(**model_kw)

    if condition == "bptt":
        model = ref
        model.custom_grad = False
        model.custom_grad_forward = False
        if bptt_subthresh is not None:
            # The repo-default Boxcar surrogate only passes gradient within
            # |mem - thresh| < 0.1, which is too narrow for BPTT to escape
            # the silent-output local minimum on this task. Widen it for the
            # BPTT ceiling condition (spike function itself is unchanged).
            model.LIF0.surrogate_function.subthresh = \
                torch.tensor(bptt_subthresh)
            model.out_node.surrogate_function.subthresh = \
                torch.tensor(bptt_subthresh)
        return model

    use_mock = condition in ("digital", "frozen")
    model = Basic_RSNN_eprop_HW_forward(
        **model_kw, hw_enabled=True, use_mock_hw=use_mock,
        mock_quantize_bits=(quantize_bits if use_mock else 0),
        mock_quantize_seed=seed, **HW_KW)
    with torch.no_grad():
        model.fc1.weight.copy_(ref.fc1.weight)
        model.recurrent.copy_(ref.recurrent)
        model.out.weight.copy_(ref.out.weight)
    if condition == "frozen":
        model.freeze_wout = True
    return model


def evaluate(model, data, targets, kernel, ksize, response_window):
    """Loss, accuracy and raw output spikes on all 4 samples."""
    with torch.no_grad():
        out = model(data, targets, training=False)
        co = apply_convolution(out, kernel, ksize)
        ct = apply_convolution(targets, kernel, ksize)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1]).item()
    pred = TemporalXORDataset.decision(out, response_window)
    labels = torch.tensor([0, 1, 1, 0])
    acc = (pred == labels).float().mean().item()
    return loss, acc, out


def run(condition, epochs, lr, seed, grad_log_path=None,
        model_kw=None, ds_kw=None, verbose=True, freeze_hidden=False,
        lr_decay=1.0, bptt_subthresh=None, repeats=1,
        batch_quadrants=False, calibrate_per_column=False,
        quantize_bits=0):
    model_kw = {**MODEL_KW, **(model_kw or {})}
    ds = TemporalXORDataset(**(ds_kw or {}))
    data, targets = ds.data, ds.targets   # (4, T, 10), (4, T, 5)

    model = build_model(condition, seed, model_kw, bptt_subthresh,
                        quantize_bits=quantize_bits)
    is_hw_model = isinstance(model, Basic_RSNN_eprop_HW_forward)
    if is_hw_model:
        # Learning window: weight updates only from response-window error.
        # Spikes outside the window are unconstrained (the decision rule
        # never reads them), which gives training a reachable fixed point.
        model.err_window = ds.response_window
        model.hw_batch_quadrants = batch_quadrants
        model.calibrate_per_column = calibrate_per_column

    if is_hw_model:
        if grad_log_path:
            os.makedirs(os.path.dirname(grad_log_path), exist_ok=True)
            if os.path.exists(grad_log_path):
                os.remove(grad_log_path)
            model.grad_log_path = grad_log_path
        if not model.connect_hardware():
            if condition == "analog":
                raise RuntimeError("Hardware connection failed -- aborting "
                                   "analog run instead of silently training "
                                   "in software")

    # W_out is updated through apply_hw_gradient (or frozen); the optimizer
    # only owns the hidden layers, matching the frozen-condition convention
    # in run_three_conditions.py. With freeze_hidden the hidden layers stay
    # at their random init (reservoir mode) and there is no optimizer at all:
    # every trainable weight lives on the crossbar.
    if freeze_hidden and not is_hw_model:
        raise ValueError("freeze_hidden only makes sense for HW-model "
                         "conditions (digital/frozen/analog)")
    if is_hw_model:
        params = [p for n, p in model.named_parameters()
                  if not n.startswith("out.")]
    else:
        params = list(model.parameters())
    if freeze_hidden:
        params = []
    opt = torch.optim.Adam(params, lr=lr) if params else None

    ksize = 3
    kernel = create_exponential_kernel(ksize, 2.0)

    losses, accs, epoch_secs = [], [], []
    best = (float("inf"), -1.0, None)
    for ep in range(epochs):
        t0 = time.time()
        if is_hw_model:
            model.reset_hardware(hard_reset=True)

        model.train()
        # `repeats` accumulates each sample's outer product K times on the
        # array before the single per-epoch weight update. The gradient
        # signal grows K-fold while the ADC read noise (one read per epoch)
        # stays fixed -- raising SNR when the residual error, and hence the
        # per-pass gradient, is small. lr is divided by K to keep the
        # effective step size unchanged.
        for i in [j % len(ds) for j in range(repeats * len(ds))]:
            x = data[i:i + 1]
            tgt = targets[i:i + 1]
            if opt is not None:
                opt.zero_grad()
            out = model(x, tgt, training=True)
            co = apply_convolution(out, kernel, ksize)
            ct = apply_convolution(tgt, kernel, ksize)
            loss = mse_acc_loss_over_time(co, ct, out.shape[1])
            if condition == "bptt":
                # Same task definition as the e-prop conditions' err_window:
                # only response-window error drives learning.
                r0, r1 = ds.response_window
                w_loss = mse_acc_loss_over_time(
                    co[:, r0:r1, :], ct[:, r0:r1, :], r1 - r0)
                # forward() already filled .grad with e-prop values; clear
                # them so the update is pure BPTT
                model.init_net()
                w_loss.backward()
            if opt is not None:
                opt.step()

        if is_hw_model:
            model.apply_hw_gradient(
                learning_rate=lr * (lr_decay ** ep) / repeats)

        ep_loss, ep_acc, out_all = evaluate(model, data, targets, kernel,
                                            ksize, ds.response_window)
        losses.append(ep_loss)
        accs.append(ep_acc)
        epoch_secs.append(time.time() - t0)
        # best epoch = highest accuracy, then lowest loss
        if (-ep_acc, ep_loss) < (-best[1], best[0]):
            best = (ep_loss, ep_acc, out_all.detach().clone())
        if verbose:
            print(f"[{condition}] epoch {ep + 1:3d}/{epochs}  "
                  f"loss {ep_loss:.4f}  acc {ep_acc:.2f}  "
                  f"({epoch_secs[-1]:.1f}s)", flush=True)

    if is_hw_model:
        model.disconnect_hardware()

    return dict(condition=condition, epochs=epochs, lr=lr, seed=seed,
                freeze_hidden=freeze_hidden, lr_decay=lr_decay,
                repeats=repeats, quantize_bits=quantize_bits,
                losses=losses, accs=accs, epoch_secs=epoch_secs,
                best_loss=best[0], best_acc=best[1],
                final_acc=accs[-1],
                best_outputs=best[2][..., :].numpy().tolist(),
                targets=targets.numpy().tolist(),
                inputs=data.numpy().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["bptt", "digital", "frozen", "analog"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/xor")
    ap.add_argument("--freeze-hidden", action="store_true",
                    help="reservoir mode: fc1/recurrent stay at random init, "
                         "only W_out (the crossbar) learns")
    ap.add_argument("--lr-decay", type=float, default=1.0,
                    help="per-epoch multiplicative decay of the W_out lr")
    ap.add_argument("--calibrate-per-column", action="store_true",
                    help="per-column gain calibration (EMA) on top of the "
                         "global ADC scale")
    ap.add_argument("--batch-quadrants", action="store_true",
                    help="queue outer products and flush quadrant-major at "
                         "epoch end (minimizes P/D alternation half-select)")
    ap.add_argument("--dno", action="store_true",
                    help="use the DNO_ (push-pull) firmware opcodes: rows "
                         "whose pulse bit is 0 are driven with the opposite "
                         "select line instead of idling, suppressing "
                         "alternating-pair half-select. Teacher-student "
                         "(2026-08-20) showed DNO erases accumulated charge "
                         "over long runs -- watch mean|hw_adc|.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="accumulate each sample K times per epoch on the "
                         "array (SNR x K); lr is divided by K internally")
    ap.add_argument("--bptt-subthresh", type=float, default=0.5,
                    help="Boxcar surrogate half-width for the bptt condition "
                         "(repo default 0.1 is too narrow to learn XOR; "
                         "0.5 + lr 0.05 converges and holds at seed 0)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.dno:
        HW_KW["dno"] = True
    grad_log = (os.path.join(args.outdir, f"grad_log_{args.condition}.csv")
                if args.condition == "analog" else None)

    r = run(args.condition, args.epochs, args.lr, args.seed,
            grad_log_path=grad_log, freeze_hidden=args.freeze_hidden,
            lr_decay=args.lr_decay, bptt_subthresh=args.bptt_subthresh,
            repeats=args.repeats, batch_quadrants=args.batch_quadrants,
            calibrate_per_column=args.calibrate_per_column)

    tag = "_res" if args.freeze_hidden else ""
    out_path = os.path.join(args.outdir,
                            f"xor_{args.condition}{tag}_seed{args.seed}.json")
    json.dump(r, open(out_path, "w"))
    print(f"\n[{args.condition}] best loss {r['best_loss']:.4f}  "
          f"best acc {r['best_acc']:.2f}  final acc {r['final_acc']:.2f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

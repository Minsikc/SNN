#!/usr/bin/env python3
"""Reload an analog-trained 4x4 checkpoint, re-evaluate it on the task it was
trained on (main_unified pipeline: CustomSpikeDataset_Teacher w_scale=1.0),
and plot input / target-vs-student raster.

NOTE the two pipelines differ: run_three_conditions.build_task hardcodes
w_scale=1.5 while base_experiment leaves the dataset default 1.0 -> different
targets. Checkpoints from main_unified MUST be evaluated at w_scale=1.0.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from datasets.customdatasets import CustomSpikeDataset_Teacher
from models.models import Basic_RSNN_eprop_forward
from models.loss import mse_acc_loss_over_time
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from utils.metrics import van_rossum_distance

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/eprop_grad_log/best_model_eprop_4x4_NR_s0.pth")
ap.add_argument("--out", default="results/eprop_grad_log/raster_4x4_s0_eval.png")
ap.add_argument("--w-scale", type=float, default=1.0)
ap.add_argument("--ds-seed", type=int, default=10)
args = ap.parse_args()

ds = CustomSpikeDataset_Teacher(num_samples=1, sequence_length=20,
                                input_size=10, output_size=4, hidden_size=4,
                                spike_prob=0.2, teacher_thresh=0.4,
                                teacher_tau=0.6, w_scale=args.w_scale,
                                seed=args.ds_seed)
x, tgt = ds.data, ds.targets

m = Basic_RSNN_eprop_forward(n_in=10, n_hidden=4, n_out=4, recurrent=True,
                             init_thresh=0.4, init_tau=0.6)
sd = torch.load(args.ckpt)
m.load_state_dict(sd)
m.eval()

with torch.no_grad():
    out = m(x, tgt, training=False)
    kernel = create_exponential_kernel(3, 2.0)
    co = apply_convolution(out, kernel, 3)
    ct = apply_convolution(tgt, kernel, 3)
    loss = mse_acc_loss_over_time(co, ct, out.shape[1]).item()
    vrd = float(van_rossum_distance(out, tgt, tau=5.0).mean())

o, t, xi = out[0], tgt[0], x[0]  # (T, n)
matched = int(((o == 1) & (t == 1)).sum())
extra = int(((o == 1) & (t == 0)).sum())
missed = int(((o == 0) & (t == 1)).sum())
print(f"ckpt {args.ckpt}  w_scale {args.w_scale}")
print(f"loss {loss:.4f}  VRD {vrd:.4f}")
print(f"target spikes {int(t.sum())}  output spikes {int(o.sum())}  "
      f"matched {matched}  extra {extra}  missed {missed}")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.6]})
ax = axes[0]
ts, ns = torch.nonzero(xi, as_tuple=True)
ax.scatter(ts, ns, marker="|", s=120, color="gray", label="input")
ax.set_ylabel("input neuron")
ax.set_yticks(range(10))
ax.set_title(f"analog-trained student (seed 0, best epoch) -- "
             f"loss {loss:.3f}, VRD {vrd:.3f}")

ax = axes[1]
ts, ns = torch.nonzero(t, as_tuple=True)
ax.scatter(ts, ns.float() + 0.15, marker="|", s=200, color="tab:blue",
           label=f"target ({int(t.sum())})")
ts, ns = torch.nonzero(o, as_tuple=True)
ax.scatter(ts, ns.float() - 0.15, marker="|", s=200, color="tab:red",
           label=f"student ({int(o.sum())})")
ax.set_ylabel("output neuron")
ax.set_xlabel("timestep")
ax.set_yticks(range(4))
ax.set_ylim(-0.6, 3.6)
ax.legend(loc="upper right", fontsize=8)
ax.set_xticks(range(0, 20, 2))
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"saved -> {args.out}")

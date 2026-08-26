"""
Compare training curves of Basic_RSNN_eprop_minsik vs Basic_RSNN_eprop_forward.

If gradients are numerically identical (verified by verify_gradients.py), and
weights are synced + dataset is identical, the loss curves should be identical
when learning rate / optimizer / batch ordering are the same.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from models.models import Basic_RSNN_eprop_minsik, Basic_RSNN_eprop_forward
from datasets.customdatasets import CustomSpikeDataset_random
from utils.kernels import create_exponential_kernel
from models.loss import mse_acc_loss_over_time


def apply_convolution(spikes, kernel, kernel_size):
    # spikes: (batch, time, channels)
    b, t, c = spikes.shape
    spikes_t = spikes.permute(0, 2, 1).reshape(b * c, 1, t)
    conv = F.conv1d(spikes_t, kernel.view(1, 1, -1), padding=kernel_size - 1)
    conv = conv[:, :, :t]
    return conv.reshape(b, c, t).permute(0, 2, 1)


def make_dataset(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return CustomSpikeDataset_random(
        num_samples=1, sequence_length=100, input_size=50, output_size=10,
        spike_prob=0.1, total_spike=5,
    )


def train_loop(model_kind, epochs, lr, seed):
    # Create dataset
    dataset = make_dataset(seed)

    # Build model with the SAME seed so initial weights match across both runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    common = dict(n_in=50, n_hidden=40, n_out=10, init_tau=0.6,
                  init_thresh=0.6, init_tau_o=0.6, gamma=0.3, recurrent=True)
    if model_kind == "minsik":
        model = Basic_RSNN_eprop_minsik(**common)
    else:
        model = Basic_RSNN_eprop_forward(**common)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    kernel_size = 5
    kernel = create_exponential_kernel(kernel_size, 2.0)

    # Use a deterministic generator for the DataLoader so batch ordering matches
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=g)

    losses = []
    weight_snapshots = []
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        # Reset generator each epoch with the same seed so two runs see identical batch order
        g.manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=g)
        for inputs, targets in loader:
            optimizer.zero_grad()
            if model_kind == "minsik":
                outputs = model(inputs)
            else:
                outputs = model(inputs, targets, training=True)

            conv_t = apply_convolution(targets, kernel, kernel_size)
            conv_o = apply_convolution(outputs, kernel, kernel_size)
            loss = mse_acc_loss_over_time(conv_o, conv_t, outputs.shape[1])

            if model_kind == "minsik":
                err = (conv_o - conv_t).permute(1, 0, 2)
                model.compute_grads(inputs, err)
            optimizer.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader))
        weight_snapshots.append({
            'fc1': model.fc1.weight.detach().clone(),
            'rec': model.recurrent.detach().clone(),
            'out': model.out.weight.detach().clone(),
        })
    return losses, model, weight_snapshots


def main():
    EPOCHS = 10
    LR = 0.01
    SEED = 42

    print("=" * 70)
    print(f"Training comparison (epochs={EPOCHS}, lr={LR}, seed={SEED})")
    print("=" * 70)

    losses_m, m_minsik, snaps_m = train_loop("minsik", EPOCHS, LR, SEED)
    losses_f, m_forward, snaps_f = train_loop("forward", EPOCHS, LR, SEED)

    print(f"\n{'epoch':>5}  {'minsik':>10}  {'forward':>10}  {'loss diff':>12}  {'fc1':>10}  {'rec':>10}  {'out':>10}")
    for i, (lm, lf) in enumerate(zip(losses_m, losses_f)):
        wd_fc1 = (snaps_m[i]['fc1'] - snaps_f[i]['fc1']).abs().max().item()
        wd_rec = (snaps_m[i]['rec'] - snaps_f[i]['rec']).abs().max().item()
        wd_out = (snaps_m[i]['out'] - snaps_f[i]['out']).abs().max().item()
        print(f"{i+1:>5}  {lm:>10.6f}  {lf:>10.6f}  {abs(lm-lf):>12.6e}  {wd_fc1:>10.2e}  {wd_rec:>10.2e}  {wd_out:>10.2e}")

    # Final weight comparison
    print("\n--- Final weight diffs ---")
    for name in ["fc1.weight", "recurrent", "out.weight"]:
        if name == "recurrent":
            wm = m_minsik.recurrent.detach()
            wf = m_forward.recurrent.detach()
        else:
            wm = dict(m_minsik.named_parameters())[name].detach()
            wf = dict(m_forward.named_parameters())[name].detach()
        max_diff = (wm - wf).abs().max().item()
        print(f"  {name}: max abs diff = {max_diff:.6e}")


if __name__ == "__main__":
    main()

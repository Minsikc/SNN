"""
Verification script for gradient comparison between Basic_RSNN_eprop_minsik and Basic_RSNN_eprop_forward.

Findings:
1. Output Layer Gradients: Match perfectly after scaling by 5.0.
   - Basic_RSNN_eprop_minsik uses learning rate factor 0.05.
   - Basic_RSNN_eprop_forward uses learning rate factor 0.01.
2. Hidden/Input Layer Gradients: Do NOT match.
   - Basic_RSNN_eprop_forward uses causal iterative updates for eligibility traces.
   - Basic_RSNN_eprop_minsik uses Conv1d for eligibility traces, which appears to have different alignment/causality (likely due to padding/slicing).
   - Recurrent trace in 'minsik' uses shifted spikes (z_{t-1}), while 'forward' uses current spikes (z_t) in the trace update loop (which might be incorrect for standard e-prop).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add snn_pattern_learning to path
sys.path.append(os.path.join(os.getcwd(), 'snn_pattern_learning'))

from models.models import Basic_RSNN_eprop_minsik, Basic_RSNN_eprop_forward

def verify_gradients():
    print("Verifying gradients between Basic_RSNN_eprop_minsik and Basic_RSNN_eprop_forward...")

    torch.manual_seed(42)
    np.random.seed(42)

    n_in = 10
    n_hidden = 20
    n_out = 5
    batch_size = 2
    steps = 10

    # Common parameters
    # Note: 'gamma' in params is passed to 'gamma' in minsik, and 'gamma' in forward.
    # However, minsik uses 'init_tau' for surrogate gradient, forward uses 'gamma'.
    # To make them comparable, we should set init_tau = gamma.

    common_tau = 0.6

    params = {
        'n_in': n_in,
        'n_hidden': n_hidden,
        'n_out': n_out,
        'init_tau': common_tau,
        'gamma': common_tau, # Setting gamma equal to init_tau to align surrogate gradient scale
        'recurrent': True
    }

    print(f"Parameters: {params}")

    # Instantiate models
    model_minsik = Basic_RSNN_eprop_minsik(**params)
    model_forward = Basic_RSNN_eprop_forward(**params)

    # Sync weights
    with torch.no_grad():
        model_forward.fc1.weight.copy_(model_minsik.fc1.weight)
        model_forward.recurrent.copy_(model_minsik.recurrent)
        model_forward.out.weight.copy_(model_minsik.out.weight)

    print("Weights synced.")

    # Input and Label
    x = torch.randn(batch_size, steps, n_in)
    # Binary spikes for input roughly (not strictly needed but better for SNN context)
    x = (x > 0).float()

    label = torch.rand(batch_size, steps, n_out)
    label = (label > 0.5).float() # Assuming label is spikes too

    # --- Run Minsik Model ---
    # Forward
    output_minsik = model_minsik(x)

    # Error calculation
    # Basic_RSNN_eprop_forward uses: err = (out_spike - label[:,step,:])
    # Basic_RSNN_eprop_minsik.compute_grads expects err shape [time, batch, n_out]

    err_minsik = output_minsik - label
    err_minsik_for_calc = err_minsik.permute(1, 0, 2)

    # Compute Grads
    model_minsik.compute_grads(x, err_minsik_for_calc)

    # --- Run Forward Model ---
    # forward(self, x, label, training)
    output_forward = model_forward(x, label, training=True)

    # --- Compare Outputs ---
    output_diff = torch.abs(output_minsik - output_forward).max().item()
    print(f"\nForward Output Max Diff: {output_diff}")

    # --- Compare Gradients ---
    grads_minsik = {
        'fc1': model_minsik.fc1.weight.grad,
        'rec': model_minsik.recurrent.grad,
        'out': model_minsik.out.weight.grad
    }

    grads_forward = {
        'fc1': model_forward.fc1.weight.grad,
        'rec': model_forward.recurrent.grad,
        'out': model_forward.out.weight.grad
    }

    print("\nGradient Comparison:")

    # We identified a learning rate factor difference:
    # Minsik: 0.05
    # Forward: 0.01
    # Expected ratio: 5.0

    expected_ratio = 5.0

    for name in ['fc1', 'rec', 'out']:
        gm = grads_minsik[name]
        gf = grads_forward[name]

        # Adjust for expected ratio
        gf_scaled = gf * expected_ratio

        diff = torch.abs(gm - gf).max().item()
        diff_scaled = torch.abs(gm - gf_scaled).max().item()

        ratio = torch.mean(gm / (gf + 1e-9)).item()

        print(f"--- {name} ---")
        print(f"Minsik Grad Norm: {gm.norm().item()}")
        print(f"Forward Grad Norm: {gf.norm().item()}")
        print(f"Max Diff (Raw): {diff}")
        print(f"Max Diff (Scaled by 5.0): {diff_scaled}")
        print(f"Mean Ratio (Minsik/Forward): {ratio:.4f}")

        if diff_scaled < 1e-5:
            print(f"✅ Gradients match (after scaling by {expected_ratio})")
        else:
            print(f"❌ Gradients do not match exactly (after scaling)")

if __name__ == "__main__":
    verify_gradients()

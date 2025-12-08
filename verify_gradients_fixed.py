"""
Verification script for gradient comparison between Basic_RSNN_eprop_minsik
and the fixed Basic_RSNN_eprop_forward_fixed.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add snn_pattern_learning to path
sys.path.append(os.path.join(os.getcwd(), 'snn_pattern_learning'))

from models.models import Basic_RSNN_eprop_minsik
from models.eprop_forward_fixed import Basic_RSNN_eprop_forward_fixed


def verify_gradients():
    print("=" * 60)
    print("Verifying gradients: Basic_RSNN_eprop_minsik vs Fixed Forward")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    n_in = 10
    n_hidden = 20
    n_out = 5
    batch_size = 2
    steps = 10

    common_tau = 0.6

    params = {
        'n_in': n_in,
        'n_hidden': n_hidden,
        'n_out': n_out,
        'init_tau': common_tau,
        'gamma': common_tau,
        'recurrent': True
    }

    print(f"Parameters: {params}")

    # Instantiate models
    model_minsik = Basic_RSNN_eprop_minsik(**params)
    model_fixed = Basic_RSNN_eprop_forward_fixed(**params)

    # Sync weights
    with torch.no_grad():
        model_fixed.fc1.weight.copy_(model_minsik.fc1.weight)
        model_fixed.recurrent.copy_(model_minsik.recurrent)
        model_fixed.out.weight.copy_(model_minsik.out.weight)

    print("Weights synced.")

    # Input and Label
    x = torch.randn(batch_size, steps, n_in)
    x = (x > 0).float()

    label = torch.rand(batch_size, steps, n_out)
    label = (label > 0.5).float()

    # --- Run Minsik Model ---
    output_minsik = model_minsik(x)
    err_minsik = output_minsik - label
    err_minsik_for_calc = err_minsik.permute(1, 0, 2)
    model_minsik.compute_grads(x, err_minsik_for_calc)

    # --- Run Fixed Forward Model ---
    output_fixed = model_fixed(x, label, training=True)

    # --- Compare Outputs ---
    output_diff = torch.abs(output_minsik - output_fixed).max().item()
    print(f"\nForward Output Max Diff: {output_diff}")

    # --- Compare Gradients ---
    grads_minsik = {
        'fc1': model_minsik.fc1.weight.grad,
        'rec': model_minsik.recurrent.grad,
        'out': model_minsik.out.weight.grad
    }

    grads_fixed = {
        'fc1': model_fixed.fc1.weight.grad,
        'rec': model_fixed.recurrent.grad,
        'out': model_fixed.out.weight.grad
    }

    print("\n" + "=" * 60)
    print("Gradient Comparison (No scaling needed - both use 0.05)")
    print("=" * 60)

    all_passed = True

    for name in ['fc1', 'rec', 'out']:
        gm = grads_minsik[name]
        gf = grads_fixed[name]

        diff = torch.abs(gm - gf).max().item()
        rel_diff = diff / (gm.abs().max().item() + 1e-9)

        # Compute correlation
        gm_flat = gm.flatten()
        gf_flat = gf.flatten()
        correlation = torch.corrcoef(torch.stack([gm_flat, gf_flat]))[0, 1].item()

        print(f"\n--- {name} ---")
        print(f"Minsik Grad Norm: {gm.norm().item():.6f}")
        print(f"Fixed  Grad Norm: {gf.norm().item():.6f}")
        print(f"Max Absolute Diff: {diff:.6e}")
        print(f"Max Relative Diff: {rel_diff:.6e}")
        print(f"Correlation: {correlation:.6f}")

        if diff < 1e-5:
            print(f"[PASS] Gradients match!")
        elif rel_diff < 1e-4:
            print(f"[PASS] Gradients match (within relative tolerance)")
        else:
            print(f"[FAIL] Gradients do not match")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL GRADIENT CHECKS PASSED!")
    else:
        print("SOME GRADIENT CHECKS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = verify_gradients()
    exit(0 if success else 1)

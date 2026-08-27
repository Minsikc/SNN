#!/usr/bin/env python3
"""
E-prop hardware gradient store/read/update test.

This test verifies the full cycle of e-prop hardware operation:
  1. Connect to the memristor crossbar (Arduino Due, COM4)
  2. Reset the gradient accumulator and measure the reference point
  3. Accumulate outer-product (err x trace) gradients across several timesteps
  4. Read the accumulated gradient back from hardware
  5. Apply it as a weight update and compare against the software-computed
     gradient that *should* have been accumulated

This is the minimal end-to-end demo: store on HW -> read from HW -> update weights.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from hardware import MemristorInterface


PORT = "COM4"
BAUD = 115200
N = 5  # 5x5 crossbar
NUM_TIMESTEPS = 8
LEARNING_RATE = 0.01
ADC_TO_GRAD_SCALE = 0.001


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    banner("E-prop Hardware Gradient Test (store / read / update)")
    print(f"Port: {PORT}  |  Crossbar: {N}x{N}  |  Timesteps: {NUM_TIMESTEPS}")

    hw = MemristorInterface(
        port=PORT,
        baud_rate=BAUD,
        timeout=30.0,
        bit_length=10,
        pulse_width=1,
        pulse_pre=100,
        pulse_post=100,
        pulse_zero=10,
        read_time=10,
        read_delay=10,
    )

    # 1) Connect
    banner("[1] Connect")
    if not hw.connect():
        print(f"[ERROR] Could not open {PORT}. Close Arduino IDE / Serial Monitor and retry.")
        return 1

    try:
        # 2) Reset + measure reference point
        banner("[2] Reset accumulator and measure reference point")
        hw.reset()
        print(f"Reference point (5x5):\n{hw.reference_point}")

        # Software-side mirror of what the hardware *should* be accumulating.
        # In real e-prop training this is `desired_gradient_accumulated`.
        sw_accum = np.zeros((N, N), dtype=np.float32)

        # Use a fixed seed so this run is reproducible
        rng = np.random.default_rng(0)

        # 3) Accumulate over several timesteps
        banner(f"[3] Accumulate outer products over {NUM_TIMESTEPS} timesteps")
        for t in range(NUM_TIMESTEPS):
            # Mock e-prop signals: error vector (n_out,) and eligibility trace (n_hidden,)
            err = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
            trace = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)

            # Split into magnitude (probability in [0,1]) and sign, exactly like
            # Basic_RSNN_eprop_HW_forward.normalize_for_hardware does.
            err_signs = np.sign(err)
            err_signs[err_signs == 0] = 1
            trace_signs = np.sign(trace)
            trace_signs[trace_signs == 0] = 1
            err_probs = np.clip(np.abs(err), 0.0, 1.0)
            trace_probs = np.clip(np.abs(trace), 0.0, 1.0)

            # Send to hardware
            hw.accumulate_outer_product(err_probs, trace_probs, err_signs, trace_signs)

            # Mirror in software
            sw_accum += np.outer(err_probs * err_signs, trace_probs * trace_signs)

            print(
                f"  t={t}: |err|_max={err_probs.max():.3f}  "
                f"|trace|_max={trace_probs.max():.3f}  "
                f"sw_accum_max={np.max(np.abs(sw_accum)):.3f}"
            )

        # 4) Read accumulated gradient from hardware
        banner("[4] Read accumulated gradient from hardware")
        hw_grad_adc = hw.read_accumulated_gradient()
        hw_grad = hw_grad_adc * ADC_TO_GRAD_SCALE
        print(f"Hardware gradient (raw ADC differential):\n{hw_grad_adc}")
        print(f"Hardware gradient (scaled, x{ADC_TO_GRAD_SCALE}):\n{hw_grad}")
        print(f"\nSoftware-mirrored gradient (what the HW *should* hold):\n{sw_accum}")

        # Comparison statistics
        diff = sw_accum - hw_grad
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        if np.std(hw_grad) > 0 and np.std(sw_accum) > 0:
            corr = float(np.corrcoef(sw_accum.flatten(), hw_grad.flatten())[0, 1])
        else:
            corr = float("nan")
        print("\n[Statistics: software vs hardware gradient]")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {corr:.4f}")

        # 5) Apply weight update with the hardware-read gradient
        banner("[5] Apply weight update using hardware gradient")
        weights = torch.zeros(N, N)
        torch.nn.init.kaiming_normal_(weights)
        weights *= 0.5
        weights_before = weights.clone()

        weights -= LEARNING_RATE * torch.from_numpy(hw_grad).float()

        delta = (weights - weights_before).abs()
        print(f"Weights before:\n{weights_before.numpy()}")
        print(f"Weights after :\n{weights.numpy()}")
        print(
            f"\nMax weight change: {delta.max().item():.6f}  "
            f"Mean weight change: {delta.mean().item():.6f}  "
            f"(lr={LEARNING_RATE})"
        )

        banner("[OK] Test completed")
        return 0

    finally:
        hw.disconnect()


if __name__ == "__main__":
    sys.exit(main())

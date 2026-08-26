"""
Plot the HW e-prop training results from the 30-epoch real-HW run, with a
SW-only baseline (mock interface, scale=1, no auto-calibration) overlaid for
reference. Both runs use the same dataset config; both use seed=42.

Bottom row shows raster plots (target vs model output) before and after
training, so target learning is visible at a glance.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

# --- 30-epoch real-HW results (seed=42, from capture_spikes.py --mode hw) ---
epochs = list(range(1, 31))

loss = [
    5.8517, 6.0627, 6.1922, 6.2564, 6.3535, 6.2335, 6.2010, 6.0778, 5.9604, 5.9797,
    5.5155, 5.2623, 4.8532, 4.3933, 4.6251, 3.5050, 4.3199, 3.8225, 3.4698, 3.6339,
    3.7783, 3.3985, 4.0395, 3.0071, 3.8922, 3.5300, 4.0126, 4.0617, 3.9046, 3.5299,
]

correlation = [
    0.3810, 0.2588, 0.2850, 0.2964, 0.4366, 0.4110, 0.4094, 0.4450, 0.6853, 0.6375,
    0.7880, 0.7214, 0.3890, 0.3355, 0.4082, 0.0672, 0.2403, 0.2326, -0.2843, 0.1285,
    0.1747, 0.5516, 0.2729, -0.4772, 0.2773, 0.3967, 0.0039, 0.1702, 0.0508, -0.0109,
]

sw_mag = [
    0.9202, 0.6864, 0.4555, 0.2185, 0.1689, 0.3383, 0.4083, 0.3934, 0.4310, 0.5766,
    0.6874, 0.9467, 1.0615, 0.9399, 1.0507, 0.3747, 0.6521, 0.6130, 0.3153, 0.4790,
    0.3102, 0.1953, 0.5019, 0.3099, 0.4060, 0.1426, 0.3268, 0.2859, 0.3081, 0.2686,
]

hw_adc_mag = [
    19.32, 11.68, 12.12, 5.44, 9.80, 6.84, 7.52, 11.24, 13.44, 13.68,
    22.64, 22.56, 20.00, 18.52, 16.00, 14.88, 11.56, 14.96, 14.36, 13.48,
    9.40, 14.40, 11.64, 14.08, 10.24, 17.16, 9.48, 10.36, 8.04, 14.16,
]

# --- SW-only baseline (mock interface, scale=1, auto_calib=off, seed=42, capture_spikes) ---
sw_loss = [
    5.8517, 5.8570, 5.4682, 5.2262, 4.3225, 4.4283, 4.0995, 4.3888, 4.0332, 4.3045,
    4.7730, 4.9401, 4.4234, 4.3631, 4.6098, 5.0261, 5.2769, 3.9570, 3.7672, 4.8809,
    5.4889, 3.7373, 3.6274, 3.5769, 4.4560, 4.1134, 3.5587, 3.6637, 3.8448, 3.7006,
]

# --- Load raster spike data ---
results_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "eprop_hardware_demo",
)
sw_npz_path = os.path.join(results_dir, "spikes_sw.npz")
hw_npz_path = os.path.join(results_dir, "spikes_hw.npz")

sw_npz = np.load(sw_npz_path) if os.path.exists(sw_npz_path) else None
hw_npz = np.load(hw_npz_path) if os.path.exists(hw_npz_path) else None


def raster_overlay(ax, target, out_sw, out_hw, title):
    """Overlay target / SW / HW output spikes on a single per-output-neuron raster."""
    n_out = target.shape[1]
    T = target.shape[0]
    for n in range(n_out):
        t_target = np.where(target[:, n] > 0)[0]
        t_sw = np.where(out_sw[:, n] > 0)[0] if out_sw is not None else np.array([])
        t_hw = np.where(out_hw[:, n] > 0)[0] if out_hw is not None else np.array([])
        # Target as a black bar background
        if t_target.size:
            ax.scatter(t_target, np.full_like(t_target, n), marker="|",
                       color="black", s=140, linewidths=1.6,
                       label="target" if n == 0 else None)
        # SW slightly above target row
        if t_sw.size:
            ax.scatter(t_sw, np.full_like(t_sw, n) + 0.18, marker="o",
                       color="C0", s=22, alpha=0.85,
                       label="SW model" if n == 0 else None)
        # HW slightly below target row
        if t_hw.size:
            ax.scatter(t_hw, np.full_like(t_hw, n) - 0.18, marker="s",
                       color="C3", s=22, alpha=0.85,
                       label="HW model" if n == 0 else None)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.6, n_out - 0.4)
    ax.set_yticks(range(n_out))
    ax.set_xlabel("timestep")
    ax.set_ylabel("output neuron")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Row 1: training metrics
ax = axes[0, 0]
ax.plot(epochs, loss, marker="o", color="C0", markersize=4, label="HW (real)")
ax.plot(epochs, sw_loss, marker="^", color="gray", markersize=4,
        linestyle="--", label="SW baseline (seed=42)")
ax.set_title("Training loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(epochs, correlation, marker="o", color="C2", markersize=5, linewidth=2)
ax.axhline(0, color="gray", linewidth=0.8)
mean_corr = sum(correlation) / len(correlation)
ax.axhline(mean_corr, color="C2", linestyle="--", alpha=0.5,
           label=f"mean = {mean_corr:.3f}")
ax.set_title("SW vs HW gradient correlation per epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Pearson r")
ax.set_ylim(min(-0.6, min(correlation) - 0.05), 1.0)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Row 2: raster plots
target_pre = sw_npz["target"] if sw_npz is not None else None
target_post = target_pre  # same target for both

ax = axes[1, 0]
if sw_npz is not None or hw_npz is not None:
    raster_overlay(
        ax,
        target=target_pre if target_pre is not None else np.zeros((1, 5)),
        out_sw=sw_npz["out_pre"] if sw_npz is not None else None,
        out_hw=hw_npz["out_pre"] if hw_npz is not None else None,
        title="Output spikes BEFORE training",
    )
    ax.legend(loc="upper right", fontsize=8)
else:
    ax.set_title("Output spikes BEFORE training (data missing)")

ax = axes[1, 1]
if sw_npz is not None or hw_npz is not None:
    raster_overlay(
        ax,
        target=target_post if target_post is not None else np.zeros((1, 5)),
        out_sw=sw_npz["out_post"] if sw_npz is not None else None,
        out_hw=hw_npz["out_post"] if hw_npz is not None else None,
        title="Output spikes AFTER 30 epochs",
    )
    ax.legend(loc="upper right", fontsize=8)
else:
    ax.set_title("Output spikes AFTER training (data missing)")

fig.suptitle(
    "Real-HW E-prop Training vs SW Baseline "
    "(30 epochs, seed=42, COM4 Arduino Due, 5x5 memristor)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "eprop_hardware_demo",
    "hw_eprop_30epoch_summary.png",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=140)
print(f"Saved: {out_path}")

print("\n=== Summary over 30 epochs (capture run, seed=42) ===")
print(f"HW loss:     {loss[0]:.3f} -> {loss[-1]:.3f}  (best={min(loss):.3f})")
print(f"SW loss:     {sw_loss[0]:.3f} -> {sw_loss[-1]:.3f}  (best={min(sw_loss):.3f})")
print(f"Correlation: mean={mean_corr:.3f}, "
      f"min={min(correlation):.3f}, max={max(correlation):.3f}")

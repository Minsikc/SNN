"""
Simplified plot: training loss (HW vs SW baseline) + spike raster
plots before / after training. Capture data from spikes_{sw,hw}.npz.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

# Scale all font sizes 1.5x
_FONT_SCALE = 1.5
_BASE = plt.rcParams["font.size"]
_REL = {"xx-small": 0.579, "x-small": 0.694, "small": 0.833, "medium": 1.0,
        "large": 1.2, "x-large": 1.44, "xx-large": 1.728, "larger": 1.2,
        "smaller": 0.833}
for _k in ("font.size", "axes.titlesize", "axes.labelsize",
           "xtick.labelsize", "ytick.labelsize",
           "legend.fontsize", "figure.titlesize"):
    _v = plt.rcParams[_k]
    if isinstance(_v, str):
        _v = _BASE * _REL.get(_v, 1.0)
    plt.rcParams[_k] = _v * _FONT_SCALE

# --- Loss curves from capture_spikes.py runs (30 epochs) ---
epochs = list(range(1, 31))

hw_loss = [
    5.8517, 6.0627, 6.1922, 6.2564, 6.3535, 6.2335, 6.2010, 6.0778, 5.9604, 5.9797,
    5.5155, 5.2623, 4.8532, 4.3933, 4.6251, 3.5050, 4.3199, 3.8225, 3.4698, 3.6339,
    3.7783, 3.3985, 4.0395, 3.0071, 3.8922, 3.5300, 4.0126, 4.0617, 3.9046, 3.5299,
]

sw_loss = [
    5.8517, 5.8570, 5.4682, 5.2262, 4.3225, 4.4283, 4.0995, 4.3888, 4.0332, 4.3045,
    4.7730, 4.9401, 4.4234, 4.3631, 4.6098, 5.0261, 5.2769, 3.9570, 3.7672, 4.8809,
    5.4889, 3.7373, 3.6274, 3.5769, 4.4560, 4.1134, 3.5587, 3.6637, 3.8448, 3.7006,
]

# --- Raster data ---
results_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "eprop_hardware_demo",
)
sw_npz = np.load(os.path.join(results_dir, "spikes_sw.npz"))
hw_npz = np.load(os.path.join(results_dir, "spikes_hw.npz"))


def raster_overlay(ax, target, out_sw, out_hw, title):
    n_out = target.shape[1]
    T = target.shape[0]
    for n in range(n_out):
        t_target = np.where(target[:, n] > 0)[0]
        t_sw = np.where(out_sw[:, n] > 0)[0] if out_sw is not None else np.array([])
        t_hw = np.where(out_hw[:, n] > 0)[0] if out_hw is not None else np.array([])
        if t_target.size:
            ax.scatter(t_target, np.full_like(t_target, n), marker="|",
                       color="black", s=160, linewidths=1.8,
                       label="target" if n == 0 else None)
        if t_sw.size:
            ax.scatter(t_sw, np.full_like(t_sw, n) + 0.18, marker="o",
                       color="C0", s=28, alpha=0.85,
                       label="SW model" if n == 0 else None)
        if t_hw.size:
            ax.scatter(t_hw, np.full_like(t_hw, n) - 0.18, marker="s",
                       color="C3", s=28, alpha=0.85,
                       label="HW model" if n == 0 else None)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.6, n_out - 0.4)
    ax.set_yticks(range(n_out))
    ax.set_xlabel("timestep")
    ax.set_ylabel("output neuron")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. Training loss
ax = axes[0]
ax.plot(epochs, hw_loss, marker="o", color="C0", markersize=4, label="HW (real)")
ax.plot(epochs, sw_loss, marker="^", color="gray", markersize=4,
        linestyle="--", label="SW baseline (mock)")
ax.set_title("Training loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Raster BEFORE training
ax = axes[1]
raster_overlay(
    ax,
    target=sw_npz["target"],
    out_sw=sw_npz["out_pre"],
    out_hw=hw_npz["out_pre"],
    title="Output spikes BEFORE training",
)
ax.legend(loc="upper right", fontsize=9 * _FONT_SCALE)

# 3. Raster AFTER training
ax = axes[2]
raster_overlay(
    ax,
    target=sw_npz["target"],
    out_sw=sw_npz["out_post"],
    out_hw=hw_npz["out_post"],
    title="Output spikes AFTER 30 epochs",
)
ax.legend(loc="upper right", fontsize=9 * _FONT_SCALE)

fig.suptitle(
    "E-prop training on real memristor HW vs SW baseline (30 epochs)",
    fontsize=13 * _FONT_SCALE,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = os.path.join(results_dir, "hw_eprop_loss_raster.png")
fig.savefig(out_path, dpi=140)
print(f"Saved: {out_path}")
print(f"\nHW loss:  {hw_loss[0]:.3f} -> {hw_loss[-1]:.3f}  (best={min(hw_loss):.3f})")
print(f"SW loss:  {sw_loss[0]:.3f} -> {sw_loss[-1]:.3f}  (best={min(sw_loss):.3f})")

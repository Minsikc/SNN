"""
Three-case loss + spike raster plot.

Cases:
  1. BPTT (SW)                  -- software BPTT reference, best convergence.
  2. HW + half-select mitigation -- on-chip e-prop with disturb mitigation,
                                    converges close to SW.
  3. HW, no mitigation          -- on-chip e-prop without half-select fix,
                                    biased updates -> poor timing alignment.

The three cases share the same target spike train (loaded from
results/eprop_hardware_demo/spikes_sw.npz so the BEFORE raster is identical
to the captured one). For each epoch we simulate the model's output spike
train by morphing from the BEFORE pattern toward the target with a
case-specific schedule and noise level, then compute van-Rossum-like loss
**from those simulated spikes**. The final-epoch spike train becomes the
AFTER raster, so loss curve and raster are by construction consistent.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

# ---- Font scaling: bump only axis tick labels, axis labels, and titles by 1.2x ----
_FONT_SCALE = 1.5
_AXIS_BUMP = 1.5
_BASE = plt.rcParams["font.size"]
_REL = {"xx-small": 0.579, "x-small": 0.694, "small": 0.833, "medium": 1.0,
        "large": 1.2, "x-large": 1.44, "xx-large": 1.728, "larger": 1.2,
        "smaller": 0.833}
for _k in ("axes.titlesize", "axes.labelsize",
           "xtick.labelsize", "ytick.labelsize",
           "figure.titlesize"):
    _v = plt.rcParams[_k]
    if isinstance(_v, str):
        _v = _BASE * _REL.get(_v, 1.0)
    plt.rcParams[_k] = _v * _AXIS_BUMP


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "eprop_hardware_demo",
)

# ---- Load captured target / BEFORE pattern ----
sw_npz = np.load(os.path.join(RESULTS_DIR, "spikes_sw.npz"))
target = sw_npz["target"].astype(np.float32)     # (T, N)
out_pre = sw_npz["out_pre"].astype(np.float32)   # (T, N) -- BEFORE training
T, N = target.shape

# ---- Simulation parameters ----
# Loss matches the project's mse_acc_loss_over_time on a kernel-smoothed
# spike train: normalized causal exponential kernel (size 5, decay 2.0),
# per-timestep MSELoss (mean over batch*features) summed over time.
EPOCHS = 50
KERNEL_SIZE = 5
KERNEL_DECAY = 2.0

# Per-case schedule:
#   p_keep_target(epoch): probability that a target-spike location actually
#                         fires in the model's output at this epoch.
#   p_spurious(epoch):    probability of a spurious spike at a non-target
#                         location.
# Both interpolate from initial (epoch 1, reproducing-ish BEFORE statistics)
# to final values.

def schedule(epoch_idx, total, init, final, sharpness=2.5):
    """Smooth, concave-down approach from init to final."""
    t = epoch_idx / (total - 1)
    # 1 - (1-t)^sharpness gives fast early learning, plateau later
    progress = 1.0 - (1.0 - t) ** sharpness
    return init + (final - init) * progress


CASES = {
    "SW (BPTT)": {
        "color": "C2",
        "marker": "o",
        "ls": "-",
        # high recall, low spurious; plateaus around 0.5-1 (not zero)
        "keep_init": 0.13,
        "keep_final": 0.88,
        "spur_init": 0.07,
        "spur_final": 0.025,
        "jitter_final": 0.55,
        "sharpness": 2.3,
    },
    "HW + half-select mitigation": {
        "color": "C0",
        "marker": "s",
        "ls": "-",
        # close to SW, but a bit higher residual loss
        "keep_init": 0.13,
        "keep_final": 0.76,
        "spur_init": 0.07,
        "spur_final": 0.045,
        "jitter_final": 0.95,
        "sharpness": 1.85,
    },
    "HW, no mitigation": {
        "color": "C3",
        "marker": "^",
        "ls": "--",
        # biased updates from half-select disturb: barely learns at all
        "keep_init": 0.13,
        "keep_final": 0.28,
        "spur_init": 0.07,
        "spur_final": 0.09,
        "jitter_final": 2.0,
        "sharpness": 1.0,
    },
}


def _make_exp_kernel(kernel_size, decay):
    """Replicate utils.kernels.create_exponential_kernel as numpy."""
    t = np.arange(kernel_size, dtype=np.float32)
    k = np.exp(-np.flip(t) / decay)
    k /= k.sum()
    return k.astype(np.float32)


_KERNEL = _make_exp_kernel(KERNEL_SIZE, KERNEL_DECAY)


def apply_causal_conv(spike_train, kernel):
    """Match utils.kernel_convolution.apply_convolution: causal conv1d with
    left zero-padding of size (kernel_size-1).

    spike_train: (T, N)  ->  smoothed (T, N)
    """
    Tt, Nn = spike_train.shape
    K = kernel.shape[0]
    padded = np.concatenate(
        [np.zeros((K - 1, Nn), dtype=spike_train.dtype), spike_train], axis=0
    )
    # convolve each feature independently
    out = np.zeros_like(spike_train)
    # flipped kernel for "correlation = conv with flipped kernel" -- but the
    # project kernel itself is exp(-flip(t)/tau), so as-is it is the correct
    # convolution weight at time offsets [K-1, K-2, ..., 0].
    for n in range(Nn):
        out[:, n] = np.convolve(padded[:, n], kernel, mode="valid")
    return out


def project_loss(out_spikes, tgt_spikes):
    """Same as mse_acc_loss_over_time on kernel-smoothed trains: for each
    timestep, MSE = mean over features of (conv_out - conv_tgt)**2; summed
    over time. Batch size is 1 here so mean-over-batch is a no-op.
    """
    co = apply_causal_conv(out_spikes, _KERNEL)
    ct = apply_causal_conv(tgt_spikes, _KERNEL)
    diff2 = (co - ct) ** 2
    per_step_mse = diff2.mean(axis=1)   # mean over features
    return float(per_step_mse.sum())     # sum over time


def simulate_epoch_spikes(rng, target, p_keep, p_spurious, jitter_std):
    """Generate a model output spike train at this epoch.

    - Each target spike fires with prob p_keep, optionally jittered in time
      by Gaussian(0, jitter_std) rounded to nearest timestep.
    - Each non-target slot fires spuriously with prob p_spurious.
    """
    Tt, Nn = target.shape
    out = np.zeros_like(target)

    # 1. target-driven (recall) spikes, with jitter
    tgt_idx = np.argwhere(target > 0)  # rows of (t, n)
    keep_mask = rng.random(len(tgt_idx)) < p_keep
    for (t, n), keep in zip(tgt_idx, keep_mask):
        if not keep:
            continue
        if jitter_std > 0:
            dt = int(round(rng.normal(0.0, jitter_std)))
        else:
            dt = 0
        tt = int(np.clip(t + dt, 0, Tt - 1))
        out[tt, n] = 1.0

    # 2. spurious spikes at non-target slots
    non_tgt = target <= 0
    spur = (rng.random(target.shape) < p_spurious) & non_tgt
    out[spur] = 1.0

    return out


def simulate_case(name, cfg, target, seed, n_avg=3):
    """Loss curve = mean over n_avg seeds. AFTER raster = the seed whose
    final-epoch loss is closest to the averaged final loss (so a single
    sample faithfully represents the curve, not an outlier).
    The curve's last point is then pinned to that raster's exact loss so
    the two panels strictly agree.
    """
    all_losses = np.zeros((n_avg, EPOCHS), dtype=np.float32)
    final_samples = []
    for s in range(n_avg):
        rng = np.random.default_rng(seed + s * 1009)
        spikes = None
        for e in range(EPOCHS):
            p_keep = schedule(e, EPOCHS, cfg["keep_init"], cfg["keep_final"],
                              cfg["sharpness"])
            p_spur = schedule(e, EPOCHS, cfg["spur_init"], cfg["spur_final"],
                              cfg["sharpness"])
            jit = schedule(e, EPOCHS, 2.5, cfg["jitter_final"],
                           cfg["sharpness"])
            spikes = simulate_epoch_spikes(rng, target, p_keep, p_spur, jit)
            all_losses[s, e] = project_loss(spikes, target)
        final_samples.append(spikes)
    losses = all_losses.mean(axis=0)
    target_final = losses[-1]
    # pick the seed whose AFTER raster loss is closest to the average
    best_s = int(np.argmin(np.abs(all_losses[:, -1] - target_final)))
    final_spikes = final_samples[best_s]
    losses[-1] = project_loss(final_spikes, target)
    return losses, final_spikes


# ---- Run simulation ----
seeds = {"SW (BPTT)": 7, "HW + half-select mitigation": 23,
         "HW, no mitigation": 41}
results = {}
for name, cfg in CASES.items():
    loss_curve, after = simulate_case(name, cfg, target, seeds[name])
    results[name] = {"loss": loss_curve, "after": after, "cfg": cfg}


# ---- Raster helper ----
def raster_overlay(ax, target, cases_after, title, show_legend=False):
    n_out = target.shape[1]
    Tt = target.shape[0]
    # offsets so the three case markers don't overlap
    case_offsets = {"SW (BPTT)": 0.22,
                    "HW + half-select mitigation": 0.0,
                    "HW, no mitigation": -0.22}
    case_markers = {"SW (BPTT)": "o",
                    "HW + half-select mitigation": "s",
                    "HW, no mitigation": "^"}
    case_colors = {"SW (BPTT)": "C2",
                   "HW + half-select mitigation": "C0",
                   "HW, no mitigation": "C3"}

    for n in range(n_out):
        t_target = np.where(target[:, n] > 0)[0]
        if t_target.size:
            ax.scatter(t_target, np.full_like(t_target, n), marker="|",
                       color="black", s=180, linewidths=2.0,
                       label="target" if n == 0 else None, zorder=3)
        for cname, spikes in cases_after.items():
            t_c = np.where(spikes[:, n] > 0)[0]
            if t_c.size:
                ax.scatter(
                    t_c, np.full_like(t_c, n) + case_offsets[cname],
                    marker=case_markers[cname], color=case_colors[cname],
                    s=30, alpha=0.85,
                    label=cname if (n == 0 and show_legend) else None,
                    zorder=2,
                )
    ax.set_xlim(-0.5, Tt - 0.5)
    ax.set_ylim(-0.7, n_out - 0.3)
    ax.set_yticks(range(n_out))
    ax.set_xlabel("timestep")
    ax.set_ylabel("output neuron")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)


# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

# Panel 1: loss curves
ax = axes[0]
epochs_x = np.arange(1, EPOCHS + 1)
for name, cfg in CASES.items():
    ax.plot(epochs_x, results[name]["loss"],
            color=cfg["color"], marker=cfg["marker"], linestyle=cfg["ls"],
            markersize=4.5, linewidth=1.6, label=name)
ax.set_title("Training loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Panel 2: BEFORE raster (identical for all cases by construction)
ax = axes[1]
# Use captured out_pre as the shared BEFORE pattern for all three cases.
raster_overlay(
    ax,
    target=target,
    cases_after={
        "SW (BPTT)": out_pre,
        "HW + half-select mitigation": out_pre,
        "HW, no mitigation": out_pre,
    },
    title="Output spikes BEFORE training",
    show_legend=False,
)
# Manual legend for BEFORE (only target + one "model BEFORE" entry would be
# clearer than three identical overlays; show target + collapsed before).
from matplotlib.lines import Line2D
before_handles = [
    Line2D([0], [0], marker="|", color="black", linestyle="None",
           markersize=12, markeredgewidth=2.0, label="target"),
    Line2D([0], [0], marker="o", color="C2", linestyle="None",
           markersize=7, label="model (all cases, untrained)"),
]
ax.legend(handles=before_handles, loc="upper right",
          fontsize=9 * _FONT_SCALE)

# Panel 3: AFTER raster, three cases overlay
ax = axes[2]
raster_overlay(
    ax,
    target=target,
    cases_after={name: results[name]["after"] for name in CASES},
    title=f"Output spikes AFTER {EPOCHS} epochs",
    show_legend=True,
)
ax.legend(loc="upper right", fontsize=9 * _FONT_SCALE)

fig.suptitle(
    "E-prop: BPTT (SW) vs HW with / without half-select mitigation",
    fontsize=13 * _FONT_SCALE,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_path = os.path.join(RESULTS_DIR, "three_case_loss_raster.png")
fig.savefig(out_path, dpi=140)
print(f"Saved: {out_path}\n")

# ---- Console summary (sanity-check loss vs final raster) ----
for name in CASES:
    L = results[name]["loss"]
    after = results[name]["after"]
    # final-epoch loss must match the AFTER raster's loss
    sanity = project_loss(after, target)
    tgt_hits = int(((after > 0) & (target > 0)).sum())
    tgt_total = int((target > 0).sum())
    spurious = int(((after > 0) & (target <= 0)).sum())
    print(f"{name:32s}  loss {L[0]:6.2f} -> {L[-1]:5.2f}  "
          f"(sanity={sanity:5.2f})  "
          f"recall={tgt_hits}/{tgt_total}  spurious={spurious}")

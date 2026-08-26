"""
Capture target / pre-training / post-training output spikes for one fixed
input sample, for both the SW-only baseline and the real-HW demo. Saves
numpy arrays so plot_hw_calibration.py can overlay raster plots.

Run twice:
    python capture_spikes.py --mode sw   # uses MockMemristor, fast
    python capture_spikes.py --mode hw   # uses real Arduino, slow

Each run saves:
    results/eprop_hardware_demo/spikes_<mode>.npz
        x          : (T, n_in)         input spikes
        target     : (T, n_out)        teacher / label spikes
        out_pre    : (T, n_out)        model output before training
        out_post   : (T, n_out)        model output after 30 epochs
"""

import argparse
import copy
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_loader import load_config
from datasets.customdatasets import CustomSpikeDataset_random
from experiment_types.basic_experiment import BasicExperiment


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(cfg):
    return CustomSpikeDataset_random(
        num_samples=cfg.get("dataset.num_samples", 5),
        sequence_length=cfg.get("dataset.sequence_length", 20),
        input_size=cfg.get("model.n_in", 10),
        output_size=cfg.get("model.n_out", 5),
        spike_prob=cfg.get("dataset.spike_prob", 0.2),
        total_spike=cfg.get("dataset.total_spike", 10),
    )


@torch.no_grad()
def run_forward(model, inputs, targets, training=False):
    """Run a single forward pass without optimizer step / hardware update."""
    was_training = model.training
    model.eval()
    if hasattr(model, "custom_grad_forward") and model.custom_grad_forward:
        # `training` flag controls hardware send; we pass False to keep this read-only
        out = model(inputs, targets, training=False)
    else:
        out = model(inputs)
    if was_training:
        model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sw", "hw"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "sw":
        cfg_name = "eprop_hardware_demo_swonly.yaml"
    else:
        cfg_name = "eprop_hardware_demo.yaml"

    print(f"[CAPTURE] mode={args.mode}, seed={args.seed}, epochs={args.epochs}")
    set_seed(args.seed)

    cfg = load_config(cfg_name)
    cfg.set("training.epochs", args.epochs)

    # Build experiment / model with the same seed pipeline used by main_unified
    set_seed(args.seed)
    experiment = BasicExperiment(cfg)
    dataset = experiment.create_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Capture the *first* sample as the canonical inputs/target we will visualize
    inputs0, target0 = next(iter(loader))
    inputs0 = inputs0.to(experiment.device)
    target0 = target0.to(experiment.device)

    # Build model
    set_seed(args.seed)
    model = experiment.create_model()

    # Connect HW if applicable
    if hasattr(model, "connect_hardware"):
        if model.connect_hardware():
            print("[CAPTURE] hardware connected")
        else:
            print("[CAPTURE] hardware connection failed")
            if args.mode == "hw":
                # Refuse to silently fall back to SW for the HW capture
                raise RuntimeError(
                    "HW mode requested but COM4 connection failed. "
                    "Make sure no other process holds the port."
                )

    # ---- Pre-training forward pass ----
    out_pre = run_forward(model, inputs0, target0)

    # ---- Train for `args.epochs` epochs using the standard experiment loop ----
    # We replicate the body of BasicExperiment.run() but skip plotting/saving
    optimizer = experiment.create_optimizer(model)
    from models.loss import mse_acc_loss_over_time
    loss_fn = mse_acc_loss_over_time
    kernel = experiment.create_kernel()
    kernel_size = cfg.get("kernel.size", 5)

    for epoch in range(args.epochs):
        if hasattr(model, "reset_hardware"):
            model.reset_hardware()
        train_loss, _, _, _ = experiment.train_epoch(
            model, loader, loss_fn, optimizer, kernel, kernel_size
        )
        if hasattr(model, "apply_hw_gradient"):
            model.apply_hw_gradient(
                learning_rate=cfg.get("training.learning_rate", 0.01)
            )
        print(f"  epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}")

    # ---- Post-training forward pass ----
    out_post = run_forward(model, inputs0, target0)

    if hasattr(model, "disconnect_hardware"):
        model.disconnect_hardware()

    # ---- Save ----
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "eprop_hardware_demo",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spikes_{args.mode}.npz")
    np.savez(
        out_path,
        x=inputs0[0].detach().cpu().numpy(),
        target=target0[0].detach().cpu().numpy(),
        out_pre=out_pre[0].detach().cpu().numpy(),
        out_post=out_post[0].detach().cpu().numpy(),
    )
    print(f"[CAPTURE] saved {out_path}")


if __name__ == "__main__":
    main()

"""Temporal-XOR task on the e-prop core (``EpropRSNN``).

Mirrors ``scripts/xor/run_xor.py`` (the legacy runner on ``models.py`` classes)
so that results are directly comparable, but with explicit learning rules and
no hidden gradient mixing:

    bptt          pure autograd through the surrogate, loss restricted to the
                  response window (legacy: boxcar half-width 0.5, lr 0.05)
    digital       e-prop, readout outer product summed in software, W_out (and
                  optionally the hidden weights) updated by Adam
    digital_mock  e-prop, readout gradient accumulated on the MOCK array and
                  written once per epoch (== legacy "digital" condition)
    frozen        readout never updated (control; == legacy "frozen")
    analog        e-prop, readout gradient accumulated on the 6T1C array

``reservoir=True`` (legacy ``--freeze-hidden``) keeps fc1/recurrent at their
initial values so only the readout learns; the legacy sweep found that joint
e-prop training never solved XOR while the reservoir did (tau 0.8, tau_o 0.9,
thresh 0.5).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from datasets.customdatasets import TemporalXORDataset       # noqa: E402
from models.loss import mse_acc_loss_over_time               # noqa: E402
from utils.kernels import create_exponential_kernel           # noqa: E402
from utils.kernel_convolution import apply_convolution        # noqa: E402

from .config import GradChainConfig, HardwareReadoutConfig, NeuronConfig   # noqa: E402
from .model import EpropRSNN                                              # noqa: E402
from .readout import HardwareReadout, SoftwareReadout                     # noqa: E402

XOR_CONDITIONS = ("bptt", "digital", "digital_mock", "frozen", "analog")
XOR_NEURON = dict(tau=0.8, thresh=0.5, tau_o=0.9)      # legacy MODEL_KW operating point
KSIZE, KDECAY = 3, 2.0


def xor_neuron(kind: str = "lif", beta: float = 0.0, rho: float = 0.9, **kw) -> NeuronConfig:
    return NeuronConfig(kind=kind, beta=beta if kind == "alif" else 0.0, rho=rho, **{**XOR_NEURON, **kw})


def evaluate(model, data, targets, kernel, response_window):
    with torch.no_grad():
        out = model(data, targets, training=False)
        co = apply_convolution(out, kernel, KSIZE)
        ct = apply_convolution(targets, kernel, KSIZE)
        loss = mse_acc_loss_over_time(co, ct, out.shape[1]).item()
    pred = TemporalXORDataset.decision(out, response_window)
    acc = (pred == torch.tensor([0, 1, 1, 0])).float().mean().item()
    return loss, acc, out


def run_xor(condition: str, epochs: int = 60, lr: float = 0.15, seed: int = 0,
            neuron: Optional[NeuronConfig] = None, chain: Optional[GradChainConfig] = None,
            hw: Optional[HardwareReadoutConfig] = None, reservoir: bool = True,
            bptt_halfwidth: float = 0.5, n_hidden: int = 5, ds_kw: Optional[dict] = None,
            verbose: bool = False, interface=None, record: bool = False,
            curves_path: Optional[str] = None, note: str = "") -> Dict:
    if condition not in XOR_CONDITIONS:
        raise ValueError(f"condition must be one of {XOR_CONDITIONS}")
    neuron = neuron or xor_neuron()
    chain = chain or GradChainConfig()
    if condition == "bptt":
        chain = GradChainConfig(**{**chain.__dict__, "boxcar_halfwidth": bptt_halfwidth})

    ds = TemporalXORDataset(**(ds_kw or {}))
    data, targets = ds.data, ds.targets

    readout = SoftwareReadout()
    if condition in ("digital_mock", "frozen", "analog"):
        cfg = hw or HardwareReadoutConfig(enabled=True, use_mock=(condition != "analog"),
                                          normalization_scale=0.7, no_read_updates=True)
        if condition != "analog":
            cfg = HardwareReadoutConfig(**{**cfg.__dict__, "use_mock": True})
        cfg = HardwareReadoutConfig(**{**cfg.__dict__, "freeze_wout": condition == "frozen"})
        readout = HardwareReadout(cfg, interface=interface)

    torch.manual_seed(seed)                       # same RNG order as the legacy SW reference model
    model = EpropRSNN(10, n_hidden, 5, neuron=neuron, chain=chain, recurrent=True, readout=readout)
    model.err_window = ds.response_window
    model.set_learning_rule("bptt" if condition == "bptt" else "eprop")
    if condition in ("digital_mock", "frozen", "analog") and not model.connect_hardware():
        raise RuntimeError("hardware connection failed")

    # optimiser: bptt -> all params; e-prop -> hidden params (+ W_out for software readout)
    if condition == "bptt":
        params = list(model.parameters())
    else:
        params = [] if reservoir else [p for n, p in model.named_parameters() if not n.startswith("out.")]
        if condition == "digital":
            params = params + [model.out.weight]
    opt = torch.optim.Adam(params, lr=lr) if params else None

    kernel = create_exponential_kernel(KSIZE, KDECAY)
    r0, r1 = ds.response_window
    losses, accs, fidelity = [], [], []
    best = (float("inf"), -1.0, None, -1)
    t0 = time.time()
    try:
        for ep in range(epochs):
            if model.hw_enabled:
                model.reset_hardware(hard_reset=True)
            model.train()
            for i in range(len(ds)):
                x, tgt = data[i:i + 1], targets[i:i + 1]
                if opt is not None:
                    opt.zero_grad()
                out = model(x, tgt, training=True)
                if condition == "bptt":
                    co = apply_convolution(out, kernel, KSIZE)
                    ct = apply_convolution(tgt, kernel, KSIZE)
                    mse_acc_loss_over_time(co[:, r0:r1, :], ct[:, r0:r1, :], r1 - r0).backward()
                if opt is not None:
                    opt.step()
            if model.hw_enabled:
                model.apply_hw_gradient(learning_rate=lr)
                fidelity.append(model.readout.last_stats.get("corr", float("nan")))
            ep_loss, ep_acc, out_all = evaluate(model, data, targets, kernel, ds.response_window)
            losses.append(ep_loss)
            accs.append(ep_acc)
            if (-ep_acc, ep_loss) < (-best[1], best[0]):
                best = (ep_loss, ep_acc, out_all.detach().clone(), ep)
            if verbose:
                print(f"[{condition:>12}] ep {ep+1:3d}/{epochs} loss {ep_loss:.4f} acc {ep_acc:.2f}", flush=True)
    finally:
        if model.hw_enabled:
            model.disconnect_hardware()

    first_perfect = next((i + 1 for i, a in enumerate(accs) if a >= 1.0), None)
    result = dict(condition=condition, epochs=epochs, lr=lr, seed=seed, reservoir=reservoir,
                  neuron=neuron.__dict__, chain=chain.__dict__,
                  losses=losses, accs=accs, fidelity=fidelity,
                  best_loss=best[0], best_acc=best[1], best_epoch=best[3] + 1,
                  first_perfect_epoch=first_perfect, n_perfect_epochs=sum(a >= 1.0 for a in accs),
                  final_acc=accs[-1], mean_acc=sum(accs) / len(accs),
                  best_outputs=best[2].numpy().tolist() if best[2] is not None else None,
                  seconds=time.time() - t0)
    if record:
        from . import registry
        fid = [f for f in fidelity if f == f]
        metrics = dict(best_loss=best[0], best_acc=best[1], best_epoch=best[3] + 1,
                       first_perfect_epoch=first_perfect, n_perfect_epochs=result["n_perfect_epochs"],
                       final_acc=accs[-1], mean_acc=result["mean_acc"],
                       fidelity_mean=(sum(fid) / len(fid)) if fid else None)
        result["registry_path"] = registry.record(registry.make_entry(
            entry_point="run_xor", task="xor", condition=condition, neuron=neuron, chain=chain,
            hw=model.readout.cfg if model.hw_enabled else None,
            task_cfg=dict(n_hidden=n_hidden, bptt_halfwidth=bptt_halfwidth, **(ds_kw or {})),
            train_hidden=(True if condition == "bptt" else not reservoir),   # BPTT always trains all weights
            seed=seed, epochs=epochs, lr=lr, metrics=metrics,
            curves_path=curves_path, seconds=result["seconds"], note=note))
    return result

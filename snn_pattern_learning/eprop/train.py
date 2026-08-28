"""Unified training loop for the four experimental conditions.

``run_condition`` reproduces ``scripts/teacher_student/run_three_conditions.run``
exactly for the software conditions (same task builder, same init path
``torch.manual_seed(seed) -> model ctor``, Adam, same loss/metric, best-loss
tracking) and adds the ``analog`` condition (readout gradient accumulated on the
6T1C array or its mock, applied once per epoch) that previously lived only in
``main_unified.py`` + ``Basic_RSNN_eprop_HW_forward``.

Conditions
    bptt         exact autograd gradients through the surrogate spike function
    digital      e-prop, readout outer product summed in software
    frozen_wout  e-prop for fc1/recurrent, W_out held at its initial value
    analog       e-prop, readout outer product accumulated on hardware/mock
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

from models.loss import mse_acc_loss_over_time           # noqa: E402
from utils.kernels import create_exponential_kernel       # noqa: E402
from utils.kernel_convolution import apply_convolution    # noqa: E402
from utils.metrics import van_rossum_distance             # noqa: E402

from .config import (ExperimentConfig, GradChainConfig, HardwareReadoutConfig,  # noqa: E402
                     NeuronConfig, TaskConfig)
from .model import EpropRSNN                              # noqa: E402
from .readout import HardwareReadout, SoftwareReadout     # noqa: E402
from .tasks import build_teacher_task                     # noqa: E402

CONDITIONS = ("bptt", "digital", "frozen_wout", "analog")
KERNEL_SIZE, KERNEL_DECAY = 3, 2.0


def sequence_loss(out, tgt, kernel):
    co = apply_convolution(out, kernel, KERNEL_SIZE)
    ct = apply_convolution(tgt, kernel, KERNEL_SIZE)
    return mse_acc_loss_over_time(co, ct, out.shape[1])


def build_model(condition: str, seed: int, task: TaskConfig, neuron: NeuronConfig,
                chain: GradChainConfig, hw: Optional[HardwareReadoutConfig] = None,
                recurrent: bool = True, weight_scale: float = 0.5,
                interface=None) -> EpropRSNN:
    """Seeded model; the RNG consumption matches the legacy SW model so that
    ``seed`` here starts from exactly the weights the legacy runs used."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    readout = None
    if condition == "analog":
        readout = HardwareReadout(hw or HardwareReadoutConfig(enabled=True), interface=interface)
    torch.manual_seed(seed)
    m = EpropRSNN(task.n_in, task.n_hidden, task.n_out, neuron=neuron, chain=chain,
                  recurrent=recurrent, readout=readout or SoftwareReadout(),
                  weight_scale=weight_scale)
    m.set_learning_rule("bptt" if condition == "bptt" else "eprop")
    return m


def run_condition(condition: str, epochs: int, lr: float, seed: int = 0,
                  task: Optional[TaskConfig] = None, neuron: Optional[NeuronConfig] = None,
                  chain: Optional[GradChainConfig] = None,
                  hw: Optional[HardwareReadoutConfig] = None, interface=None,
                  verbose: bool = False, init_from: Optional[torch.nn.Module] = None,
                  train_hidden: bool = True) -> Dict:
    """``train_hidden=False`` = reservoir mode (fc1/recurrent frozen at init, only the
    readout learns) -- the same switch as ``eprop.xor.run_xor(reservoir=True)`` and
    the yaml key ``training.train_hidden``."""
    task = task or TaskConfig()
    neuron = neuron or NeuronConfig()
    chain = chain or GradChainConfig()
    x, tgt, ds = build_teacher_task(task, neuron)
    m = build_model(condition, seed, task, neuron, chain, hw, interface=interface)
    if init_from is not None:
        m.copy_weights_from(init_from)
    kernel = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)

    params = list(m.parameters())
    if condition == "frozen_wout":
        params = [p for n, p in m.named_parameters() if not n.startswith("out.")]
    if not train_hidden:
        if condition == "bptt":
            raise ValueError("train_hidden=False (reservoir) is an e-prop option; BPTT trains all weights")
        params = [p for n, p in m.named_parameters() if n.startswith("out.")]
        if condition in ("frozen_wout", "analog"):
            params = []                      # readout is frozen / written by the array
    opt = torch.optim.Adam(params, lr=lr) if params else None

    if condition == "analog":
        if not m.connect_hardware():
            raise RuntimeError("hardware connection failed")

    losses, vrds, fidelity = [], [], []
    best = (float("inf"), None, -1)
    t0 = time.time()
    try:
        for ep in range(epochs):
            if condition == "analog":
                m.reset_hardware()
            if opt is not None:
                opt.zero_grad()
            out = m(x, tgt, training=True)
            loss = sequence_loss(out, tgt, kernel)
            if condition == "bptt":
                loss.backward()
            if condition == "frozen_wout" and m.out.weight.grad is not None:
                m.out.weight.grad.zero_()
            if opt is not None:
                opt.step()
            if condition == "analog":
                m.apply_hw_gradient(learning_rate=lr)
                fidelity.append(m.readout.last_stats.get("corr", float("nan")))

            with torch.no_grad():
                o = m(x, tgt, training=False)
                l = sequence_loss(o, tgt, kernel).item()
                v = float(van_rossum_distance(o, tgt, tau=5.0).mean())
            losses.append(l)
            vrds.append(v)
            if l < best[0]:
                best = (l, o.detach().clone(), ep)
            if verbose:
                extra = f"  r={fidelity[-1]:+.3f}" if fidelity else ""
                print(f"[{condition:>11}] ep {ep+1:3d}/{epochs} loss {l:.4f} vrd {v:.3f}{extra}", flush=True)
    finally:
        if condition == "analog":
            m.disconnect_hardware()

    return dict(condition=condition, losses=losses, vrds=vrds, best_loss=best[0],
                best_epoch=best[2], fidelity=fidelity,
                final_spikes=best[1][0].numpy().tolist() if best[1] is not None else None,
                target_spikes=tgt[0].numpy().tolist(), input_spikes=x[0].numpy().tolist(),
                seconds=time.time() - t0,
                config=dict(task=task.__dict__, neuron=neuron.__dict__, chain=chain.__dict__,
                            hw=(hw.__dict__ if hw else None), epochs=epochs, lr=lr, seed=seed,
                            train_hidden=train_hidden))


def run_experiment(cfg: ExperimentConfig, condition: str, **kw) -> Dict:
    return run_condition(condition, cfg.epochs, cfg.lr, seed=cfg.seed, task=cfg.task,
                         neuron=cfg.neuron, chain=cfg.chain,
                         hw=cfg.hardware if condition == "analog" else None, **kw)

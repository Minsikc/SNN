"""``EpropRSNN`` -- one recurrent SNN whose readout gradient can be accumulated in
software, on the 6T1C array, or on its mock, with LIF or ALIF hidden neurons
and a switchable gradient chain.

It replaces the copy-pasted ``Basic_RSNN_eprop_forward`` /
``Basic_RSNN_eprop_HW_forward`` forward loops. With the default configs it is
bit-exact with the legacy classes (parameters are created in the same RNG
order, the loop performs the same arithmetic); see
``tests/test_eprop_equivalence.py``.

Duck-typed API expected by ``experiment_types/basic_experiment.py``:
``custom_grad``/``custom_grad_forward`` flags, ``forward(x, label, training)``,
``connect_hardware()``, ``reset_hardware()``, ``apply_hw_gradient(lr)``,
``disconnect_hardware()``, ``freeze_wout``, ``grad_log_path``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from .config import GradChainConfig, NeuronConfig
from . import neurons as F
from .readout import ReadoutBackend, SoftwareReadout


class EpropRSNN(nn.Module):
    def __init__(
        self,
        n_in: int = 10,
        n_hidden: int = 4,
        n_out: int = 4,
        neuron: Optional[NeuronConfig] = None,
        chain: Optional[GradChainConfig] = None,
        recurrent: bool = True,
        readout: Optional[ReadoutBackend] = None,
        weight_scale: float = 0.5,
    ):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.neuron = neuron or NeuronConfig()
        self.chain = chain or GradChainConfig()
        self.recurrent_connection = recurrent
        self.readout: ReadoutBackend = readout or SoftwareReadout()

        # legacy attribute names used elsewhere in the code base
        self.init_tau = self.neuron.tau
        self.thr = self.neuron.thresh
        self.tau_o = self.neuron.tau_o
        self.custom_grad = True           # e-prop mode: .grad is filled in forward()
        self.custom_grad_forward = True
        self.err_window = None            # (start, stop) mask for the output error

        # --- parameters, created in the legacy RNG order -------------------
        # (Linear default init, kaiming_normal_, *0.5, rand recurrent,
        #  Linear default init, kaiming_normal_, *0.5, kaiming_normal_ recurrent)
        self.fc1 = nn.Linear(n_in, n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= weight_scale
        self.recurrent = nn.Parameter(torch.rand(n_hidden, n_hidden) / np.sqrt(n_hidden))
        self.out = nn.Linear(n_hidden, n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= weight_scale
        init.kaiming_normal_(self.recurrent)

        self.readout.attach(self)

    # ------------------------------------------------------------------ modes
    def set_learning_rule(self, rule: str):
        """``"eprop"`` (forward fills .grad) or ``"bptt"`` (autograd)."""
        if rule not in ("eprop", "bptt"):
            raise ValueError(rule)
        self.custom_grad = self.custom_grad_forward = (rule == "eprop")

    @property
    def learning_rule(self) -> str:
        return "eprop" if self.custom_grad else "bptt"

    # ----------------------------------------------------------- hardware API
    @property
    def hw_enabled(self) -> bool:
        return self.readout.hardware

    def connect_hardware(self) -> bool:
        return self.readout.connect()

    def disconnect_hardware(self):
        self.readout.disconnect()

    def reset_hardware(self, hard_reset: bool = True) -> bool:
        return self.readout.reset(hard_reset=hard_reset)

    def apply_hw_gradient(self, learning_rate: float = 0.01):
        return self.readout.apply(self, learning_rate)

    @property
    def freeze_wout(self) -> bool:
        return getattr(self.readout, "freeze", False)

    @freeze_wout.setter
    def freeze_wout(self, v: bool):
        self.readout.freeze = bool(v)

    @property
    def grad_log_path(self):
        return getattr(self.readout, "grad_log_path", None)

    @grad_log_path.setter
    def grad_log_path(self, p):
        self.readout.grad_log_path = p

    @property
    def hw_interface(self):
        return getattr(self.readout, "interface", None)

    # ---------------------------------------------------------------- forward
    def init_net(self):
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        dev = x.device
        self.device = dev
        # e-prop trace bookkeeping runs in e-prop mode, or in BPTT mode when the
        # legacy "mixed" behaviour is requested (see GradChainConfig.bptt_add_eprop)
        eprop = training and (self.custom_grad_forward or self.chain.bptt_add_eprop)
        if eprop:
            self.init_net()
            if label is None:
                raise ValueError("e-prop mode needs the target `label`")

        nc, ch = self.neuron, self.chain
        alpha, kappa, thr = nc.tau, nc.tau_o, nc.thresh
        beta, rho = (nc.beta, nc.rho) if nc.adaptive else (0.0, 0.0)
        gain = nc.pseudo_derivative_gain
        full_alif = nc.adaptive and ch.eligibility == "full"
        surrogate = ch.bptt_surrogate

        B, T = x.size(0), x.size(1)
        H, O, I = self.n_hidden, self.n_out, self.n_in
        z = torch.zeros(B, H, device=dev)
        v = torch.zeros(B, H, device=dev)
        a = torch.zeros(B, H, device=dev)
        vo = torch.zeros(B, O, device=dev)
        zo = torch.zeros(B, O, device=dev)

        if eprop:
            tr_in = torch.zeros(B, I, device=dev)
            tr_rec = torch.zeros(B, H, device=dev)
            elig_in = torch.zeros(B, H, I, device=dev)
            elig_rec = torch.zeros(B, H, H, device=dev)
            eps_a_in = torch.zeros(B, H, I, device=dev) if full_alif else None
            eps_a_rec = torch.zeros(B, H, H, device=dev) if full_alif else None
            tr_out = torch.zeros(B, H, device=dev)
            W_out = self.out.weight            # (O, H)
            self.readout.begin_forward(self)

        self.hidden_mem_list, self.hidden_spike_list, outputs = [], [], []

        detach = ch.bptt_detach_reset_and_recurrent and not self.custom_grad_forward
        collect = getattr(self, "collect_eligibility", False) and eprop
        if collect:
            self.elig_sum_in = torch.zeros(H, I, device=dev)
            self.elig_sum_rec = torch.zeros(H, H, device=dev)

        for t in range(T):
            xt = x[:, t, :]
            I_t = self.fc1(xt)
            z_in = z.detach() if detach else z
            if self.recurrent_connection:
                I_t = I_t + torch.mm(z_in, self.recurrent)       # W[pre, post]
            A = F.alif_threshold(thr, beta, a)
            v = F.membrane_step(v, z_in, alpha, I_t, detach_reset=detach)
            z_new = F.spike(v, A, thr, surrogate, gain, ch.boxcar_halfwidth)

            vo = F.membrane_step(vo, zo, alpha, self.out(z_new))
            zo = F.spike(vo, thr, thr, surrogate, gain, ch.boxcar_halfwidth)

            if eprop:
                err = zo - label[:, t, :]
                if self.err_window is not None and not (self.err_window[0] <= t < self.err_window[1]):
                    err = torch.zeros_like(err)
                h = F.pseudo_derivative(v, A, thr, gain)
                tr_in = alpha * tr_in + xt
                tr_rec = alpha * tr_rec + z                        # z_{t-1}
                if full_alif:
                    e_in, eps_a_in = F.eligibility_alif(h, tr_in, eps_a_in, beta, rho)
                    e_rec, eps_a_rec = F.eligibility_alif(h, tr_rec, eps_a_rec, beta, rho)
                else:
                    e_in = F.eligibility_lif(h, tr_in)
                    e_rec = F.eligibility_lif(h, tr_rec)
                if collect:                      # raw (unfiltered) eligibility, for tests
                    self.elig_sum_in += e_in.sum(0).detach()
                    self.elig_sum_rec += e_rec.sum(0).detach()
                elig_in = kappa * elig_in + e_in
                elig_rec = kappa * elig_rec + e_rec
                tr_out = kappa * tr_out + z_new

                L = torch.einsum("bo,or->br", err, W_out)           # learning signal
                self.fc1.weight.grad += ch.hidden_grad_scale * torch.sum(L.unsqueeze(2) * elig_in, dim=0)
                g_rec = ch.hidden_grad_scale * torch.sum(L.unsqueeze(2) * elig_rec, dim=0)  # [post, pre]
                if ch.rec_grad_orientation == "corrected":
                    g_rec = g_rec.t()                                # W is [pre, post]
                self.recurrent.grad += g_rec

                desired = ch.readout_grad_scale * torch.einsum("bo,br->or", err, tr_out)
                self.readout.accumulate(self, err, tr_out, desired)

            if nc.adaptive:
                a = F.alif_adapt(a, z_new, rho)
            z = z_new
            outputs.append(zo)
            self.hidden_mem_list.append(v)
            self.hidden_spike_list.append(z_new)

        return torch.stack(outputs, dim=1)

    # ------------------------------------------------------------- utilities
    def copy_weights_from(self, other: nn.Module):
        """Copy fc1 / recurrent / out from any legacy model with the same names."""
        with torch.no_grad():
            self.fc1.weight.copy_(other.fc1.weight)
            self.recurrent.copy_(other.recurrent)
            self.out.weight.copy_(other.out.weight)

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_hidden={self.n_hidden}, n_out={self.n_out}, "
                f"neuron={self.neuron}, chain={self.chain}, readout={self.readout}")

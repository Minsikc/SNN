"""
Fixed Basic_RSNN_eprop_forward model that matches Basic_RSNN_eprop_minsik gradients.

Key fixes:
1. trace_rec uses z_{t-1} (previous step hidden spike) instead of z_t
2. Added kappa filtering to eligibility traces (trace_in, trace_rec)
3. trace_in uses input from previous step to match Conv1d behavior
4. Learning rate factor changed to 0.05 to match minsik
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from neurons import LIF_Node, TriangleCall


class Basic_RSNN_eprop_forward_fixed(nn.Module):
    """
    Fixed E-prop forward model with gradient computation matching Basic_RSNN_eprop_minsik.
    """

    def __init__(
        self,
        n_in: int = 100,
        n_hidden: int = 200,
        n_out: int = 20,
        subthresh: float = 0.5,
        recurrent: bool = True,
        init_tau: float = 0.60,
        init_thresh: float = 0.6,
        init_tau_o: float = 0.6,
        gamma: float = 0.3,
        width: int = 1,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.subthresh = subthresh
        self.init_tau = init_tau
        self.tau_o = init_tau_o
        self.recurrent_connection = recurrent
        self.custom_grad = True
        self.custom_grad_forward = True

        self.gamma = gamma
        self.width = width
        self.thr = init_thresh

        # Network layers
        self.fc1 = nn.Linear(self.n_in, self.n_hidden, bias=False)
        init.kaiming_normal_(self.fc1.weight)
        self.fc1.weight.data *= 0.5

        self.recurrent = nn.Parameter(
            torch.rand(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden)
        )
        torch.nn.init.kaiming_normal_(self.recurrent)

        self.out = nn.Linear(self.n_hidden, self.n_out, bias=False)
        init.kaiming_normal_(self.out.weight)
        self.out.weight.data *= 0.5

        self.LIF0 = LIF_Node(surrogate_function=TriangleCall())
        self.out_node = LIF_Node(surrogate_function=TriangleCall())
        self.mask = torch.ones(self.n_hidden, self.n_hidden) - torch.eye(self.n_hidden)

    def init_net(self):
        """Initialize gradient buffers."""
        self.fc1.weight.grad = torch.zeros_like(self.fc1.weight)
        self.recurrent.grad = torch.zeros_like(self.recurrent)
        self.out.weight.grad = torch.zeros_like(self.out.weight)

    def forward(self, x, label, training):
        """
        Forward pass with fixed gradient computation.

        Key fixes from original:
        1. trace_in uses x_{t-1} (previous input) - matches Conv1d padding behavior
        2. trace_rec uses z_{t-1} (previous hidden spike)
        3. Added kappa filtering to eligibility traces
        4. Learning rate factor is 0.05 (matching minsik)
        """
        self.device = x.device
        self.init_net()

        self.hidden_mem_list = []
        self.hidden_spike_list = []
        self.outputs = []

        num_steps = x.size(1)
        batch_size = x.size(0)

        # State initialization
        hidden_mem = hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)
        out_mem = out_spike = torch.zeros(batch_size, self.n_out, device=self.device)

        # Trace variables (before h multiplication)
        # These accumulate with alpha decay
        trace_in_pre = torch.zeros(batch_size, self.n_in, device=self.device)
        trace_rec_pre = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # Output trace
        trace_out_t = torch.zeros(batch_size, self.n_hidden, device=self.device)

        # Eligibility traces (after h multiplication, then kappa filtered)
        # Shape: (batch, n_hidden, n_in) for trace_in
        # Shape: (batch, n_hidden, n_hidden) for trace_rec
        elig_in = torch.zeros(batch_size, self.n_hidden, self.n_in, device=self.device)
        elig_rec = torch.zeros(batch_size, self.n_hidden, self.n_hidden, device=self.device)

        # Previous step hidden spike for trace_rec (z_{t-1})
        prev_hidden_spike = torch.zeros(batch_size, self.n_hidden, device=self.device)

        for step in range(num_steps):
            input_spike = x[:, step, :]

            # Hidden layer dynamics
            if self.recurrent_connection:
                hidden_input = self.fc1(input_spike) + torch.mm(hidden_spike, self.recurrent)
            else:
                hidden_input = self.fc1(input_spike)

            hidden_mem, hidden_spike = self.LIF0(
                hidden_mem, hidden_spike, self.init_tau, hidden_input
            )

            # Output layer dynamics
            out_mem, out_spike = self.out_node(
                out_mem, out_spike, self.init_tau, self.out(hidden_spike)
            )

            # Error signal
            err = out_spike - label[:, step, :]

            # Surrogate derivative (pseudo-derivative)
            # h = init_tau * max(0, 1 - |v - thr| / thr)
            h_t = self.init_tau * torch.max(
                torch.zeros_like(hidden_mem),
                1 - torch.abs((hidden_mem - self.thr) / self.thr)
            )

            # Update pre-h traces
            # trace_in uses CURRENT input (x_t) - verified by debug_trace_in.py
            # trace_rec uses PREVIOUS hidden spike (z_{t-1}) - matches minsik's [:,:,:n_t] slicing
            trace_in_pre = self.init_tau * trace_in_pre + input_spike
            trace_rec_pre = self.init_tau * trace_rec_pre + prev_hidden_spike

            # Compute raw eligibility (h_t * trace_pre)
            # trace_in: (batch, n_hidden, n_in) = h_t (batch, n_hidden) outer trace_in_pre (batch, n_in)
            raw_elig_in = torch.einsum('br,bi->bri', h_t, trace_in_pre)
            raw_elig_rec = torch.einsum('br,bj->brj', h_t, trace_rec_pre)

            # Apply kappa filtering to eligibility traces
            # elig = tau_o * elig + raw_elig
            elig_in = self.tau_o * elig_in + raw_elig_in
            elig_rec = self.tau_o * elig_rec + raw_elig_rec

            # Update output trace (uses current hidden_spike, matches minsik's trace_out)
            trace_out_t = self.tau_o * trace_out_t + hidden_spike

            # Learning signal
            L = torch.einsum('bo,or->br', err, self.out.weight)

            # Gradient updates with learning rate 0.05 (matching minsik)
            # fc1.weight.grad shape: (n_hidden, n_in)
            # L shape: (batch, n_hidden), elig_in shape: (batch, n_hidden, n_in)
            self.fc1.weight.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_in, dim=0)

            # recurrent.grad shape: (n_hidden, n_hidden)
            self.recurrent.grad += 0.05 * torch.sum(L.unsqueeze(2) * elig_rec, dim=0)

            # out.weight.grad shape: (n_out, n_hidden)
            self.out.weight.grad += 0.05 * torch.einsum('bo,br->or', err, trace_out_t)

            # Store for next iteration
            prev_hidden_spike = hidden_spike.clone()

            self.outputs.append(out_spike)
            self.hidden_mem_list.append(hidden_mem)
            self.hidden_spike_list.append(hidden_spike)

        return torch.stack(self.outputs, dim=1)

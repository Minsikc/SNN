"""Task builders for the e-prop core.

Teacher--student spike sequence: a frozen teacher with the *same* neuron model
as the student maps a fixed Bernoulli input to a target spike train, so the
loss floor is provably 0 (planting the teacher's weights reproduces the
target). For ALIF students the teacher is ALIF with the same ``beta``/``rho``.
"""
from __future__ import annotations

import os
import sys

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from datasets.customdatasets import CustomSpikeDataset_Teacher  # noqa: E402
from .config import NeuronConfig, TaskConfig                     # noqa: E402


def build_teacher_task(task: TaskConfig, neuron: NeuronConfig):
    """Return ``(x, target, dataset)``; ``x``/``target`` are (n_seq, T, n) tensors."""
    ds = CustomSpikeDataset_Teacher(
        num_samples=task.n_seq, sequence_length=task.T, input_size=task.n_in,
        output_size=task.n_out, hidden_size=task.n_hidden, spike_prob=task.spike_prob,
        teacher_thresh=neuron.thresh, teacher_tau=neuron.tau, w_scale=task.w_scale,
        seed=task.ds_seed,
        beta=neuron.beta if neuron.adaptive else 0.0,
        rho=neuron.rho if neuron.adaptive else 0.0,
    )
    return ds.data, ds.targets, ds


def planted_teacher_loss(model, ds, loss_fn) -> float:
    """Plant the teacher's weights into ``model`` (copy) and return the loss --
    must be 0 for a realizable task. Used by the tests as a sanity check."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.fc1.weight.copy_(ds.teacher_weights["w_in"].t())
        m.recurrent.copy_(ds.teacher_weights["w_rec"])
        m.out.weight.copy_(ds.teacher_weights["w_out"].t())
    m.eval()
    with torch.no_grad():
        out = m(ds.data, ds.targets, training=False)
    return float(loss_fn(out, ds.targets))

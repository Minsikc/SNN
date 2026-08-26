"""ALIF e-prop: the full eligibility chain (with the threshold-adaptation trace
eps_a) must equal the autograd derivative of the hidden spikes when the two
paths e-prop drops (reset term, recurrent input) are stopped; the truncated
chain must not. Plus task-realizability and sanity checks for the extension
experiment (ALIF full vs truncated).
"""
import numpy as np
import pytest
import torch

from eprop import (EpropRSNN, GradChainConfig, NeuronConfig, TaskConfig, build_teacher_task,
                   planted_teacher_loss, run_condition, sequence_loss)
from eprop.train import KERNEL_DECAY, KERNEL_SIZE
from utils.kernels import create_exponential_kernel

KERNEL = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)
TASK_BIG = TaskConfig(n_in=20, n_hidden=10, n_out=5, T=40, w_scale=1.5, ds_seed=25)


def _spike_sum_autograd_grads(seed, task, neuron, chain):
    """d(sum_t sum_j z_j^t)/dW with reset & recurrent paths detached, triangle surrogate
    with the e-prop pseudo-derivative gain -> should equal sum_t e_{ji}^t."""
    x, _, _ = build_teacher_task(task, neuron)
    torch.manual_seed(seed)
    m = EpropRSNN(task.n_in, task.n_hidden, task.n_out, neuron=neuron, chain=chain)
    m.set_learning_rule("bptt")
    m(x, x[:, :, : task.n_out] * 0, training=True)
    torch.stack(m.hidden_spike_list, 1).sum().backward()
    return m.fc1.weight.grad.clone(), m.recurrent.grad.clone()


def _eligibility_sums(seed, task, neuron, chain):
    x, tgt, _ = build_teacher_task(task, neuron)
    torch.manual_seed(seed)
    m = EpropRSNN(task.n_in, task.n_hidden, task.n_out, neuron=neuron, chain=chain)
    m.collect_eligibility = True
    m(x, tgt, training=True)
    return m.elig_sum_in.clone(), m.elig_sum_rec.clone()


@pytest.mark.parametrize("beta", [0.3, 0.8, 1.5])
@pytest.mark.parametrize("seed", [0, 1])
def test_full_alif_eligibility_equals_autograd_with_eprop_assumptions(beta, seed):
    neuron = NeuronConfig(kind="alif", beta=beta, rho=0.9, tau=0.6, thresh=0.4)
    chain = GradChainConfig(eligibility="full", bptt_surrogate="triangle",
                            bptt_detach_reset_and_recurrent=True, rec_grad_orientation="corrected")
    g_in, g_rec = _spike_sum_autograd_grads(seed, TASK_BIG, neuron, chain)
    e_in, e_rec = _eligibility_sums(seed, TASK_BIG, neuron, chain)
    assert torch.allclose(g_in, e_in, atol=1e-5, rtol=1e-5)
    # recurrent: autograd is [pre, post] (forward uses W[pre, post]); eligibility is [post, pre]
    assert torch.allclose(g_rec, e_rec.t(), atol=1e-5, rtol=1e-5)
    assert e_in.abs().sum() > 0


@pytest.mark.parametrize("beta", [0.8, 1.5])
def test_truncated_alif_eligibility_differs_from_autograd(beta):
    neuron = NeuronConfig(kind="alif", beta=beta, rho=0.9)
    chain_t = GradChainConfig(eligibility="truncated", bptt_surrogate="triangle",
                              bptt_detach_reset_and_recurrent=True, rec_grad_orientation="corrected")
    g_in, _ = _spike_sum_autograd_grads(0, TASK_BIG, neuron, chain_t)
    e_in, _ = _eligibility_sums(0, TASK_BIG, neuron, chain_t)
    assert not torch.allclose(g_in, e_in, atol=1e-5, rtol=1e-5)
    # ... but is a decent approximation (this is the rank-1-mappable chain)
    c = np.corrcoef(g_in.flatten().numpy(), e_in.flatten().numpy())[0, 1]
    assert c > 0.9


def test_lif_full_and_truncated_are_identical():
    neuron = NeuronConfig(kind="lif")
    a, _ = _eligibility_sums(0, TASK_BIG, neuron, GradChainConfig(eligibility="full"))
    b, _ = _eligibility_sums(0, TASK_BIG, neuron, GradChainConfig(eligibility="truncated"))
    assert torch.equal(a, b)
    # kind="alif" with beta=0 is a LIF too
    c, _ = _eligibility_sums(0, TASK_BIG, NeuronConfig(kind="alif", beta=0.0), GradChainConfig())
    assert torch.equal(a, c)


def test_alif_teacher_task_is_realizable_and_adaptation_matters():
    neuron = NeuronConfig(kind="alif", beta=0.8, rho=0.9)
    x, tgt, ds = build_teacher_task(TASK_BIG, neuron)
    m = EpropRSNN(TASK_BIG.n_in, TASK_BIG.n_hidden, TASK_BIG.n_out, neuron=neuron)
    assert planted_teacher_loss(m, ds, lambda o, t: sequence_loss(o, t, KERNEL)) == 0.0
    # same teacher weights with a LIF student are NOT a solution
    lif = EpropRSNN(TASK_BIG.n_in, TASK_BIG.n_hidden, TASK_BIG.n_out, neuron=NeuronConfig(kind="lif"))
    assert planted_teacher_loss(lif, ds, lambda o, t: sequence_loss(o, t, KERNEL)) > 0.0


def test_neuron_config_validation():
    with pytest.raises(ValueError):
        NeuronConfig(kind="lif", beta=0.5)
    with pytest.raises(ValueError):
        GradChainConfig(eligibility="half")


def test_alif_training_runs_for_all_conditions():
    """Smoke: every condition trains an ALIF network for a few epochs (mock hardware
    for analog) and both chains produce finite, different trajectories."""
    from eprop import HardwareReadoutConfig
    neuron = NeuronConfig(kind="alif", beta=0.5, rho=0.9)
    task = TaskConfig()
    out = {}
    for cond in ("bptt", "digital", "frozen_wout"):
        r = run_condition(cond, epochs=3, lr=0.05, seed=0, task=task, neuron=neuron,
                          chain=GradChainConfig(eligibility="full"))
        out[cond] = r["losses"]
        assert all(np.isfinite(r["losses"]))
    r = run_condition("analog", epochs=3, lr=0.05, seed=0, task=task, neuron=neuron,
                      chain=GradChainConfig(eligibility="truncated"),
                      hw=HardwareReadoutConfig(enabled=True, use_mock=True))
    assert all(np.isfinite(r["losses"])) and len(r["fidelity"]) == 3
    # the two chains give different hidden-layer gradients (losses may coincide
    # for a few epochs because the output is a discrete spike train)
    e_full, _ = _eligibility_sums(0, task, neuron, GradChainConfig(eligibility="full"))
    e_trunc, _ = _eligibility_sums(0, task, neuron, GradChainConfig(eligibility="truncated"))
    assert not torch.equal(e_full, e_trunc)

"""Mixed LIF/ALIF hidden populations (NeuronConfig.n_adaptive or a beta list).

Guarantees:
* n_adaptive=0 / all-zero beta list  ==  plain LIF, bit-exact
* n_adaptive=H                       ==  scalar-beta ALIF, bit-exact
* full eligibility chain == autograd under the e-prop assumptions for a mixed
  population; the LIF rows of the full chain == truncated chain rows
* the mixed teacher task is realizable (planted teacher -> loss 0)
"""
import pytest
import torch

from eprop import (EpropRSNN, GradChainConfig, NeuronConfig, TaskConfig, build_teacher_task,
                   planted_teacher_loss, run_condition, sequence_loss)
from eprop.train import KERNEL_DECAY, KERNEL_SIZE
from utils.kernels import create_exponential_kernel
from tests.test_alif_gradients import _eligibility_sums, _spike_sum_autograd_grads, TASK_BIG

KERNEL = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)
H = TASK_BIG.n_hidden


def test_beta_list_resolution():
    assert NeuronConfig(kind="alif", beta=0.5, n_adaptive=3).beta_list(5) == [0, 0, 0.5, 0.5, 0.5]
    assert NeuronConfig(kind="alif", beta=0.5, n_adaptive=0).beta_list(5) == [0.0] * 5
    assert NeuronConfig(kind="alif", beta=0.5).beta_list(3) == [0.5] * 3
    assert NeuronConfig(kind="alif", beta=[0, 0.2, 0.4]).beta_list(3) == [0, 0.2, 0.4]
    assert NeuronConfig(kind="alif", beta=0.5, n_adaptive=0).adaptive is False
    assert NeuronConfig(kind="alif", beta=[0, 0, 0]).adaptive is False
    with pytest.raises(ValueError):
        NeuronConfig(kind="alif", beta=[0, 0.5], n_adaptive=1)
    with pytest.raises(ValueError):
        NeuronConfig(kind="lif", beta=[0, 0.5])
    with pytest.raises(ValueError):
        NeuronConfig(kind="alif", beta=[0.5, 0.5]).beta_list(3)


def _grads_and_out(neuron, seed=0, chain=None):
    x, tgt, _ = build_teacher_task(TASK_BIG, neuron)
    torch.manual_seed(seed)
    m = EpropRSNN(TASK_BIG.n_in, H, TASK_BIG.n_out, neuron=neuron, chain=chain or GradChainConfig())
    out = m(x, tgt, training=True)
    return out, m.fc1.weight.grad.clone(), m.recurrent.grad.clone(), m.out.weight.grad.clone()


def test_n_adaptive_zero_is_lif_bit_exact():
    a = _grads_and_out(NeuronConfig(kind="lif"))
    b = _grads_and_out(NeuronConfig(kind="alif", beta=0.7, n_adaptive=0))
    c = _grads_and_out(NeuronConfig(kind="alif", beta=[0.0] * H))
    for ta, tb, tc in zip(a, b, c):
        assert torch.equal(ta, tb) and torch.equal(ta, tc)


def test_n_adaptive_all_is_scalar_alif_bit_exact():
    a = _grads_and_out(NeuronConfig(kind="alif", beta=0.7))
    b = _grads_and_out(NeuronConfig(kind="alif", beta=0.7, n_adaptive=H))
    c = _grads_and_out(NeuronConfig(kind="alif", beta=[0.7] * H))
    for ta, tb, tc in zip(a, b, c):
        assert torch.equal(ta, tb) and torch.equal(ta, tc)


@pytest.mark.parametrize("n_adaptive", [1, 4, 7])
def test_mixed_full_chain_equals_autograd(n_adaptive):
    neuron = NeuronConfig(kind="alif", beta=0.8, rho=0.9, n_adaptive=n_adaptive)
    chain = GradChainConfig(eligibility="full", bptt_surrogate="triangle",
                            bptt_detach_reset_and_recurrent=True, rec_grad_orientation="corrected")
    g_in, g_rec = _spike_sum_autograd_grads(0, TASK_BIG, neuron, chain)
    e_in, e_rec = _eligibility_sums(0, TASK_BIG, neuron, chain)
    assert torch.allclose(g_in, e_in, atol=1e-5, rtol=1e-5)
    assert torch.allclose(g_rec, e_rec.t(), atol=1e-5, rtol=1e-5)


def test_mixed_lif_rows_match_truncated_and_alif_rows_differ():
    neuron = NeuronConfig(kind="alif", beta=0.8, rho=0.9, n_adaptive=4)
    e_full, _ = _eligibility_sums(0, TASK_BIG, neuron, GradChainConfig(eligibility="full"))
    e_trunc, _ = _eligibility_sums(0, TASK_BIG, neuron, GradChainConfig(eligibility="truncated"))
    mask = torch.tensor(neuron.adaptive_mask(H))
    assert torch.equal(e_full[~mask], e_trunc[~mask])          # LIF neurons: identical chains
    assert not torch.allclose(e_full[mask], e_trunc[mask])     # ALIF neurons: eps_a term active


def test_mixed_teacher_is_realizable():
    neuron = NeuronConfig(kind="alif", beta=0.8, rho=0.9, n_adaptive=4)
    x, tgt, ds = build_teacher_task(TASK_BIG, neuron)
    assert ds.beta_vec.tolist() == pytest.approx(neuron.beta_list(H))   # float32 tensor vs python floats
    m = EpropRSNN(TASK_BIG.n_in, H, TASK_BIG.n_out, neuron=neuron)
    assert planted_teacher_loss(m, ds, lambda o, t: sequence_loss(o, t, KERNEL)) == 0.0
    # a homogeneous ALIF student with the same weights is NOT a solution
    homo = EpropRSNN(TASK_BIG.n_in, H, TASK_BIG.n_out, neuron=NeuronConfig(kind="alif", beta=0.8))
    assert planted_teacher_loss(homo, ds, lambda o, t: sequence_loss(o, t, KERNEL)) > 0.0


def test_mixed_training_smoke_all_conditions():
    from eprop import HardwareReadoutConfig
    neuron = NeuronConfig(kind="alif", beta=0.5, n_adaptive=2)
    for cond in ("bptt", "digital", "frozen_wout"):
        r = run_condition(cond, epochs=2, lr=0.05, seed=0, task=TaskConfig(), neuron=neuron)
        assert len(r["losses"]) == 2
    r = run_condition("analog", epochs=2, lr=0.05, seed=0, task=TaskConfig(), neuron=neuron,
                      chain=GradChainConfig(eligibility="truncated"),
                      hw=HardwareReadoutConfig(enabled=True, use_mock=True))
    assert len(r["fidelity"]) == 2

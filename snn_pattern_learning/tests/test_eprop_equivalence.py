"""The new ``eprop`` core must be bit-exact with the legacy model classes under
the default configuration, and must reproduce the paper's SW baseline numbers
(4x4, dataset seed 25, w_scale 1.5, thresh 0.4, tau 0.6, lr 0.05, 50 epochs).
"""
import pytest
import torch

from eprop import (EpropRSNN, GradChainConfig, HardwareReadoutConfig, NeuronConfig,
                   TaskConfig, build_teacher_task, planted_teacher_loss, run_condition,
                   sequence_loss)
from eprop.train import KERNEL_DECAY, KERNEL_SIZE
from models.models import Basic_RSNN_eprop_forward, Basic_RSNN_eprop_HW_forward
from utils.kernels import create_exponential_kernel

TASK = TaskConfig(n_in=10, n_hidden=4, n_out=4, T=20, w_scale=1.5, ds_seed=25)
NEURON = NeuronConfig(kind="lif", tau=0.6, thresh=0.4, tau_o=0.6)


def legacy_model(seed):
    torch.manual_seed(seed)
    return Basic_RSNN_eprop_forward(n_in=TASK.n_in, n_hidden=TASK.n_hidden, n_out=TASK.n_out,
                                    recurrent=True, init_thresh=NEURON.thresh, init_tau=NEURON.tau)


def new_model(seed, chain=None):
    torch.manual_seed(seed)
    return EpropRSNN(TASK.n_in, TASK.n_hidden, TASK.n_out, neuron=NEURON, chain=chain or GradChainConfig())


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_same_initial_weights(seed):
    a, b = legacy_model(seed), new_model(seed)
    for k in ("fc1.weight", "recurrent", "out.weight"):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), k


@pytest.mark.parametrize("seed", [0, 3])
def test_forward_and_eprop_grads_identical(seed):
    x, tgt, _ = build_teacher_task(TASK, NEURON)
    a, b = legacy_model(seed), new_model(seed)
    oa = a(x, tgt, training=True)
    ob = b(x, tgt, training=True)
    assert torch.equal(oa, ob)
    assert torch.equal(a.fc1.weight.grad, b.fc1.weight.grad)
    assert torch.equal(a.recurrent.grad, b.recurrent.grad)
    assert torch.equal(a.out.weight.grad, b.out.weight.grad)


def _grads(m):
    return (m.fc1.weight.grad.clone(), m.recurrent.grad.clone(), m.out.weight.grad.clone())


def test_legacy_bptt_condition_mixes_eprop_and_autograd_grads():
    """INTEGRITY FINDING (2026-08-26): the legacy forward fills .grad with the
    e-prop gradient even when custom_grad is False, so the legacy 'bptt'
    condition trained on e-prop + autograd. GradChainConfig(bptt_add_eprop=True)
    reproduces it exactly; the default is pure BPTT."""
    x, tgt, _ = build_teacher_task(TASK, NEURON)
    kernel = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)

    a = legacy_model(0)
    a.custom_grad = a.custom_grad_forward = False
    sequence_loss(a(x, tgt, training=True), tgt, kernel).backward()
    legacy_mixed = _grads(a)

    b = new_model(0, GradChainConfig(bptt_add_eprop=True))
    b.set_learning_rule("bptt")
    sequence_loss(b(x, tgt, training=True), tgt, kernel).backward()
    for ga, gb in zip(legacy_mixed, _grads(b)):
        assert torch.equal(ga, gb)

    c = new_model(0)                      # pure BPTT
    c.set_learning_rule("bptt")
    sequence_loss(c(x, tgt, training=True), tgt, kernel).backward()
    pure = _grads(c)
    # pure autograd == legacy mixed minus the e-prop part
    d = new_model(0)                      # e-prop only
    d(x, tgt, training=True)
    for gm, gp, ge in zip(legacy_mixed, pure, _grads(d)):
        assert torch.allclose(gm, gp + ge, atol=1e-6)
    assert not torch.allclose(pure[0], legacy_mixed[0])


def test_corrected_orientation_is_transpose_of_legacy():
    x, tgt, _ = build_teacher_task(TASK, NEURON)
    a = new_model(0, GradChainConfig(rec_grad_orientation="legacy"))
    b = new_model(0, GradChainConfig(rec_grad_orientation="corrected"))
    a(x, tgt, training=True); b(x, tgt, training=True)
    assert torch.equal(a.recurrent.grad.t(), b.recurrent.grad)
    assert torch.equal(a.fc1.weight.grad, b.fc1.weight.grad)


def test_planted_teacher_is_a_zero_loss_optimum():
    x, tgt, ds = build_teacher_task(TASK, NEURON)
    kernel = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)
    m = new_model(0)
    assert planted_teacher_loss(m, ds, lambda o, t: sequence_loss(o, t, kernel)) == 0.0
    assert int(tgt.sum()) == 17          # the paper task: 17 target spikes


# -------- paper SW baselines (seed 0, lr 0.05) -----------------------------
# values recorded from scripts/teacher_student/run_three_conditions.run on 2026-08-26
# (== results/eprop_grad_log/sw_sweep_4x4_ds25.json, seed 0, lr 0.05)
EXPECTED = [
    ("digital", GradChainConfig(), 0.9088573455810547),
    ("frozen_wout", GradChainConfig(), 1.446),
    ("bptt", GradChainConfig(bptt_add_eprop=True), 0.4992),   # paper v0.1 "BPTT" (mixed, see above)
    ("bptt", GradChainConfig(), 1.3868),                       # pure BPTT, boxcar surrogate
]


@pytest.mark.parametrize("condition,chain,expected", EXPECTED)
def test_reproduces_paper_sw_baselines(condition, chain, expected):
    r = run_condition(condition, epochs=50, lr=0.05, seed=0, task=TASK, neuron=NEURON, chain=chain)
    assert r["best_loss"] == pytest.approx(expected, abs=5e-4)


def test_mock_hardware_path_matches_legacy_hw_model():
    """EpropRSNN + HardwareReadout(mock) == Basic_RSNN_eprop_HW_forward(use_mock_hw) for
    several epochs, starting from identical weights (weights copied as
    basic_experiment does with training.init_seed)."""
    x, tgt, _ = build_teacher_task(TASK, NEURON)
    kernel = create_exponential_kernel(KERNEL_SIZE, KERNEL_DECAY)
    ref = legacy_model(0)

    legacy = Basic_RSNN_eprop_HW_forward(n_in=TASK.n_in, n_hidden=4, n_out=4, recurrent=True,
                                         init_tau=0.6, init_thresh=0.4, init_tau_o=0.6,
                                         hw_enabled=True, use_mock_hw=True, bit_length=10,
                                         normalization_scale=0.7, no_read_updates=True)
    legacy.fc1.weight.data.copy_(ref.fc1.weight); legacy.recurrent.data.copy_(ref.recurrent)
    legacy.out.weight.data.copy_(ref.out.weight)

    from eprop import HardwareReadout
    hw = HardwareReadoutConfig(enabled=True, use_mock=True, normalization_scale=0.7)
    torch.manual_seed(0)
    new = EpropRSNN(TASK.n_in, 4, 4, neuron=NEURON, readout=HardwareReadout(hw))
    new.copy_weights_from(ref)

    for m in (legacy, new):
        assert m.connect_hardware()
    opt_l = torch.optim.Adam(legacy.parameters(), lr=0.05)
    opt_n = torch.optim.Adam(new.parameters(), lr=0.05)
    for ep in range(4):
        for m, opt in ((legacy, opt_l), (new, opt_n)):
            m.reset_hardware()
            opt.zero_grad()
            m(x, tgt, training=True)
            opt.step()
            m.apply_hw_gradient(learning_rate=0.05)
        assert torch.allclose(legacy.out.weight, new.out.weight, atol=1e-6), ep
        assert torch.allclose(legacy.fc1.weight, new.fc1.weight, atol=1e-6), ep
    with torch.no_grad():
        la = sequence_loss(legacy(x, tgt, training=False), tgt, kernel).item()
        ln = sequence_loss(new(x, tgt, training=False), tgt, kernel).item()
    assert la == pytest.approx(ln, abs=1e-6)

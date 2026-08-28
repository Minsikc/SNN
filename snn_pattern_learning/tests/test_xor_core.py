"""Temporal XOR on the e-prop core reproduces the legacy scripts/xor/run_xor.py
results (seed 0, 60 epochs) recorded on 2026-08-26/28."""
import pytest

from eprop.xor import run_xor, xor_neuron
from eprop import GradChainConfig

# (condition, lr) -> (best_loss, first epoch with acc 1.00, best_acc)
LEGACY = {
    ("digital_mock", 0.15): (1.9686, 19, 1.0),   # legacy "digital": mock array + epoch-end write
    ("bptt", 0.05): (1.2268, 37, 1.0),           # pure BPTT, boxcar half-width 0.5, window loss
    ("frozen", 0.15): (2.1215, None, 0.25),      # readout frozen -> chance
}


@pytest.mark.parametrize("cond_lr,expected", list(LEGACY.items()))
def test_xor_matches_legacy_runner(cond_lr, expected):
    cond, lr = cond_lr
    r = run_xor(cond, epochs=60, lr=lr, seed=0)
    assert r["best_loss"] == pytest.approx(expected[0], abs=5e-4)
    assert r["first_perfect_epoch"] == expected[1]
    assert r["best_acc"] == expected[2]


def test_xor_alif_and_truncated_run():
    n = xor_neuron("alif", beta=0.5)
    r_full = run_xor("digital", 3, 0.15, seed=0, neuron=n, chain=GradChainConfig(eligibility="full"), reservoir=False)
    r_trunc = run_xor("digital", 3, 0.15, seed=0, neuron=n, chain=GradChainConfig(eligibility="truncated"), reservoir=False)
    assert len(r_full["accs"]) == 3 and len(r_trunc["accs"]) == 3
    assert all(0.0 <= a <= 1.0 for a in r_full["accs"])

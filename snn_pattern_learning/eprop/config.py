"""Configuration dataclasses for the e-prop core (``eprop`` package).

Defaults reproduce the legacy ``Basic_RSNN_eprop_forward`` /
``Basic_RSNN_eprop_HW_forward`` behaviour bit-for-bit (see
``tests/test_eprop_equivalence.py``); every extension is an explicit switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class NeuronConfig:
    """Hidden-layer neuron model.

    kind:   ``"lif"`` (legacy) or ``"alif"`` -- adaptive-threshold LIF
            (Bellec et al. 2020): ``A_t = thresh + beta * a_t``,
            ``a_{t+1} = rho * a_t + z_t``.
    tau:    membrane decay ``alpha`` (legacy ``init_tau``); also the legacy
            pseudo-derivative height when ``pd_gamma`` is None.
    thresh: firing threshold (legacy ``init_thresh``); also the
            pseudo-derivative half-width.
    tau_o:  ``kappa`` -- readout / eligibility low-pass constant (``init_tau_o``).
    beta:   ALIF adaptation strength (0 -> plain LIF even when kind="alif").
    rho:    ALIF adaptation decay ``exp(-dt / tau_a)``.
    pd_gamma: pseudo-derivative height; None -> ``tau`` (legacy quirk, kept
            for reproducibility).
    """
    kind: str = "lif"
    tau: float = 0.6
    thresh: float = 0.4
    tau_o: float = 0.6
    beta: float = 0.0
    rho: float = 0.9
    pd_gamma: Optional[float] = None

    def __post_init__(self):
        if self.kind not in ("lif", "alif"):
            raise ValueError(f"neuron kind must be 'lif' or 'alif', got {self.kind!r}")
        if self.kind == "lif" and self.beta != 0.0:
            raise ValueError("beta must be 0 for kind='lif' (use kind='alif')")

    @property
    def pseudo_derivative_gain(self) -> float:
        return self.tau if self.pd_gamma is None else self.pd_gamma

    @property
    def adaptive(self) -> bool:
        return self.kind == "alif" and self.beta != 0.0


@dataclass
class GradChainConfig:
    """Which gradient chain the e-prop update follows.

    eligibility: ``"full"`` keeps the ALIF threshold-adaptation eligibility
        ``eps_a`` (exact e-prop for ALIF); ``"truncated"`` drops it and uses
        the LIF trace ``h_t * eps_v`` only -- the approximation that maps onto
        rank-1 hardware accumulation. Identical for LIF neurons.
    rec_grad_orientation: ``"legacy"`` accumulates the recurrent gradient as
        ``[post, pre]`` although the forward pass uses ``W[pre, post]`` (the
        transposition documented 2026-08-16, kept for reproducibility);
        ``"corrected"`` transposes it.
    hidden_grad_scale / readout_grad_scale: the fixed ``0.05`` factors of the
        legacy implementation.
    bptt_surrogate: surrogate used when the model is trained by autograd
        (``condition="bptt"``): ``"boxcar"`` (legacy, half-width 0.1) or
        ``"triangle"`` (same shape as the e-prop pseudo-derivative).
    bptt_add_eprop: INTEGRITY FINDING 2026-08-26 -- the legacy
        ``Basic_RSNN_eprop_forward.forward`` fills ``.grad`` with the e-prop
        gradient unconditionally, so the legacy "bptt" condition
        (``run_three_conditions.py``) trained on e-prop + autograd gradients
        summed. ``True`` reproduces that behaviour (paper v0.1 BPTT rows);
        ``False`` (default) is pure BPTT.
    """
    eligibility: str = "full"
    rec_grad_orientation: str = "legacy"
    hidden_grad_scale: float = 0.05
    readout_grad_scale: float = 0.05
    bptt_surrogate: str = "boxcar"
    boxcar_halfwidth: float = 0.1
    bptt_add_eprop: bool = False
    # e-prop-consistent autograd reference: stop gradients through the reset
    # term and through the recurrent input (the two paths e-prop drops), so the
    # remaining autograd gradient of the hidden spikes equals the sum of the
    # e-prop eligibility traces exactly (used by tests/test_alif_gradients.py).
    bptt_detach_reset_and_recurrent: bool = False

    def __post_init__(self):
        if self.eligibility not in ("full", "truncated"):
            raise ValueError("eligibility must be 'full' or 'truncated'")
        if self.rec_grad_orientation not in ("legacy", "corrected"):
            raise ValueError("rec_grad_orientation must be 'legacy' or 'corrected'")
        if self.bptt_surrogate not in ("boxcar", "triangle"):
            raise ValueError("bptt_surrogate must be 'boxcar' or 'triangle'")


@dataclass
class HardwareReadoutConfig:
    """Analog readout-gradient accumulator (6T1C array or its mock)."""
    enabled: bool = False
    use_mock: bool = True
    serial_port: str = "COM4"
    baud_rate: int = 115200
    bit_length: int = 10
    pulse_width: int = 15
    pulse_pre: int = 100
    pulse_post: int = 100
    pulse_zero: int = 10
    read_time: int = 20
    read_delay: int = 10
    no_read_updates: bool = True
    dno: bool = False
    normalization_scale: float = 0.7
    fixed_norm: Optional[Tuple[float, float]] = None
    adc_to_grad_scale: float = 0.001
    auto_calibrate_scale: bool = True
    calibrate_ema: float = 0.5
    calibrate_per_column: bool = False
    batch_quadrants: bool = False
    mock_quantize_bits: int = 0
    mock_quantize_seed: int = 0
    freeze_wout: bool = False
    grad_log_path: Optional[str] = None
    array_size: int = 5


@dataclass
class TaskConfig:
    """Teacher--student spike-sequence task (``CustomSpikeDataset_Teacher``)."""
    n_in: int = 10
    n_hidden: int = 4
    n_out: int = 4
    n_seq: int = 1
    T: int = 20
    spike_prob: float = 0.2
    w_scale: float = 1.5
    ds_seed: int = 25
    # teacher neuron parameters default to the student's NeuronConfig


@dataclass
class ExperimentConfig:
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    chain: GradChainConfig = field(default_factory=GradChainConfig)
    hardware: HardwareReadoutConfig = field(default_factory=HardwareReadoutConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    epochs: int = 50
    lr: float = 0.05
    seed: int = 0
    recurrent: bool = True
    weight_scale: float = 0.5

    def to_dict(self):
        return asdict(self)


def neuron_from_dict(d: dict) -> NeuronConfig:
    d = dict(d or {})
    # accept legacy yaml keys
    d.setdefault("tau", d.pop("init_tau", 0.6))
    d.setdefault("thresh", d.pop("init_thresh", 0.4))
    d.setdefault("tau_o", d.pop("init_tau_o", 0.6))
    allowed = NeuronConfig.__dataclass_fields__.keys()
    return NeuronConfig(**{k: v for k, v in d.items() if k in allowed})


def chain_from_dict(d: dict) -> GradChainConfig:
    allowed = GradChainConfig.__dataclass_fields__.keys()
    return GradChainConfig(**{k: v for k, v in (d or {}).items() if k in allowed})


def hardware_from_dict(d: dict) -> HardwareReadoutConfig:
    d = dict(d or {})
    if "use_mock_hw" in d:
        d["use_mock"] = d.pop("use_mock_hw")
    if "fixed_norm" in d and d["fixed_norm"] is not None:
        d["fixed_norm"] = tuple(d["fixed_norm"])
    allowed = HardwareReadoutConfig.__dataclass_fields__.keys()
    return HardwareReadoutConfig(**{k: v for k, v in d.items() if k in allowed})

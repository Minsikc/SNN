"""e-prop core: LIF/ALIF recurrent SNN with software / 6T1C-array readout-gradient
accumulation and a switchable gradient chain.

Quick start (from ``snn_pattern_learning/``)::

    from eprop import NeuronConfig, GradChainConfig, TaskConfig, run_condition
    r = run_condition("digital", epochs=50, lr=0.05, seed=0)          # legacy-exact
    r = run_condition("digital", 50, 0.05, neuron=NeuronConfig(kind="alif", beta=0.5),
                      chain=GradChainConfig(eligibility="truncated"))
    r = run_condition("analog", 50, 0.05, hw=HardwareReadoutConfig(use_mock=True))
"""
from .config import (ExperimentConfig, GradChainConfig, HardwareReadoutConfig,
                     NeuronConfig, TaskConfig, chain_from_dict, hardware_from_dict,
                     neuron_from_dict)
from .model import EpropRSNN
from .readout import HardwareReadout, ReadoutBackend, SoftwareReadout
from .tasks import build_teacher_task, planted_teacher_loss
from .train import CONDITIONS, build_model, run_condition, run_experiment, sequence_loss
from .xor import XOR_CONDITIONS, run_xor, xor_neuron

__all__ = [
    "ExperimentConfig", "GradChainConfig", "HardwareReadoutConfig", "NeuronConfig", "TaskConfig",
    "chain_from_dict", "hardware_from_dict", "neuron_from_dict",
    "EpropRSNN", "HardwareReadout", "ReadoutBackend", "SoftwareReadout",
    "build_teacher_task", "planted_teacher_loss",
    "CONDITIONS", "build_model", "run_condition", "run_experiment", "sequence_loss",
]

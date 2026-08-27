# Consolidation & refactor guide (branch `refactor/eprop-core`, 2026-08-26)

Everything e-prop / 6T1C related now lives in **`SNN/snn_pattern_learning/`**.
Baseline before the refactor: commit `a1d2de8` (`main`) — the exact code of the
paper draft v0.1.

## New layout

```
snn_pattern_learning/
├── main_unified.py            entry point (unchanged CLI)
├── neurons.py                 legacy surrogate/LIF nodes (imported by models/)
├── eprop/                     NEW core library
│   ├── config.py              NeuronConfig / GradChainConfig / HardwareReadoutConfig / TaskConfig
│   ├── neurons.py             functional LIF/ALIF steps, pseudo-derivative, surrogates
│   ├── model.py               EpropRSNN  (LIF|ALIF, full|truncated chain, sw|hw|mock readout)
│   ├── readout.py             SoftwareReadout, HardwareReadout (port of the HW_forward logic)
│   ├── tasks.py               teacher–student task builder (ALIF-aware), planted-teacher check
│   └── train.py               run_condition(bptt|digital|frozen_wout|analog) — legacy-exact
├── models/                    legacy classes (kept, untouched) + model_factory (new type "EpropRSNN")
├── hardware/
│   ├── hw_interface.py        MemristorInterface / MockMemristorInterface (unchanged API)
│   ├── device_model.py        ← measurements/aihwkit_model/model_6t1c.py (+ its params json)
│   ├── firmware/              ← measurements/6T1C_5x5_add_stochastic.ino (2026-08-20 build)
│   └── measure/               ← measurements/{01..04 scripts, DNO/ABAB, col5 diagnosis}
│       ├── common.py          shared serial/command helpers (tested)
│       ├── run.py             `python -m hardware.measure.run <group>/<script>.py …`
│       ├── 0N_*/              scripts verbatim + README
│       └── data/              copied CSV/NPZ (git-ignored)
├── scripts/                   experiment entry points, moved from the package root
│   ├── teacher_student/       run_three_conditions, sweep_sw_lr, run_alif_conditions (NEW), plots…
│   ├── xor/  analysis/  hw/  legacy/
├── tests/                     pytest (36 tests, ~65 s, no hardware)
├── docs/                      INTEGRITY_REPORT.md, this file, HARDWARE_SETUP.md, PROJECT_STRUCTURE.md,
│                              device_parameters.tex, measurement_overview.md
├── configs/                   + eprop_alif_digital.yaml, eprop_alif_mock.yaml
└── datasets/ experiment_types/ utils/ results/   (as before; CustomSpikeDataset_Teacher gained beta/rho)
```

## Where things went

| before (2026-08-25) | after |
|---|---|
| `snn_pattern_learning/run_three_conditions.py`, `sweep_sw_lr.py`, … (51 files at the package root) | `scripts/<group>/` — run from the package root: `python scripts/teacher_student/run_three_conditions.py …` |
| `measurements/uv_grid_sweep.py`, `read_disturb_all25.py`, … (01–04 sets, 53 files) | `hardware/measure/0N_*/` — run via `python -m hardware.measure.run 02_grad_fidelity_uv/uv_grid_sweep.py …` |
| `measurements/scripts/0[1-6]_*` (organised copies) | deleted (05/06 code already lived in the package; 01–04 moved) |
| `measurements/aihwkit_model/model_6t1c.py`, `aihwkit_linearstep_params.json` | `hardware/device_model.py`, `hardware/aihwkit_linearstep_params.json` |
| `measurements/6T1C_5x5_add_stochastic.ino` | `hardware/firmware/` |
| `measurements/device_parameters.tex`, `measurements/scripts/README.md` | `docs/` |
| `measurements/2026-*.csv`, `*.npz` | copied to `hardware/measure/data/` (originals kept — `scripts/07_svdp` still reads them) |
| `HARDWARE_SETUP.md`, `PROJECT_STRUCTURE.md` | `docs/` |

Not moved (out of scope, still in `measurements/`): `eprop_rl_sim/` (reward-based
e-prop CartPole), `scripts/07_svdp/`, `scripts/08_eprop_lif_rl/`,
`scripts/common/capture_env.py`, `col5_diagnosis.py`'s sibling one-off scripts
(`_smoke_overlap.py`, `pyserial_example.py`), `synapse-instrument-kit/`, `mndl/`.

## Running things

```bash
cd SNN/snn_pattern_learning
python -m pytest tests -q                                   # everything, no hardware

# legacy pipeline (unchanged)
python main_unified.py --config eprop_4x4_ds25_s0.yaml       # analog run on COM4
python scripts/teacher_student/sweep_sw_lr.py                # SW baselines

# new core
python -c "from eprop import run_condition; print(run_condition('digital',50,0.05,seed=0)['best_loss'])"
python main_unified.py --config eprop_alif_digital.yaml      # ALIF, model.type: EpropRSNN
python scripts/teacher_student/run_alif_conditions.py --betas 0 0.5 1.0 --seeds 0 1 2 3 4
python scripts/teacher_student/run_alif_conditions.py --conditions analog --mock   # or --port COM4

# device measurements
python -m hardware.measure.run --list
python -m hardware.measure.run 03_read_disturbance/fit_read_disturb_modelB.py
```

## Using `EpropRSNN` from yaml

```yaml
model:
  type: "EpropRSNN"
  learning_rule: "eprop"            # or "bptt"
  neuron:      {kind: alif, beta: 0.5, rho: 0.9, pd_gamma: null}
  grad_chain:  {eligibility: full, rec_grad_orientation: legacy, bptt_surrogate: boxcar}
  hardware:    {enabled: true, use_mock_hw: true, no_read_updates: true, ...}   # same keys as before
dataset:
  type: "CustomSpikeDataset_Teacher"  # teacher automatically gets the student's beta/rho
```

`model.type: RSNN_eprop_HW_forward` / `RSNN_eprop_forward` still work unchanged.

## Behaviour switches and their legacy defaults

| switch | legacy value | note |
|---|---|---|
| `NeuronConfig.kind` | `lif` | `alif` adds `A_t = ϑ + β a_t`, `a_{t+1} = ρ a_t + z_t` |
| `GradChainConfig.eligibility` | `full` | `truncated` drops `ε_a` (rank-1-mappable); identical for LIF |
| `GradChainConfig.rec_grad_orientation` | `legacy` (transposed) | `corrected` |
| `GradChainConfig.bptt_add_eprop` | `False` (pure BPTT) | `True` reproduces paper-v0.1 "bptt" (see INTEGRITY_REPORT §1) |
| `GradChainConfig.bptt_surrogate` | `boxcar` | `triangle` = e-prop pseudo-derivative |
| `NeuronConfig.pd_gamma` | `None` (= tau) | |

## Extension experiment: ALIF + gradient chain

`scripts/teacher_student/run_alif_conditions.py` sweeps
`beta × {full, truncated} × condition × seed` on the ALIF teacher task and writes
`results/eprop_alif/alif_conditions.json` (per-epoch curves + config). The
unit-level guarantee behind it: `tests/test_alif_gradients.py` shows the full
chain equals the autograd derivative of the hidden spikes under the e-prop
assumptions (stop-gradient through reset and recurrent input) for β ∈ {0.3, 0.8,
1.5}, while the truncated chain deviates (r ≈ 0.9+). On the 20-10-5 / T=40 task
plain BPTT vs e-prop gradient correlations are 0.5–0.8 for both chains — the
extension experiment measures whether that difference matters for learning.

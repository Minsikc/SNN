# Code integrity report — e-prop / 6T1C code base (2026-08-26)

Scope: everything that produced the paper draft v0.1 (`measurements/paper/`),
checked before and during the `refactor/eprop-core` consolidation. Baseline
commit `a1d2de8` on `main` is the exact code that produced the paper numbers.

## What was checked

| check | result |
|---|---|
| `pyflakes` on all tracked `.py` (models, hardware, experiment types, scripts) | no undefined names / syntax errors; unused-variable noise only |
| `compileall` | all files compile |
| Reproduction of paper SW baselines (`run_three_conditions.run`, seed 0, lr 0.05, 4×4, ds 25) | digital **0.9089**, frozen **1.446**, bptt **0.4992** — matches `sw_sweep_4x4_ds25.json` |
| Mock-hardware pipeline (`main_unified.py --config eprop_4x4_mock.yaml`) | runs end to end |
| `check_transpose_local.py` | recurrent e-prop gradient r = 0.186 vs BPTT, 0.566 vs BPTT^T — transposition confirmed (see finding 2) |
| New `eprop` core vs legacy classes | bit-exact: initial weights, forward outputs, e-prop grads, mock-hardware trajectory (4 epochs), 50-epoch best losses (`tests/test_eprop_equivalence.py`) |
| ALIF eligibility chain vs autograd | exact under e-prop assumptions (`tests/test_alif_gradients.py`) |
| Full test-suite | `python -m pytest tests -q` → 36 passed (~65 s) |

## Findings

### 1. The legacy "BPTT" condition is BPTT **plus** e-prop (affects paper Table 2, BPTT row)

`Basic_RSNN_eprop_forward.forward()` fills `fc1.weight.grad`, `recurrent.grad`
and `out.weight.grad` with the e-prop gradient **unconditionally** — it never
looks at `custom_grad`. `run_three_conditions.run(condition="bptt")` only sets
`custom_grad=False` and then calls `loss.backward()`, which *adds* the autograd
gradient to the e-prop gradient already in `.grad`. Verified numerically:

| seed | legacy "bptt" (mixed) | pure BPTT (grads zeroed before backward) |
|---|---|---|
| 0 | 0.4992 | 1.3868 |
| 1 | 1.0517 | 2.0625 |

Pure BPTT with the legacy boxcar surrogate (half-width 0.1) is far worse than
the mixed rule; the paper's BPTT rows (median 0.983) therefore describe an
e-prop+BPTT hybrid, not BPTT. The `basic_experiment` pipeline is not affected
(its BPTT path uses `Basic_RSNN_spike`, a different class), and the digital /
analog / frozen rows are unaffected.

*Status:* reproducible on purpose via `GradChainConfig(bptt_add_eprop=True)`;
the default of the new core is pure BPTT. **Action for the paper:** re-run the
BPTT baseline with `eprop.run_condition("bptt", …)` — and choose the surrogate
deliberately (`bptt_surrogate="triangle"` gives BPTT the same pseudo-derivative
e-prop uses; the boxcar with half-width 0.1 is very sparse), then replace the
BPTT column or drop it.

### 2. Recurrent e-prop gradient is transposed relative to the forward pass (known, kept)

Forward: `I = z @ W_rec` → `W_rec[pre, post]`. E-prop accumulates
`L ⊗ elig_rec` = `[post, pre]`. Confirmed again by
`tests/test_alif_gradients.py::test_full_alif_eligibility_equals_autograd…`: the
autograd gradient equals the **transposed** eligibility sum exactly. Readout
(`W_out`) gradients — everything that ran on the array — are correctly oriented.
`GradChainConfig(rec_grad_orientation="corrected")` fixes it; default stays
`"legacy"` so paper runs reproduce. Not yet re-evaluated on the task.

### 3. Multiplicative reset vs e-prop trace

The legacy neuron resets multiplicatively (`v ← α v (1−z) + I`) while the e-prop
trace `eps_v ← α eps_v + x` ignores the reset entirely. Under the e-prop
convention (stop-gradient through the reset) the autograd derivative and the
trace agree only if the reset *amount* is treated as a constant
(`v − (v z).detach()`), which is what `bptt_detach_reset_and_recurrent=True`
does. This is a modelling choice, not a bug, but it is why plain BPTT and e-prop
gradients correlate only moderately (0.5–0.8) on this neuron.

### 4. Pseudo-derivative height is `init_tau`, not `gamma`

`h_t = init_tau · max(0, 1 − |v−ϑ|/ϑ)`; the yaml `gamma`/`width` only reach the
BPTT surrogate object. Documented in `scripts/05` README; exposed as
`NeuronConfig.pd_gamma` (None = legacy).

### 5. Two model classes, two RNG orders

`Basic_RSNN_eprop_HW_forward` draws `recurrent` with an extra `kaiming_normal_`
before creating `out`, so the same `torch.manual_seed` gives different weights
from `Basic_RSNN_eprop_forward`; `basic_experiment` works around it by copying
weights (`training.init_seed`). `EpropRSNN` uses the SW class's RNG order for
both software and hardware readouts, so the workaround is no longer needed
(but still harmless).

### 6. Legacy dead code

`models/models.py` still contains 12 unused / broken classes
(`Basic_RSNN_ALIF` has its methods at module level and hard-codes `.cuda()`;
`RSNN_eprop`, `_minsik`, `_minsik_`, `_analog_forward`, `_aihwkit` are
unreferenced by any config). Left in place (git history preserves them) — the
new core supersedes the e-prop ones. `01_potdep_curve/measurement_functions.py`
is a notebook fragment (undefined `UPDATE_NUM`, `SET_NUM`, `csv`, `datetime`).

### 7. Firmware

`is_potentiation = (update_function != "STOCHASTIC_DEPRESSION")` — exact string
compare, so `STOCHASTIC_DNO_DEPRESSION(_NR)` streams are routed to N1/N2. The
NR-opcode parser condition must list every opcode name (both fixed on
2026-08-20 for the `_NR` family; the DNO-depression compare is still open).
Canonical source: `hardware/firmware/6T1C_5x5_add_stochastic.ino` (2026-08-20).

## Not verified

* Hardware scripts under `hardware/measure/0*` were moved verbatim and are
  exercised only through the runner on recorded data (`fit_read_disturb_modelB`
  reproduced V∞ = 67.4 LSB on the 2026-08-06 file); no serial session was run.
* `experiment_types/teacher_student_experiment.py`, `weight_init_experiment.py`,
  `statistical_ablation_experiment.py` and the wandb sweeps (`scripts/legacy`)
  were compiled and linted only.

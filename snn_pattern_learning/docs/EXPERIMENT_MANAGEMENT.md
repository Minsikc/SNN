# Experiment management — review and recommendations (2026-08-28)

Question asked: *is a refactor needed to keep experiments manageable now that
neuron type (LIF / ALIF / mixed), gradient chain (full / truncated, recurrent
orientation, BPTT variants), readout backend (software / mock / 6T1C) and
training mode (joint / reservoir) are all switchable?*

## Where we are

Three entry points build the same `EpropRSNN` but read their switches from
different places:

| entry point | task | switches come from | output |
|---|---|---|---|
| `eprop.run_condition()` / `scripts/teacher_student/run_alif_conditions.py` | teacher–student | function kwargs / CLI flags | one JSON per sweep (`results/eprop_alif/*.json`) |
| `eprop.xor.run_xor()` / `scripts/xor/run_xor_eprop_core.py` | temporal XOR | function kwargs / CLI flags | one JSON per sweep (`results/xor_core/*.json`) |
| `main_unified.py --config *.yaml` (`BasicExperiment`) | any dataset in `base_experiment.create_dataset` | yaml (`model.neuron`, `model.grad_chain`, `model.hardware`, `training.train_hidden`) | per-run folder under `experiment.results_dir` + gradient CSV |

What is already unified (this branch): the **semantics** of every switch live in
one place — `eprop/config.py` (`NeuronConfig`, `GradChainConfig`,
`HardwareReadoutConfig`, `TaskConfig`) — and all three entry points resolve to
those dataclasses (`neuron_from_dict` / `chain_from_dict` / `hardware_from_dict`
for yaml). Mixed populations (`n_adaptive` / beta list) flow to the teacher
through the same `NeuronConfig`, so a switch can no longer mean different
things in different runners. `train_hidden` is now the single name for
joint/reservoir in all three.

## Assessment

A **large refactor is not needed**; the core is already config-driven and
tested. Three specific gaps remain, in order of value:

### 1. Run registry — IMPLEMENTED (2026-08-28)
`eprop/registry.py` appends one JSON line per run to `results/registry.jsonl`
(override with `$SNN_REGISTRY`; the test-suite always points it at a temp file).
Stable schema: `run_id` (sha1 of the resolved task/condition/neuron/chain/hw/
train_hidden/seed/epochs/lr — identical settings ⇒ identical id), timestamp,
`git_commit`, `entry_point`, `task`, `condition`, resolved config dicts,
`metrics` (best_loss, best_epoch, best_acc, first_perfect_epoch, fidelity_mean,
…), `curves_path` (the sweep JSON / results dir holding the full curves),
`seconds`, `note`; `main_unified` entries also carry a per-epoch `history`.

Hooks: `run_condition(..., record=True, curves_path=...)`,
`run_xor(..., record=True, ...)` (both sweep scripts pass these),
`main_unified` (always, unless `experiment.registry: false`).

Query:
```
python scripts/analysis/registry_table.py --task xor --group condition neuron.kind train_hidden \
    --metrics metrics.best_acc metrics.first_perfect_epoch --agg mean
python scripts/analysis/registry_table.py --filter neuron.kind=alif --filter chain.eligibility=full --list
python scripts/analysis/registry_table.py --latest-only          # newest entry per run_id
```
or in Python: `registry.load(task="xor", **{"neuron.kind": "alif"})`,
`registry.table(rows, group=[...], metrics=[...])`.
The legacy per-sweep JSONs remain the curve store; the registry is the index.

### 2. One task interface (recommended, medium)
`run_condition` (teacher–student) and `run_xor` duplicate the epoch loop and
differ only in: dataset builder, loss window, metric (VRD vs XOR accuracy),
per-sample vs whole-batch stepping. Proposal: `eprop/tasks.py` exposes a small
`Task` protocol (`build() -> (x, tgt, meta)`, `loss(out, tgt)`,
`metrics(out, tgt) -> dict`, `err_window`) with `TeacherStudentTask` and
`TemporalXORTask`; a single `eprop/train.py::run(condition, task, ...)` loop.
Both regression test-sets (paper numbers, XOR legacy numbers) must stay
bit-exact — they are the guard rail for doing this safely. Not urgent: the two
loops are ~80 lines each and both are tested.

### 3. yaml for sweeps (optional)
`main_unified` runs one config per invocation; sweeps are Python scripts.
A `sweep:` block (lists for `seed`, `model.neuron.beta`, `model.grad_chain.eligibility`,
…) expanded by `main_unified --sweep` would let hardware sweeps be declared in
yaml like single runs. Only worth it once hardware ALIF runs start; the Python
sweep scripts are fine for software.

## Not recommended
* Splitting `EpropRSNN` into per-neuron-type classes — conditions must share
  init RNG order and forward arithmetic; switches inside one class are what
  make the bit-exact comparisons possible.
* Touching `models/models.py` — frozen for paper reproducibility
  (`paper-v0.1-baseline`).

## Conventions to keep (already in place)
* New behaviour = new field on a config dataclass with the **legacy value as
  default**, plus a test that the default is bit-exact with before.
* Every runner records the fully resolved config next to its results.
* Hardware runs go through `HardwareReadoutConfig` only (no ad-hoc kwargs);
  `analog` conditions are never part of the test-suite.

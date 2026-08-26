# 6T1C 5×5 array — measurement scripts

Consolidated 2026-08-26 from `measurements/*.py` (originals) and the organised
copies in `measurements/scripts/0[1-4]_*`. Every script is kept **verbatim**
(so the recorded CSVs stay reproducible) and is run through `run.py`, which
changes into the data directory before executing it:

```
python -m hardware.measure.run --list
python -m hardware.measure.run 03_read_disturbance/fit_read_disturb_modelB.py
python -m hardware.measure.run 02_grad_fidelity_uv/uv_random_signed.py --port COM4 --trials 40
python -m hardware.measure.run --data-dir D:/somewhere 01_potdep_curve/plot_cycle_test.py x.csv
```

Data directory: `hardware/measure/data/` (override with `$SNN_MEASURE_DATA`).
All dated CSV/NPZ files recorded in `measurements/` up to 2026-08-26 were
copied there (see `data/README.md`).

New scripts should import the shared helpers instead of re-implementing the
serial protocol:

```python
from hardware.measure import common as C
with C.open_port("COM4") as ard:
    C.hard_reset(ard)                       # drains the 31 Reset blocks
    before = C.read_array(ard)              # one READ_ROW (5x5 N5-N6, LSB)
    rows = C.bernoulli_streams(u, 10, rng); cols = C.bernoulli_streams(v, 10, rng)
    C.send(ard, C.stochastic_command(rows, cols, "POTENTIATION", no_read=True))
    delta = C.read_array(ard) - before
    C_real = C.coincidence_matrix(rows, cols)
```

`common.py` is unit-tested (`tests/test_measure_common.py`) for the pure
parts: command formats, Reset block count, stream/coincidence bookkeeping,
reply parsing with a fake serial port. Hardware scripts are never run by the
test-suite.

| folder | what it measures | README |
|---|---|---|
| `01_potdep_curve` | full-range POT/DEP cycle curve, soft-bound (aihwkit LinearStep) fit | yes |
| `02_grad_fidelity_uv` | desired vs realised outer-product update over (u,v) conditions | yes |
| `03_read_disturbance` | per-read decay (τ ≈ 120 reads, 0.79 %/read) and time leakage | yes |
| `04_halfselect_sequence` | single-line and alternating-pair half-select; attractor map + decay fit | yes |
| `05_dno_abab` | NORMAL vs DNO interleaved (ABAB) fidelity, half-select explanation of Δr | see `docs/measurement_overview.md` and memory notes |
| `06_diagnostics` | column-5 fault diagnosis (uses `hardware.hw_interface`) | — |

`01_potdep_curve/measurement_functions.py` is a notebook fragment (it refers
to notebook globals) and is not runnable standalone; it documents the
`GLOBAL_PD_SEQ_READ` cycle-test command.

Cross-cutting cautions (details in `docs/measurement_overview.md`): never
compare across sessions; reads are destructive; `STOCHASTIC_*` (non-`_NR`)
commands read twice per call; `Reset` emits 31 blocks; `bit_length` is clamped
to 20 by the firmware; Pearson r inflates with the range of the desired update.

Firmware: `hardware/firmware/6T1C_5x5_add_stochastic.ino` (2026-08-20 build
with `STOCHASTIC_*_NR`, `READ_ROW`, `HS_SEQ`, `PROG_NR`). Known bug: the
`is_potentiation` test is an exact string compare, so `STOCHASTIC_DNO_DEPRESSION`
routes streams to N1/N2 (potentiates) — fix with `indexOf("DEPRESSION") < 0`
before relying on the DNO depression opcode.

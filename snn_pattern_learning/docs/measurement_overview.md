# Measurement and experiment scripts

6T1C 5×5 memristor crossbar (Arduino Due on COM4), used as analog gradient
memory for e-prop learning. Scripts are grouped by experiment; each folder has
its own README with the run order and the result.

Scripts were copied here from `measurements/` and `SNN/snn_pattern_learning/`.
The originals are still in place — these are organised copies, so a script that
reads a data file expects to be run from the directory holding that data
(usually `measurements/`). Paths inside the scripts were not rewritten.

| # | Folder | What it measures |
|---|--------|------------------|
| 1 | `01_potdep_curve` | POT/DEP cycle curves, full-range; device model fits |
| 2 | `02_grad_fidelity_uv` | desired vs actual update over many (u,v) conditions |
| 3 | `03_read_disturbance` | decay per read, and per unit time (leakage) |
| 4 | `04_halfselect_sequence` | disturbance from ordered half-select pairs |
| 5 | `05_teacher_student` | e-prop demo on a teacher-generated spike sequence |
| 6 | `06_temporal_xor` | e-prop demo on temporal XOR |
| — | `common` | firmware, environment capture |

## Device operating point

Everything below assumes the operating point validated 2026-08-05
(`uv_grid_sweep`, r ≈ 0.92):

```
bit_length 10, pulse_width 15 us, pre/post 100 us, zero 10 us,
read_time 20, read_delay 10, normalization_scale 0.7
```

## Signal roles (firmware `6T1C_5x5_add_stochastic.ino`)

| Lines | Effect |
|-------|--------|
| N1 (row) + N2 (col) | POTENTIATION coincidence |
| N3 (row) + N4 (col) | DEPRESSION coincidence |
| N1 + N3 | Reset — shorts the storage cap |
| N5 − N6 | differential read, in ADC LSB |

Array rails: **+455 / −444 LSB**.

## Cross-cutting cautions

These cost real time to discover; they apply to more than one experiment.

- **Never compare across sessions.** The device degrades measurably within a
  day (update fidelity r fell 0.91 → 0.48 in one session; the read-disturbance
  floor `V_inf` moved 89% between two sessions while `tau` moved 0.7%). Any
  A/B comparison must have both arms measured in the same session, ideally
  interleaved.
- **Reads are destructive.** Each array read costs ~0.8% of the distance to
  the read attractor. A protocol that reads more in one arm than the other has
  a confound, not a result. See `03_read_disturbance`.
- **`STOCHASTIC_*` commands each perform 2 array reads.** A 20-cycle exposure
  therefore carries 80 reads. Use the `HS_*` opcodes with
  `read_period > update_num` when you want exposure without reads.
- **Serial termination differs by opcode.** `READ_ROW` and `STOCHASTIC_*` end
  with `EOD>`; `HS_*` ends with a bare `operation end` (no newline) and
  `Reset` has no terminator at all. Stopping early on a block count leaves
  `EOD>` in the buffer, which the next command then consumes as its first
  line.
- **`Reset` emits `1 + set_num*(update_num//read_period)` = 31 blocks**, not
  11. Draining the wrong number corrupts the next command's parse.
- **Firmware clamps `bit_length` to `MAX_BIT_LENGTH` = 20.** BL=40 runs are
  silently identical to BL=20.
- **Pearson r inflates with the range of the desired update.** For comparing
  designs use a range-independent metric — residual_sd / gain, i.e. the error
  in units of one coincidence.

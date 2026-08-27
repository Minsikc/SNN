# Measurement data (6T1C 5×5 array)

Copied 2026-08-26 from `measurements/` (root of the lab folder): every dated
`2025-*` / `2026-*` CSV, every `*.npz`, and the `*_CycleTest_Data.csv` files.
File names are unchanged; the `hardware/measure/run.py` runner executes the
analysis scripts with this folder as the working directory so their relative
globs (`glob("*_seq_attractor.csv")`, hard-coded `2026-08-10_14-29_CycleTest_Data.csv`, …)
resolve here.

Key files used in the paper draft (`measurements/paper/README.md` has the
number → file map):

| file | experiment |
|---|---|
| `2026-08-10_14-29_CycleTest_Data.csv` | full-range POT/DEP cycle test (Table 1 update parameters) |
| `2026-08-10_14-43_uv_random_signed.csv` | random signed u⊗v fidelity (r 0.978, 13 LSB/coincidence) |
| `2026-08-06_23-07_disturb_all25.csv`, `*_disturb_signed.csv` | read disturbance, 1000 reads, both polarities |
| `2026-08-07_14-4x_seq_attractor.csv`, `2026-08-07_12-3x_half_select_seq16.csv` | half-select pair attractors |
| `2026-08-20_12-01_abab_dno_cells.csv` | NORMAL vs DNO ABAB (n = 10) |
| `2026-08-24_17-33_col5_diagnosis.npz` | column-5 fault diagnosis |

This folder is git-ignored (`*.csv`, `*.npz`, `*.json` in `.gitignore`); keep a
backup outside the repository. `V_inf` of the read-disturbance model changes
between sessions (+67 → +7 LSB between 2026-08-06 and 08-10) — always fit it
on same-session data.

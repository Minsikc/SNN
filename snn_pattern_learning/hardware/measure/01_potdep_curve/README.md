# 1. POT/DEP curve, and the device model fitted to it

Drives the whole array to one rail and back, reading periodically, so the
full-range update curve is visible. This is the dataset that decides what the
device model may claim outside the small range a single burst covers.

## Measure

```
python measurement_functions.py        # run_cycle_test(arduino)
```

Firmware opcode `GLOBAL_PD_SEQ_READ`: `update_num` potentiation pulses with a
read every `read_period`, then the same for depression. Writes
`*_CycleTest_Data.csv`.

**CSV layout, easy to get wrong.** `Read_all_rows_sequentially` emits FIVE
fields per read event, one per array row, so the field axis is
`(read_event × row)`. Reshaping it as `(step × half-block)` interleaves the
potentiation and depression ramps and produces nonsense (it inflated a model
RMS from 61 to 270 LSB before it was caught). Also: fields contain `[...]`
lists with commas inside, so `pandas.read_csv` mis-splits them — use
`csv.reader` + `ast.literal_eval`.

## Plot

```
python plot_cycle_test.py <csv>              # the 5x5 grid of raw curves
python plot_cycletest_vs_aihwkit.py <csv>    # measured vs fitted LinearStep
python plot_percell_cycletest.py             # shared vs per-cell parameters
```

## Fit

```
python fit_update_model.py            # saturating burst model + half-select
python fit_aihwkit_linearstep.py      # map onto aihwkit LinearStepDevice
python fit_percell_cycletest.py       # per-cell parameters from the cycle test
```

## Result

The per-coincidence step shrinks along a burst — POT 15.0 → 9.9 LSB over 10
pulses, DEP 15.3 → 7.4. **DEP saturates about twice as fast as POT**
(k = 0.081 vs 0.046), so DEP is stronger at C=1 but weaker by C=9.

aihwkit's `LinearStepDevice` applies `w += slope·w + scale` per coincidence,
which iterates to `A·exp(slope·n)` — the *same* form as the measured
within-burst decay. So the mapping is exact, not a curve-shape coincidence:

```python
LinearStepDevice(dw_min=0.036142, up_down=-0.2859,
                 gamma_up=1.7824, gamma_down=1.7089,
                 w_max=1.0, w_min=-0.9758,
                 dw_min_dtod=0.157, dw_min_std=0.564)
```

1 normalised weight unit = 455 LSB.

**The fitted gamma does not survive the cycle test.** gamma ≈ 1.78 was
extrapolated from bursts that move the cell only ~16% of its range; it
predicts the step vanishing at ±255 LSB, so the cell would stall well short of
the ±455 rails. The cycle test drives rail to rail and rejects it:

| model | cycle-test RMS |
|-------|----------------|
| fitted gamma (1.78 / 1.71) | 82.4 LSB |
| **gamma = 1 (soft bounds)** | **61.3 LSB** |

Use gamma ≈ 1 for full-range simulation; the fitted gamma is only valid for
short bursts near reset (the e-prop case).

**Most of the remaining error is cell spread, not model form.** Fitting per
cell drops the RMS 61.3 → 12.9 LSB (79%). Per-cell gamma is 1.20 ± 0.13, i.e.
scattered around 1. Step scales vary most (σ/μ = 0.12–0.21); gammas and bounds
are uniform (0.03–0.09), so in simulation `dw_min_dtod` matters and
`gamma_*_dtod` / `w_*_dtod` can stay small.

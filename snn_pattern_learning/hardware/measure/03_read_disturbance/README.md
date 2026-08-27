# 3. Read disturbance, and leakage

Reading is destructive. This measures how much, separates it from time-driven
leakage, and fits a model usable in simulation.

## Run order

```
python read_disturb_all25.py --n-reads 1000 --reps 3   # per-cell decay, positive start
python read_disturb_signed.py --n-reads 1000 --reps 3  # BOTH polarities, interleaved
python read_disturb_leakage.py                         # separates time from read count
```

then

```
python fit_read_disturb_modelB.py    # tau, V_inf, with c2c/d2d split
python fit_disturb_signed.py         # decides what V_inf means
python plot_decay_both_signs.py      # all 25 cells, both polarities
python plot_disturb_signed.py
```

## Model

```
V_k = V_inf + (V_0 - V_inf)·exp(-k/tau)          k = cumulative read count
per read:  V <- V_inf + (V - V_inf)·exp(-1/tau)
```

Each read removes a fixed fraction of the **distance to an attractor**, not of
the level. The simpler `V_k = V_0·exp(-k/tau)` was tested and **rejected**:
cross-validated RMSE 23.1 vs 4.8 LSB. The traces plainly stop at a non-zero
level after 7.9 time constants — a pure exponential predicts 0.1 LSB where 60
LSB was measured.

## Parameters (2026-08-10 session, 150 traces, model RMSE 4.98 LSB)

| | POT arm | DEP arm | d2d | c2c |
|---|---|---|---|---|
| tau (reads) | 127.80 | 117.15 | 3.81 | 3.68 |
| V_inf (LSB) | +7.17 | +15.04 | 19.32 | 3.78 |

`d2d` is corrected for the sampling noise in each cell mean:
`var_d2d = var(cell means) − var_c2c / n_rep`.

Simulation recipe:

```python
tau_cell  = N(127.8 if pot else 117.2, 3.81**2)   # reads
Vinf_cell = N(+7.2  if pot else +15.0, 19.32**2)  # LSB, per cell
# per read:
V = Vinf_cell + (V - Vinf_cell) * exp(-1 / max(N(tau_cell, 3.68**2), 1))
```

Replaying this read-by-read reproduces the 150 measured traces at RMS 6.7 LSB
when per-cell parameters are used, 16.6 LSB when they are drawn at random —
the difference is `V_inf` spread, which is the dominant uncertainty.

## V_inf is an absolute attractor

Fitted only to positive starts, `V_inf` could equally have been "a fixed level"
or "a fixed fraction of `V_0`". A negative-start arm separates them, since a
cell at −400 LSB must *rise through zero* under the first reading and stay
negative under the second. It rose to **+15 LSB**, and the pooled regression
over both arms gives slope **−0.008** (flat). So `V_inf` does not depend on
where the cell started, and no sign branch is needed in simulation.

## Other findings

- **Per read: 0.79%** of the distance to `V_inf`.
- **tau differs by polarity** — 10.7 reads, highly significant (Welch
  t = 12.4, p = 3e-24) and 3× the c2c spread. Keep the arms separate; pooling
  inflates c2c from 3.7 to 7.0 reads by leaking the polarity difference into
  the within-cell term.
- **tau is uniform across cells** (d2d ≈ c2c), **V_inf is not**
  (d2d/c2c ≈ 5). Calibrate `V_inf` per cell if you can.
- **V_inf is not stable across sessions** (+67.4 → +7.2 LSB in four days,
  −89%) while tau moved 0.7%. Re-measure `V_inf` each session; reuse tau.
- **Some cells have negative V_inf** (−61 to +52 LSB). A negative asymptote
  cannot be stored charge, so `V_inf` is a read-path offset, not a level.

## Leakage is a separate, smaller mechanism

`read_disturb_leakage.py` breaks the collinearity between read count and
elapsed time by inserting waits (one array read costs ~0.21 s).

```
Exp C  reads fixed at 2, wait swept 0.5-30 s
       drop = 6.00·(1 - exp(-t/5.18 s)) + 7.39      rms 0.59 LSB
Exp A  elapsed time matched (~8.5 s), reads 8 vs 40
       -> 2.18 LSB per extra read
```

So leakage saturates at ~6 LSB with tau ≈ 5.2 s, while reading keeps
accumulating (100 LSB by 40 reads). **tau in the model above is in reads, not
seconds** — do not add the two time constants together.

Not measured: retention beyond 30 s, and leakage from negative levels.

## Caution

Averaging cells with slightly different rates manufactures a fake double
exponential — a simulated mixture with only 0.4% rate spread fits a double
exponential 11000× better than a single one, though every constituent is a pure
single exponential. Fit per cell.

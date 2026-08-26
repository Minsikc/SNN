# 2. Desired vs actual update, across (u,v) conditions

The core fidelity question: when the algorithm asks for an outer-product
update `u ⊗ v`, how close is what the array actually stores?

Two definitions of "desired" are kept apart throughout:

| | meaning |
|---|---|
| **ideal** | `u_r · v_c · L` — what the algorithm asked for |
| **realized** | `sign · (C_pot + C_dep)` — what the sampled pulses delivered |

The gap between them is Bernoulli sampling noise, which is not the device's
fault, so **the device is judged against `realized`**.

## Run order

```
python uv_grid_sweep.py --levels 9 --mode normal   # 9x9 (u,v) grid, scalar u,v
python uv_random_signed.py --trials 40             # u,v ~ U(-1,1), all 4 quadrants
python signed_uv_exposure.py                       # fixed magnitude, mixed signs
python uv_halfselect_corrected.py                  # all-positive, P only
```

then

```
python plot_uv_random.py
python plot_signed_heatmaps.py
python plot_single_uv_exposure.py
```

## Result

Random signed `u,v ~ U(-1,1)`, BL=10, 40 trials × 25 cells:

```
r(delta, realized)  = +0.978      gain 13.0 LSB/coincidence
r(delta, ideal)     = +0.929      residual 0.73 coincidences
```

The 0.05 drop from realized to ideal is sampling, not the device — it shrinks
as BL grows.

Sign handling is clean: no cell ever moved the wrong way, and P/D gain
asymmetry is only 9% (+12.47 vs −11.32 LSB per coincidence).

**Cross-design comparison must use `residual/gain`, not r.** Pearson r inflates
with the range of `desired`, so a design that happens to span a wider range
scores higher for free:

| design | r | gain | residual/gain |
|--------|---|------|---------------|
| random signed U(−1,1) | 0.978 | 12.96 | 0.73 |
| fixed-magnitude signed | 0.990 | 15.60 | 0.73 |
| all-positive (P only) | 0.914 | 12.24 | 0.82 |

Read by r, the fixed-magnitude design looks best; read by residual/gain the
first two are identical. Mixing signs does **not** degrade fidelity.

## Cautions

- `hard_reset()` must drain 31 blocks, not 11 (see top-level README).
- Drift over a sweep is small (≈ −1.4 LSB) and correcting for it *hurt*
  (r 0.859 → 0.755) because the correction added more noise than it removed.
  `--drift-correct` therefore defaults **off**; drift is logged only.
- Include low-C_mean grid points in the aggregate — dropping them biases the
  fit toward the easy regime.

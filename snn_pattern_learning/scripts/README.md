# Experiment scripts

Moved here from the package root on 2026-08-26 (`git mv`, history preserved).
Every script inserts the package root into `sys.path`, so run them **from
`snn_pattern_learning/`** (relative output paths such as `results/eprop_grad_log/…`
are resolved against the cwd):

```
python scripts/teacher_student/run_three_conditions.py --help
python scripts/teacher_student/sweep_sw_lr.py
python scripts/teacher_student/run_alif_conditions.py --betas 0 0.5 1.0
```

| folder | contents |
|---|---|
| `teacher_student/` | paper experiment 1: `run_three_conditions.py` (bptt / digital / frozen SW baselines — note the legacy "bptt" mixes e-prop + autograd, see `docs/INTEGRITY_REPORT.md`), `sweep_sw_lr.py` (lr × seed sweep → `sw_sweep_4x4_ds25.json`), `run_alif_conditions.py` (**new** ALIF / gradient-chain extension on the `eprop` core), `spikegen_baseline.py`, `run_teacher_perfect.py`, `sweep_teacher_perfect.py`, `sweep_bptt_perfect.py`, `verify_vrd0.py`, `check_eprop_seeds.py`, `check_bptt_orig.py`, `check_intersection.py`, `eval_checkpoint_raster.py`, `capture_spikes.py`, `compare_*.py`, plots (`plot_4x4_ds25_summary.py` → paper fig. 4, `plot_teacher_student_postfix.py`, `plot_three_conditions.py`, …) |
| `xor/` | temporal-XOR demo: `run_xor.py` (`--condition bptt|digital|frozen|analog`), `run_xor_quantized.py`, `sweep_xor*.py`, `plot_xor*.py`, `debug_xor_*.py` |
| `analysis/` | gradient-fidelity analysis of the per-epoch CSV logs: `analyze_grad_fidelity.py`, `plot_grad_fidelity.py`, `plot_hw_calibration.py`, `plot_hw_loss_raster.py`, `check_grad_saturation.py`, `check_transpose_local.py`, `debug_grad_structure.py` |
| `hw/` | hardware helpers: `hw_preflight.py` (ADC liveness — read ≈0 means a dead setup), `test_hardware.py`, `test_eprop_hw_gradient.py`, `run_analog_seeds.sh`, `run_analog_scale.sh` (`bash scripts/hw/run_analog_seeds.sh` from the package root) |
| `legacy/` | pre-e-prop demos and wandb sweeps (`run_demos.py`, `run_ablation_demo.py`, `train_sweep*.py`) |

Device characterisation scripts (POT/DEP curve, u·v fidelity, read disturbance,
half-select, DNO/ABAB, column diagnosis) are under `hardware/measure/` and run
through `python -m hardware.measure.run …`.

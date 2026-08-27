#!/usr/bin/env python3
"""aihwkit device model of the 6T1C cell, built from the measured fits.

Combines the three measured behaviours into one importable module:

  1. update curve   -> LinearStepDevice, parameters from
                       aihwkit_linearstep_params.json (fit_aihwkit_linearstep.py)
  2. read disturb   -> V <- V_inf + (V - V_inf) * exp(-n/tau), applied to the
                       tile weights per read (fit_read_disturb_modelB.py /
                       scripts/03_read_disturbance/README.md).  aihwkit has no
                       per-read disturb concept, so this is a host-side step,
                       not a device parameter.
  3. half-select    -> reported as an effective gain factor (~ +6% on the net
                       drive) and the per-pair coefficients; NOT injected into
                       the device.  See scripts/04_halfselect_sequence.

Session dependence: dw_min / V_inf move between sessions (V_inf by 89% in four
days), tau and the curve SHAPE do not.  Re-run fit_aihwkit_linearstep.py and
the disturb fit for the session you want to model, then point this module at
the fresh JSON / pass fresh disturb numbers.

Units: aihwkit works in normalised weights; 1.0 = lsb_per_w LSB (455 by
default).  ReadDisturbModel takes and returns normalised weights but its
parameters are quoted in LSB, matching the fit output.

Verification (run this file inside the WSL aihwkit env):

    ~/aihwkit-env/bin/python model_6t1c.py

The tile must be driven so ONE update = ONE coincidence, i.e. learning rate =
dw_min with x = d = 1; the default lr 0.1 makes aihwkit fire ~3 pulses per
update (lr*x*d/dw_min) which triples every step.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS = os.path.join(HERE, "aihwkit_linearstep_params.json")

# read-disturb fit, 2026-08-10 session (150 traces, model RMSE 4.98 LSB).
# tau is stable across sessions (0.7% in four days) and may be reused;
# V_inf is NOT (moved 89%) and should be re-measured per session.
READ_DISTURB_DEFAULT = dict(
    tau_pot=127.80, tau_dep=117.15,   # reads; arm = polarity of the level
    tau_d2d=3.81, tau_c2c=3.68,
    vinf_pot_lsb=7.17, vinf_dep_lsb=15.04,
    vinf_d2d_lsb=19.32, vinf_c2c_lsb=3.78,
)

# half-select in-update effect (scripts/04): ~1.7 LSB/pair, net 1.6 pairs per
# cell-update following the requested sign -> amplifies the effective gain.
HALFSELECT_GAIN = 1.06


def load_params(path=DEFAULT_PARAMS):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_device(params, full_range=False, variability=True):
    """LinearStepDevice from the fitted parameter set.

    full_range=False keeps the fitted gamma (valid for short bursts near the
    reset state, the e-prop case).  full_range=True sets gamma = 1 (soft
    bounds), which the rail-to-rail cycle test prefers (RMS 61.3 vs 82.4 LSB);
    the fitted gamma stalls the cell at ~56% of the rails.
    variability=False zeroes every dtod/std term for deterministic replays.
    """
    from aihwkit.simulator.configs.devices import LinearStepDevice

    cfg = dict(params["linearstep"])
    if full_range:
        cfg["gamma_up"] = 1.0
        cfg["gamma_down"] = 1.0
    if not variability:
        for k in ("dw_min_dtod", "dw_min_std", "up_down_dtod",
                  "gamma_up_dtod", "gamma_down_dtod",
                  "w_max_dtod", "w_min_dtod"):
            cfg[k] = 0.0
    return LinearStepDevice(**cfg)


def make_rpu_config(params, full_range=False, variability=True):
    """SingleRPUConfig with the update discretisation matched to the hardware:
    stochastic pulse trains, BL = 10 (firmware bit_length at the validated
    operating point), no BL management so the coincidence count stays
    Binomial(BL, px*pd) like the STOCHASTIC_* opcodes."""
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.parameters.enums import PulseType

    rpu = SingleRPUConfig(
        device=make_device(params, full_range=full_range,
                           variability=variability))
    rpu.update.desired_bl = 10
    rpu.update.update_bl_management = False
    rpu.update.update_management = False
    rpu.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    return rpu


class ReadDisturbModel:
    """Per-read decay toward an absolute attractor, per cell.

    V_k = V_inf + (V_0 - V_inf) * exp(-k/tau), k = cumulative read count.
    tau depends on the polarity of the level (POT/DEP arms differ by 10.7
    reads, p=3e-24); V_inf does not depend on the starting level (pooled
    slope -0.008), so only tau gets a sign branch here.

    Weights in/out are normalised; internal parameters are LSB.
    """

    def __init__(self, shape, lsb_per_w, disturb=None, seed=None):
        d = dict(READ_DISTURB_DEFAULT)
        if disturb:
            d.update(disturb)
        rng = np.random.default_rng(seed)
        self.lsb_per_w = float(lsb_per_w)
        self.tau_pot = rng.normal(d["tau_pot"], d["tau_d2d"], shape)
        self.tau_dep = rng.normal(d["tau_dep"], d["tau_d2d"], shape)
        # V_inf is a read-path offset (some cells fit negative), one draw per
        # cell; the POT/DEP arm means differ, keep the arm of the level's sign
        self.vinf_pot = rng.normal(d["vinf_pot_lsb"], d["vinf_d2d_lsb"], shape)
        self.vinf_dep = rng.normal(d["vinf_dep_lsb"], d["vinf_d2d_lsb"], shape)
        self.tau_c2c = d["tau_c2c"]
        self._rng = rng

    def apply(self, w, n_reads=1):
        """Return w after n_reads array reads (normalised units)."""
        v = np.asarray(w, dtype=float) * self.lsb_per_w
        pot = v >= 0.0
        tau = np.where(pot, self.tau_pot, self.tau_dep)
        tau = np.maximum(self._rng.normal(tau, self.tau_c2c), 1.0)
        vinf = np.where(pot, self.vinf_pot, self.vinf_dep)
        v = vinf + (v - vinf) * np.exp(-float(n_reads) / tau)
        return v / self.lsb_per_w


def apply_read_disturb(tile, model, n_reads=1):
    """Disturb an aihwkit tile in place; returns the new weights."""
    import torch
    w = tile.get_weights()[0]
    w2 = torch.as_tensor(model.apply(w.numpy(), n_reads), dtype=w.dtype)
    tile.set_weights(w2)
    return w2


def _verify():
    import torch
    from aihwkit.simulator.tiles import AnalogTile

    p = load_params()
    lsb = p["lsb_per_w"]
    w0 = p["w0_reset"]
    m = p["measured"]
    dw_min = p["linearstep"]["dw_min"]

    print("=" * 64)
    print("tile burst replay (deterministic, lr = dw_min -> 1 pulse/update)")
    print("=" * 64)
    rpu = make_rpu_config(p, variability=False)
    # deterministic single-coincidence drive for the check
    from aihwkit.simulator.parameters.enums import PulseType
    rpu.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
    rpu.update.desired_bl = 1
    tile = AnalogTile(1, 1, rpu_config=rpu)
    tile.set_learning_rate(dw_min)

    print(f"{'C':>3s} {'closed form':>12s} {'tile POT':>10s} "
          f"{'closed form':>12s} {'tile DEP':>10s}   (LSB)")
    ok = True
    for C in (1, 3, 5, 10):
        row = []
        for sgn, A, k in ((+1, m["A_pot"], m["k_pot"]),
                          (-1, m["A_dep"], m["k_dep"])):
            tile.set_weights(torch.tensor([[w0]]))
            for _ in range(C):
                tile.update(torch.ones(1, 1), -sgn * torch.ones(1, 1))
            dw = (float(tile.get_weights()[0][0, 0]) - w0) * lsb
            closed = sgn * A * (1 - np.exp(-k * C)) / k
            row.extend([closed, dw])
            if abs(dw - closed) > 0.15 * abs(closed):
                ok = False
        print(f"{C:3d} {row[0]:12.1f} {row[1]:10.1f} "
              f"{row[2]:12.1f} {row[3]:10.1f}")
    print("burst replay:", "OK" if ok else "MISMATCH > 15%")

    print()
    print("=" * 64)
    print("read disturb replay (expect ~0.79%/read toward V_inf)")
    print("=" * 64)
    rd = ReadDisturbModel((1, 1), lsb, seed=0)
    v = 400.0 / lsb
    print(f"{'reads':>6s} {'level (LSB)':>12s}")
    for k in (0, 10, 100, 300, 1000):
        w = rd.apply(np.array([[v]]), n_reads=k) if k else np.array([[v]])
        print(f"{k:6d} {float(w[0, 0]) * lsb:12.1f}")
    print(f"per-cell V_inf draw: {float(rd.vinf_pot[0, 0]):+.1f} LSB, "
          f"tau {float(rd.tau_pot[0, 0]):.1f} reads")


if __name__ == "__main__":
    _verify()

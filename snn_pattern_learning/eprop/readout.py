"""Readout-gradient backends for :class:`eprop.model.EpropRSNN`.

The e-prop readout update is a sum over time of rank-1 outer products
``err_t (x) trace_out_t`` (config.readout_grad_scale included). Where that sum
is formed is the only difference between the training conditions:

* :class:`SoftwareReadout`  -- exact sum into ``out.weight.grad`` (digital e-prop,
  BPTT, frozen-readout control).
* :class:`HardwareReadout`  -- every outer product is sent to the 6T1C array
  (``hardware.hw_interface.MemristorInterface``) or its mock as stochastic
  pulse-coincidence updates; the stored charge is read once per epoch,
  scaled and written to ``out.weight``. This is a faithful port of the
  ``Basic_RSNN_eprop_HW_forward`` logic (normalisation, 4-quadrant split,
  auto-calibration, per-column calibration, gradient CSV log, freeze switch).
"""
from __future__ import annotations

import csv
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import HardwareReadoutConfig

logger = logging.getLogger(__name__)


class ReadoutBackend:
    hardware = False

    def attach(self, model):
        self.model = model

    # per-epoch hooks -----------------------------------------------------
    def connect(self) -> bool:
        return False

    def disconnect(self):
        pass

    def reset(self, hard_reset: bool = True) -> bool:
        return True

    def begin_forward(self, model):
        pass

    def accumulate(self, model, err: torch.Tensor, trace_out: torch.Tensor, desired: torch.Tensor):
        raise NotImplementedError

    def apply(self, model, learning_rate: float):
        """End-of-epoch hook (no-op for software)."""
        return None

    def __repr__(self):
        return self.__class__.__name__


class SoftwareReadout(ReadoutBackend):
    """Exact outer-product accumulation into ``out.weight.grad`` (legacy digital path)."""

    def accumulate(self, model, err, trace_out, desired):
        model.out.weight.grad += desired


class HardwareReadout(ReadoutBackend):
    """Accumulate the readout gradient on the 6T1C array (or its mock)."""
    hardware = True

    def __init__(self, cfg: Optional[HardwareReadoutConfig] = None, interface=None):
        self.cfg = cfg or HardwareReadoutConfig(enabled=True)
        self.interface = interface if interface is not None else self._make_interface(self.cfg)
        self.freeze = self.cfg.freeze_wout
        self.grad_log_path = self.cfg.grad_log_path
        self.grad_log_epoch = 0
        self.adc_to_grad_scale = self.cfg.adc_to_grad_scale
        self._calibrated_once = False
        self._col_gain: Optional[np.ndarray] = None
        self.running_max_err = 1e-8
        self.running_max_trace = 1e-8
        self._queue: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.desired_accumulated: Optional[torch.Tensor] = None
        self.last_stats = {}

    @staticmethod
    def _make_interface(cfg: HardwareReadoutConfig):
        from hardware import MemristorInterface, MockMemristorInterface
        if cfg.use_mock:
            return MockMemristorInterface(port=cfg.serial_port, baud_rate=cfg.baud_rate,
                                          bit_length=cfg.bit_length,
                                          quantize_bits=cfg.mock_quantize_bits,
                                          quantize_seed=cfg.mock_quantize_seed)
        return MemristorInterface(port=cfg.serial_port, baud_rate=cfg.baud_rate,
                                  bit_length=cfg.bit_length, pulse_width=cfg.pulse_width,
                                  pulse_pre=cfg.pulse_pre, pulse_post=cfg.pulse_post,
                                  pulse_zero=cfg.pulse_zero, read_time=cfg.read_time,
                                  read_delay=cfg.read_delay,
                                  no_read_updates=cfg.no_read_updates, dno=cfg.dno)

    # ------------------------------------------------------------------ hooks
    def attach(self, model):
        super().attach(model)
        n = self.cfg.array_size
        if not (1 <= model.n_hidden <= n and 1 <= model.n_out <= n):
            raise ValueError(f"n_hidden/n_out must be 1..{n} for the {n}x{n} array")
        self.desired_accumulated = torch.zeros(model.n_out, model.n_hidden)

    def connect(self) -> bool:
        return bool(self.interface.connect())

    def disconnect(self):
        self.interface.disconnect()

    def reset(self, hard_reset: bool = True) -> bool:
        ok = self.interface.reset(hard_reset=hard_reset)
        self.running_max_err = 1e-8
        self.running_max_trace = 1e-8
        self._queue = []
        return bool(ok)

    def begin_forward(self, model):
        if self.desired_accumulated is None or self.desired_accumulated.device != model.device:
            self.desired_accumulated = torch.zeros(model.n_out, model.n_hidden, device=model.device)

    # -------------------------------------------------------------- per step
    def _normalize(self, err: torch.Tensor, trace: torch.Tensor):
        err_avg = err.mean(dim=0).detach()
        trace_avg = trace.mean(dim=0).detach()
        err_signs = torch.sign(err_avg)
        trace_signs = torch.sign(trace_avg)
        err_signs[err_signs == 0] = 1
        trace_signs[trace_signs == 0] = 1
        err_abs, trace_abs = err_avg.abs(), trace_avg.abs()
        if self.cfg.fixed_norm is not None:
            err_max, trace_max = self.cfg.fixed_norm
        else:
            self.running_max_err = max(self.running_max_err, err_abs.max().item())
            self.running_max_trace = max(self.running_max_trace, trace_abs.max().item())
            err_max, trace_max = self.running_max_err, self.running_max_trace
        s = self.cfg.normalization_scale
        u = (err_abs / err_max).clamp(0, 1) * s
        v = (trace_abs / trace_max).clamp(0, 1) * s
        return (u.cpu().numpy().astype(np.float32), v.cpu().numpy().astype(np.float32),
                err_signs.cpu().numpy().astype(np.float32), trace_signs.cpu().numpy().astype(np.float32))

    def accumulate(self, model, err, trace_out, desired):
        self.desired_accumulated += desired.detach()
        u_p, v_p, u_s, v_s = self._normalize(err, trace_out)
        n = self.cfg.array_size
        if u_p.size < n:          # smaller model on the top-left block: pad with p=0
            u_p = np.pad(u_p, (0, n - u_p.size)); u_s = np.pad(u_s, (0, n - u_s.size), constant_values=1.0)
        if v_p.size < n:
            v_p = np.pad(v_p, (0, n - v_p.size)); v_s = np.pad(v_s, (0, n - v_s.size), constant_values=1.0)
        item = (u_p, v_p, u_s, v_s)
        if self.cfg.batch_quadrants:
            self._queue.append(item)
        else:
            self.interface.accumulate_outer_product(*item)

    # ------------------------------------------------------------- per epoch
    def apply(self, model, learning_rate: float):
        """Read the array, calibrate, log, write ``W_out <- W_out - lr * G``.
        Returns the scaled hardware gradient (or None when frozen)."""
        if self.freeze:
            self.desired_accumulated.zero_()
            self._queue = []
            return None
        if self.cfg.batch_quadrants and self._queue:
            self.interface.accumulate_outer_products_grouped(self._queue)
            self._queue = []

        adc = self.interface.read_accumulated_gradient()[: model.n_out, : model.n_hidden]
        desired = self.desired_accumulated.detach().cpu().numpy()

        if self.cfg.auto_calibrate_scale:
            sw_mag, hw_mag = float(np.abs(desired).mean()), float(np.abs(adc).mean())
            if hw_mag > 1e-9 and sw_mag > 1e-9:
                target = sw_mag / hw_mag
                if not self._calibrated_once:
                    self.adc_to_grad_scale, self._calibrated_once = target, True
                else:
                    a = self.cfg.calibrate_ema
                    self.adc_to_grad_scale = (1 - a) * self.adc_to_grad_scale + a * target
        gain = np.ones(model.n_hidden)
        if self.cfg.calibrate_per_column:
            sw_col = np.abs(desired).mean(axis=0)
            hw_col = np.abs(adc).mean(axis=0)
            if self._col_gain is None:
                self._col_gain = np.ones(model.n_hidden)
            for j in range(model.n_hidden):
                if hw_col[j] > 1e-9 and sw_col[j] > 1e-9:
                    tgt = (sw_col[j] / hw_col[j]) / self.adc_to_grad_scale
                    a = self.cfg.calibrate_ema
                    self._col_gain[j] = (1 - a) * self._col_gain[j] + a * tgt
            gain = np.clip(self._col_gain, 0.2, 5.0)

        hw_grad_np = adc * self.adc_to_grad_scale * gain[None, :]
        hw_grad = torch.tensor(hw_grad_np, dtype=torch.float32, device=model.device)

        corr = float(np.corrcoef(desired.ravel(), adc.ravel())[0, 1]) if np.std(adc) > 0 and np.std(desired) > 0 else 0.0
        self.last_stats = dict(corr=corr, mean_abs_adc=float(np.abs(adc).mean()),
                               mean_abs_desired=float(np.abs(desired).mean()),
                               adc_to_grad_scale=self.adc_to_grad_scale)
        logger.info(f"[HW-READOUT] epoch {self.grad_log_epoch}: r={corr:+.3f} "
                    f"|adc|={self.last_stats['mean_abs_adc']:.1f} |desired|={self.last_stats['mean_abs_desired']:.4f}")

        if self.grad_log_path:
            self._log(desired, adc, hw_grad_np)
        self.grad_log_epoch += 1
        self.desired_accumulated.zero_()

        with torch.no_grad():
            model.out.weight.data -= learning_rate * hw_grad
        return hw_grad

    def _log(self, desired, adc, hw_scaled):
        new = not os.path.exists(self.grad_log_path)
        d = os.path.dirname(self.grad_log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.grad_log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "row", "col", "desired", "hw_adc", "hw_scaled", "adc_to_grad_scale"])
            for i in range(desired.shape[0]):
                for j in range(desired.shape[1]):
                    w.writerow([self.grad_log_epoch, i + 1, j + 1, float(desired[i, j]),
                                float(adc[i, j]), float(hw_scaled[i, j]), float(self.adc_to_grad_scale)])

    def __repr__(self):
        kind = "mock" if self.cfg.use_mock else self.cfg.serial_port
        return f"HardwareReadout({kind}, BL={self.cfg.bit_length}, NR={self.cfg.no_read_updates}, DNO={self.cfg.dno})"

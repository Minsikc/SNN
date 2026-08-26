"""Pre-flight check before an analog run: COM4 reachable + ADC alive.

Healthy: raw ADC reads in the few-hundreds (historically ~400-600).
Dead setup symptom: reads ~0 (see memory note adc-near-zero-symptom).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hardware import MemristorInterface

hw = MemristorInterface(port="COM4", baud_rate=115200, bit_length=10,
                        pulse_width=15, pulse_pre=100, pulse_post=100,
                        pulse_zero=10, read_time=20, read_delay=10,
                        timeout=10.0)
if not hw.connect():
    print("FAIL: could not open COM4 (busy or unplugged)")
    sys.exit(1)

zero = np.zeros(5)
pre, post = hw._send_command("POTENTIATION", zero, zero)
hw.disconnect()

if pre.shape != (5, 10):
    print(f"FAIL: bad response shape {pre.shape}")
    sys.exit(1)

n5, n6 = pre[:, :5], pre[:, 5:]
diff = n5 - n6
print("raw N5 reads:\n", n5)
print("raw N6 reads:\n", n6)
print(f"N5 mean {n5.mean():.1f}  N6 mean {n6.mean():.1f}  "
      f"diff mean {diff.mean():.1f} std {diff.std():.1f}")
if n5.mean() < 50 and n6.mean() < 50:
    print("WARNING: ADC near zero -- dead setup symptom, do NOT start the run")
    sys.exit(2)
print("OK: ADC alive")

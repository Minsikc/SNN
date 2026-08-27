"""pytest configuration: make ``snn_pattern_learning/`` importable and keep the
tests deterministic and hardware-free."""
import os
import sys

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _cpu_threads():
    import torch
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    yield

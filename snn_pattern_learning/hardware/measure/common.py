"""Shared helpers for the 6T1C 5x5 array measurement scripts.

Every measurement script under ``hardware/measure/0*_*/`` was written as a
self-contained file with its own copy of the serial helpers (``_drain``,
``send``, ``to_diff`` ...). Those copies are kept verbatim so the recorded
data stays reproducible; *new* scripts should import from here instead.

The functions below are the union of the verified helper variants
(``uv_random_signed.py`` / ``signed_uv_exposure.py`` family, 2026-08-07..10)
and ``hardware/hw_interface.py``. They encode every firmware quirk that cost
real time to discover:

* ``STOCHASTIC_*`` commands perform a pre- AND post-read (2 array reads per
  call); the ``_NR`` variants perform none. Reads are destructive
  (~0.79 %/read toward a cell-specific attractor), so exposure experiments
  must use ``_NR`` / ``HS_*`` / ``PROG_NR`` and measure with ``READ_ROW``.
* ``Reset`` emits ``1 + set_num * (update_num // read_period)`` blocks and no
  ``EOD`` -- 31 blocks for the canonical command. Draining 11 corrupts the
  next command.
* ``READ_ROW`` and ``STOCHASTIC_*`` end with ``EOD>``; ``HS_*`` ends with a
  bare ``operation end`` (no newline); ``Reset`` has no terminator.
* The firmware clamps ``bit_length`` to ``MAX_BIT_LENGTH`` (20) silently.
* Firmware ``is_potentiation`` is an exact string compare against
  ``STOCHASTIC_DEPRESSION`` (see ``firmware/`` and the 04 README): use the
  opcode names exactly as produced here.

Data location: measurement scripts read/write CSVs in the *current working
directory*. :func:`data_dir` resolves the canonical data folder
(``hardware/measure/data`` or ``$SNN_MEASURE_DATA``); ``run.py`` changes into
it before executing a script so the legacy relative globs keep working.
"""
from __future__ import annotations

import contextlib
import datetime
import os
import re
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

N = 5                      # array is 5 x 5
MAX_BIT_LENGTH = 20        # firmware clamp
RAILS_LSB = (+455, -444)   # measured saturation levels (2026-08-08)

# Operating point validated 2026-08-05 (uv_grid_sweep, r ~ 0.92)
OP_POINT = dict(bit_length=10, width=15, pre=100, post=100, zero=10,
                read_time=20, read_delay=10)

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# paths / bookkeeping
# --------------------------------------------------------------------------
def data_dir() -> str:
    """Directory holding the measurement CSV/NPZ files.

    ``$SNN_MEASURE_DATA`` overrides the default ``hardware/measure/data``.
    Legacy data recorded before the consolidation (2026-08-26) lives in
    ``measurements/`` and was copied here; new runs land here directly.
    """
    d = os.environ.get("SNN_MEASURE_DATA") or os.path.join(HERE, "data")
    os.makedirs(d, exist_ok=True)
    return d


def timestamp(seconds: bool = False) -> str:
    fmt = "%Y-%m-%d_%H-%M-%S" if seconds else "%Y-%m-%d_%H-%M"
    return datetime.datetime.now().strftime(fmt)


# --------------------------------------------------------------------------
# command builders (pure functions -- unit tested)
# --------------------------------------------------------------------------
def _fmt_streams(streams: Iterable[Sequence[int]]) -> List[str]:
    out = []
    for s in streams:
        out.append("".join("1" if int(b) else "0" for b in s))
    return out


def stochastic_command(row_streams, col_streams, mode: str = "POTENTIATION",
                       bit_length: int = OP_POINT["bit_length"],
                       width: int = OP_POINT["width"], pre: int = OP_POINT["pre"],
                       post: int = OP_POINT["post"], zero: int = OP_POINT["zero"],
                       read_time: int = OP_POINT["read_time"],
                       read_delay: int = OP_POINT["read_delay"],
                       no_read: bool = True, dno: bool = False) -> str:
    """Outer-product pulse command.

    ``mode`` is ``POTENTIATION`` or ``DEPRESSION``; ``no_read=True`` selects the
    ``_NR`` opcode (no pre/post array read); ``dno=True`` the push-pull
    drive-non-selected variant. Streams are 5 row and 5 column bit vectors of
    length ``bit_length`` (<= 20).
    """
    mode = mode.upper()
    if mode not in ("POTENTIATION", "DEPRESSION"):
        raise ValueError(f"mode must be POTENTIATION or DEPRESSION, got {mode}")
    if bit_length > MAX_BIT_LENGTH:
        raise ValueError(f"bit_length {bit_length} > firmware MAX_BIT_LENGTH {MAX_BIT_LENGTH} "
                         "(the firmware would silently truncate)")
    rows = _fmt_streams(row_streams)
    cols = _fmt_streams(col_streams)
    if len(rows) != N or len(cols) != N:
        raise ValueError("need exactly 5 row streams and 5 column streams")
    if any(len(s) != bit_length for s in rows + cols):
        raise ValueError("every stream must have length bit_length")
    op = f"STOCHASTIC_{'DNO_' if dno else ''}{mode}{'_NR' if no_read else ''}"
    return ",".join(["F", "0", "0", op, "N56", str(bit_length), str(width),
                     str(pre), str(post), str(zero), str(read_time), str(read_delay)]
                    + rows + cols)


def read_row_command(row: int = 5, read_time: int = OP_POINT["read_time"],
                     read_delay: int = OP_POINT["read_delay"]) -> str:
    """``READ_ROW``: row 0-4 reads one row, row 5 reads the whole array.
    Ends with ``EOD>``. One array read (not the pre+post pair)."""
    if not 0 <= row <= 5:
        raise ValueError("row must be 0..5")
    return f"F,{row},5,READ_ROW,N56,1,1,1,F,1,1,1,1,{read_time},0,{read_delay}"


RESET_SET_NUM, RESET_UPDATE_NUM, RESET_READ_PERIOD = 10, 30, 10


def reset_command(set_num: int = RESET_SET_NUM) -> str:
    """Array reset (N1+N3 on every row shorts the storage capacitors).
    Emits :func:`reset_expected_blocks` blocks and NO terminator."""
    return ",".join(["F", "5", "5", "Reset", "N56", str(set_num),
                     str(RESET_UPDATE_NUM), str(RESET_READ_PERIOD), "F",
                     "5", "100", "100", "10", "10", "20", "10"])


def reset_expected_blocks(set_num: int = RESET_SET_NUM,
                          update_num: int = RESET_UPDATE_NUM,
                          read_period: int = RESET_READ_PERIOD) -> int:
    """Block count the firmware emits for a Reset (verified on the wire 2026-08-05)."""
    return 1 + set_num * (update_num // read_period)


def hs_seq_command(pair: int, cycles: int, width: int = OP_POINT["width"]) -> str:
    """``HS_SEQ``: alternate an ordered line pair (tens digit = first line,
    units digit = second; lines 1-4 = N1..N4) for ``cycles`` cycles with no
    reads inside the exposure. Ends with a bare ``operation end``."""
    if not (11 <= pair <= 44 and 1 <= pair % 10 <= 4):
        raise ValueError("pair must be two digits in 1..4, e.g. 12 for N1->N2")
    return f"F,{pair},5,HS_SEQ,N56,1,{cycles},99999,F,{width},100,100,10,20,0,10"


def prog_nr_command(direction: str, n_pulses: int, width: int = OP_POINT["width"]) -> str:
    """``PROG_NR``: ``n_pulses`` whole-array potentiation (``P``) or depression
    (``D``) pulses with no reads at all (firmware 2026-08-20+)."""
    d = direction.upper()
    if d not in ("P", "D"):
        raise ValueError("direction must be 'P' or 'D'")
    return f"F,5,5,PROG_NR,{d},1,{n_pulses},99999,F,{width},100,100,10,20,0,10"


# --------------------------------------------------------------------------
# stream generation / bookkeeping
# --------------------------------------------------------------------------
def bernoulli_streams(probs: Sequence[float], bit_length: int, rng: np.random.Generator,
                      mask: Optional[Sequence[bool]] = None) -> List[np.ndarray]:
    """One Bernoulli stream per line; masked-out lines are all-zero."""
    out = []
    for k in range(N):
        if mask is not None and not mask[k]:
            out.append(np.zeros(bit_length, int))
        else:
            out.append((rng.random(bit_length) < probs[k]).astype(int))
    return out


def coincidence_matrix(row_streams, col_streams) -> np.ndarray:
    """Realised coincidence count C[r, c] = number of slots where both bits are 1."""
    R = np.asarray(row_streams, int)
    Cc = np.asarray(col_streams, int)
    return R @ Cc.T


def quadrant_parts(u: np.ndarray, v: np.ndarray):
    """Signed outer product u (x) v -> four sign-pure (mode, |u| part, |v| part)
    commands, matching ``hw_interface.MemristorInterface._quadrant_parts``.
    Empty quadrants are dropped."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    up, un = np.abs(u) * (u > 0), np.abs(u) * (u < 0)
    vp, vn = np.abs(v) * (v > 0), np.abs(v) * (v < 0)
    parts = [("POTENTIATION", up, vp), ("POTENTIATION", un, vn),
             ("DEPRESSION", up, vn), ("DEPRESSION", un, vp)]
    return [(m, a, b) for m, a, b in parts if np.any(a) and np.any(b)]


# --------------------------------------------------------------------------
# serial I/O
# --------------------------------------------------------------------------
@contextlib.contextmanager
def open_port(port: str = "COM4", baud: int = 115200, timeout: float = 5.0,
              settle: float = 2.0):
    """Open the Arduino Due serial port and wait for it to settle."""
    import serial  # imported lazily so the pure helpers work without pyserial
    with serial.Serial(port, baud, timeout=timeout) as ard:
        time.sleep(settle)
        yield ard


_NUM = re.compile(r"-?\d+")


def drain(ard, timeout_s: float, expect: Optional[int] = None, quiet: float = 1.2):
    """Read reply lines until ``EOD``/``operation end``, ``expect`` numeric
    blocks, or ``quiet`` seconds of silence. Returns the numeric blocks (the
    last 10 integers of every line carrying >= 11 integers: 5 x N5 then 5 x N6).
    """
    blocks, last = [], time.time()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = ard.readline()
        line = raw.decode("utf-8", "ignore").strip() if raw else ""
        if line:
            last = time.time()
            if "EOD" in line or "operation end" in line:
                break
            nums = [int(x) for x in _NUM.findall(line)]
            if len(nums) >= 11:
                blocks.append(nums[-10:])
                if expect and len(blocks) >= expect:
                    break
            continue
        if time.time() - last > quiet:
            break
    return blocks


def send(ard, cmd: str, timeout_s: float = 60.0, expect: Optional[int] = None,
         settle: float = 0.1):
    """Write one command line and drain its reply."""
    ard.reset_input_buffer()
    ard.write((cmd + "\n").encode())
    b = drain(ard, timeout_s, expect)
    time.sleep(settle)
    ard.reset_input_buffer()
    return b


def blocks_to_diff(blocks) -> np.ndarray:
    """First 5 blocks -> (5, 5) differential N5 - N6 in ADC LSB."""
    if len(blocks) < N:
        raise ValueError(f"need {N} row blocks, got {len(blocks)}")
    a = np.array(blocks[:N], float)
    return a[:, :N] - a[:, N:]


def read_array(ard) -> np.ndarray:
    """One READ_ROW of the whole array -> (5, 5) differential."""
    return blocks_to_diff(send(ard, read_row_command(5)))


def hard_reset(ard, set_num: int = RESET_SET_NUM, timeout_s: float = 40.0,
               settle: float = 0.15):
    """Reset the array and drain exactly the blocks the firmware emits."""
    b = send(ard, reset_command(set_num), timeout_s=timeout_s,
             expect=reset_expected_blocks(set_num))
    time.sleep(settle)
    return b


def realized_update(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.asarray(after, float) - np.asarray(before, float)


__all__ = [n for n in dir() if not n.startswith("_") and n not in
           ("annotations", "contextlib", "datetime", "os", "re", "time", "np",
            "Iterable", "List", "Optional", "Sequence")]

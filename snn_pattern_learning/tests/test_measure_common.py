"""Pure-function tests for ``hardware.measure.common`` (no serial port)."""
import numpy as np
import pytest

from hardware.measure import common as C


def test_stochastic_command_format_matches_hw_interface():
    rows = [[1, 0] * 5] * 5
    cols = [[0, 1] * 5] * 5
    cmd = C.stochastic_command(rows, cols, "POTENTIATION", bit_length=10, no_read=True)
    f = cmd.split(",")
    assert f[:5] == ["F", "0", "0", "STOCHASTIC_POTENTIATION_NR", "N56"]
    assert f[5:12] == ["10", "15", "100", "100", "10", "20", "10"]
    assert f[12:17] == ["1010101010"] * 5 and f[17:22] == ["0101010101"] * 5
    assert "STOCHASTIC_DNO_DEPRESSION" in C.stochastic_command(rows, cols, "DEPRESSION", no_read=False, dno=True)


def test_bit_length_clamp_is_rejected():
    with pytest.raises(ValueError):
        C.stochastic_command([[1] * 40] * 5, [[1] * 40] * 5, bit_length=40)


def test_reset_block_count_is_31():
    assert C.reset_expected_blocks() == 31
    assert C.reset_command().split(",")[:8] == ["F", "5", "5", "Reset", "N56", "10", "30", "10"]


def test_read_row_and_hs_seq_commands():
    assert C.read_row_command(5).startswith("F,5,5,READ_ROW,N56,")
    assert C.read_row_command(2).startswith("F,2,5,READ_ROW,")
    with pytest.raises(ValueError):
        C.read_row_command(6)
    assert C.hs_seq_command(12, 300).split(",")[:7] == ["F", "12", "5", "HS_SEQ", "N56", "1", "300"]
    with pytest.raises(ValueError):
        C.hs_seq_command(15, 10)
    assert C.prog_nr_command("D", 12).split(",")[3:5] == ["PROG_NR", "D"]


def test_streams_and_coincidences():
    rng = np.random.default_rng(0)
    rows = C.bernoulli_streams([1, 1, 0, 0, 0], 10, rng)
    cols = C.bernoulli_streams([1, 0, 1, 0, 0], 10, rng)
    Cm = C.coincidence_matrix(rows, cols)
    assert Cm.shape == (5, 5)
    assert Cm[0, 0] == 10 and Cm[0, 2] == 10 and Cm[2, 0] == 0 and Cm[0, 1] == 0
    masked = C.bernoulli_streams([1] * 5, 10, rng, mask=[True, False, False, False, False])
    assert masked[1].sum() == 0


def test_quadrant_parts_matches_sign_rule():
    u = np.array([0.5, -0.5, 0, 0, 0]); v = np.array([0.2, 0, -0.3, 0, 0])
    parts = C.quadrant_parts(u, v)
    modes = [m for m, _, _ in parts]
    assert modes == ["POTENTIATION", "POTENTIATION", "DEPRESSION", "DEPRESSION"]
    # (+,+) -> row0/col0, (-,-) -> row1/col2, (+,-) -> row0/col2, (-,+) -> row1/col0
    assert parts[0][1][0] == 0.5 and parts[0][2][0] == 0.2
    assert parts[1][1][1] == 0.5 and parts[1][2][2] == 0.3


class FakeSerial:
    def __init__(self, lines):
        self._lines = [l.encode() for l in lines]
        self.written = []

    def readline(self):
        return self._lines.pop(0) if self._lines else b""

    def write(self, b):
        self.written.append(b)

    def reset_input_buffer(self):
        pass


def test_drain_parses_row_blocks_and_stops_at_eod():
    row = lambda i: f"{i}," + ",".join(str(100 + i * 10 + k) for k in range(5)) + "," + ",".join(str(50 + k) for k in range(5)) + ">"
    ard = FakeSerial([row(i) for i in range(5)] + ["EOD>", "garbage"])
    blocks = C.send(ard, C.read_row_command(), timeout_s=1.0, settle=0.0)
    assert len(blocks) == 5
    diff = C.blocks_to_diff(blocks)
    assert diff.shape == (5, 5)
    assert diff[0, 0] == 100 - 50 and diff[4, 4] == 144 - 54
    assert ard.written[0].endswith(b"\n")


def test_blocks_to_diff_requires_five_rows():
    with pytest.raises(ValueError):
        C.blocks_to_diff([[0] * 10] * 3)

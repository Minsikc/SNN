"""
Hardware interface for memristor crossbar array communication.

This module provides interfaces for communicating with a 5x5 memristor crossbar
array via Arduino serial communication for E-prop gradient accumulation.
"""

import serial
import time
import numpy as np
import re
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemristorInterface:
    """
    Interface for real memristor crossbar array hardware via Arduino.

    The hardware accumulates outer products of error and trace vectors using
    stochastic pulse-based updates over multiple timesteps.
    """

    def __init__(
        self,
        port: str = 'COM7',
        baud_rate: int = 115200,
        timeout: float = 10.0,
        bit_length: int = 10,
        pulse_width: int = 1,
        pulse_pre: int = 100,
        pulse_post: int = 100,
        pulse_zero: int = 10,
        read_time: int = 10,
        read_delay: int = 10,
        no_read_updates: bool = False,
        dno: bool = False
    ):
        """
        Initialize hardware interface.

        Args:
            port: Serial port name (e.g., 'COM7')
            baud_rate: Serial communication baud rate
            timeout: Serial read timeout in seconds
            bit_length: Length of stochastic pulse streams
            pulse_width: Width of each pulse in microseconds
            pulse_pre: Pre-pulse delay in microseconds
            pulse_post: Post-pulse delay in microseconds
            pulse_zero: Zero-pulse delay in microseconds
            no_read_updates: use the _NR firmware opcodes, which deliver the
                same pulses without the per-command pre/post read. Cuts the
                reads per epoch from ~21 to 2 (one reference read after
                reset, one when the accumulated gradient is read), so the
                stored gradient decays ~1.6% per epoch instead of ~15%.
                Requires firmware with STOCHASTIC_*_NR (2026-08-20).
            dno: use the DNO_ opcode variants. In DNO potentiation, a row
                whose pulse bit is 0 gets N3 asserted instead of being left
                idle, so every row is driven every slot -- N1 to potentiate
                or N3 to depress -- making the update push-pull rather than
                one-sided. Depression is the mirror image.
            read_time: ADC read time in microseconds
            read_delay: Delay between reads in microseconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout

        # Pulse timing parameters
        self.bit_length = bit_length
        self.no_read_updates = no_read_updates
        self.dno = dno
        self.pulse_timing = {
            'width': pulse_width,
            'pre': pulse_pre,
            'post': pulse_post,
            'zero': pulse_zero
        }
        self.read_timing = {
            'time': read_time,
            'delay': read_delay
        }

        # Hardware connection
        self.arduino: Optional[serial.Serial] = None
        self.connected = False

        # Accumulated gradient buffer (5x5)
        self.gradient_accumulator = np.zeros((5, 5), dtype=np.float32)

        # Reference point for differential measurement (5x5)
        self.reference_point = np.zeros((5, 5), dtype=np.float32)

    def connect(self) -> bool:
        """
        Connect to Arduino hardware.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.arduino = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for Arduino to initialize
            self.connected = True
            logger.info(f"[OK] Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Arduino hardware."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            self.connected = False
            logger.info("Arduino disconnected")

    def reset(self, hard_reset: bool = False) -> bool:
        """
        Reset accumulated gradient to zero and measure new reference point.

        Args:
            hard_reset: If True, send a physical Reset command to the device
                first to push all 25 cells back toward their baseline
                conductance, then measure the reference point afterwards.
                Use at epoch start to avoid cumulative saturation.

        Returns:
            True if reset successful
        """
        self.gradient_accumulator.fill(0)

        if self.connected:
            if hard_reset:
                self.hard_reset()
            self.reference_point = self._measure_reference_point()
            logger.info(
                "Gradient accumulator reset"
                + (" (with hard device reset)" if hard_reset else "")
                + " and reference point measured"
            )
        else:
            self.reference_point.fill(0)
            logger.info("Gradient accumulator reset to zero")

        return True

    def hard_reset(self, set_num: int = 10, silence_timeout: float = 1.5) -> bool:
        """
        Send a physical Reset command to the device. Mirrors the
        perform_reset() routine in stochastic_update.ipynb.

        Command format:
            F,5,5,Reset,N56,10,30,10,F,5,100,100,10,10,20,10
        Field meaning (from stochastic_update.ipynb):
            F,5,5      : full 5x5 array
            Reset      : reset opcode (no EOD is emitted by firmware)
            N56        : differential read (N5 - N6)
            10         : set_num (reset cycles, also expected '>' count + 1)
            30,10,F,5  : reset pulse timing parameters
            100,100,10 : pulse pre/post/zero (us)
            10,20,10   : ADC time / delay / extra delay

        Termination detection: the firmware Reset branch emits one
        '<idx>,<read>>' block at startup plus one block per
        (k+1) % read_period == 0 hit within each set -- i.e.
        update_num // read_period blocks per set, NOT one per set. With the
        command below (update_num=30, read_period=10) that is
        1 + set_num * 3 = 31 blocks, verified by counting lines on the wire
        (2026-08-05). The previous version waited for set_num + 1 = 11 and
        left 20 blocks in the serial buffer, corrupting the parse of the
        next command. There is still no EOD marker, so a silence-based
        fallback remains, and the buffer is drained before returning.

        Args:
            set_num: Expected number of reset cycles (sixth field of the
                command, default 10).
            silence_timeout: Seconds of inactivity after which we declare the
                reset finished even if the '>' count is below the budget.
        """
        if not self.connected or not self.arduino:
            logger.warning("[RESET] Hardware not connected, skipping hard reset")
            return False

        reset_update_num = 30
        reset_read_period = 10
        reset_params = [
            "F", "5", "5", "Reset", "N56",
            str(set_num), str(reset_update_num), str(reset_read_period), "F",
            "5", "100", "100", "10",
            "10", "20", "10",
        ]
        cmd = ",".join(reset_params) + "\n"

        logger.info(f"[RESET] Sending hard reset command: {cmd.strip()}")
        self.arduino.reset_input_buffer()
        self.arduino.write(cmd.encode("utf-8"))

        reads_per_set = max(1, reset_update_num // reset_read_period)
        expected_blocks = 1 + set_num * reads_per_set
        seen_blocks = 0
        last_data_time = time.time()

        while True:
            # readline() returns b'' when the per-call serial timeout elapses;
            # we use that as our silence signal rather than failing immediately
            raw = self.arduino.readline()
            if raw:
                line = raw.decode("utf-8", "ignore").strip()
                if line:
                    logger.debug(f"[RESET] {line}")
                    if ">" in line:
                        seen_blocks += 1
                        last_data_time = time.time()
                        if seen_blocks >= expected_blocks:
                            logger.info(
                                f"[RESET] Hard reset complete "
                                f"({seen_blocks}/{expected_blocks} blocks)"
                            )
                            return True
                    continue
            # No bytes this round — check silence threshold
            if time.time() - last_data_time > silence_timeout:
                # Drain any stragglers so they cannot bleed into the parse
                # of the next command.
                self.arduino.reset_input_buffer()
                if seen_blocks > 0:
                    logger.info(
                        f"[RESET] Hard reset complete by silence timeout "
                        f"({seen_blocks}/{expected_blocks} blocks)"
                    )
                    return True
                logger.error("[RESET] No response from device")
                return False

    def _measure_reference_point(self) -> np.ndarray:
        """
        Measure current array state as reference point.
        Uses zero pulse streams (no update) to read current state.

        Returns:
            Reference point matrix (N5 - N6), shape (5, 5)
        """
        if not self.connected or not self.arduino:
            logger.warning("Hardware not connected, returning zero reference")
            return np.zeros((5, 5), dtype=np.float32)

        # With _NR opcodes a zero-pulse command performs no read at all, so
        # it cannot serve as a measurement; READ_ROW is also half the read
        # cost (1 array read instead of the pre+post pair).
        if self.no_read_updates:
            ref = self.read_array()
            logger.info(f"[REF] Reference point measured (READ_ROW), "
                        f"mean={np.mean(ref):.2f}")
            return ref.astype(np.float32)

        # Zero pulse streams (no update, just read)
        zero_probs = np.zeros(5)

        try:
            # Send a POTENTIATION command with zero probabilities (effectively just a read)
            pre_data, _ = self._send_command("POTENTIATION", zero_probs, zero_probs)

            # Compute differential (N5 - N6)
            if pre_data.shape == (5, 10):
                ref_diff = pre_data[:, :5] - pre_data[:, 5:]
                logger.info(f"[REF] Reference point measured, mean={np.mean(ref_diff):.2f}")
                return ref_diff.astype(np.float32)
            else:
                logger.error(f"[REF] Invalid reference data shape: {pre_data.shape}")
                return np.zeros((5, 5), dtype=np.float32)

        except Exception as e:
            logger.error(f"[REF] Failed to measure reference point: {e}")
            return np.zeros((5, 5), dtype=np.float32)

    def _generate_pulse_stream(self, prob_value: float) -> str:
        """
        Generate stochastic pulse stream from probability value.

        Args:
            prob_value: Probability value in [0, 1]

        Returns:
            Binary string of length bit_length (e.g., "0110100101")
        """
        pulse_array = (np.random.rand(self.bit_length) < prob_value).astype(int)
        return "".join(map(str, pulse_array))

    def read_array(self) -> np.ndarray:
        """Read the array once with READ_ROW and return the (5,5) differential.

        One array read, versus the two that a STOCHASTIC command performs
        (pre + post). Used for the epoch-boundary measurements when
        no_read_updates is on.
        """
        if not self.connected or not self.arduino:
            raise RuntimeError("Arduino not connected")
        self.arduino.write(b"F,5,5,READ_ROW,N56,1,1,1,F,1,1,1,1,20,0,10\n")
        rows = []
        deadline = time.time() + 20.0
        while time.time() < deadline:
            line = self.arduino.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            if "EOD" in line:
                break
            nums = [int(n) for n in re.findall(r"-?\d+", line)]
            if len(nums) >= 11:
                rows.append(nums[-10:])
        if len(rows) < 5:
            logger.error(f"[HW] read_array got {len(rows)} rows, expected 5")
            return np.zeros((5, 5), dtype=np.float32)
        a = np.array(rows[:5], dtype=np.float32)
        return a[:, :5] - a[:, 5:]

    def _send_command(
        self,
        update_mode: str,
        u_probs: np.ndarray,
        v_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Send stochastic update command to Arduino and receive pre/post readings.

        Args:
            update_mode: "POTENTIATION" or "DEPRESSION"
            u_probs: Array of 5 probability values for first vector
            v_probs: Array of 5 probability values for second vector

        Returns:
            Tuple of (pre_update_matrix, post_update_matrix), each shape (5, 10)
            with columns [N5_ADC0...N5_ADC4, N6_ADC0...N6_ADC4]

        With no_read_updates set, the _NR firmware opcodes are used instead:
        identical pulse delivery, but the command performs no pre/post read
        and returns only EOD>. Verified on hardware 2026-08-20 -- 3 commands
        gave P +195.6 / D -190.5 LSB, against +187.2 / -199.4 for the
        read-wrapped opcodes, i.e. the same update within cell noise.

        Motivation: each read moves the cell ~0.79% of the way to the read
        attractor. Training issued ~10.5 update commands per epoch, so the
        built-in reads cost ~21 array reads and ~15% gradient decay per
        epoch -- and the host discarded every one of them, since it reads the
        accumulated gradient once after all updates are flushed. Both
        returned matrices are empty in this mode; callers must not use them.
        """
        if not self.connected or not self.arduino:
            raise RuntimeError("Arduino not connected")

        # Generate pulse streams
        u_streams = [self._generate_pulse_stream(p) for p in u_probs]
        v_streams = [self._generate_pulse_stream(p) for p in v_probs]
        _dno = "DNO_" if self.dno else ""

        if self.no_read_updates:
            command = ",".join([
                "F", "0", "0", f"STOCHASTIC_{_dno}{update_mode}_NR", "N56",
                str(self.bit_length),
                str(self.pulse_timing['width']), str(self.pulse_timing['pre']),
                str(self.pulse_timing['post']), str(self.pulse_timing['zero']),
                str(self.read_timing['time']), str(self.read_timing['delay']),
            ] + u_streams + v_streams)
            self.arduino.write((command + "\n").encode("utf-8"))
            deadline = time.time() + 30.0
            while time.time() < deadline:
                line = self.arduino.readline().decode("utf-8", "ignore").strip()
                if line and "EOD" in line:
                    break
            else:
                logger.warning(f"[HW] {update_mode}_NR: no EOD before timeout")
            return np.empty((0, 10)), np.empty((0, 10))

        # Assemble command
        # Format: F,0,0,STOCHASTIC_<MODE>,N56,bit_length,width,pre,post,zero,time,delay,<5 u_streams>,<5 v_streams>
        command_parts = [
            "F", "0", "0", f"STOCHASTIC_{_dno}{update_mode}",
            "N56",  # Read function (differential N5-N6)
            str(self.bit_length),
            str(self.pulse_timing['width']),
            str(self.pulse_timing['pre']),
            str(self.pulse_timing['post']),
            str(self.pulse_timing['zero']),
            str(self.read_timing['time']),
            str(self.read_timing['delay'])
        ] + u_streams + v_streams

        command = ",".join(command_parts)

        # Send command
        self.arduino.write((command + "\n").encode('utf-8'))
        logger.info(f"[HW] Sending {update_mode} command to Arduino...")
        logger.debug(f"Sent command: {command[:80]}...")

        # Receive pre-update and post-update data
        # STOCHASTIC mode uses "Row_X,..." format without PRE/POST markers
        pre_update_data = []
        post_update_data = []

        # Read all Row_X data until EOD
        all_rows = []
        while True:
            line = self.arduino.readline().decode('utf-8', 'ignore').strip()
            if not line:
                continue
            if "EOD" in line:
                break
            if "Row_" in line:
                # Extract data from "Row_0,val1,val2,...>" format
                numbers = [int(n) for n in re.findall(r'-?\d+', line)]
                if len(numbers) >= 11:  # Row index + 10 values
                    all_rows.append(numbers[1:11])  # Skip row index, take 10 values

        # Split into pre (first 5) and post (next 5) update data
        if len(all_rows) >= 10:
            pre_update_data = all_rows[:5]
            post_update_data = all_rows[5:10]
        elif len(all_rows) >= 5:
            # Only pre-update data available
            pre_update_data = all_rows[:5]
            post_update_data = [[0]*10 for _ in range(5)]
            logger.warning("[HW] Only received pre-update data")
        else:
            logger.error(f"[HW] Insufficient data received: {len(all_rows)} rows")

        logger.info(f"[HW] Received {update_mode} response from Arduino ({len(all_rows)} rows)")
        return np.array(pre_update_data), np.array(post_update_data)

    def accumulate_outer_product(
        self,
        err_probs: np.ndarray,
        trace_probs: np.ndarray,
        err_signs: np.ndarray,
        trace_signs: np.ndarray
    ):
        """
        Accumulate outer product gradient using hardware via 4-quadrant sign
        decomposition.

        Splits the outer product u ⊗ v (where u = err_probs * err_signs,
        v = trace_probs * trace_signs) into four sign-pure outer products and
        sends each as a separate Arduino command:

            (+,+) -> POTENTIATION(u+, v+)   produces  +u+v+ on (+,+) cells
            (-,-) -> POTENTIATION(u-, v-)   produces  +u-v- on (-,-) cells (positive)
            (+,-) -> DEPRESSION (u+, v-)    produces  -u+v- on (+,-) cells
            (-,+) -> DEPRESSION (u-, v+)    produces  -u-v+ on (-,+) cells

        Because each command's pulse streams are zero on the irrelevant rows /
        columns, only the intended cells receive coincident pulses, so no
        spurious conductance drift occurs on the other cells.

        Args:
            err_probs: |err| vector, shape (5,), values in [0, 1]
            trace_probs: |trace| vector, shape (5,), values in [0, 1]
            err_signs: err signs, shape (5,), values in {-1, +1}
            trace_signs: trace signs, shape (5,), values in {-1, +1}
        """
        if not self.connected:
            logger.warning("Hardware not connected, skipping update")
            return

        logger.info("[HW] Accumulating outer product on hardware (4-quadrant)...")

        for mode, u, v in self._quadrant_parts(err_probs, trace_probs,
                                               err_signs, trace_signs):
            self._apply_quadrant(mode, u, v)

    @staticmethod
    def _quadrant_parts(err_probs, trace_probs, err_signs, trace_signs):
        """Decompose a signed outer product into 4 sign-pure (mode, u, v)
        commands. Quadrants whose u or v is all-zero are omitted."""
        err_pos = err_probs * (err_signs > 0).astype(err_probs.dtype)
        err_neg = err_probs * (err_signs < 0).astype(err_probs.dtype)
        trc_pos = trace_probs * (trace_signs > 0).astype(trace_probs.dtype)
        trc_neg = trace_probs * (trace_signs < 0).astype(trace_probs.dtype)
        parts = [
            ("POTENTIATION", err_pos, trc_pos),   # (+,+)
            ("POTENTIATION", err_neg, trc_neg),   # (-,-) -> positive product
            ("DEPRESSION", err_pos, trc_neg),     # (+,-)
            ("DEPRESSION", err_neg, trc_pos),     # (-,+)
        ]
        return [(m, u, v) for m, u, v in parts if np.any(u) and np.any(v)]

    def _apply_quadrant(self, mode: str, u: np.ndarray, v: np.ndarray):
        """Send one sign-pure outer-product command and fold the measured
        conductance delta into the software-side accumulator."""
        pre, post = self._send_command(mode, u, v)
        if self.no_read_updates:
            # _NR commands perform no read, so there is no per-command delta
            # to fold in. gradient_accumulator is only a software mirror used
            # when the hardware is absent; the value that actually drives
            # training comes from read_accumulated_gradient() at the epoch
            # boundary, which measures the array directly.
            return
        pre_diff = pre[:, :5] - pre[:, 5:]
        post_diff = post[:, :5] - post[:, 5:]
        delta = post_diff - pre_diff
        if mode == "POTENTIATION":
            self.gradient_accumulator += delta
        else:
            self.gradient_accumulator -= delta

    def accumulate_outer_products_grouped(self, items):
        """Accumulate a batch of outer products, reordered quadrant-major.

        Motivation (measured 2026-08-10, temporal-XOR runs 1-2): when a
        column's desired update has mixed signs across rows, the per-timestep
        4-quadrant scheme alternates POTENTIATION and DEPRESSION commands on
        that column hundreds of times per epoch. Alternating select-line
        exposure programs half-selected cells (see half-select notes), and
        the P/D contributions cancel: the column's net accumulated gradient
        read ~10-20x smaller than desired while single-sign columns were
        faithful. Sending all commands of one quadrant back-to-back reduces
        the number of P<->D alternations per epoch from O(timesteps) to 3.
        The command multiset (and thus the ideal sum) is unchanged.

        Args:
            items: list of (err_probs, trace_probs, err_signs, trace_signs)
        """
        if not self.connected:
            logger.warning("Hardware not connected, skipping grouped update")
            return
        grouped = [[], [], [], []]
        order = {("POTENTIATION", 0): 0, ("POTENTIATION", 1): 1,
                 ("DEPRESSION", 0): 2, ("DEPRESSION", 1): 3}
        for it in items:
            for mode, u, v in self._quadrant_parts(*it):
                # index quadrants by (mode, err-sign-negative?)
                neg = int(np.any(u * (it[2] < 0)))
                grouped[order[(mode, neg)]].append((mode, u, v))
        n = sum(len(g) for g in grouped)
        logger.info(f"[HW] Grouped flush: {n} commands in 4 quadrant blocks "
                    f"({[len(g) for g in grouped]})")
        for g in grouped:
            for mode, u, v in g:
                self._apply_quadrant(mode, u, v)

    def read_accumulated_gradient(self) -> np.ndarray:
        """
        Read the accumulated gradient from hardware and compute differential.

        The actual gradient is the difference between current state and reference point:
        gradient = (current_state - reference_point)

        Returns:
            Differential gradient matrix, shape (5, 5)
        """
        if not self.connected or not self.arduino:
            logger.warning("Hardware not connected, returning software gradient")
            return self.gradient_accumulator.copy()

        # With _NR opcodes a zero-pulse command performs no read, so measure
        # with READ_ROW (also 1 array read rather than the pre+post pair).
        if self.no_read_updates:
            gradient = self.read_array() - self.reference_point
            logger.info(f"[GRAD] Read gradient (READ_ROW), "
                        f"mean={np.mean(gradient):.2f}, std={np.std(gradient):.2f}")
            return gradient.astype(np.float32)

        # Measure current state using zero pulse streams
        try:
            zero_probs = np.zeros(5)
            current_data, _ = self._send_command("POTENTIATION", zero_probs, zero_probs)

            # Compute current differential (N5 - N6)
            if current_data.shape == (5, 10):
                current_diff = current_data[:, :5] - current_data[:, 5:]

                # Compute gradient = current - reference
                gradient = current_diff - self.reference_point

                logger.info(f"[GRAD] Read gradient, mean={np.mean(gradient):.2f}, std={np.std(gradient):.2f}")
                return gradient.astype(np.float32)
            else:
                logger.error(f"[GRAD] Invalid current data shape: {current_data.shape}")
                return self.gradient_accumulator.copy()

        except Exception as e:
            logger.error(f"[GRAD] Failed to read gradient: {e}")
            return self.gradient_accumulator.copy()

    def send_outer_product_update(
        self,
        u_probs: np.ndarray,
        v_probs: np.ndarray,
        direction: str = 'POTENTIATION'
    ):
        """
        Simplified interface for single update (used by models.py).

        Args:
            u_probs: First vector probabilities, shape (5,)
            v_probs: Second vector probabilities, shape (5,)
            direction: 'POTENTIATION' or 'DEPRESSION'
        """
        # All probabilities assumed positive, direction determines the sign
        u_signs = np.ones(5)
        v_signs = np.ones(5)

        if direction == 'DEPRESSION':
            # For depression, flip the sign
            u_signs = -np.ones(5)

        # Call the main accumulation function
        self.accumulate_outer_product(u_probs, v_probs, u_signs, v_signs)


class MockMemristorInterface:
    """
    Mock hardware interface for testing without real hardware.

    Simulates the behavior of the memristor crossbar array using
    software-based outer product computation.
    """

    def __init__(self, port: str = 'COM7', quantize_bits: int = 0,
                 quantize_seed: int = 0, **kwargs):
        """Initialize mock interface.

        Args:
            quantize_bits: if > 0, reproduce the stochastic pulse encoding
                instead of the exact real-valued outer product. Each
                probability is drawn as a Bernoulli bit stream of this length
                (the real device's `bit_length`) and the coincidence count is
                accumulated, so a coordinate is quantised to multiples of
                1/bit_length and any probability far below 1/bit_length
                rounds to zero most of the time. Everything else about the
                device -- nonlinearity, half-select, read disturbance,
                cell-to-cell spread -- is still absent, so this isolates the
                encoding from the physics.
            quantize_seed: seed for the Bernoulli draws.
        """
        self.port = port
        self.connected = False
        self.gradient_accumulator = np.zeros((5, 5), dtype=np.float32)
        self.reference_point = np.zeros((5, 5), dtype=np.float32)
        self.quantize_bits = int(quantize_bits)
        self._rng = np.random.default_rng(quantize_seed)
        logger.info(
            f"[MOCK] Initialized MOCK memristor interface (no hardware)"
            + (f", stochastic encoding BL={self.quantize_bits}"
               if self.quantize_bits > 0 else ""))

    def connect(self) -> bool:
        """Simulate connection."""
        self.connected = True
        logger.info(f"[MOCK] Connected to simulated hardware on {self.port}")
        return True

    def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        logger.info("[MOCK] Disconnected")

    def reset(self, hard_reset: bool = False) -> bool:
        """Reset accumulated gradient and reference point.

        For the mock interface a hard reset is equivalent to clearing the
        accumulator and reference (i.e. simulating a freshly-baselined device).
        """
        if hard_reset:
            # Simulate hardware fully returning to baseline
            self.reference_point.fill(0)
        else:
            # Save current state as reference (drift carries over)
            self.reference_point = self.gradient_accumulator.copy()
        self.gradient_accumulator.fill(0)
        logger.info(
            "[MOCK] Gradient accumulator reset"
            + (" (hard reset)" if hard_reset else "")
        )
        return True

    def hard_reset(self) -> bool:
        """No-op for mock; provided for API parity with MemristorInterface."""
        self.reference_point.fill(0)
        logger.info("[MOCK] Hard reset (simulated)")
        return True

    def accumulate_outer_product(
        self,
        err_probs: np.ndarray,
        trace_probs: np.ndarray,
        err_signs: np.ndarray,
        trace_signs: np.ndarray
    ):
        """
        Simulate the 4-quadrant sign-decomposed outer product accumulation in
        software. Mathematically equivalent to (err_probs*err_signs) outer
        (trace_probs*trace_signs), but implemented with the same 4 sub-outer-
        products that the real hardware path uses.
        """
        err_pos = err_probs * (err_signs > 0).astype(err_probs.dtype)
        err_neg = err_probs * (err_signs < 0).astype(err_probs.dtype)
        trc_pos = trace_probs * (trace_signs > 0).astype(trace_probs.dtype)
        trc_neg = trace_probs * (trace_signs < 0).astype(trace_probs.dtype)

        if self.quantize_bits > 0:
            # Reproduce the device's stochastic encoding: each probability
            # becomes a Bernoulli bit stream and the cell accumulates the
            # COINCIDENCE COUNT, normalised back by bit_length so the result
            # stays on the same scale as the exact product. This is the same
            # arithmetic as MemristorInterface, minus the physics.
            L = self.quantize_bits

            def coincide(u, v):
                bu = self._rng.random((len(u), L)) < np.asarray(u)[:, None]
                bv = self._rng.random((len(v), L)) < np.asarray(v)[:, None]
                return (bu.astype(np.float32) @ bv.astype(np.float32).T) / L

            self.gradient_accumulator += coincide(err_pos, trc_pos)
            self.gradient_accumulator += coincide(err_neg, trc_neg)
            self.gradient_accumulator -= coincide(err_pos, trc_neg)
            self.gradient_accumulator -= coincide(err_neg, trc_pos)
        else:
            # POTENTIATION (positive contributions)
            self.gradient_accumulator += np.outer(err_pos, trc_pos)
            self.gradient_accumulator += np.outer(err_neg, trc_neg)

            # DEPRESSION (negative contributions)
            self.gradient_accumulator -= np.outer(err_pos, trc_neg)
            self.gradient_accumulator -= np.outer(err_neg, trc_pos)

        logger.debug(
            f"[MOCK] Accumulated outer product, "
            f"max grad = {np.max(np.abs(self.gradient_accumulator)):.4f}"
        )

    def accumulate_outer_products_grouped(self, items):
        """API parity with MemristorInterface: command order is irrelevant
        for the exact software outer product, so just accumulate each item."""
        for err_probs, trace_probs, err_signs, trace_signs in items:
            self.accumulate_outer_product(err_probs, trace_probs,
                                          err_signs, trace_signs)

    def read_accumulated_gradient(self) -> np.ndarray:
        """
        Read accumulated gradient.

        For mock interface, this returns the software-accumulated gradient
        (which already represents the differential from the reset point).
        """
        logger.info(f"[MOCK] Read gradient, mean={np.mean(self.gradient_accumulator):.2f}")
        return self.gradient_accumulator.copy()

    def send_outer_product_update(
        self,
        u_probs: np.ndarray,
        v_probs: np.ndarray,
        direction: str = 'POTENTIATION'
    ):
        """
        Simplified interface for single update (used by models.py).

        Args:
            u_probs: First vector probabilities, shape (5,)
            v_probs: Second vector probabilities, shape (5,)
            direction: 'POTENTIATION' or 'DEPRESSION'
        """
        u_signs = np.ones(5)
        v_signs = np.ones(5)

        if direction == 'DEPRESSION':
            u_signs = -np.ones(5)

        self.accumulate_outer_product(u_probs, v_probs, u_signs, v_signs)

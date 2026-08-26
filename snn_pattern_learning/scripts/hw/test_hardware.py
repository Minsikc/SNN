#!/usr/bin/env python3
"""
Test script for hardware interface.

This script tests the memristor interface without running a full experiment.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
from hardware import MemristorInterface, MockMemristorInterface


def test_mock_interface():
    """Test mock hardware interface."""
    print("="*60)
    print("Testing MOCK Hardware Interface")
    print("="*60)

    # Create mock interface
    hw = MockMemristorInterface(port='COM7')

    # Connect
    if not hw.connect():
        print("❌ Failed to connect to mock hardware")
        return False

    # Reset
    hw.reset()

    # Test accumulation
    print("\n--- Test 1: Simple outer product accumulation ---")
    err_probs = np.array([0.5, 0.3, 0.7, 0.2, 0.9])
    trace_probs = np.array([0.4, 0.6, 0.1, 0.8, 0.5])
    err_signs = np.array([1, -1, 1, -1, 1])
    trace_signs = np.array([1, 1, -1, 1, -1])

    hw.accumulate_outer_product(err_probs, trace_probs, err_signs, trace_signs)

    gradient = hw.read_accumulated_gradient()
    print(f"Accumulated gradient:\n{gradient}")
    print(f"Max gradient: {np.max(np.abs(gradient)):.4f}")

    # Test multiple accumulations
    print("\n--- Test 2: Multiple accumulations ---")
    for i in range(5):
        err_probs = np.random.rand(5)
        trace_probs = np.random.rand(5)
        err_signs = np.random.choice([-1, 1], size=5)
        trace_signs = np.random.choice([-1, 1], size=5)

        hw.accumulate_outer_product(err_probs, trace_probs, err_signs, trace_signs)

    gradient = hw.read_accumulated_gradient()
    print(f"Final accumulated gradient after 5 updates:\n{gradient}")
    print(f"Max gradient: {np.max(np.abs(gradient)):.4f}")

    # Disconnect
    hw.disconnect()

    print("\n[OK] Mock interface test completed successfully!")
    return True


def test_real_interface():
    """Test real hardware interface."""
    print("\n" + "="*60)
    print("Testing REAL Hardware Interface")
    print("="*60)

    # Create real interface
    hw = MemristorInterface(
        port='COM7',
        baud_rate=115200,
        timeout=10.0,
        bit_length=10
    )

    # Connect
    print("\nAttempting to connect to Arduino on COM7...")
    if not hw.connect():
        print("[ERROR] Failed to connect to real hardware")
        print("[INFO] Make sure Arduino is connected to COM7")
        return False

    # Reset
    hw.reset()

    # Test single update
    print("\n--- Test: Single outer product update ---")
    err_probs = np.array([0.5, 0.3, 0.7, 0.2, 0.9])
    trace_probs = np.array([0.4, 0.6, 0.1, 0.8, 0.5])
    err_signs = np.array([1, 1, 1, 1, 1])
    trace_signs = np.array([1, 1, 1, 1, 1])

    print("Sending update to hardware...")
    hw.accumulate_outer_product(err_probs, trace_probs, err_signs, trace_signs)

    gradient = hw.read_accumulated_gradient()
    print(f"Accumulated gradient:\n{gradient}")
    print(f"Max gradient: {np.max(np.abs(gradient)):.4f}")

    # Disconnect
    hw.disconnect()

    print("\n[OK] Real hardware test completed successfully!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test hardware interface')
    parser.add_argument('--mock', action='store_true',
                       help='Test mock interface only')
    parser.add_argument('--real', action='store_true',
                       help='Test real hardware interface')

    args = parser.parse_args()

    if args.real:
        # Test real hardware only
        test_real_interface()
    elif args.mock:
        # Test mock only
        test_mock_interface()
    else:
        # Test both by default
        test_mock_interface()

        print("\n\n")
        response = input("Do you want to test REAL hardware? (y/N): ")
        if response.lower() == 'y':
            test_real_interface()
        else:
            print("\nSkipping real hardware test.")

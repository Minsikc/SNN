#!/usr/bin/env python3
"""
Demo script to show different experiment types
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run demo experiments"""
    print("SNN Experiment System Demo")
    print("="*60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Demo 1: List available configurations
    run_command("python3 main_unified.py --list_configs", 
                "List available configurations")
    
    # Demo 2: Basic experiment with default config
    run_command("python3 main_unified.py --config default.yaml --epochs 5 --verbose", 
                "Basic experiment (5 epochs)")
    
    # Demo 3: Teacher-student experiment
    run_command("python3 main_unified.py --config teacher_student.yaml --epochs 3 --verbose", 
                "Teacher-student experiment (3 epochs)")
    
    # Demo 4: Weight initialization experiment
    run_command("python3 main_unified.py --config weight_init.yaml --epochs 3 --verbose", 
                "Weight initialization experiment (3 epochs)")
    
    # Demo 5: Basic experiment with boxcar neurons
    run_command("python3 main_unified.py --config default.yaml --neuron_type boxcar --epochs 5 --verbose", 
                "Basic experiment with boxcar neurons (5 epochs)")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("Check the results/ directory for experiment outputs.")
    print("="*60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Demo script for running the statistical ablation study
"""

import subprocess
import sys
import os

def run_ablation_demo():
    """Run the ablation study demo"""
    print("🔬 SNN Pattern Learning - Statistical Ablation Study Demo")
    print("=" * 60)
    print()
    
    print("This demo will run a statistical ablation study for Basic_RSNN_spike model")
    print("tracking the best normalized distance across different parameter combinations.")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("main_unified.py"):
        print("❌ Error: Please run this script from the snn_pattern_learning directory")
        sys.exit(1)
    
    print("🎯 Running ablation study with test configuration...")
    print("   - 2 learning rates: [0.05, 0.1]")
    print("   - 2 hidden sizes: [10, 20]") 
    print("   - 2 neuron thresholds: [0.5, 0.7]")
    print("   - 2 repetitions per combination")
    print("   - Total: 8 parameter combinations")
    print()
    
    try:
        # Run the ablation study
        result = subprocess.run([
            sys.executable, "main_unified.py", 
            "--config", "ablation_test.yaml", 
            "--verbose"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ablation study completed successfully!")
            print()
            print("📊 Results saved to:")
            print("   - ablation_results/ablation_study_results.csv")
            print("   - ablation_results/ablation_lr_vs_hidden.png")
            print("   - ablation_results/ablation_lr_vs_thresh.png") 
            print("   - ablation_results/ablation_results_with_std.png")
            print()
            
            # Extract and display summary from output
            output_lines = result.stdout.split('\n')
            summary_start = False
            for line in output_lines:
                if "ABLATION STUDY SUMMARY" in line:
                    summary_start = True
                if summary_start and line.strip():
                    print(line)
                    
        else:
            print("❌ Error running ablation study:")
            print(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_full_ablation():
    """Run the full ablation study with more comprehensive parameters"""
    print()
    print("🔬 SNN Pattern Learning - Full Statistical Ablation Study")
    print("=" * 60)
    print()
    
    print("This will run a comprehensive ablation study with:")
    print("   - 4 learning rates: [0.01, 0.05, 0.1, 0.2]")
    print("   - 4 hidden sizes: [20, 40, 80, 120]")
    print("   - 3 neuron thresholds: [0.4, 0.6, 0.8]")
    print("   - 5 repetitions per combination")
    print("   - Total: 48 parameter combinations")
    print("   - Estimated time: 30-60 minutes")
    print()
    
    response = input("Do you want to proceed with the full study? (y/N): ")
    if response.lower() != 'y':
        print("Full study cancelled.")
        return
    
    try:
        # Run the full ablation study
        result = subprocess.run([
            sys.executable, "main_unified.py", 
            "--config", "ablation_basic_rsnn.yaml", 
            "--verbose"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Full ablation study completed successfully!")
            print()
            print("📊 Results saved to ablation_results/")
        else:
            print("❌ Error running full ablation study:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_ablation()
    else:
        run_ablation_demo()
        
        print()
        print("💡 To run the full comprehensive ablation study, use:")
        print("   python3 run_ablation_demo.py --full")
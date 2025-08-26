#!/usr/bin/env python3
"""
Unified main script for all SNN experiments
Supports basic, teacher-student, and weight initialization experiments
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_loader import load_config, get_available_configs
from experiment_types import BasicExperiment, TeacherStudentExperiment, WeightInitExperiment, StatisticalAblationExperiment


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified SNN Experiment Runner')
    
    parser.add_argument('--config', type=str, default='default.yaml',
                       help='Configuration file name (default: default.yaml)')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['basic', 'teacher_student', 'weight_init', 'statistical_ablation'],
                       help='Override experiment type from config')
    parser.add_argument('--name', type=str, help='Override experiment name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--model_type', type=str, help='Override model type')
    parser.add_argument('--neuron_type', type=str, choices=['triangular', 'boxcar'],
                       help='Override neuron type')
    parser.add_argument('--list_configs', action='store_true',
                       help='List available configuration files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def apply_cli_overrides(config, args):
    """Apply command line argument overrides to configuration"""
    if args.experiment_type:
        config.set('experiment.type', args.experiment_type)
    if args.name:
        config.set('experiment.name', args.name)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    if args.model_type:
        config.set('model.type', args.model_type)
    if args.neuron_type:
        config.set('model.neuron_type', args.neuron_type)
    
    return config


def create_experiment(config):
    """Create experiment instance based on configuration"""
    experiment_type = config.get('experiment.type', 'basic')
    
    if experiment_type == 'basic':
        return BasicExperiment(config)
    elif experiment_type == 'teacher_student':
        return TeacherStudentExperiment(config)
    elif experiment_type == 'weight_init':
        return WeightInitExperiment(config)
    elif experiment_type == 'statistical_ablation':
        return StatisticalAblationExperiment(config)
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")


def print_config_summary(config):
    """Print a summary of the configuration"""
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("="*50)
    
    print(f"Experiment Type: {config.get('experiment.type', 'basic')}")
    print(f"Experiment Name: {config.get('experiment.name', 'default')}")
    print(f"Model Type: {config.get('model.type', 'unknown')}")
    print(f"Neuron Type: {config.get('model.neuron_type', 'triangular')}")
    print(f"Input Size: {config.get('model.n_in', 50)}")
    print(f"Hidden Size: {config.get('model.n_hidden', 40)}")
    print(f"Output Size: {config.get('model.n_out', 10)}")
    print(f"Dataset Type: {config.get('dataset.type', 'unknown')}")
    print(f"Batch Size: {config.get('training.batch_size', 1)}")
    print(f"Learning Rate: {config.get('training.learning_rate', 0.1)}")
    print(f"Epochs: {config.get('training.epochs', 200)}")
    print(f"Results Directory: {config.get('experiment.results_dir', 'results')}")
    print("="*50)


def main():
    """Main function"""
    args = parse_arguments()
    
    # List available configs if requested
    if args.list_configs:
        print("Available configuration files:")
        configs = get_available_configs()
        for config_file in configs:
            print(f"  - {config_file}")
        return
    
    try:
        # Load configuration
        if args.verbose:
            print(f"Loading configuration from: {args.config}")
        
        config = load_config(args.config)
        
        # Apply CLI overrides
        config = apply_cli_overrides(config, args)
        
        # Print configuration summary
        if args.verbose:
            print_config_summary(config)
        
        # Create and run experiment
        experiment = create_experiment(config)
        
        print(f"\nStarting {config.get('experiment.type', 'basic')} experiment...")
        results = experiment.run()
        
        print("\nExperiment completed successfully!")
        print(f"Results: {results}")
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        print("Available configs:")
        for config_file in get_available_configs():
            print(f"  - {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
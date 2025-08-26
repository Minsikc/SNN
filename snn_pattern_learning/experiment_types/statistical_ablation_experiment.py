import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import copy
import os
import itertools
from typing import Dict, List, Any, Tuple

from experiment_types.base_experiment import BaseExperiment
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance


class StatisticalAblationExperiment(BaseExperiment):
    """Statistical ablation study experiment for tracking best normalized distances"""
    
    def __init__(self, config):
        super().__init__(config)
        self.ablation_results = []
        self.parameter_combinations = []
        
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for ablation study"""
        ablation_config = self.config.get('ablation', {})
        
        # Extract parameter ranges
        learning_rates = ablation_config.get('learning_rates', [0.1])
        hidden_sizes = ablation_config.get('hidden_sizes', [40])
        neuron_thresholds = ablation_config.get('neuron_thresholds', [0.6])
        
        # Generate all combinations
        combinations = []
        for lr, hidden, thresh in itertools.product(learning_rates, hidden_sizes, neuron_thresholds):
            combinations.append({
                'learning_rate': lr,
                'n_hidden': hidden,
                'neuron_thresh': thresh
            })
        
        return combinations
    
    def run_single_experiment(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run a single experiment with given parameters and track best normalized distance"""
        # Create a new config with updated parameters
        from configs.config_loader import ExperimentConfig
        temp_config_dict = copy.deepcopy(self.config.to_dict())
        temp_config = ExperimentConfig(temp_config_dict)
        
        # Update config with current parameters
        temp_config.set('training.learning_rate', params['learning_rate'])
        temp_config.set('model.n_hidden', params['n_hidden'])
        neuron_type = temp_config.get("model.neuron_type", "triangular")
        temp_config.set(f'neuron.{neuron_type}.thresh', params['neuron_thresh'])
        
        # Create dataset and dataloader
        dataset = self.create_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=temp_config.get('training.batch_size', 1),
            shuffle=True
        )
        
        # Create model with updated config
        model_config = {
            'n_in': temp_config.get('model.n_in'),
            'n_hidden': params['n_hidden'],
            'n_out': temp_config.get('model.n_out'),
            'recurrent': temp_config.get('model.recurrent', False)
        }
        
        neuron_type = temp_config.get('model.neuron_type', 'triangular')
        neuron_config = temp_config.get(f'neuron.{neuron_type}', {})
        neuron_config['thresh'] = params['neuron_thresh']
        
        from models.model_factory import create_configurable_model
        model = create_configurable_model(
            model_type=temp_config.get('model.type'),
            model_config=model_config,
            neuron_type=neuron_type,
            neuron_config=neuron_config
        ).to(self.device)
        
        # Create optimizer with updated learning rate
        optimizer_type = temp_config.get('training.optimizer', 'Adam')
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
        
        loss_fn = mse_acc_loss_over_time
        
        # Create kernel
        kernel = self.create_kernel()
        kernel_size = temp_config.get('kernel.size', 5)
        
        # Training loop - track best normalized distance
        epochs = temp_config.get('training.epochs', 200)
        best_normalized_distance = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            train_loss, outputs, targets, inputs = self.train_epoch(
                model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )
            
            val_metric, val_metric_per_spike, normalized_distance = self.evaluate(
                model, dataloader, van_rossum_distance, kernel, kernel_size
            )
            
            # Track best normalized distance
            if normalized_distance < best_normalized_distance:
                best_normalized_distance = normalized_distance
                best_epoch = epoch + 1
        
        return {
            'best_normalized_distance': float(best_normalized_distance),
            'best_epoch': best_epoch,
            'final_loss': float(train_loss)
        }
    
    def run_statistical_analysis(self, param_combo: Dict[str, Any], num_repetitions: int) -> Dict[str, Any]:
        """Run multiple repetitions for statistical analysis"""
        results = []
        
        print(f"Running {num_repetitions} repetitions for parameters: {param_combo}")
        
        for rep in range(num_repetitions):
            print(f"  Repetition {rep + 1}/{num_repetitions}")
            result = self.run_single_experiment(param_combo)
            results.append(result)
        
        # Calculate statistics
        distances = [r['best_normalized_distance'] for r in results]
        epochs = [r['best_epoch'] for r in results]
        
        stats = {
            'learning_rate': param_combo['learning_rate'],
            'n_hidden': param_combo['n_hidden'],
            'neuron_thresh': param_combo['neuron_thresh'],
            'mean_best_distance': np.mean(distances),
            'std_best_distance': np.std(distances),
            'min_best_distance': np.min(distances),
            'max_best_distance': np.max(distances),
            'mean_best_epoch': np.mean(epochs),
            'std_best_epoch': np.std(epochs),
            'num_repetitions': num_repetitions
        }
        
        # Add 95% confidence interval
        if len(distances) > 1:
            confidence_interval = 1.96 * stats['std_best_distance'] / np.sqrt(len(distances))
            stats['ci_95_lower'] = stats['mean_best_distance'] - confidence_interval
            stats['ci_95_upper'] = stats['mean_best_distance'] + confidence_interval
        else:
            stats['ci_95_lower'] = stats['mean_best_distance']
            stats['ci_95_upper'] = stats['mean_best_distance']
        
        return stats
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: str = None):
        """Save ablation results to CSV file"""
        if filename is None:
            results_dir = self.config.get('experiment.results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, 'ablation_study_results.csv')
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return filename
    
    def create_visualization(self, results: List[Dict[str, Any]]):
        """Create heatmap visualization of parameter effects"""
        df = pd.DataFrame(results)
        
        # Create pivot tables for different parameter combinations
        results_dir = self.config.get('experiment.results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Heatmap: Learning Rate vs Hidden Size (averaged over thresholds)
        if len(df['learning_rate'].unique()) > 1 and len(df['n_hidden'].unique()) > 1:
            plt.figure(figsize=(10, 8))
            
            pivot_lr_hidden = df.groupby(['learning_rate', 'n_hidden'])['mean_best_distance'].mean().unstack()
            sns.heatmap(pivot_lr_hidden, annot=True, fmt='.4f', cmap='viridis_r')
            plt.title('Mean Best Normalized Distance\n(Learning Rate vs Hidden Size)')
            plt.ylabel('Learning Rate')
            plt.xlabel('Hidden Size')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'ablation_lr_vs_hidden.png'))
            plt.close()
        
        # Heatmap: Learning Rate vs Neuron Threshold (averaged over hidden sizes)
        if len(df['learning_rate'].unique()) > 1 and len(df['neuron_thresh'].unique()) > 1:
            plt.figure(figsize=(10, 8))
            
            pivot_lr_thresh = df.groupby(['learning_rate', 'neuron_thresh'])['mean_best_distance'].mean().unstack()
            sns.heatmap(pivot_lr_thresh, annot=True, fmt='.4f', cmap='viridis_r')
            plt.title('Mean Best Normalized Distance\n(Learning Rate vs Neuron Threshold)')
            plt.ylabel('Learning Rate')
            plt.xlabel('Neuron Threshold')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'ablation_lr_vs_thresh.png'))
            plt.close()
        
        # Bar plot with error bars
        plt.figure(figsize=(12, 6))
        
        # Create x-axis labels from parameter combinations
        df['param_combo'] = df.apply(lambda row: f"LR:{row['learning_rate']}\nH:{row['n_hidden']}\nT:{row['neuron_thresh']}", axis=1)
        
        plt.errorbar(range(len(df)), df['mean_best_distance'], yerr=df['std_best_distance'], 
                    fmt='o-', capsize=5, capthick=2)
        plt.xticks(range(len(df)), df['param_combo'], rotation=45, ha='right')
        plt.ylabel('Best Normalized Distance')
        plt.title('Ablation Study Results with Standard Deviation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ablation_results_with_std.png'))
        plt.close()
        
        print(f"Visualizations saved to {results_dir}")
    
    def run(self):
        """Run the complete ablation study"""
        print(f"Starting statistical ablation study: {self.config.get('experiment.name', 'ablation_study')}")
        
        # Generate parameter combinations
        self.parameter_combinations = self.generate_parameter_combinations()
        print(f"Generated {len(self.parameter_combinations)} parameter combinations")
        
        # Get number of repetitions
        num_repetitions = self.config.get('ablation.num_repetitions', 3)
        
        # Run ablation study
        all_results = []
        
        for i, param_combo in enumerate(self.parameter_combinations):
            print(f"\nParameter combination {i+1}/{len(self.parameter_combinations)}")
            print(f"Parameters: {param_combo}")
            
            # Run statistical analysis for this parameter combination
            stats = self.run_statistical_analysis(param_combo, num_repetitions)
            all_results.append(stats)
            
            print(f"Results: Mean best distance = {stats['mean_best_distance']:.4f} ± {stats['std_best_distance']:.4f}")
        
        # Save results
        csv_file = self.save_results_to_csv(all_results)
        
        # Create visualizations
        self.create_visualization(all_results)
        
        # Store results
        self.ablation_results = all_results
        
        # Print summary
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        
        # Find best parameter combination
        best_result = min(all_results, key=lambda x: x['mean_best_distance'])
        print(f"Best parameter combination:")
        print(f"  Learning Rate: {best_result['learning_rate']}")
        print(f"  Hidden Size: {best_result['n_hidden']}")
        print(f"  Neuron Threshold: {best_result['neuron_thresh']}")
        print(f"  Mean Best Distance: {best_result['mean_best_distance']:.4f} ± {best_result['std_best_distance']:.4f}")
        print(f"  95% CI: [{best_result['ci_95_lower']:.4f}, {best_result['ci_95_upper']:.4f}]")
        
        print(f"\nResults saved to: {csv_file}")
        print("="*60)
        
        return {
            'all_results': all_results,
            'best_parameters': {
                'learning_rate': best_result['learning_rate'],
                'n_hidden': best_result['n_hidden'],
                'neuron_thresh': best_result['neuron_thresh']
            },
            'best_mean_distance': best_result['mean_best_distance'],
            'best_std_distance': best_result['std_best_distance']
        }
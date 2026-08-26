import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import copy
import os
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.model_factory import create_configurable_model
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance, max_van_rossum_distance
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from utils.plots import plot_raster, plot_comparison_raster
from datasets.customdatasets import (
    CustomSpikeDataset_random, CustomSpikeDataset_Teacher,
    SuperSpikePatternDataset, TemporalXORDataset,
    CustomSpikeDataset_Probabilistic, NMNISTDataset, NMNISTDatasetRateEncoded,
    NMNISTDatasetClassification
)


class BaseExperiment(ABC):
    """Base class for all experiments"""

    def __init__(self, config):
        self.config = config
        # Force CPU usage to avoid aihwkit CUDA issues
        self.device = torch.device("cpu")
        self.best_loss = float("inf")
        self.best_model_state = None
        self.results = {}

        # WandB integration
        self.use_wandb = self.config.get('wandb.enabled', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB logging"""
        wandb.init(
            project=self.config.get('wandb.project', 'snn-experiments'),
            entity=self.config.get('wandb.entity', None),
            name=self.config.get('experiment.name', None),
            config=self.config.to_dict(),
            tags=self.config.get('wandb.tags', [])
        )

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to WandB if enabled"""
        if self.use_wandb:
            if step is not None:
                metrics['step'] = step
            wandb.log(metrics)

    def finish_wandb(self):
        """Finish WandB run"""
        if self.use_wandb:
            wandb.finish()
        
    def create_model(self):
        """Create model based on configuration"""
        model_type = self.config.get('model.type')
        model_config = {
            'n_in': self.config.get('model.n_in'),
            'n_hidden': self.config.get('model.n_hidden'),
            'n_out': self.config.get('model.n_out'),
            'recurrent': self.config.get('model.recurrent', False),
            'init_tau': self.config.get('model.init_tau', 0.6),
            # init_thresh feeds BOTH the student model (firing threshold,
            # after the alignment fix) and the teacher dataset -- keeping the
            # task realizable. It was previously missing here, so the model
            # ran at its constructor default regardless of the yaml.
            'init_thresh': self.config.get('model.init_thresh', 0.5),
            'init_tau_o': self.config.get('model.init_tau_o', 0.6),
            'gamma': self.config.get('model.gamma', 0.3),
            'width': self.config.get('model.width', 1),
            'weight_scale': self.config.get('model.weight_scale', 0.5),
            'hardware': self.config.get('model.hardware', {})  # Include hardware config
        }

        # Get neuron configuration
        neuron_type = self.config.get('model.neuron_type', 'triangular')
        neuron_config = self.config.get(f'neuron.{neuron_type}', {})

        # Create model using factory
        model = create_configurable_model(
            model_type=model_type,
            model_config=model_config,
            neuron_type=neuron_type,
            neuron_config=neuron_config
        )

        return model.to(self.device)
    
    
    def create_dataset(self, train=None):
        """Create dataset based on configuration

        Args:
            train: Override train/test split. If None, uses config value.
        """
        dataset_type = self.config.get('dataset.type')

        # Use provided train parameter or fall back to config
        is_train = train if train is not None else self.config.get('dataset.train', True)

        if dataset_type == "CustomSpikeDataset_random":
            return CustomSpikeDataset_random(
                num_samples=self.config.get('dataset.num_samples', 1),
                sequence_length=self.config.get('dataset.sequence_length', 100),
                input_size=self.config.get('model.n_in', 50),
                output_size=self.config.get('model.n_out', 10),
                spike_prob=self.config.get('dataset.spike_prob', 0.1)
            )
        elif dataset_type == "CustomSpikeDataset_Teacher":
            return CustomSpikeDataset_Teacher(
                num_samples=self.config.get('dataset.num_samples', 5),
                sequence_length=self.config.get('dataset.sequence_length', 20),
                input_size=self.config.get('model.n_in', 10),
                output_size=self.config.get('model.n_out', 5),
                hidden_size=self.config.get('model.n_hidden', 5),
                spike_prob=self.config.get('dataset.spike_prob', 0.2),
                teacher_thresh=self.config.get('model.init_thresh', 0.2),
                teacher_tau=self.config.get('model.init_tau', 0.6),
                seed=self.config.get('dataset.seed', 0),
                # teacher weight scale; MUST match whatever the SW baselines
                # use (run_three_conditions defaulted to 1.5 while this
                # pipeline silently used the dataset default 1.0 -> the
                # 2026-08-25 cross-task mismatch)
                w_scale=self.config.get('dataset.w_scale', 1.0),
            )
        elif dataset_type == "TemporalXORDataset":
            return TemporalXORDataset(
                sequence_length=self.config.get('dataset.sequence_length', 20),
                input_size=self.config.get('model.n_in', 10),
                output_size=self.config.get('model.n_out', 5),
                seed=self.config.get('dataset.seed', 0),
            )
        elif dataset_type == "CustomSpikeDataset_Probabilistic":
            return CustomSpikeDataset_Probabilistic(
                num_samples=self.config.get('dataset.num_samples', 1),
                sequence_length=self.config.get('dataset.sequence_length', 100),
                input_size=self.config.get('model.n_in', 50),
                output_size=self.config.get('model.n_out', 10),
                input_spike_prob=self.config.get('dataset.spike_prob', 0.1),
                target_spike_prob=self.config.get('dataset.spike_prob', 0.1) / 2
            )
        elif dataset_type == "SuperSpikePatternDataset":
            return SuperSpikePatternDataset(
                num_samples=self.config.get('dataset.num_samples', 1),
                sequence_length=self.config.get('dataset.sequence_length', 100),
                input_size=self.config.get('model.n_in', 50),
                output_size=self.config.get('model.n_out', 10),
                input_spike_prob=self.config.get('dataset.spike_prob', 0.1)
            )
        elif dataset_type == "NMNISTDataset":
            return NMNISTDataset(
                root=self.config.get('dataset.root', './data/nmnist'),
                train=is_train,
                time_window=self.config.get('dataset.time_window', 1000.0),
                num_time_bins=self.config.get('dataset.sequence_length', 300),
                flatten=self.config.get('dataset.flatten', True),
                max_samples=self.config.get('dataset.max_samples', None)
            )
        elif dataset_type == "NMNISTDatasetRateEncoded":
            return NMNISTDatasetRateEncoded(
                root=self.config.get('dataset.root', './data/nmnist'),
                train=is_train,
                time_window=self.config.get('dataset.time_window', 1000.0),
                num_time_bins=self.config.get('dataset.sequence_length', 300),
                flatten=self.config.get('dataset.flatten', True),
                max_samples=self.config.get('dataset.max_samples', None),
                output_duration=self.config.get('dataset.output_duration', 50),
                output_spike_prob=self.config.get('dataset.output_spike_prob', 0.1)
            )
        elif dataset_type == "NMNISTDatasetClassification":
            return NMNISTDatasetClassification(
                root=self.config.get('dataset.root', './data/nmnist'),
                train=is_train,
                time_window=self.config.get('dataset.time_window', 1000.0),
                num_time_bins=self.config.get('dataset.sequence_length', 300),
                flatten=self.config.get('dataset.flatten', True),
                max_samples=self.config.get('dataset.max_samples', None)
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def create_optimizer(self, model):
        """Create optimizer based on configuration"""
        optimizer_type = self.config.get('training.optimizer', 'Adam')
        learning_rate = self.config.get('training.learning_rate', 0.1)
        
        if optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def create_kernel(self):
        """Create kernel based on configuration"""
        kernel_size = self.config.get('kernel.size', 5)
        decay_rate = self.config.get('kernel.decay_rate', 2.0)
        return create_exponential_kernel(kernel_size, decay_rate).to(self.device)
    
    def train_epoch(self, model, dataloader, loss_fn, optimizer, kernel, kernel_size):
        """Train for one epoch"""
        model.train()
        total_loss = 0

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            
            if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
                outputs = model(inputs, targets, training=True)
            else:
                outputs = model(inputs)

            # Apply convolution
            conv_targets = apply_convolution(targets, kernel, kernel_size)
            conv_outputs = apply_convolution(outputs, kernel, kernel_size)

            loss = loss_fn(conv_outputs, conv_targets, outputs.shape[1])
            
            if hasattr(model, 'custom_grad') and model.custom_grad:
                if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
                    optimizer.step()
                else:
                    err = (conv_outputs - conv_targets).permute(1, 0, 2)
                    model.compute_grads(inputs, err)
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader), outputs, targets, inputs
    
    def evaluate(self, model, dataloader, metric_fn, kernel, kernel_size):
        """Evaluate model"""
        model.eval()
        total_metric = 0
        total_distance_per_spike = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
                    outputs = model(inputs, targets, training=False)
                else:
                    outputs = model(inputs)

                # Apply convolution
                targets = apply_convolution(targets, kernel, kernel_size)
                outputs = apply_convolution(outputs, kernel, kernel_size)

                # Calculate distance
                distance = metric_fn(outputs, targets, tau=10, dt=1.0)
                total_metric += distance.mean().item()

                if torch.sum(targets) != 0:
                    distance_per_spike = distance / torch.sum(targets)
                    total_distance_per_spike += distance_per_spike.mean().item()
                    
                    # Calculate max distance
                    max_distance = metric_fn(torch.zeros_like(targets), targets, tau=10, dt=1.0).mean()
                    normalized_distance = distance.mean() / (max_distance + 1e-6)
                
                count += 1

        return total_metric / count, total_distance_per_spike / count, normalized_distance / count
    
    def save_results(self):
        """Save experiment results"""
        if self.config.get('experiment.save_results', True):
            results_dir = self.config.get('experiment.results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save best model
            if self.best_model_state is not None:
                model_path = os.path.join(results_dir, f"best_model_{self.config.get('experiment.name', 'experiment')}.pth")
                torch.save(self.best_model_state, model_path)
    
    def visualize_results(self, inputs, outputs, targets):
        """Visualize experiment results"""
        if self.config.get('logging.save_plots', True):
            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            
            # Target vs Output comparison
            plot_comparison_raster(targets, outputs, ax=axes[0], title="Target vs Output Spikes")
            
            # Distance calculation
            distance = van_rossum_distance(outputs, targets, tau=10, dt=1.0)
            max_distance = van_rossum_distance(torch.zeros_like(targets), targets, tau=10, dt=1.0).mean()
            normalized_distance = distance.mean() / (max_distance + 1e-6)
            print(f"Normalized Distance: {normalized_distance.item():.4f}")
            
            # Input and output spikes
            plot_raster(inputs, ax=axes[1], title="Input Spikes")
            plot_raster(outputs, ax=axes[2], title="Model Output Spikes")

            plt.tight_layout()
            
            # Save or show plot
            if self.config.get('logging.save_plots', True):
                results_dir = self.config.get('experiment.results_dir', 'results')
                os.makedirs(results_dir, exist_ok=True)
                plot_format = self.config.get('logging.plot_format', 'png')
                plot_path = os.path.join(results_dir, f"results_{self.config.get('experiment.name', 'experiment')}.{plot_format}")
                plt.savefig(plot_path)
            else:
                plt.show()
            
            plt.close()
    
    @abstractmethod
    def run(self):
        """Run the experiment - must be implemented by subclasses"""
        pass
"""
WandB Sweep Training Script for E-prop Minsik Model Hyperparameter Optimization
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
import wandb
import torch
from torch.utils.data import DataLoader

from configs.config_loader import load_config, ExperimentConfig
from experiment_types.base_experiment import BaseExperiment
from models.model_factory import create_model
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance


class EpropSweepExperiment(BaseExperiment):
    """Experiment class for E-prop Minsik WandB Sweep"""

    def __init__(self, config, wandb_config):
        super().__init__(config)
        self.wandb_config = wandb_config
        self._apply_sweep_params()

    def _apply_sweep_params(self):
        """Apply sweep parameters to config for eprop_minsik model"""
        # Learning rate
        if hasattr(self.wandb_config, 'learning_rate'):
            self.config.set('training.learning_rate', self.wandb_config.learning_rate)

        # Network structure
        if hasattr(self.wandb_config, 'n_hidden'):
            self.config.set('model.n_hidden', self.wandb_config.n_hidden)

        # Membrane time constants
        if hasattr(self.wandb_config, 'init_tau'):
            self.config.set('model.init_tau', self.wandb_config.init_tau)

        if hasattr(self.wandb_config, 'init_tau_o'):
            self.config.set('model.init_tau_o', self.wandb_config.init_tau_o)

        # Spike threshold
        if hasattr(self.wandb_config, 'init_thresh'):
            self.config.set('model.init_thresh', self.wandb_config.init_thresh)

        # Surrogate gradient parameters
        if hasattr(self.wandb_config, 'gamma'):
            self.config.set('model.gamma', self.wandb_config.gamma)

        if hasattr(self.wandb_config, 'subthresh'):
            self.config.set('model.subthresh', self.wandb_config.subthresh)

        # Weight initialization
        if hasattr(self.wandb_config, 'weight_scale'):
            self.config.set('model.weight_scale', self.wandb_config.weight_scale)

        # Kernel parameters
        if hasattr(self.wandb_config, 'kernel_size'):
            self.config.set('kernel.size', self.wandb_config.kernel_size)

        if hasattr(self.wandb_config, 'kernel_decay_rate'):
            self.config.set('kernel.decay_rate', self.wandb_config.kernel_decay_rate)

    def create_model(self):
        """Create eprop_minsik model with sweep parameters"""
        model_config = {
            'n_in': self.config.get('model.n_in'),
            'n_hidden': self.config.get('model.n_hidden'),
            'n_out': self.config.get('model.n_out'),
            'recurrent': self.config.get('model.recurrent', True),
            'init_tau': self.config.get('model.init_tau', 0.6),
            'init_thresh': self.config.get('model.init_thresh', 0.6),
            'init_tau_o': self.config.get('model.init_tau_o', 0.6),
            'gamma': self.config.get('model.gamma', 0.3),
            'subthresh': self.config.get('model.subthresh', 0.5),
            'weight_scale': self.config.get('model.weight_scale', 0.5),
        }

        # Use create_model from factory with RSNN_eprop type
        model = create_model(
            model_type="RSNN_eprop",
            model_config=model_config
        )

        return model.to(self.device)

    def run(self):
        """Run training with WandB logging"""
        # Create model and optimizer
        model = self.create_model()
        optimizer = self.create_optimizer(model)
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)

        # Create dataset and dataloader
        dataset = self.create_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('training.batch_size', 1),
            shuffle=True
        )

        # Loss function
        loss_fn = mse_acc_loss_over_time

        # Training loop
        epochs = self.config.get('training.epochs', 200)
        best_loss = float('inf')
        best_normalized_distance = float('inf')

        for epoch in range(epochs):
            # Train one epoch
            avg_loss, outputs, targets, inputs = self.train_epoch(
                model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )

            # Evaluate
            eval_distance, distance_per_spike, normalized_distance = self.evaluate(
                model, dataloader, van_rossum_distance, kernel, kernel_size
            )

            # Track best metrics
            if avg_loss < best_loss:
                best_loss = avg_loss
            if normalized_distance < best_normalized_distance:
                best_normalized_distance = normalized_distance

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                'best_loss': best_loss,
                'van_rossum_distance': eval_distance,
                'distance_per_spike': distance_per_spike,
                'normalized_distance': normalized_distance,
                'best_normalized_distance': best_normalized_distance
            })

        # Log final metrics
        wandb.log({
            'final_loss': avg_loss,
            'final_best_loss': best_loss,
            'final_normalized_distance': normalized_distance,
            'final_best_normalized_distance': best_normalized_distance
        })

        return best_normalized_distance


def train():
    """Main training function for WandB Sweep"""
    # Initialize WandB
    wandb.init()

    # Load base config for eprop (filename only - config_loader adds configs/ path)
    base_config = load_config('eprop_default.yaml')

    # Create experiment with sweep config
    experiment = EpropSweepExperiment(base_config, wandb.config)

    # Run training
    final_metric = experiment.run()

    # Finish WandB run
    wandb.finish()

    return final_metric


def main():
    """Entry point for sweep agent"""
    train()


if __name__ == '__main__':
    main()

"""
WandB Sweep Training Script for SNN Hyperparameter Optimization
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb
import torch
from torch.utils.data import DataLoader

from configs.config_loader import load_config, ExperimentConfig
from experiment_types.base_experiment import BaseExperiment
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance


class SweepExperiment(BaseExperiment):
    """Experiment class for WandB Sweep"""

    def __init__(self, config, wandb_config):
        super().__init__(config)
        self.wandb_config = wandb_config
        self._apply_sweep_params()

    def _apply_sweep_params(self):
        """Apply sweep parameters to config"""
        # Learning rate
        if hasattr(self.wandb_config, 'learning_rate'):
            self.config.set('training.learning_rate', self.wandb_config.learning_rate)

        # Neuron threshold
        if hasattr(self.wandb_config, 'neuron_threshold'):
            self.config.set('neuron.triangular.thresh', self.wandb_config.neuron_threshold)

        # Weight scale
        if hasattr(self.wandb_config, 'weight_scale'):
            self.config.set('model.weight_scale', self.wandb_config.weight_scale)

        # Hidden size
        if hasattr(self.wandb_config, 'n_hidden'):
            self.config.set('model.n_hidden', self.wandb_config.n_hidden)

        # Init tau
        if hasattr(self.wandb_config, 'init_tau'):
            self.config.set('model.init_tau', self.wandb_config.init_tau)

        # Kernel size
        if hasattr(self.wandb_config, 'kernel_size'):
            self.config.set('kernel.size', self.wandb_config.kernel_size)

        # Kernel decay rate
        if hasattr(self.wandb_config, 'kernel_decay_rate'):
            self.config.set('kernel.decay_rate', self.wandb_config.kernel_decay_rate)

        # Subthreshold width
        if hasattr(self.wandb_config, 'subthresh'):
            self.config.set('neuron.triangular.subthresh', self.wandb_config.subthresh)

        # Gamma
        if hasattr(self.wandb_config, 'gamma'):
            self.config.set('neuron.triangular.gamma', self.wandb_config.gamma)

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

        for epoch in range(epochs):
            # Train one epoch
            avg_loss, outputs, targets, inputs = self.train_epoch(
                model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )

            # Evaluate
            eval_distance, distance_per_spike, normalized_distance = self.evaluate(
                model, dataloader, van_rossum_distance, kernel, kernel_size
            )

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                'best_loss': best_loss,
                'van_rossum_distance': eval_distance,
                'distance_per_spike': distance_per_spike,
                'normalized_distance': normalized_distance
            })

        # Log final metrics
        wandb.log({
            'final_loss': avg_loss,
            'final_best_loss': best_loss,
            'final_normalized_distance': normalized_distance
        })

        return best_loss


def train():
    """Main training function for WandB Sweep"""
    # Initialize WandB
    wandb.init()

    # Load base config
    base_config = load_config('default.yaml')

    # Create experiment with sweep config
    experiment = SweepExperiment(base_config, wandb.config)

    # Run training
    final_loss = experiment.run()

    # Finish WandB run
    wandb.finish()

    return final_loss


def main():
    """Entry point for sweep agent"""
    train()


if __name__ == '__main__':
    main()

import torch
from torch.utils.data import DataLoader
import copy

from experiment_types.base_experiment import BaseExperiment
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance


class BasicExperiment(BaseExperiment):
    """Basic experiment implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        
    def run(self):
        """Run the basic experiment"""
        print(f"Running basic experiment: {self.config.get('experiment.name', 'basic_experiment')}")
        
        # Create dataset and dataloader
        dataset = self.create_dataset()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.get('training.batch_size', 1), 
            shuffle=True
        )
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and loss function
        optimizer = self.create_optimizer(model)
        loss_fn = mse_acc_loss_over_time
        
        # Create kernel
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)
        
        # Training loop
        epochs = self.config.get('training.epochs', 200)
        final_outputs, final_targets, final_inputs = None, None, None
        
        for epoch in range(epochs):
            train_loss, outputs, targets, inputs = self.train_epoch(
                model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )

            
            val_metric, val_metric_per_spike, normalized_distance_per_spike = self.evaluate(
                model, dataloader, van_rossum_distance, kernel, kernel_size
            )
            
            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_model_state = copy.deepcopy(model.state_dict())
                final_outputs, final_targets, final_inputs = outputs, targets, inputs
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
                  f"Metric: {val_metric:.4f}, Per Spike: {val_metric_per_spike:.4f}, "
                  f"Normalized Per Spike: {normalized_distance_per_spike:.4f}")
        
        # Save results
        print(inputs.shape)
        self.save_results()
        
        # Visualize results
        if final_outputs is not None:
            self.visualize_results(final_inputs, final_outputs, final_targets)
        
        # Store results
        self.results = {
            'best_loss': float(self.best_loss),
            'final_metric': float(val_metric),
            'final_metric_per_spike': float(val_metric_per_spike),
            'normalized_distance_per_spike': float(normalized_distance_per_spike)
        }
        
        return self.results
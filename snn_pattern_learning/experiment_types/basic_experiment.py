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

        # Connect to hardware if the model supports it
        if hasattr(model, 'connect_hardware'):
            print("Connecting to hardware...")
            if model.connect_hardware():
                print("[OK] Hardware connected successfully")
            else:
                print("[WARNING] Hardware connection failed, will use software fallback")

        # Create optimizer and loss function
        optimizer = self.create_optimizer(model)
        loss_fn = mse_acc_loss_over_time

        # Create kernel
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)
        
        # Training loop
        epochs = self.config.get('training.epochs', 200)
        final_outputs, final_targets, final_inputs = None, None, None

        # Match the software model's initial weights when a seed is given.
        # torch.manual_seed alone is NOT enough: Basic_RSNN_eprop_HW_forward
        # calls kaiming_normal_ on `recurrent` where the SW model only uses
        # torch.rand, so the two consume different amounts of RNG and every
        # weight after fc1 diverges. Rebuilding the SW model under the same
        # seed and copying its tensors gives a genuinely identical start.
        init_seed = self.config.get('training.init_seed', None)
        if init_seed is not None:
            import torch as _torch
            from models.models import Basic_RSNN_eprop_forward as _SW
            _torch.manual_seed(init_seed)
            ref = _SW(n_in=model.n_in, n_hidden=model.n_hidden,
                      n_out=model.n_out, recurrent=model.recurrent_connection,
                      init_thresh=model.thr)
            with _torch.no_grad():
                model.fc1.weight.copy_(ref.fc1.weight)
                model.recurrent.copy_(ref.recurrent)
                model.out.weight.copy_(ref.out.weight)
            print(f"[INIT] weights copied from SW model built with "
                  f"seed {init_seed}")

        for epoch in range(epochs):
            # Reset hardware gradient accumulator and measure reference point at epoch start
            if hasattr(model, 'reset_hardware'):
                model.reset_hardware()

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

            # Apply hardware gradient (for hardware-enabled models)
            if hasattr(model, 'apply_hw_gradient'):
                model.apply_hw_gradient(learning_rate=self.config.get('training.learning_rate', 0.01))

        # Save results
        print(inputs.shape)
        self.save_results()
        
        # Visualize results
        if final_outputs is not None:
            self.visualize_results(final_inputs, final_outputs, final_targets)
        
        # Disconnect hardware if connected
        if hasattr(model, 'disconnect_hardware'):
            model.disconnect_hardware()
            print("[OK] Hardware disconnected")

        # Store results
        self.results = {
            'best_loss': float(self.best_loss),
            'final_metric': float(val_metric),
            'final_metric_per_spike': float(val_metric_per_spike),
            'normalized_distance_per_spike': float(normalized_distance_per_spike)
        }

        return self.results
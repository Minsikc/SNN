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

        self.history = []          # per-epoch summary, written to the run registry at the end
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
            
            self.history.append(dict(epoch=epoch + 1, train_loss=float(train_loss),
                                     metric=float(val_metric),
                                     hw_fidelity=(getattr(getattr(model, 'readout', None), 'last_stats', {})
                                                  or {}).get('corr')))

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

        # Run registry (results/registry.jsonl, $SNN_REGISTRY): one summary line per
        # run with the resolved config, shared with eprop.run_condition / run_xor.
        if self.config.get('experiment.registry', True):
            try:
                self._record_registry(model, epochs)
            except Exception as e:          # never let bookkeeping kill a finished run
                print(f"[registry] skipped: {e}")

        return self.results
    def _record_registry(self, model, epochs):
        """Append this run to the shared run registry (see eprop/registry.py)."""
        from eprop import registry
        cfg = self.config
        hw_cfg = cfg.get('model.hardware', {}) or {}
        hw_on = bool(hw_cfg.get('enabled', False))
        if hw_on:
            condition = 'digital_mock' if hw_cfg.get('use_mock_hw', False) else 'analog'
            if hw_cfg.get('freeze_wout', False):
                condition = 'frozen'
        else:
            condition = 'bptt' if cfg.get('model.learning_rule', 'eprop') == 'bptt' else 'digital'
        neuron = getattr(model, 'neuron', None)
        chain = getattr(model, 'chain', None)
        if neuron is None:                      # legacy model classes
            neuron = dict(kind='lif', legacy_model=cfg.get('model.type'),
                          tau=cfg.get('model.init_tau'), thresh=cfg.get('model.init_thresh'),
                          tau_o=cfg.get('model.init_tau_o'))
        fid = [h['hw_fidelity'] for h in self.history if h.get('hw_fidelity') is not None]
        best_ep = min(self.history, key=lambda h: h['train_loss'])['epoch'] if self.history else None
        metrics = dict(best_loss=float(self.best_loss), best_epoch=best_ep,
                       final_loss=self.history[-1]['train_loss'] if self.history else None,
                       final_metric=self.results.get('final_metric'),
                       normalized_distance_per_spike=self.results.get('normalized_distance_per_spike'),
                       fidelity_mean=(sum(fid) / len(fid)) if fid else None,
                       fidelity_min=min(fid) if fid else None)
        task_cfg = dict(dataset=cfg.get('dataset.type'), n_in=cfg.get('model.n_in'),
                        n_hidden=cfg.get('model.n_hidden'), n_out=cfg.get('model.n_out'),
                        T=cfg.get('dataset.sequence_length'), n_seq=cfg.get('dataset.num_samples'),
                        ds_seed=cfg.get('dataset.seed'), w_scale=cfg.get('dataset.w_scale'),
                        spike_prob=cfg.get('dataset.spike_prob'))
        entry = registry.make_entry(
            entry_point='main_unified', task=str(cfg.get('dataset.type')), condition=condition,
            neuron=neuron, chain=chain, hw=(hw_cfg if hw_on else None), task_cfg=task_cfg,
            train_hidden=bool(cfg.get('training.train_hidden', True)),
            seed=cfg.get('training.init_seed', -1) if cfg.get('training.init_seed') is not None else -1,
            epochs=epochs, lr=cfg.get('training.learning_rate', 0.01), metrics=metrics,
            curves_path=cfg.get('experiment.results_dir', 'results'),
            note=f"config={cfg.get('experiment.name', '')}")
        entry['history'] = self.history
        path = registry.record(entry)
        print(f"[registry] {entry['run_id']} -> {path}")

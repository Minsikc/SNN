"""
Classification Experiment for N-MNIST and similar classification tasks.
Uses cumulative spike count as class prediction with CrossEntropyLoss.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import os
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from experiment_types.base_experiment import BaseExperiment


class ClassificationExperiment(BaseExperiment):
    """Classification experiment using cumulative spike counts"""

    def __init__(self, config):
        super().__init__(config)

    def train_epoch_classification(self, model, dataloader, optimizer):
        """Train for one epoch using CrossEntropyLoss on cumulative spikes"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # [batch, time, n_out]

            # Sum spikes over time to get firing rates
            spike_count = outputs.sum(dim=1)  # [batch, n_out]

            # Classification loss
            loss = F.cross_entropy(spike_count, labels)

            if hasattr(model, 'custom_grad') and model.custom_grad:
                # E-prop: 매 타임스텝에서 output spike와 target 비교 (online learning)
                # Rate-coded target: 정답 클래스는 1, 나머지는 0 (매 타임스텝)
                num_timesteps = outputs.shape[1]
                num_classes = outputs.shape[2]

                # target: [batch, n_out] -> [time, batch, n_out]
                one_hot = F.one_hot(labels, num_classes=num_classes).float()  # [batch, n_out]
                target = one_hot.unsqueeze(0).expand(num_timesteps, -1, -1)  # [time, batch, n_out]

                # err[t] = output_spike[t] - target[t] for each timestep
                err = outputs.permute(1, 0, 2) - target  # [time, batch, n_out]

                model.compute_grads(inputs, err)
            else:
                # BPTT: 기존 방식
                loss.backward()

            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = spike_count.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                current_acc = correct / total if total > 0 else 0
                print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {current_acc*100:.2f}%", end='\r')

        print()  # New line after batch loop

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy, outputs, labels, inputs

    def evaluate_classification(self, model, dataloader):
        """Evaluate model on classification task"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        print("  Evaluating...", end=' ')
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)  # [batch, time, n_out]
                spike_count = outputs.sum(dim=1)  # [batch, n_out]

                loss = F.cross_entropy(spike_count, labels)
                total_loss += loss.item()

                predictions = spike_count.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        print("Done")
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def visualize_classification_results(self, inputs, outputs, labels, predictions):
        """Visualize classification results"""
        if not self.config.get('logging.save_plots', True):
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Input spike raster (first sample)
        ax1 = axes[0, 0]
        input_sample = inputs[0].cpu().numpy()  # [time, n_in]
        spike_times, neuron_ids = input_sample.nonzero()
        ax1.scatter(spike_times, neuron_ids, s=1, c='black', marker='|')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Input Neuron')
        ax1.set_title(f'Input Spikes (Label: {labels[0].item()})')

        # 2. Output spike raster (first sample)
        ax2 = axes[0, 1]
        output_sample = outputs[0].detach().cpu().numpy()  # [time, n_out]
        spike_times, neuron_ids = output_sample.nonzero()
        ax2.scatter(spike_times, neuron_ids, s=10, c='blue', marker='|')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Output Neuron (Class)')
        ax2.set_title(f'Output Spikes (Prediction: {predictions[0].item()})')
        ax2.set_yticks(range(10))

        # 3. Spike count per class (first sample)
        ax3 = axes[1, 0]
        spike_count = outputs[0].sum(dim=0).detach().cpu().numpy()  # [n_out]
        colors = ['green' if i == labels[0].item() else 'gray' for i in range(len(spike_count))]
        colors[predictions[0].item()] = 'blue' if predictions[0].item() != labels[0].item() else 'green'
        ax3.bar(range(len(spike_count)), spike_count, color=colors)
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Spike Count')
        ax3.set_title('Cumulative Spikes per Class')
        ax3.set_xticks(range(10))

        # 4. Batch accuracy summary
        ax4 = axes[1, 1]
        batch_correct = (predictions == labels).cpu().numpy()
        ax4.bar(['Correct', 'Incorrect'],
                [batch_correct.sum(), len(batch_correct) - batch_correct.sum()],
                color=['green', 'red'])
        ax4.set_ylabel('Count')
        ax4.set_title(f'Batch Accuracy: {batch_correct.mean()*100:.1f}%')

        plt.tight_layout()

        # Save plot
        results_dir = self.config.get('experiment.results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plot_format = self.config.get('logging.plot_format', 'png')
        plot_path = os.path.join(results_dir, f"results_{self.config.get('experiment.name', 'classification')}.{plot_format}")
        plt.savefig(plot_path)
        print(f"Results saved to {plot_path}")
        plt.close()

    def run(self):
        """Run the classification experiment"""
        print(f"Running classification experiment: {self.config.get('experiment.name', 'classification_experiment')}")

        # Initialize wandb if enabled and available
        use_wandb = self.config.get('logging.use_wandb', False)
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get('logging.wandb_project', 'snn-training'),
                name=self.config.get('experiment.name', 'classification_experiment'),
                config={
                    'model_type': self.config.get('model.type'),
                    'n_in': self.config.get('model.n_in'),
                    'n_hidden': self.config.get('model.n_hidden'),
                    'n_out': self.config.get('model.n_out'),
                    'neuron_type': self.config.get('model.neuron_type'),
                    'batch_size': self.config.get('training.batch_size'),
                    'learning_rate': self.config.get('training.learning_rate'),
                    'epochs': self.config.get('training.epochs'),
                    'optimizer': self.config.get('training.optimizer'),
                    'dataset_type': self.config.get('dataset.type'),
                    'sequence_length': self.config.get('dataset.sequence_length'),
                }
            )
            print("wandb initialized")
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")

        # Create train dataset and dataloader
        train_dataset = self.create_dataset(train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('training.batch_size', 32),
            shuffle=True
        )

        # Create test dataset and dataloader if available
        try:
            test_dataset = self.create_dataset(train=False)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.config.get('training.batch_size', 32),
                shuffle=False
            )
            has_test_set = True
        except Exception as e:
            print(f"No test set available: {e}")
            test_dataloader = train_dataloader
            has_test_set = False

        # Create model
        model = self.create_model()
        print(f"Model created: {type(model).__name__}")
        print(f"  Input size: {self.config.get('model.n_in')}")
        print(f"  Hidden size: {self.config.get('model.n_hidden')}")
        print(f"  Output size: {self.config.get('model.n_out')}")

        # Create optimizer
        optimizer = self.create_optimizer(model)

        # Training loop
        epochs = self.config.get('training.epochs', 20)
        best_accuracy = 0
        final_outputs, final_labels, final_inputs = None, None, None

        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)

        for epoch in range(epochs):
            # Train
            train_loss, train_acc, outputs, labels, inputs = self.train_epoch_classification(
                model, train_dataloader, optimizer
            )

            # Evaluate
            if has_test_set:
                test_loss, test_acc = self.evaluate_classification(model, test_dataloader)
            else:
                test_loss, test_acc = train_loss, train_acc

            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.best_model_state = copy.deepcopy(model.state_dict())
                final_outputs, final_labels, final_inputs = outputs, labels, inputs

            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'test_loss': test_loss,
                    'test_accuracy': test_acc,
                    'best_accuracy': best_accuracy
                })

        print("-" * 60)
        print(f"Best Test Accuracy: {best_accuracy*100:.2f}%")

        # Save best model
        if self.best_model_state is not None:
            results_dir = self.config.get('experiment.results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, f"best_model_{self.config.get('experiment.name', 'classification')}.pth")
            torch.save(self.best_model_state, model_path)
            print(f"Best model saved to {model_path}")

        # Visualize results
        if final_outputs is not None:
            spike_count = final_outputs.sum(dim=1)
            predictions = spike_count.argmax(dim=1)
            self.visualize_classification_results(final_inputs, final_outputs, final_labels, predictions)

        # Store results
        self.results = {
            'best_accuracy': float(best_accuracy),
            'final_train_loss': float(train_loss),
            'final_train_acc': float(train_acc),
            'final_test_loss': float(test_loss),
            'final_test_acc': float(test_acc)
        }

        # Log final results and finish wandb
        if use_wandb and WANDB_AVAILABLE:
            # Log final summary metrics
            wandb.summary['best_accuracy'] = best_accuracy
            wandb.summary['final_train_loss'] = train_loss
            wandb.summary['final_test_loss'] = test_loss

            # Save model artifact
            if self.best_model_state is not None:
                results_dir = self.config.get('experiment.results_dir', 'results')
                model_path = os.path.join(results_dir, f"best_model_{self.config.get('experiment.name', 'classification')}.pth")
                wandb.save(model_path)

            # Save result plot if available
            plot_path = os.path.join(results_dir, f"results_{self.config.get('experiment.name', 'classification')}.png")
            if os.path.exists(plot_path):
                wandb.log({"results_plot": wandb.Image(plot_path)})

            wandb.finish()
            print("wandb logging finished")

        return self.results

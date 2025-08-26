import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt

from experiment_types.base_experiment import BaseExperiment
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance
from datasets.customdatasets import CustomSpikeDataset_from_model


class TeacherStudentExperiment(BaseExperiment):
    """Teacher-Student experiment implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.teacher_model = None
        self.student_model = None
        self.logger = None
        self.pattern_analyzer = None
        
        # Initialize logger if enabled
        if self.config.get('logging.enable_logger', False):
            self.logger = self._create_logger()
        
        # Initialize pattern analyzer if enabled
        if self.config.get('logging.enable_pattern_analyzer', False):
            self.pattern_analyzer = self._create_pattern_analyzer()
    
    def _create_logger(self):
        """Create experiment logger"""
        # Import here to avoid circular imports
        try:
            from utils.experiment_logger import ExperimentLogger
            return ExperimentLogger(
                experiment_name=self.config.get('experiment.name', 'teacher_student'),
                save_dir=self.config.get('experiment.results_dir', 'results')
            )
        except ImportError:
            print("Warning: ExperimentLogger not available. Logging disabled.")
            return None
    
    def _create_pattern_analyzer(self):
        """Create pattern analyzer"""
        try:
            from utils.pattern_analyzer import PatternAnalyzer
            return PatternAnalyzer()
        except ImportError:
            print("Warning: PatternAnalyzer not available. Pattern analysis disabled.")
            return None
    
    def create_teacher_model(self):
        """Create teacher model"""
        teacher_model_type = self.config.get('teacher_student.teacher_model_type', 'Basic_RSNN_spike')
        
        # Temporarily override model type for teacher
        original_model_type = self.config.get('model.type')
        self.config.set('model.type', teacher_model_type)
        
        teacher_model = self.create_model()
        
        # Restore original model type
        self.config.set('model.type', original_model_type)
        
        return teacher_model
    
    def create_student_model(self):
        """Create student model"""
        student_model_type = self.config.get('teacher_student.student_model_type', 'Basic_RSNN_spike')
        
        # Temporarily override model type for student
        original_model_type = self.config.get('model.type')
        self.config.set('model.type', student_model_type)
        
        student_model = self.create_model()
        
        # Restore original model type
        self.config.set('model.type', original_model_type)
        
        return student_model
    
    def train_teacher(self):
        """Train teacher model"""
        print("Training teacher model...")
        
        # Create dataset and dataloader
        dataset = self.create_dataset()
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.get('training.batch_size', 1), 
            shuffle=True
        )
        
        # Create teacher model
        self.teacher_model = self.create_teacher_model()
        
        # Create optimizer and loss function
        optimizer = self.create_optimizer(self.teacher_model)
        loss_fn = mse_acc_loss_over_time
        
        # Create kernel
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)
        
        # Training loop
        teacher_epochs = self.config.get('teacher_student.teacher_epochs', 100)
        best_teacher_loss = float("inf")
        
        for epoch in range(teacher_epochs):
            train_loss, outputs, targets, inputs = self.train_epoch(
                self.teacher_model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )
            
            if train_loss < best_teacher_loss:
                best_teacher_loss = train_loss
                self.teacher_model.load_state_dict(copy.deepcopy(self.teacher_model.state_dict()))
            
            if epoch % 10 == 0:
                print(f"Teacher Epoch {epoch+1}/{teacher_epochs}, Loss: {train_loss:.4f}")
        
        print(f"Teacher training completed. Best loss: {best_teacher_loss:.4f}")
        return self.teacher_model
    
    def generate_teacher_data(self):
        """Generate data from trained teacher model"""
        print("Generating teacher data...")
        
        # Create base dataset for input patterns
        base_dataset = self.create_dataset()
        
        # Generate teacher outputs
        teacher_dataset = CustomSpikeDataset_from_model(
            model=self.teacher_model,
            num_samples=self.config.get('dataset.num_samples', 1),
            sequence_length=self.config.get('dataset.sequence_length', 100),
            input_size=self.config.get('model.n_in', 50),
            output_size=self.config.get('model.n_out', 10),
            spike_prob=self.config.get('dataset.spike_prob', 0.1)
        )
        
        # Analyze patterns if analyzer is available
        if self.pattern_analyzer is not None:
            # Get a sample from the dataset to analyze
            sample_data = next(iter(torch.utils.data.DataLoader(teacher_dataset, batch_size=1)))
            sample_targets = sample_data[1]  # targets
            
            pattern_analysis = self.pattern_analyzer.analyze_pattern(sample_targets)
            pattern_difficulty = self.pattern_analyzer.pattern_difficulty_score(pattern_analysis)
            print(f"Teacher pattern difficulty: {pattern_difficulty:.4f}")
            
            if self.logger is not None:
                self.logger.log_metadata(teacher_pattern_difficulty=pattern_difficulty)
        
        return teacher_dataset
    
    def train_student(self, teacher_dataset):
        """Train student model on teacher data"""
        print("Training student model...")
        
        # Create dataloader
        dataloader = DataLoader(
            teacher_dataset,
            batch_size=self.config.get('training.batch_size', 1),
            shuffle=True
        )
        
        # Create student model
        self.student_model = self.create_student_model()
        
        # Create optimizer and loss function
        optimizer = self.create_optimizer(self.student_model)
        loss_fn = mse_acc_loss_over_time
        
        # Create kernel
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)
        
        # Training loop
        student_epochs = self.config.get('teacher_student.student_epochs', 200)
        best_student_loss = float("inf")
        best_student_state = None
        
        for epoch in range(student_epochs):
            train_loss, outputs, targets, inputs = self.train_epoch(
                self.student_model, dataloader, loss_fn, optimizer, kernel, kernel_size
            )
            
            if train_loss < best_student_loss:
                best_student_loss = train_loss
                best_student_state = copy.deepcopy(self.student_model.state_dict())
            
            # Calculate similarities if tracking is enabled
            if self.config.get('teacher_student.track_weight_diff', False):
                teacher_student_sim = self.calculate_similarity(
                    self.teacher_model, self.student_model, inputs
                )
                student_target_sim = self.calculate_output_similarity(outputs, targets)
                
                if self.logger is not None:
                    self.logger.log_training_step(
                        epoch=epoch,
                        train_loss=train_loss,
                        teacher_student_sim=teacher_student_sim,
                        student_target_sim=student_target_sim
                    )
            
            if epoch % 10 == 0:
                print(f"Student Epoch {epoch+1}/{student_epochs}, Loss: {train_loss:.4f}")
        
        # Load best student model
        if best_student_state is not None:
            self.student_model.load_state_dict(best_student_state)
            self.best_model_state = best_student_state
            self.best_loss = best_student_loss
        
        print(f"Student training completed. Best loss: {best_student_loss:.4f}")
        return self.student_model
    
    def calculate_similarity(self, teacher_model, student_model, inputs):
        """Calculate similarity between teacher and student outputs"""
        teacher_model.eval()
        student_model.eval()
        
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            
            similarity = van_rossum_distance(teacher_outputs, student_outputs, tau=10, dt=1.0)
            return similarity.mean().item()
    
    def calculate_output_similarity(self, outputs, targets):
        """Calculate similarity between outputs and targets"""
        similarity = van_rossum_distance(outputs, targets, tau=10, dt=1.0)
        return similarity.mean().item()
    
    def evaluate_final_performance(self, teacher_dataset):
        """Evaluate final performance"""
        dataloader = DataLoader(teacher_dataset, batch_size=1, shuffle=False)
        kernel = self.create_kernel()
        kernel_size = self.config.get('kernel.size', 5)
        
        # Evaluate student model
        val_metric, val_metric_per_spike, normalized_distance = self.evaluate(
            self.student_model, dataloader, van_rossum_distance, kernel, kernel_size
        )
        
        print(f"Final Student Performance:")
        print(f"  Metric: {val_metric:.4f}")
        print(f"  Per Spike: {val_metric_per_spike:.4f}")
        print(f"  Normalized Distance: {normalized_distance:.4f}")
        
        return {
            'final_metric': float(val_metric),
            'final_metric_per_spike': float(val_metric_per_spike),
            'normalized_distance': float(normalized_distance)
        }
    
    def visualize_teacher_student_comparison(self, teacher_dataset):
        """Visualize teacher-student comparison"""
        if not self.config.get('logging.save_plots', True):
            return
        
        # Get sample data
        sample_data = next(iter(DataLoader(teacher_dataset, batch_size=1)))
        inputs, targets = sample_data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Get outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            student_outputs = self.student_model(inputs)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Teacher vs Target
        from utils.plots import plot_comparison_raster
        plot_comparison_raster(targets, teacher_outputs, ax=axes[0, 0], title="Teacher vs Target")
        
        # Student vs Target
        plot_comparison_raster(targets, student_outputs, ax=axes[0, 1], title="Student vs Target")
        
        # Teacher vs Student
        plot_comparison_raster(teacher_outputs, student_outputs, ax=axes[1, 0], title="Teacher vs Student")
        
        # Input spikes
        from utils.plots import plot_raster
        plot_raster(inputs, ax=axes[1, 1], title="Input Spikes")
        
        plt.tight_layout()
        
        # Save plot
        if self.logger is not None:
            self.logger.save_plot(fig, "teacher_student_comparison.png")
        else:
            results_dir = self.config.get('experiment.results_dir', 'results')
            import os
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f"teacher_student_comparison_{self.config.get('experiment.name', 'experiment')}.png")
            plt.savefig(plot_path)
        
        plt.close()
    
    def run(self):
        """Run the teacher-student experiment"""
        print(f"Running teacher-student experiment: {self.config.get('experiment.name', 'teacher_student_experiment')}")
        
        # Log metadata
        if self.logger is not None:
            self.logger.log_metadata(
                experiment_type="teacher_student",
                config=self.config.to_dict()
            )
        
        # Step 1: Train teacher model
        teacher_model = self.train_teacher()
        
        # Step 2: Generate teacher data
        teacher_dataset = self.generate_teacher_data()
        
        # Step 3: Train student model
        student_model = self.train_student(teacher_dataset)
        
        # Step 4: Evaluate final performance
        final_results = self.evaluate_final_performance(teacher_dataset)
        
        # Step 5: Visualize results
        self.visualize_teacher_student_comparison(teacher_dataset)
        
        # Step 6: Save results
        self.save_results()
        
        if self.logger is not None:
            self.logger.log_final_results(**final_results)
        
        # Store results
        self.results = final_results
        
        print("Teacher-Student experiment completed!")
        return self.results
import json
import os
import datetime
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt


class ExperimentLogger:
    """실험 결과를 체계적으로 기록하고 저장하는 클래스"""
    
    def __init__(self, experiment_name, save_dir="experiment_results"):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 타임스탬프로 고유한 실험 폴더 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 실험 메타데이터
        self.metadata = {}
        self.results = {
            'train_losses': [],
            'similarities': [],
            'weight_differences': [],
            'final_metrics': {}
        }
        
        print(f"Experiment directory created: {self.experiment_dir}")
    
    def log_metadata(self, **kwargs):
        """실험 설정 및 메타데이터 기록"""
        self.metadata.update(kwargs)
        self.metadata['timestamp'] = datetime.datetime.now().isoformat()
        
        # numpy 타입을 Python 기본 타입으로 변환
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 메타데이터를 JSON으로 저장
        with open(self.experiment_dir / "metadata.json", 'w') as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=2)
    
    def log_training_step(self, epoch, train_loss, teacher_student_sim=None, student_target_sim=None, weight_diffs=None):
        """각 에포크의 학습 결과 기록"""
        self.results['train_losses'].append({
            'epoch': epoch,
            'loss': train_loss
        })
        
        if teacher_student_sim is not None and student_target_sim is not None:
            self.results['similarities'].append({
                'epoch': epoch,
                'teacher_student_similarity': teacher_student_sim,
                'student_target_similarity': student_target_sim
            })
        
        if weight_diffs is not None:
            weight_diff_entry = {'epoch': epoch}
            weight_diff_entry.update(weight_diffs)
            self.results['weight_differences'].append(weight_diff_entry)
    
    def log_final_results(self, **metrics):
        """최종 실험 결과 기록"""
        self.results['final_metrics'].update(metrics)
        
        # 결과를 JSON으로 저장
        with open(self.experiment_dir / "results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def save_plot(self, fig, filename):
        """그래프를 실험 폴더에 저장"""
        fig.savefig(self.experiment_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {self.experiment_dir / filename}")
    
    def save_model(self, model, filename):
        """모델을 실험 폴더에 저장"""
        torch.save(model.state_dict(), self.experiment_dir / filename)
        print(f"Model saved: {self.experiment_dir / filename}")
    
    def get_experiment_dir(self):
        """실험 디렉토리 경로 반환"""
        return self.experiment_dir
    
    def create_summary_report(self):
        """실험 요약 보고서 생성"""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': self.metadata.get('timestamp', 'unknown'),
            'total_epochs': len(self.results['train_losses']),
            'final_loss': self.results['train_losses'][-1]['loss'] if self.results['train_losses'] else None,
            'final_metrics': self.results['final_metrics']
        }
        
        # 요약 보고서 저장
        with open(self.experiment_dir / "summary_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def plot_training_curves(self):
        """학습 곡선 플롯 생성"""
        if not self.results['train_losses']:
            return
        
        epochs = [entry['epoch'] for entry in self.results['train_losses']]
        losses = [entry['loss'] for entry in self.results['train_losses']]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        axes[0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Similarity curves (if available)
        if self.results['similarities']:
            sim_epochs = [entry['epoch'] for entry in self.results['similarities']]
            teacher_student_sims = [entry['teacher_student_similarity'] for entry in self.results['similarities']]
            student_target_sims = [entry['student_target_similarity'] for entry in self.results['similarities']]
            
            axes[1].plot(sim_epochs, teacher_student_sims, 'r-', linewidth=2, label='Teacher-Student')
            axes[1].plot(sim_epochs, student_target_sims, 'g-', linewidth=2, label='Student-Target')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Similarity')
            axes[1].set_title('Similarity Curves')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No similarity data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        self.save_plot(fig, "training_curves.png")
        plt.close()
    
    def __del__(self):
        """소멸자: 실험 종료 시 요약 보고서 생성"""
        try:
            if hasattr(self, 'experiment_dir') and self.experiment_dir.exists():
                self.create_summary_report()
                self.plot_training_curves()
        except Exception as e:
            print(f"Warning: Error during logger cleanup: {e}")


def calculate_weight_differences(teacher, student):
    """Teacher와 Student 모델 간의 가중치 차이 계산"""
    weight_diffs = {}
    
    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    
    total_diff = 0
    total_params = 0
    
    for name in teacher_params:
        if name in student_params:
            diff = torch.norm(teacher_params[name] - student_params[name]).item()
            weight_diffs[name] = diff
            total_diff += diff
            total_params += 1
    
    weight_diffs['total_average'] = total_diff / total_params if total_params > 0 else 0
    
    return weight_diffs
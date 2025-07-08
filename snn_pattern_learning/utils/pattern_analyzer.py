"""
스파이킹 패턴의 특성을 분석하는 유틸리티 모듈
Teacher 모델이 생성하는 target pattern의 복잡도와 특성을 정량적으로 측정
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import warnings

class PatternAnalyzer:
    """스파이킹 패턴 분석을 위한 클래스"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_pattern(self, spike_pattern: torch.Tensor, tau: float = 10.0, dt: float = 1.0) -> Dict[str, float]:
        """
        스파이킹 패턴의 종합적인 분석을 수행
        
        Args:
            spike_pattern: [batch, time, neurons] 형태의 스파이크 패턴
            tau: Van Rossum distance 계산용 시상수
            dt: 시간 간격
            
        Returns:
            분석 결과 딕셔너리
        """
        # GPU 텐서를 CPU로 이동
        if spike_pattern.is_cuda:
            spike_pattern = spike_pattern.cpu()
        
        pattern_np = spike_pattern.numpy()
        
        results = {}
        
        # 기본 통계
        results.update(self._compute_basic_stats(pattern_np))
        
        # 시공간 특성
        results.update(self._compute_spatiotemporal_stats(pattern_np))
        
        # 복잡도 측정
        results.update(self._compute_complexity_measures(pattern_np))
        
        # 상관관계 측정
        results.update(self._compute_correlation_measures(pattern_np))
        
        # 동역학 특성
        results.update(self._compute_dynamics_measures(pattern_np))
        
        return results
    
    def _compute_basic_stats(self, pattern: np.ndarray) -> Dict[str, float]:
        """기본 통계 계산"""
        results = {}
        
        # Sparsity (0인 값의 비율)
        total_elements = pattern.size
        zero_elements = np.sum(pattern == 0)
        results['sparsity'] = zero_elements / total_elements
        
        # Activity level (평균 활성화)
        results['activity_level'] = np.mean(pattern)
        
        # Spike rate (시간당 스파이크 수)
        total_spikes = np.sum(pattern)
        total_time_neurons = pattern.shape[1] * pattern.shape[2]  # time * neurons
        results['spike_rate'] = total_spikes / total_time_neurons
        
        # Activity distribution
        results['activity_std'] = np.std(pattern)
        results['activity_max'] = np.max(pattern)
        results['activity_min'] = np.min(pattern)
        
        return results
    
    def _compute_spatiotemporal_stats(self, pattern: np.ndarray) -> Dict[str, float]:
        """시공간 통계 계산"""
        results = {}
        
        batch_size, time_steps, num_neurons = pattern.shape
        
        # Temporal statistics (시간축 통계)
        temporal_activity = np.mean(pattern, axis=(0, 2))  # [time]
        results['temporal_variance'] = np.var(temporal_activity)
        results['temporal_peak_ratio'] = np.max(temporal_activity) / (np.mean(temporal_activity) + 1e-8)
        
        # Spatial statistics (공간축 통계)
        spatial_activity = np.mean(pattern, axis=(0, 1))  # [neurons]
        results['spatial_variance'] = np.var(spatial_activity)
        results['spatial_peak_ratio'] = np.max(spatial_activity) / (np.mean(spatial_activity) + 1e-8)
        
        # Active neuron ratio
        active_neurons = np.sum(np.sum(pattern, axis=(0, 1)) > 0)
        results['active_neuron_ratio'] = active_neurons / num_neurons
        
        # Burst detection (연속 스파이크 감지)
        results['burst_coefficient'] = self._compute_burst_coefficient(pattern)
        
        return results
    
    def _compute_complexity_measures(self, pattern: np.ndarray) -> Dict[str, float]:
        """복잡도 측정"""
        results = {}
        
        # Shannon entropy
        # 전체 패턴의 히스토그램 기반 엔트로피
        flat_pattern = pattern.flatten()
        if len(np.unique(flat_pattern)) > 1:
            hist, _ = np.histogram(flat_pattern, bins=10, density=True)
            hist = hist[hist > 0]  # 0이 아닌 값만
            results['shannon_entropy'] = -np.sum(hist * np.log2(hist + 1e-8))
        else:
            results['shannon_entropy'] = 0.0
        
        # Temporal entropy (시간축 패턴의 엔트로피)
        temporal_patterns = np.mean(pattern, axis=2)  # [batch, time]
        results['temporal_entropy'] = self._compute_sequence_entropy(temporal_patterns)
        
        # Spatial entropy (공간축 패턴의 엔트로피)
        spatial_patterns = np.mean(pattern, axis=1)  # [batch, neurons]
        results['spatial_entropy'] = self._compute_sequence_entropy(spatial_patterns)
        
        # Pattern regularity (패턴의 규칙성)
        results['pattern_regularity'] = self._compute_pattern_regularity(pattern)
        
        return results
    
    def _compute_correlation_measures(self, pattern: np.ndarray) -> Dict[str, float]:
        """상관관계 측정"""
        results = {}
        
        batch_size, time_steps, num_neurons = pattern.shape
        
        # Spatial correlation (뉴런 간 상관관계)
        spatial_corr_sum = 0
        valid_batches = 0
        
        for b in range(batch_size):
            neuron_activities = pattern[b].T  # [neurons, time]
            if neuron_activities.std() > 1e-8:  # 활성화가 있는 경우만
                corr_matrix = np.corrcoef(neuron_activities)
                # 대각선 제외한 상관계수의 평균
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                valid_corrs = corr_matrix[mask]
                valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
                if len(valid_corrs) > 0:
                    spatial_corr_sum += np.mean(np.abs(valid_corrs))
                    valid_batches += 1
        
        results['spatial_correlation'] = spatial_corr_sum / max(valid_batches, 1)
        
        # Temporal correlation (시간축 상관관계)
        temporal_corr_sum = 0
        valid_neurons = 0
        
        for n in range(num_neurons):
            neuron_time_series = pattern[:, :, n]  # [batch, time]
            if neuron_time_series.std() > 1e-8:
                # 연속된 시간 스텝 간의 상관관계
                for b in range(batch_size):
                    series = neuron_time_series[b]
                    if len(series) > 1 and series.std() > 1e-8:
                        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                        if not np.isnan(autocorr):
                            temporal_corr_sum += abs(autocorr)
                            valid_neurons += 1
        
        results['temporal_correlation'] = temporal_corr_sum / max(valid_neurons, 1)
        
        return results
    
    def _compute_dynamics_measures(self, pattern: np.ndarray) -> Dict[str, float]:
        """동역학 특성 측정"""
        results = {}
        
        batch_size, time_steps, num_neurons = pattern.shape
        
        # Temporal variability (시간적 변동성)
        temporal_diffs = np.diff(pattern, axis=1)  # [batch, time-1, neurons]
        results['temporal_variability'] = np.mean(np.abs(temporal_diffs))
        
        # Pattern stability (패턴 안정성)
        # 연속된 시간 구간 간의 유사도
        window_size = min(10, time_steps // 4)
        if window_size > 1:
            similarities = []
            for i in range(time_steps - window_size):
                window1 = pattern[:, i:i+window_size, :]
                window2 = pattern[:, i+1:i+1+window_size, :]
                similarity = np.corrcoef(window1.flatten(), window2.flatten())[0, 1]
                if not np.isnan(similarity):
                    similarities.append(abs(similarity))
            results['pattern_stability'] = np.mean(similarities) if similarities else 0.0
        else:
            results['pattern_stability'] = 0.0
        
        # Oscillation detection (진동 패턴 감지)
        results['oscillation_strength'] = self._detect_oscillations(pattern)
        
        return results
    
    def _compute_burst_coefficient(self, pattern: np.ndarray) -> float:
        """버스트 계수 계산"""
        burst_scores = []
        
        for b in range(pattern.shape[0]):
            for n in range(pattern.shape[2]):
                spike_train = pattern[b, :, n]
                if np.sum(spike_train) > 1:  # 스파이크가 있는 경우만
                    # 연속된 스파이크 구간 찾기
                    diff = np.diff(np.concatenate(([0], spike_train, [0])))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    if len(starts) > 0 and len(ends) > 0:
                        burst_lengths = ends - starts
                        avg_burst = np.mean(burst_lengths)
                        burst_scores.append(avg_burst)
        
        return np.mean(burst_scores) if burst_scores else 0.0
    
    def _compute_sequence_entropy(self, sequences: np.ndarray) -> float:
        """시퀀스의 엔트로피 계산"""
        entropies = []
        
        for seq in sequences:
            if seq.std() > 1e-8:
                # 값을 빈으로 나누어 히스토그램 계산
                hist, _ = np.histogram(seq, bins=10, density=True)
                hist = hist[hist > 0]
                if len(hist) > 1:
                    entropy = -np.sum(hist * np.log2(hist + 1e-8))
                    entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0
    
    def _compute_pattern_regularity(self, pattern: np.ndarray) -> float:
        """패턴의 규칙성 계산"""
        regularities = []
        
        for b in range(pattern.shape[0]):
            for n in range(pattern.shape[2]):
                spike_train = pattern[b, :, n]
                if np.sum(spike_train) > 2:  # 충분한 스파이크가 있는 경우
                    spike_times = np.where(spike_train > 0)[0]
                    if len(spike_times) > 2:
                        intervals = np.diff(spike_times)
                        if len(intervals) > 1:
                            # 간격의 변동계수 (낮을수록 규칙적)
                            cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                            regularity = 1.0 / (1.0 + cv)  # 0~1 사이로 정규화
                            regularities.append(regularity)
        
        return np.mean(regularities) if regularities else 0.0
    
    def _detect_oscillations(self, pattern: np.ndarray) -> float:
        """진동 패턴 강도 감지"""
        oscillation_strengths = []
        
        for b in range(pattern.shape[0]):
            # 전체 활성화의 시간 변화
            total_activity = np.sum(pattern[b], axis=1)  # [time]
            
            if len(total_activity) > 4 and total_activity.std() > 1e-8:
                # FFT를 이용한 주파수 분석
                fft = np.fft.fft(total_activity - np.mean(total_activity))
                power_spectrum = np.abs(fft[:len(fft)//2])
                
                if len(power_spectrum) > 1:
                    # 가장 강한 주파수 성분의 강도
                    max_power = np.max(power_spectrum[1:])  # DC 성분 제외
                    total_power = np.sum(power_spectrum[1:])
                    
                    if total_power > 1e-8:
                        oscillation_strength = max_power / total_power
                        oscillation_strengths.append(oscillation_strength)
        
        return np.mean(oscillation_strengths) if oscillation_strengths else 0.0
    
    def visualize_pattern_analysis(self, spike_pattern: torch.Tensor, 
                                 analysis_results: Dict[str, float],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """패턴 분석 결과를 시각화"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # GPU 텐서를 CPU로 이동
        if spike_pattern.is_cuda:
            spike_pattern = spike_pattern.cpu()
        
        pattern_np = spike_pattern.numpy()
        
        # 1. 래스터 플롯 (첫 번째 배치)
        if pattern_np.shape[0] > 0:
            ax = axes[0, 0]
            spike_times, neuron_ids = np.where(pattern_np[0].T > 0)
            ax.scatter(neuron_ids, spike_times, s=1, alpha=0.6)
            ax.set_xlabel('Time')
            ax.set_ylabel('Neuron ID')
            ax.set_title('Raster Plot')
            ax.grid(True, alpha=0.3)
        
        # 2. 시간축 활성화
        ax = axes[0, 1]
        temporal_activity = np.mean(pattern_np, axis=(0, 2))
        ax.plot(temporal_activity, linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Activity')
        ax.set_title(f'Temporal Activity (Var: {analysis_results.get("temporal_variance", 0):.3f})')
        ax.grid(True, alpha=0.3)
        
        # 3. 공간축 활성화
        ax = axes[0, 2]
        spatial_activity = np.mean(pattern_np, axis=(0, 1))
        ax.bar(range(len(spatial_activity)), spatial_activity, alpha=0.7)
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Average Activity')
        ax.set_title(f'Spatial Activity (Var: {analysis_results.get("spatial_variance", 0):.3f})')
        ax.grid(True, alpha=0.3)
        
        # 4. 활성화 히스토그램
        ax = axes[1, 0]
        ax.hist(pattern_np.flatten(), bins=20, alpha=0.7, density=True)
        ax.set_xlabel('Activity Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Activity Distribution (Entropy: {analysis_results.get("shannon_entropy", 0):.3f})')
        ax.grid(True, alpha=0.3)
        
        # 5. 주요 메트릭 표시
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
Pattern Analysis Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sparsity: {analysis_results.get('sparsity', 0):.3f}
Activity Level: {analysis_results.get('activity_level', 0):.3f}
Spike Rate: {analysis_results.get('spike_rate', 0):.3f}

Complexity:
Shannon Entropy: {analysis_results.get('shannon_entropy', 0):.3f}
Pattern Regularity: {analysis_results.get('pattern_regularity', 0):.3f}
Temporal Entropy: {analysis_results.get('temporal_entropy', 0):.3f}

Correlations:
Spatial Correlation: {analysis_results.get('spatial_correlation', 0):.3f}
Temporal Correlation: {analysis_results.get('temporal_correlation', 0):.3f}

Dynamics:
Temporal Variability: {analysis_results.get('temporal_variability', 0):.3f}
Pattern Stability: {analysis_results.get('pattern_stability', 0):.3f}
Oscillation Strength: {analysis_results.get('oscillation_strength', 0):.3f}
        """
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 6. 복잡도 레이더 차트
        ax = axes[1, 2]
        metrics = ['Sparsity', 'Activity', 'Temporal\nEntropy', 'Spatial\nCorrelation', 
                  'Pattern\nStability', 'Oscillation\nStrength']
        values = [
            analysis_results.get('sparsity', 0),
            analysis_results.get('activity_level', 0),
            analysis_results.get('temporal_entropy', 0) / 5.0,  # 정규화
            analysis_results.get('spatial_correlation', 0),
            analysis_results.get('pattern_stability', 0),
            analysis_results.get('oscillation_strength', 0)
        ]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 원형으로 만들기
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Pattern Characteristics Radar')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_patterns(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> Dict[str, float]:
        """두 패턴 간의 차이를 분석"""
        analysis1 = self.analyze_pattern(pattern1)
        analysis2 = self.analyze_pattern(pattern2)
        
        differences = {}
        for key in analysis1:
            if key in analysis2:
                differences[f'{key}_diff'] = abs(analysis1[key] - analysis2[key])
                differences[f'{key}_ratio'] = analysis1[key] / (analysis2[key] + 1e-8)
        
        return differences
    
    def pattern_difficulty_score(self, analysis_results: Dict[str, float]) -> float:
        """패턴의 학습 난이도 점수 계산 (0=쉬움, 1=어려움)"""
        
        # 각 특성이 학습 난이도에 미치는 영향을 가중치로 반영
        weights = {
            'sparsity': -0.3,  # 더 sparse할수록 쉬움
            'shannon_entropy': 0.4,  # 높은 엔트로피일수록 어려움
            'spatial_correlation': 0.2,  # 높은 상관관계일수록 어려움
            'temporal_correlation': 0.2,  # 높은 시간 상관관계일수록 어려움
            'temporal_variability': 0.3,  # 높은 변동성일수록 어려움
            'pattern_stability': -0.2,  # 안정적일수록 쉬움
            'oscillation_strength': 0.1  # 강한 진동일수록 어려움
        }
        
        score = 0.5  # 기본 점수
        
        for metric, weight in weights.items():
            if metric in analysis_results:
                score += weight * analysis_results[metric]
        
        # 0~1 사이로 클리핑
        return max(0.0, min(1.0, score))

def quick_pattern_analysis(spike_pattern: torch.Tensor) -> Dict[str, float]:
    """빠른 패턴 분석 (기본 메트릭만)"""
    analyzer = PatternAnalyzer()
    
    if spike_pattern.is_cuda:
        spike_pattern = spike_pattern.cpu()
    
    pattern_np = spike_pattern.numpy()
    
    results = {}
    results['sparsity'] = np.sum(pattern_np == 0) / pattern_np.size
    results['activity_level'] = np.mean(pattern_np)
    results['spike_rate'] = np.sum(pattern_np) / (pattern_np.shape[1] * pattern_np.shape[2])
    
    # 간단한 복잡도 측정
    flat_pattern = pattern_np.flatten()
    if len(np.unique(flat_pattern)) > 1:
        hist, _ = np.histogram(flat_pattern, bins=10, density=True)
        hist = hist[hist > 0]
        results['entropy'] = -np.sum(hist * np.log2(hist + 1e-8))
    else:
        results['entropy'] = 0.0
    
    return results
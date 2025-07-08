"""
다양한 가중치 초기화 전략을 구현한 유틸리티 모듈
Teacher 모델의 가중치를 다양한 방식으로 초기화하여 다양한 target pattern을 생성
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple

class WeightInitializer:
    """다양한 가중치 초기화 전략을 제공하는 클래스"""
    
    @staticmethod
    def normal_init(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
        """정규분포 초기화"""
        with torch.no_grad():
            tensor.normal_(mean=mean, std=std)
    
    @staticmethod
    def uniform_init(tensor: torch.Tensor, low: float = -1.0, high: float = 1.0):
        """균등분포 초기화"""
        with torch.no_grad():
            tensor.uniform_(low, high)
    
    @staticmethod
    def exponential_init(tensor: torch.Tensor, lambd: float = 1.0, sign_prob: float = 0.5):
        """지수분포 초기화 (양수/음수 랜덤)"""
        with torch.no_grad():
            # 지수분포에서 샘플링
            exp_values = torch.empty_like(tensor).exponential_(lambd)
            # 랜덤하게 부호 결정
            signs = torch.randint(0, 2, tensor.shape, dtype=torch.float, device=tensor.device)
            signs = 2 * signs - 1  # 0,1 -> -1,1
            # 부호 확률 적용
            random_mask = torch.rand_like(tensor) < sign_prob
            signs = torch.where(random_mask, signs, -signs)
            tensor.copy_(exp_values * signs)
    
    @staticmethod
    def binary_init(tensor: torch.Tensor, low: float = -1.0, high: float = 1.0, prob: float = 0.5):
        """이진 가중치 초기화 (두 값 중 하나)"""
        with torch.no_grad():
            binary_values = torch.where(
                torch.rand_like(tensor) < prob,
                torch.full_like(tensor, high),
                torch.full_like(tensor, low)
            )
            tensor.copy_(binary_values)
    
    @staticmethod
    def sparse_init(tensor: torch.Tensor, sparsity: float = 0.8, scale: float = 1.0):
        """Sparse 초기화 (많은 0과 일부 큰 값)"""
        with torch.no_grad():
            # 대부분을 0으로 설정
            tensor.zero_()
            # sparsity만큼 0이 아닌 값으로 설정
            mask = torch.rand_like(tensor) > sparsity
            non_zero_values = torch.randn_like(tensor) * scale
            tensor.masked_fill_(mask, 0)
            tensor.add_(mask.float() * non_zero_values)
    
    @staticmethod
    def bimodal_init(tensor: torch.Tensor, mean1: float = -1.0, std1: float = 0.3, 
                    mean2: float = 1.0, std2: float = 0.3, mix_prob: float = 0.5):
        """이중모드 정규분포 초기화"""
        with torch.no_grad():
            # 첫 번째 모드
            values1 = torch.normal(mean1, std1, tensor.shape, device=tensor.device)
            # 두 번째 모드
            values2 = torch.normal(mean2, std2, tensor.shape, device=tensor.device)
            # 믹싱
            mask = torch.rand_like(tensor) < mix_prob
            tensor.copy_(torch.where(mask, values1, values2))
    
    @staticmethod
    def scaled_kaiming_init(tensor: torch.Tensor, scale_factor: float = 1.0, mode: str = 'fan_in'):
        """스케일된 Kaiming 초기화"""
        with torch.no_grad():
            nn.init.kaiming_normal_(tensor, mode=mode)
            tensor.mul_(scale_factor)
    
    @staticmethod
    def shifted_normal_init(tensor: torch.Tensor, mean_shift: float = 0.0, base_std: float = 1.0):
        """평균이 이동된 정규분포 초기화"""
        with torch.no_grad():
            tensor.normal_(mean=mean_shift, std=base_std)

def get_init_config(init_type: str, **kwargs) -> Dict[str, Any]:
    """초기화 설정을 반환하는 헬퍼 함수"""
    
    configs = {
        # 기본 분포들
        'normal_zero': {'type': 'normal', 'mean': 0.0, 'std': 1.0},
        'normal_positive': {'type': 'normal', 'mean': 0.5, 'std': 1.0},
        'normal_negative': {'type': 'normal', 'mean': -0.5, 'std': 1.0},
        'normal_large': {'type': 'normal', 'mean': 0.0, 'std': 2.0},
        'normal_small': {'type': 'normal', 'mean': 0.0, 'std': 0.5},
        
        # 균등분포
        'uniform_standard': {'type': 'uniform', 'low': -1.0, 'high': 1.0},
        'uniform_positive': {'type': 'uniform', 'low': 0.0, 'high': 2.0},
        'uniform_large': {'type': 'uniform', 'low': -2.0, 'high': 2.0},
        
        # 특수 분포들
        'exponential': {'type': 'exponential', 'lambd': 1.0, 'sign_prob': 0.5},
        'binary_standard': {'type': 'binary', 'low': -1.0, 'high': 1.0, 'prob': 0.5},
        'binary_asymmetric': {'type': 'binary', 'low': -0.5, 'high': 2.0, 'prob': 0.3},
        'sparse_90': {'type': 'sparse', 'sparsity': 0.9, 'scale': 2.0},
        'sparse_80': {'type': 'sparse', 'sparsity': 0.8, 'scale': 1.5},
        'bimodal': {'type': 'bimodal', 'mean1': -1.0, 'std1': 0.3, 'mean2': 1.0, 'std2': 0.3},
        
        # 스케일된 Kaiming
        'kaiming_small': {'type': 'scaled_kaiming', 'scale_factor': 0.1},
        'kaiming_large': {'type': 'scaled_kaiming', 'scale_factor': 3.0},
    }
    
    if init_type in configs:
        config = configs[init_type].copy()
        config.update(kwargs)
        return config
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")

def apply_weight_init(model: nn.Module, init_config: Dict[str, Any], 
                     layer_types: Optional[list] = None) -> None:
    """
    모델에 특정 초기화를 적용
    
    Args:
        model: 초기화할 모델
        init_config: 초기화 설정
        layer_types: 초기화를 적용할 레이어 타입들 (None이면 모든 Linear 레이어)
    """
    if layer_types is None:
        layer_types = [nn.Linear]
    
    init_type = init_config['type']
    initializer = WeightInitializer()
    
    for module in model.modules():
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            if hasattr(module, 'weight') and module.weight is not None:
                if init_type == 'normal':
                    initializer.normal_init(module.weight, 
                                          init_config.get('mean', 0.0),
                                          init_config.get('std', 1.0))
                elif init_type == 'uniform':
                    initializer.uniform_init(module.weight,
                                           init_config.get('low', -1.0),
                                           init_config.get('high', 1.0))
                elif init_type == 'exponential':
                    initializer.exponential_init(module.weight,
                                               init_config.get('lambd', 1.0),
                                               init_config.get('sign_prob', 0.5))
                elif init_type == 'binary':
                    initializer.binary_init(module.weight,
                                          init_config.get('low', -1.0),
                                          init_config.get('high', 1.0),
                                          init_config.get('prob', 0.5))
                elif init_type == 'sparse':
                    initializer.sparse_init(module.weight,
                                          init_config.get('sparsity', 0.8),
                                          init_config.get('scale', 1.0))
                elif init_type == 'bimodal':
                    initializer.bimodal_init(module.weight,
                                           init_config.get('mean1', -1.0),
                                           init_config.get('std1', 0.3),
                                           init_config.get('mean2', 1.0),
                                           init_config.get('std2', 0.3),
                                           init_config.get('mix_prob', 0.5))
                elif init_type == 'scaled_kaiming':
                    initializer.scaled_kaiming_init(module.weight,
                                                  init_config.get('scale_factor', 1.0),
                                                  init_config.get('mode', 'fan_in'))
                else:
                    raise ValueError(f"Unknown initialization type: {init_type}")

def get_available_inits() -> list:
    """사용 가능한 초기화 타입들을 반환"""
    return [
        'normal_zero', 'normal_positive', 'normal_negative', 'normal_large', 'normal_small',
        'uniform_standard', 'uniform_positive', 'uniform_large',
        'exponential', 'binary_standard', 'binary_asymmetric',
        'sparse_90', 'sparse_80', 'bimodal',
        'kaiming_small', 'kaiming_large'
    ]

def create_layer_specific_config(input_init: str, recurrent_init: str, output_init: str) -> Dict[str, Dict]:
    """레이어별로 다른 초기화를 사용하는 설정을 생성"""
    return {
        'input_layer': get_init_config(input_init),
        'recurrent_layer': get_init_config(recurrent_init), 
        'output_layer': get_init_config(output_init)
    }
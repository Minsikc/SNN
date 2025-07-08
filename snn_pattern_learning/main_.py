import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.models import RSNN_eprop, Basic_RSNN_spike, Basic_RSNN_eprop_minsik, Basic_RSNN_eprop_minsik_, Basic_RSNN_eprop_forward, Basic_RSNN_eprop_analog_forward,RSNN_merged_hidden_output,RSNN_fixed_w_in
from models.loss import mse_acc_loss_over_time
from utils.metrics import van_rossum_distance, max_van_rossum_distance
from utils.kernels import create_exponential_kernel
from utils.kernel_convolution import apply_convolution
from utils.plots import plot_raster, plot_comparison_raster, plot_spike_trains
from datasets.customdatasets import CustomSpikeDataset_random, SuperSpikePatternDataset, CustomSpikeDataset_from_model
from utils.weight_init import apply_weight_init, get_init_config

import argparse
import copy
import matplotlib.pyplot as plt

def create_teacher_model(teacher_model_type, n_in, n_hidden, n_out, weight_init=None, device='cpu'):
    """Teacher 모델 생성 및 초기화"""
    print(f"Creating teacher model: {teacher_model_type}")
    
    # Teacher 모델 생성
    if teacher_model_type == "RSNN_eprop":
        teacher = Basic_RSNN_eprop_minsik_(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    elif teacher_model_type == "Basic_RSNN_spike":
        teacher = Basic_RSNN_spike(n_in=n_in, n_hidden=n_hidden, n_out=n_out, recurrent=True)
    elif teacher_model_type == "RSNN_eprop_forward":
        teacher = Basic_RSNN_eprop_forward(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    elif teacher_model_type == "RSNN_eprop_analog_forward":
        teacher = Basic_RSNN_eprop_analog_forward(n_in=n_in, n_hidden=n_hidden, n_out=n_out, recurrent=True)
    elif teacher_model_type == "RSNN_merged_hidden_output":
        teacher = RSNN_merged_hidden_output(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    elif teacher_model_type == "RSNN_fixed_w_in":
        teacher = RSNN_fixed_w_in(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    else:
        raise ValueError(f"Unsupported teacher model type: {teacher_model_type}")
    
    # Weight 초기화 적용
    if weight_init:
        print(f"Applying weight initialization: {weight_init}")
        init_config = get_init_config(weight_init)
        apply_weight_init(teacher, init_config)
    
    teacher.to(device)
    teacher.eval()  # Teacher는 추론 모드로 고정
    
    print(f"Teacher model created and initialized successfully")
    return teacher

def create_dataset_from_model(dataset_type, teacher_model=None, device='cpu', **dataset_kwargs):
    """데이터셋 생성"""
    print(f"Creating dataset: {dataset_type}")
    
    if dataset_type == "CustomSpikeDataset_random":
        dataset = CustomSpikeDataset_random(**dataset_kwargs)
    elif dataset_type == "SuperSpikePatternDataset":
        dataset = SuperSpikePatternDataset(**dataset_kwargs)
    elif dataset_type == "CustomSpikeDataset_from_model":
        if teacher_model is None:
            raise ValueError("Teacher model is required for CustomSpikeDataset_from_model")
        # CustomSpikeDataset_from_model을 위한 특별한 처리
        dataset = create_model_based_dataset(teacher_model, device, **dataset_kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    print(f"Dataset created with {len(dataset)} samples")
    return dataset

def create_model_based_dataset(teacher_model, device, num_samples=100, sequence_length=50, 
                              input_size=100, output_size=2, spike_prob=0.05):
    """Teacher 모델로부터 데이터셋 생성 (device 호환성 보장)"""
    
    # 입력 데이터 생성
    input_data = (torch.rand(num_samples, sequence_length, input_size) < spike_prob).float().to(device)
    
    # Teacher 모델로 target 생성
    teacher_model.eval()
    with torch.no_grad():
        # 모델 타입에 따라 다른 호출 방식 사용
        target_data = teacher_model(input_data)
    
    # CPU로 이동해서 dataset 생성
    input_data = input_data.cpu()
    target_data = target_data.cpu()
    
    # 간단한 dataset 클래스 생성
    class ModelBasedDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]
    
    return ModelBasedDataset(input_data, target_data)

def train(model, dataloader, loss_fn, optimizer, kernel, kernel_size, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
            outputs = model(inputs, targets, training=True)
        else:
            outputs = model(inputs)

        # 컨볼루션 적용
        conv_targets = apply_convolution(targets, kernel, kernel_size)
        conv_outputs = apply_convolution(outputs, kernel, kernel_size)

        loss = loss_fn(conv_outputs, conv_targets, outputs.shape[1])
        if hasattr(model, 'custom_grad') and model.custom_grad == True:
            if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
                pass
            else:
                err = (conv_outputs-conv_targets).permute(1,0,2)
                model.compute_grads(inputs, err)
                optimizer.step()
        
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), outputs, targets, inputs


def evaluate(model, dataloader, metric_fn, kernel, kernel_size, device):
    model.eval()
    total_metric = 0
    total_distance_per_spike = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            if hasattr(model, 'custom_grad_forward') and model.custom_grad_forward:
                outputs = model(inputs, targets, training=False)
            else:
                outputs = model(inputs)

            # 컨볼루션 적용
            targets = apply_convolution(targets, kernel, kernel_size)
            outputs = apply_convolution(outputs, kernel, kernel_size)

            # 거리 계산
            distance = metric_fn(outputs, targets, tau=10, dt=1.0)
            total_metric += distance.mean().item()

            if torch.sum(targets) != 0:
                distance_per_spike = distance / torch.sum(targets)
                total_distance_per_spike += distance_per_spike.mean().item()
                # 거리 최댓값 계산 (스파이크 하나는 0초, 다른 하나는 T초에 있을 때)
                max_distance = metric_fn(torch.zeros_like(targets), targets, tau=10, dt=1.0).mean()

                # 0~1 사이로 정규화
                normalized_distance = distance.mean() / (max_distance + 1e-6) 
            
            count += 1

    return total_metric / count, total_distance_per_spike / count, normalized_distance/count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Teacher 모델 생성 (CustomSpikeDataset_from_model용)
    teacher_model = None
    if args.dataset_type == "CustomSpikeDataset_from_model":
        teacher_model = create_teacher_model(
            teacher_model_type=args.teacher_model_type,
            n_in=args.n_in,
            n_hidden=args.n_hidden,
            n_out=args.n_out,
            weight_init=args.teacher_weight_init,
            device=device
        )

    # 데이터셋 준비
    dataset_kwargs = {
        'num_samples': args.num_samples,
        'sequence_length': args.seq_len,
        'input_size': args.n_in,
        'output_size': args.n_out,
        'spike_prob': args.spike_prob,
    }
    
    # CustomSpikeDataset_random의 경우 total_spike 추가
    if args.dataset_type == "CustomSpikeDataset_random":
        dataset_kwargs['total_spike'] = args.total_spike
    
    # SuperSpikePatternDataset의 경우 input_spike_prob 사용
    if args.dataset_type == "SuperSpikePatternDataset":
        dataset_kwargs.pop('spike_prob')  # spike_prob 제거
        dataset_kwargs['input_spike_prob'] = args.spike_prob
    
    train_dataset = create_dataset_from_model(
        dataset_type=args.dataset_type,
        teacher_model=teacher_model,
        device=device,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Student 모델 초기화
    print(f"Creating student model: {args.model_type}")
    if args.model_type == "RSNN_eprop":
        model = Basic_RSNN_eprop_minsik_(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
    elif args.model_type == "Basic_RSNN_spike":
        model = Basic_RSNN_spike(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
    elif args.model_type == "RSNN_eprop_forward":
        model = Basic_RSNN_eprop_forward(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
    elif args.model_type == "RSNN_eprop_analog_forward":
        model = Basic_RSNN_eprop_analog_forward(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out, recurrent=True)
    elif args.model_type == "RSNN_merged_hidden_output":
        model = RSNN_merged_hidden_output(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
    elif args.model_type == "RSNN_fixed_w_in":
        model = RSNN_fixed_w_in(n_in=args.n_in, n_hidden=args.n_hidden, n_out=args.n_out)
    else:
        raise ValueError("Unsupported model type")

    model.to(device)

    # 손실 함수 및 옵티마이저 설정
    loss_fn = mse_acc_loss_over_time
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 커널 생성
    kernel = create_exponential_kernel(args.kernel_size, args.decay_rate).to(device)

    # 실험 정보 출력
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Dataset Type: {args.dataset_type}")
    if args.dataset_type == "CustomSpikeDataset_from_model":
        print(f"Teacher Model: {args.teacher_model_type}")
        print(f"Teacher Weight Init: {args.teacher_weight_init}")
    print(f"Student Model: {args.model_type}")
    print(f"Dataset Size: {len(train_dataset)} samples")
    print(f"Network Architecture: {args.n_in} -> {args.n_hidden} -> {args.n_out}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("="*60)

    # 학습 및 평가
    best_loss = float("inf")
    best_model_state = None
    final_outputs, final_targets = None, None  # 시각화를 위한 최종 결과 저장

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, outputs, targets, inputs = train(model, train_loader, loss_fn, optimizer, kernel, args.kernel_size, device)
        val_metric, val_metric_per_spike, normalized_distance_per_spike = evaluate(model, train_loader, van_rossum_distance, kernel, args.kernel_size, device)

        if train_loss < best_loss:
            best_loss = train_loss
            best_model_state = copy.deepcopy(model.state_dict())
            final_outputs, final_targets = outputs, targets  # 현재 가장 좋은 결과 저장

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Metric: {val_metric:.4f}, "
                  f"Per Spike: {val_metric_per_spike:.4f}, Normalized: {normalized_distance_per_spike:.4f}")

    # 최적 모델 저장
    if best_model_state is not None:
        model_filename = f"best_model_{args.dataset_type}_{args.model_type}.pth"
        if args.dataset_type == "CustomSpikeDataset_from_model":
            model_filename = f"best_model_{args.dataset_type}_{args.teacher_model_type}_{args.model_type}.pth"
        torch.save(best_model_state, model_filename)
        print(f"\nBest model saved as: {model_filename}")

    # 학습 결과 시각화
    if final_outputs is not None and final_targets is not None:
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # 1️⃣ 타겟 vs 출력 비교 (레이스터 플롯)
        plot_comparison_raster(final_targets, final_outputs, ax=axes[0], title="Target vs Output Spikes")
        distance = van_rossum_distance(final_outputs, final_targets, tau=10, dt=1.0)
        max_distance = van_rossum_distance(torch.zeros_like(final_targets), final_targets, tau=10, dt=1.0).mean()
        normalized_distance = distance.mean() / (max_distance + 1e-6)  # 정규화된 거리 계산
        
        # 2️⃣ 모델의 스파이크 패턴
        plot_raster(inputs, ax=axes[1], title="Input Spikes")
        plot_raster(final_outputs, ax=axes[2], title="Model Output Spikes")

        plt.suptitle(f"Results: {args.dataset_type} -> {args.model_type}")
        plt.tight_layout()
        
        plot_filename = f"results_{args.dataset_type}_{args.model_type}.png"
        if args.dataset_type == "CustomSpikeDataset_from_model":
            plot_filename = f"results_{args.dataset_type}_{args.teacher_model_type}_{args.model_type}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as: {plot_filename}")
        print(f"Final Normalized Distance: {normalized_distance.item():.4f}")

        # 3️⃣ 스파이크 강도 비교
        plot_spike_trains(final_outputs, final_targets, epoch=args.epochs)

    print(f"\n✅ Training completed!")
    print(f"Final loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Training with Various Datasets and Models")

    # 데이터셋 선택
    parser.add_argument("--dataset_type", type=str, 
                       choices=["CustomSpikeDataset_random", "SuperSpikePatternDataset", "CustomSpikeDataset_from_model"], 
                       default="CustomSpikeDataset_random",
                       help="Dataset type to use for training")

    # Teacher 모델 설정 (CustomSpikeDataset_from_model용)
    parser.add_argument("--teacher_model_type", type=str, 
                       choices=["RSNN_eprop", "Basic_RSNN_spike", "RSNN_eprop_forward", 
                               "RSNN_eprop_analog_forward", "RSNN_merged_hidden_output", "RSNN_fixed_w_in"], 
                       default="Basic_RSNN_spike",
                       help="Teacher model type for dataset generation")
    
    parser.add_argument("--teacher_weight_init", type=str, default="uniform_positive",
                       help="Weight initialization method for teacher model")

    # Student 모델 파라미터
    parser.add_argument("--model_type", type=str, 
                       choices=["RSNN_eprop", "Basic_RSNN_spike", "RSNN_eprop_forward", 
                               "RSNN_eprop_analog_forward", "RSNN_merged_hidden_output", "RSNN_fixed_w_in"], 
                       default="Basic_RSNN_spike",
                       help="Student model type")
    
    parser.add_argument("--n_in", type=int, default=50)
    parser.add_argument("--n_hidden", type=int, default=40)
    parser.add_argument("--n_out", type=int, default=10)

    # 데이터셋 파라미터
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--spike_prob", type=float, default=0.1)
    parser.add_argument("--total_spike", type=int, default=5)

    # 학습 파라미터
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)

    # 커널 파라미터
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--decay_rate", type=float, default=2.0)

    args = parser.parse_args()
    
    # CustomSpikeDataset_from_model 사용 시 필수 파라미터 검증
    if args.dataset_type == "CustomSpikeDataset_from_model":
        if not args.teacher_model_type:
            parser.error("--teacher_model_type is required when using CustomSpikeDataset_from_model")
    
    main(args)
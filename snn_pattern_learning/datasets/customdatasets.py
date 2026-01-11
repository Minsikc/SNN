from torch.utils.data import Dataset
import torch
import numpy as np

try:
    import tonic
    import tonic.transforms as transforms
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False

class CustomSpikeDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=50, input_size=100, output_size=2, spike_prob=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.spike_prob = spike_prob
        period_range = (3,15)

        # 스파이크 확률을 기반으로 0 또는 1의 값을 갖는 스파이크 데이터 생성
        self.data = (torch.rand(num_samples, sequence_length, input_size) < 0.05).float()
        # 출력 데이터를 위한 빈 텐서 생성
        self.targets = torch.zeros(num_samples, sequence_length, output_size).float()
        
        for output_neuron in range(output_size):
            # 각 출력 뉴런마다 주기 결정: 주어진 범위 내에서 랜덤하게 선택
            period = torch.randint(low=period_range[0], high=period_range[1] + 1, size=(1,)).item()
            
            # 설정된 주기에 따라 스파이크 생성
            spike_times = torch.arange(0, sequence_length, period)
            for time in spike_times:
                self.targets[:, time, output_neuron] = 1.0     
            # 각 출력 뉴런의 첫 번째 스파이크 삭제
            if len(spike_times) > 0:  # 스파이크 시간 배열에 요소가 있는 경우에만
                first_spike_time = spike_times[0]  # 첫 번째 스파이크 시간
                self.targets[:, first_spike_time, output_neuron] = 0.0  # 첫 번째 스파이크 삭제


    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
class CustomSpikeDataset_random(Dataset):
    def __init__(self, num_samples=1000, sequence_length=50, input_size=100, output_size=2, spike_prob=0.02, total_spike=10):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.spike_prob = spike_prob

        # 스파이크 확률을 기반으로 0 또는 1의 값을 갖는 스파이크 데이터 생성
        self.data = (torch.rand(num_samples, sequence_length, input_size) < 0.05).float()

        # 출력 데이터를 위한 빈 텐서 생성 (전체적으로 0으로 초기화)
        self.targets = torch.zeros(num_samples, sequence_length, output_size)

        # 각 샘플에 대해 total_spike 개수만큼 랜덤하게 스파이크를 생성
        for i in range(num_samples):
            for j in range(output_size):
                # sequence_length 내에서 total_spike개의 랜덤한 시간 인덱스를 선택
                spike_times = torch.randperm(sequence_length)[:total_spike]
                # 해당 인덱스에서 스파이크 발생 (1로 설정)
                self.targets[i, spike_times, j] = 1.0


    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class CustomSpikeDataset_Probabilistic(Dataset):
    """
    모든 타임스텝에서 주어진 확률(spike_prob)에 따라 
    랜덤하게 스파이크를 생성하는 데이터셋 클래스입니다.

    Args:
        num_samples (int): 생성할 샘플의 총 개수
        sequence_length (int): 각 샘플의 시퀀스 길이 (타임스텝 수)
        input_size (int): 입력 데이터의 특성(feature) 차원
        output_size (int): 타겟 데이터의 특성(feature) 차원
        input_spike_prob (float): 입력 데이터(data)에서 스파이크가 발생할 확률
        target_spike_prob (float): 타겟 데이터(targets)에서 스파이크가 발생할 확률
    """
    def __init__(self, num_samples=1000, sequence_length=50, input_size=100, output_size=2, input_spike_prob=0.05, target_spike_prob=0.02):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.input_spike_prob = input_spike_prob
        self.target_spike_prob = target_spike_prob

        # 입력 데이터 생성: 각 타임스텝에서 input_spike_prob 확률로 스파이크(1) 발생
        self.data = (torch.rand(num_samples, sequence_length, input_size) < self.input_spike_prob).float()

        # 타겟 데이터 생성: 각 타임스텝에서 target_spike_prob 확률로 스파이크(1) 발생
        # 이 부분이 요청하신 "완전 랜덤하게 모든 타임스텝에 대해서 spike_prob 확률로 spike가 만들어지도록" 하는 핵심 로직입니다.
        self.targets = (torch.rand(num_samples, sequence_length, output_size) < self.target_spike_prob).float()
        #self.targets=self.data.clone()  # 입력 데이터와 동일한 구조로 초기화
        
    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return self.num_samples

    def __getitem__(self, idx):
        """주어진 인덱스(idx)에 해당하는 샘플(데이터와 타겟)을 반환합니다."""
        return self.data[idx], self.targets[idx]
    
class pattern_generation_Dataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=1000, input_size=100, output_size=1, spike_prob=0.05, frequency=1):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.frequency = frequency  # 주기 함수의 주파수
        
        # 입력 데이터 생성
        self.data = (torch.rand(num_samples, sequence_length, input_size) < spike_prob).float()
        

        # 사인 함수를 사용한 타겟 데이터 생성
        time_steps = torch.linspace(0, 2 * np.pi, sequence_length)
        self.targets = torch.sin(frequency * time_steps).unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, output_size)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

class CustomSpikeDataset_from_model(Dataset):
    def __init__(self, model, num_samples=100, sequence_length=50, input_size=100, output_size=2, spike_prob=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.spike_prob = spike_prob

        # 스파이크 확률을 기반으로 0 또는 1의 값을 갖는 스파이크 데이터 생성
        self.data = (torch.rand(num_samples, sequence_length, input_size) < spike_prob).float()
        
        # Ensure the model is in evaluation mode to disable dropout, batchnorm updates etc.
        model.eval()
        with torch.no_grad():  # Ensure gradients are not calculated to save memory and computations
            # 모델을 사용하여 타겟 텐서 생성
            self.targets = model(self.data)

        # Ensure targets tensor is correctly shaped and of the correct dtype
        # If necessary, add additional transformations to ensure its shape is (num_samples, sequence_length, output_size)
        assert self.targets.shape == (num_samples, sequence_length, output_size), "The output shape of the model does not match the expected target shape."

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class pattern_generation_Dataset_phase(Dataset):
    def __init__(self, num_samples=1000, sequence_length=1000, input_size=100, output_size=1, spike_prob=0.05, frequency=1):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.frequency = frequency  # 주기 함수의 주파수
        
        # 입력 데이터 생성
        self.data = (torch.rand(num_samples, sequence_length, input_size) < spike_prob).float()
        
        # 사인 함수를 사용한 타겟 데이터 생성
        time_steps = torch.linspace(0, 2 * np.pi, sequence_length)
        #phase_offsets = torch.rand(output_size) * 2 * np.pi  # 출력마다 랜덤한 위상 # 출력마다 다른 위상
        self.targets = torch.sin(frequency * time_steps.unsqueeze(-1)).unsqueeze(0).repeat(num_samples, 1, 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# --- SuperSpike 논문 패턴 생성을 위한 새로운 데이터셋 클래스 ---
class SuperSpikePatternDataset(Dataset):
    """
    SuperSpike 논문의 패턴 생성 실험(Fig 6)과 유사한 데이터셋 클래스.
    - 입력: 고정된(frozen) 푸아송 노이즈
    - 목표: 리사주 곡선을 이용한 복잡한 시공간적 패턴
    """
    def __init__(self, num_samples=1, sequence_length=3500, input_size=100, output_size=100, input_spike_prob=0.02):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.input_spike_prob = input_spike_prob

        # 1. 고정된 푸아송 노이즈 입력 생성
        # 모든 샘플이 동일한 "frozen" 노이즈를 사용하도록 __init__에서 한 번만 생성합니다.
        self.data = (torch.rand(sequence_length, input_size) < self.input_spike_prob).float()

        # 2. 복잡한 목표 스파이크 패턴 생성 (리사주 곡선)
        # num_samples 만큼의 다양한 목표 패턴을 생성할 수 있습니다.
        self.targets = torch.zeros(num_samples, sequence_length, output_size)
        
        for i in range(num_samples):
            # 매 샘플마다 다른 리사주 곡선 파라미터를 사용하여 다양한 패턴 생성
            time = torch.linspace(0, 1, sequence_length)
            a = 3 # 주파수 비율 x (1~4)
            b = 3 # 주파수 비율 y (1~4)
            delta = torch.rand(1).item() * np.pi # 위상차
            
            # y축(뉴런 인덱스)을 시간에 따라 계산
            neuron_indices = (output_size / 2.5 * (1 + torch.sin(2 * np.pi * a * time + delta)) + output_size / 5).long()
            
            # 패턴에 두께를 주기 위해, 중심 뉴런 주변으로 가우시안 확률에 따라 스파이크 생성
            for t, center_neuron_idx in enumerate(neuron_indices):
                neuron_axis = torch.arange(output_size)
                distance = torch.abs(neuron_axis - center_neuron_idx)
                spike_prob = 0.8 * torch.exp(-torch.pow(distance, 2) / (2 * 2**2)) # 표준편차=2
                self.targets[i, t, :] = torch.bernoulli(spike_prob)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 입력 데이터는 항상 동일한 고정된 패턴을 반환하고,
        # 목표 데이터는 해당 인덱스의 패턴을 반환합니다.
        return self.data, self.targets[idx]


# --- N-MNIST Dataset ---
# Two implementations: one using tonic (if available), one standalone

class NMNISTDataset(Dataset):
    """
    N-MNIST (Neuromorphic MNIST) dataset wrapper.

    Uses tonic library if available, otherwise provides instructions for manual setup.

    Args:
        root (str): Root directory for dataset storage
        train (bool): If True, use training set, else test set
        time_window (float): Time window in microseconds for binning events (default: 1000 = 1ms)
        num_time_bins (int): Number of time bins for the output tensor
        flatten (bool): If True, flatten spatial dimensions (34x34x2 -> 2312)
        max_samples (int): Maximum number of samples to use (None for all)
        transform: Additional transforms to apply
    """

    def __init__(
        self,
        root: str = './data/nmnist',
        train: bool = True,
        time_window: float = 1000.0,
        num_time_bins: int = 300,
        flatten: bool = True,
        max_samples: int = None,
        transform = None
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.time_window = time_window
        self.num_time_bins = num_time_bins
        self.flatten = flatten
        self.max_samples = max_samples
        self.additional_transform = transform
        self.sensor_size = (34, 34, 2)

        if flatten:
            self.input_size = 34 * 34 * 2  # 2312
        else:
            self.input_size = (2, 34, 34)

        if TONIC_AVAILABLE:
            self._init_with_tonic()
        else:
            self._init_standalone()

    def _init_with_tonic(self):
        """Initialize using tonic library."""
        frame_transform = transforms.ToFrame(
            sensor_size=self.sensor_size,
            time_window=self.time_window
        )
        self.dataset = tonic.datasets.NMNIST(
            save_to=self.root,
            train=self.train,
            transform=frame_transform
        )
        self.use_tonic = True

    def _init_standalone(self):
        """Initialize without tonic - uses pre-processed data."""
        import os

        # Check for pre-processed data
        split = 'train' if self.train else 'test'
        data_file = os.path.join(self.root, f'nmnist_{split}_frames.pt')
        label_file = os.path.join(self.root, f'nmnist_{split}_labels.pt')

        if os.path.exists(data_file) and os.path.exists(label_file):
            self.frames_data = torch.load(data_file)
            self.labels_data = torch.load(label_file)
            self.use_tonic = False
        else:
            raise ImportError(
                f"tonic library is not available and pre-processed data not found.\n"
                f"Please either:\n"
                f"1. Install tonic: pip install tonic\n"
                f"2. Or provide pre-processed data at:\n"
                f"   - {data_file}\n"
                f"   - {label_file}"
            )

    def __len__(self):
        if self.use_tonic:
            length = len(self.dataset)
        else:
            length = len(self.labels_data)

        if self.max_samples is not None:
            return min(self.max_samples, length)
        return length

    def __getitem__(self, idx):
        if self.use_tonic:
            frames, label = self.dataset[idx]
        else:
            frames = self.frames_data[idx].numpy()
            label = self.labels_data[idx].item()

        # Pad or truncate to num_time_bins
        num_frames = frames.shape[0]

        if num_frames < self.num_time_bins:
            padding = np.zeros((self.num_time_bins - num_frames, *frames.shape[1:]))
            frames = np.concatenate([frames, padding], axis=0)
        elif num_frames > self.num_time_bins:
            frames = frames[:self.num_time_bins]

        data = torch.from_numpy(frames).float()

        if self.flatten:
            data = data.view(self.num_time_bins, -1)

        data = (data > 0).float()

        if self.additional_transform is not None:
            data = self.additional_transform(data)

        # One-hot target at last timestep
        num_classes = 10
        target = torch.zeros(self.num_time_bins, num_classes)
        target[-1, label] = 1.0

        return data, target

    def get_class_label(self, idx):
        """Get the integer class label for a sample."""
        if self.use_tonic:
            _, label = self.dataset[idx]
        else:
            label = self.labels_data[idx].item()
        return label


class NMNISTDatasetClassification(Dataset):
    """
    N-MNIST dataset for classification tasks.

    Returns integer labels instead of one-hot encoded targets,
    suitable for use with CrossEntropyLoss.

    Args:
        root (str): Root directory for dataset storage
        train (bool): If True, use training set, else test set
        time_window (float): Time window in microseconds for binning
        num_time_bins (int): Number of time bins
        flatten (bool): Flatten spatial dimensions
        max_samples (int): Maximum samples
        shuffle (bool): If True, shuffle indices to get balanced class distribution
    """

    def __init__(
        self,
        root: str = './data/nmnist',
        train: bool = True,
        time_window: float = 1000.0,
        num_time_bins: int = 300,
        flatten: bool = True,
        max_samples: int = None,
        transform = None,
        shuffle: bool = True
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.time_window = time_window
        self.num_time_bins = num_time_bins
        self.flatten = flatten
        self.max_samples = max_samples
        self.additional_transform = transform
        self.sensor_size = (34, 34, 2)
        self.shuffle = shuffle

        if flatten:
            self.input_size = 34 * 34 * 2  # 2312
        else:
            self.input_size = (2, 34, 34)

        if TONIC_AVAILABLE:
            self._init_with_tonic()
        else:
            self._init_standalone()

        # Create shuffled indices for balanced class sampling
        self._create_indices()

    def _init_with_tonic(self):
        """Initialize using tonic library."""
        frame_transform = transforms.ToFrame(
            sensor_size=self.sensor_size,
            time_window=self.time_window
        )
        self.dataset = tonic.datasets.NMNIST(
            save_to=self.root,
            train=self.train,
            transform=frame_transform
        )
        self.use_tonic = True

    def _init_standalone(self):
        """Initialize without tonic - uses pre-processed data."""
        import os

        split = 'train' if self.train else 'test'
        data_file = os.path.join(self.root, f'nmnist_{split}_frames.pt')
        label_file = os.path.join(self.root, f'nmnist_{split}_labels.pt')

        if os.path.exists(data_file) and os.path.exists(label_file):
            self.frames_data = torch.load(data_file)
            self.labels_data = torch.load(label_file)
            self.use_tonic = False
        else:
            raise ImportError(
                f"tonic library is not available and pre-processed data not found.\n"
                f"Please either:\n"
                f"1. Install tonic: pip install tonic\n"
                f"2. Or provide pre-processed data at:\n"
                f"   - {data_file}\n"
                f"   - {label_file}"
            )

    def _create_indices(self):
        """Create shuffled indices for balanced class sampling."""
        if self.use_tonic:
            total_len = len(self.dataset)
        else:
            total_len = len(self.labels_data)

        if self.shuffle:
            # Create shuffled indices with fixed seed for reproducibility
            rng = np.random.RandomState(42)
            self.indices = rng.permutation(total_len)
        else:
            self.indices = np.arange(total_len)

        # Apply max_samples limit after shuffling
        if self.max_samples is not None:
            self.indices = self.indices[:self.max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map to shuffled index
        real_idx = self.indices[idx]

        if self.use_tonic:
            frames, label = self.dataset[real_idx]
        else:
            frames = self.frames_data[real_idx].numpy()
            label = self.labels_data[real_idx].item()

        # Pad or truncate to num_time_bins
        num_frames = frames.shape[0]

        if num_frames < self.num_time_bins:
            padding = np.zeros((self.num_time_bins - num_frames, *frames.shape[1:]))
            frames = np.concatenate([frames, padding], axis=0)
        elif num_frames > self.num_time_bins:
            frames = frames[:self.num_time_bins]

        data = torch.from_numpy(frames).float()

        if self.flatten:
            data = data.view(self.num_time_bins, -1)

        # Binarize to spikes
        data = (data > 0).float()

        if self.additional_transform is not None:
            data = self.additional_transform(data)

        # Return integer label for classification
        return data, label


class NMNISTDatasetRateEncoded(Dataset):
    """
    N-MNIST dataset with rate-based target encoding.

    Instead of a single spike at the last timestep, this version
    produces a sustained output for the correct class over the
    last portion of the sequence.

    Args:
        root (str): Root directory for dataset storage
        train (bool): If True, use training set, else test set
        time_window (float): Time window in microseconds for binning
        num_time_bins (int): Number of time bins
        flatten (bool): Flatten spatial dimensions
        max_samples (int): Maximum samples
        output_duration (int): Number of timesteps at the end to have output
        output_spike_prob (float): Probability of spike in output window
    """

    def __init__(
        self,
        root: str = './data/nmnist',
        train: bool = True,
        time_window: float = 1000.0,
        num_time_bins: int = 300,
        flatten: bool = True,
        max_samples: int = None,
        output_duration: int = 50,
        output_spike_prob: float = 0.1
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.time_window = time_window
        self.num_time_bins = num_time_bins
        self.flatten = flatten
        self.max_samples = max_samples
        self.output_duration = output_duration
        self.output_spike_prob = output_spike_prob
        self.sensor_size = (34, 34, 2)

        if flatten:
            self.input_size = 34 * 34 * 2
        else:
            self.input_size = (2, 34, 34)

        if TONIC_AVAILABLE:
            self._init_with_tonic()
        else:
            self._init_standalone()

    def _init_with_tonic(self):
        """Initialize using tonic library."""
        frame_transform = transforms.ToFrame(
            sensor_size=self.sensor_size,
            time_window=self.time_window
        )
        self.dataset = tonic.datasets.NMNIST(
            save_to=self.root,
            train=self.train,
            transform=frame_transform
        )
        self.use_tonic = True

    def _init_standalone(self):
        """Initialize without tonic - uses pre-processed data."""
        import os

        split = 'train' if self.train else 'test'
        data_file = os.path.join(self.root, f'nmnist_{split}_frames.pt')
        label_file = os.path.join(self.root, f'nmnist_{split}_labels.pt')

        if os.path.exists(data_file) and os.path.exists(label_file):
            self.frames_data = torch.load(data_file)
            self.labels_data = torch.load(label_file)
            self.use_tonic = False
        else:
            raise ImportError(
                f"tonic library is not available and pre-processed data not found.\n"
                f"Please either:\n"
                f"1. Install tonic: pip install tonic\n"
                f"2. Or provide pre-processed data at:\n"
                f"   - {data_file}\n"
                f"   - {label_file}"
            )

    def __len__(self):
        if self.use_tonic:
            length = len(self.dataset)
        else:
            length = len(self.labels_data)

        if self.max_samples is not None:
            return min(self.max_samples, length)
        return length

    def __getitem__(self, idx):
        if self.use_tonic:
            frames, label = self.dataset[idx]
        else:
            frames = self.frames_data[idx].numpy()
            label = self.labels_data[idx].item()

        num_frames = frames.shape[0]

        if num_frames < self.num_time_bins:
            padding = np.zeros((self.num_time_bins - num_frames, *frames.shape[1:]))
            frames = np.concatenate([frames, padding], axis=0)
        elif num_frames > self.num_time_bins:
            frames = frames[:self.num_time_bins]

        data = torch.from_numpy(frames).float()

        if self.flatten:
            data = data.view(self.num_time_bins, -1)

        data = (data > 0).float()

        # Rate-encoded target
        num_classes = 10
        target = torch.zeros(self.num_time_bins, num_classes)

        start_idx = self.num_time_bins - self.output_duration
        for t in range(start_idx, self.num_time_bins):
            if torch.rand(1).item() < self.output_spike_prob:
                target[t, label] = 1.0

        return data, target
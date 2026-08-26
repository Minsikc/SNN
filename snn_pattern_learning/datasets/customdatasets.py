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

class TemporalXORDataset(Dataset):
    """Temporal XOR: two bits presented sequentially, XOR answered on a go cue.

    Timeline (default T=20):
        steps  1-3   bit A window
        steps  9-11  bit B window   (gap of 5 silent steps after A: membrane
                                     tau=0.6 decays to ~0.08 over the gap, so
                                     the network must hold A in recurrent
                                     activity -- a feedforward net cannot)
        steps 14-19  go cue + response window

    Input coding (n_in = 10):
        ch 0-3  fire every step of a window whose bit is 1
        ch 4-7  fire every step of a window whose bit is 0
        ch 8-9  go cue, fire every step of the response window

    Target coding (n_out = 5), spikes only inside the response window:
        XOR = 0  ->  output neurons 0,1 fire every response step
        XOR = 1  ->  output neurons 3,4 fire every response step
        neuron 2 is always silent.

    Why XOR needs the hidden layer: over the whole sequence the "1"-channels
    fire in 0, 1 or 2 windows and the class is (count == 1) -- non-monotonic
    in the input rates, hence not linearly separable from them.

    Exactly 4 deterministic samples: (0,0), (0,1), (1,0), (1,1).
    """

    def __init__(self, sequence_length=20, input_size=10, output_size=5,
                 a_window=(1, 4), b_window=(9, 12), response_window=(14, 20),
                 seed=0):
        super().__init__()
        assert input_size >= 10 and output_size >= 5
        self.num_samples = 4
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.response_window = response_window

        bits = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.labels = torch.tensor([a ^ b for a, b in bits])

        self.data = torch.zeros(4, sequence_length, input_size)
        self.targets = torch.zeros(4, sequence_length, output_size)

        for i, (a, b) in enumerate(bits):
            for bit, (t0, t1) in ((a, a_window), (b, b_window)):
                chans = slice(0, 4) if bit == 1 else slice(4, 8)
                self.data[i, t0:t1, chans] = 1.0
            r0, r1 = response_window
            self.data[i, r0:r1, 8:10] = 1.0  # go cue
            out_group = (0, 1) if (a ^ b) == 0 else (3, 4)
            for n in out_group:
                self.targets[i, r0:r1, n] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @staticmethod
    def decision(outputs, response_window):
        """Classify from output spikes: sum group {0,1} vs {3,4} in window.

        Args:
            outputs: (batch, time, n_out) spike tensor
            response_window: (r0, r1)
        Returns:
            (batch,) predicted class tensor; ties predict -1 (always wrong).
        """
        r0, r1 = response_window
        win = outputs[:, r0:r1, :]
        g0 = win[:, :, 0:2].sum(dim=(1, 2))
        g1 = win[:, :, 3:5].sum(dim=(1, 2))
        pred = torch.where(g1 > g0, torch.ones_like(g0, dtype=torch.long),
                           torch.zeros_like(g0, dtype=torch.long))
        pred[g0 == g1] = -1
        return pred


class CustomSpikeDataset_Teacher(Dataset):
    """Targets generated by a teacher SNN driven by the same input.

    Why this exists: CustomSpikeDataset_random draws target spike times with
    torch.randperm, so the target is statistically independent of the input.
    No weight setting can reproduce it, and the loss floors well above zero --
    measured 2026-08-06, a BPTT upper-bound run reached 0.579 on random
    targets vs 0.013 on teacher targets with the identical model and loss.
    That floor hides whatever the analog hardware is actually doing.

    Here a frozen teacher network of the SAME architecture as the student maps
    input -> target, so a perfect solution provably exists (the teacher's own
    weights). This mirrors the teacher-student setup in snn-spikegen-win.

    Args:
        num_samples, sequence_length, input_size, output_size: as usual.
        spike_prob: input spike probability.
        teacher_thresh / teacher_tau: teacher LIF params. Keep them equal to
            the student's config so the task stays realizable.
        seed: fixes both the input draw and the teacher weights, so runs are
            comparable across conditions (the random dataset was unseeded,
            which made HW/SW comparisons non-sample-matched).
    """

    def __init__(self, num_samples=5, sequence_length=20, input_size=10,
                 output_size=5, spike_prob=0.2, hidden_size=5,
                 teacher_thresh=0.2, teacher_tau=0.6, w_scale=1.0, seed=0,
                 beta=0.0, rho=0.0):
        # beta/rho: adaptive-threshold (ALIF) teacher, A_t = thresh + beta*a_t,
        # a_{t+1} = rho*a_t + z_t (Bellec 2020). beta=0 is the plain LIF
        # teacher and is bit-identical to the pre-2026-08-26 behaviour.
        super().__init__()
        import torch as _torch
        self.beta, self.rho = float(beta), float(rho)

        g = _torch.Generator().manual_seed(seed)
        self.data = (_torch.rand(num_samples, sequence_length, input_size,
                                 generator=g) < spike_prob).float()

        # Teacher LIF is rolled out explicitly rather than reusing a model
        # class: Basic_RSNN_eprop_forward stores init_thresh only for the
        # surrogate gradient and builds LIF_Node() with its default threshold
        # of 0.5, so a teacher built from it silently ignores teacher_thresh.
        # Doing it here keeps the threshold (and hence target density)
        # controllable, which is what makes the task tunable.
        _torch.manual_seed(seed)
        w_in = _torch.randn(input_size, hidden_size, generator=g) / (input_size ** 0.5)
        w_rec = _torch.randn(hidden_size, hidden_size, generator=g) / (hidden_size ** 0.5)
        w_out = _torch.randn(hidden_size, output_size, generator=g) / (hidden_size ** 0.5)
        w_in *= w_scale
        w_rec *= w_scale
        w_out *= w_scale

        v = _torch.zeros(num_samples, hidden_size)
        z = _torch.zeros(num_samples, hidden_size)
        a = _torch.zeros(num_samples, hidden_size)      # ALIF adaptation (unused if beta == 0)
        vo = _torch.zeros(num_samples, output_size)
        outs = []
        with _torch.no_grad():
            for t in range(sequence_length):
                I = self.data[:, t, :] @ w_in + z @ w_rec
                v = teacher_tau * v * (1 - z) + I
                if self.beta != 0.0:
                    A = teacher_thresh + self.beta * a
                    z = (v > A).float()
                    a = self.rho * a + z
                else:
                    z = (v > teacher_thresh).float()
                vo = teacher_tau * vo * (1 - (vo > teacher_thresh).float()) + z @ w_out
                outs.append((vo > teacher_thresh).float())
        self.targets = _torch.stack(outs, 1)

        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.teacher_weights = {'w_in': w_in, 'w_rec': w_rec, 'w_out': w_out}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

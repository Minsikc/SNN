import torch
import torch.nn.functional as F

def apply_convolution(spike_train, kernel, kernel_size):
    """
    Apply 1D convolution to spike train.

    Args:
        spike_train: Tensor of shape [batch, time, features]
        kernel: Convolution kernel of shape [1, 1, kernel_size]
        kernel_size: Size of the kernel

    Returns:
        Smoothed spike train of shape [batch, time, features]
    """
    batch_size, time_steps, n_features = spike_train.shape

    # Reshape to [batch * features, 1, time] for conv1d
    # This applies the same kernel independently to each feature
    spike_train_reshaped = spike_train.permute(0, 2, 1).reshape(batch_size * n_features, 1, time_steps)

    # Pad and apply convolution
    padded = F.pad(spike_train_reshaped, (kernel_size - 1, 0), 'constant', 0)
    smoothed = F.conv1d(padded, kernel)

    # Reshape back to [batch, time, features]
    smoothed_out = smoothed.reshape(batch_size, n_features, time_steps).permute(0, 2, 1)
    return smoothed_out


def apply_convolution_gaussian(spike_train, kernel, kernel_size):
    """
    Apply Gaussian convolution to spike train.

    Args:
        spike_train: Tensor of shape [batch, time, features]
        kernel: Convolution kernel of shape [1, 1, kernel_size]
        kernel_size: Size of the kernel

    Returns:
        Smoothed spike train of shape [batch, time, features]
    """
    batch_size, time_steps, n_features = spike_train.shape

    # Reshape to [batch * features, 1, time] for conv1d
    spike_train_reshaped = spike_train.permute(0, 2, 1).reshape(batch_size * n_features, 1, time_steps)

    # Symmetric padding for Gaussian
    pad_left = int((kernel_size - 1) / 2)
    pad_right = int((kernel_size - 1) / 2)
    padded = F.pad(spike_train_reshaped, (pad_left, pad_right), 'constant', 0)
    smoothed = F.conv1d(padded, kernel)

    # Reshape back to [batch, time, features]
    smoothed_out = smoothed.reshape(batch_size, n_features, time_steps).permute(0, 2, 1)
    return smoothed_out

def apply_convolution_with_ones_kernel(spike_train, kernel, kernel_size):
    """
    Applies a ones kernel convolution to a spike train.
    Args:
        spike_train (Tensor): Input spike train of shape [batch, time, features].
        kernel (Tensor): Ones kernel of shape [1, 1, kernel_size].
        kernel_size (int): Size of the kernel.
    Returns:
        Tensor: Smoothed spike train of shape [batch, time, features].
    """
    batch_size, time_steps, n_features = spike_train.shape

    # Reshape to [batch * features, 1, time] for conv1d
    spike_train_reshaped = spike_train.permute(0, 2, 1).reshape(batch_size * n_features, 1, time_steps)

    # Pad and apply convolution
    padded = F.pad(spike_train_reshaped, (kernel_size - 1, 0), 'constant', 0)
    smoothed = F.conv1d(padded, kernel)

    # Reshape back to [batch, time, features]
    smoothed_out = smoothed.reshape(batch_size, n_features, time_steps).permute(0, 2, 1)
    return smoothed_out

import torch.nn.functional as F

def apply_convolution(spike_train, kernel, kernel_size):
    spike_train_permuted = spike_train.permute(2,0,1)
    padded_spike_train = F.pad(spike_train_permuted, (kernel_size - 1, 0), 'constant', 0)
    smoothed_spike_train = F.conv1d(padded_spike_train, kernel)
    smoothed_spike_train_out = smoothed_spike_train.permute(1,2,0)
    return smoothed_spike_train_out


def apply_convolution_gaussian(spike_train, kernel, kernel_size):
    spike_train_permuted = spike_train.permute(2,0,1)
    padded_spike_train = F.pad(spike_train_permuted, (int((kernel_size - 1)/2), 0), 'constant', 0)
    padded_spike_train = F.pad(padded_spike_train, (int((kernel_size - 1)/2), 0), 'constant', kernel_size - 1)
    smoothed_spike_train = F.conv1d(padded_spike_train, kernel)
    smoothed_spike_train_out = smoothed_spike_train.permute(1,2,0)
    return smoothed_spike_train_out

def apply_convolution_with_ones_kernel(spike_train, kernel, kernel_size):
    """
    Applies a ones kernel convolution to a spike train.
    Args:
        spike_train (Tensor): Input spike train of shape [batch, time, neuron].
        kernel (Tensor): Ones kernel of shape [1, 1, kernel_size].
        kernel_size (int): Size of the kernel.
    Returns:
        Tensor: Smoothed spike train of shape [batch, time, neuron].
    """
    spike_train_permuted = spike_train.permute(2, 0, 1)  # [neuron, batch, time]
    padded_spike_train = F.pad(spike_train_permuted, (kernel_size - 1, 0), 'constant', 0)
    smoothed_spike_train = F.conv1d(padded_spike_train, kernel)
    return smoothed_spike_train.permute(1, 2, 0)  # Back to [batch, time, neuron]

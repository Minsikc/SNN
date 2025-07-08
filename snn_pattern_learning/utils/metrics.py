import torch


def van_rossum_distance(spike_train1, spike_train2, tau, dt=1.0):
    """
    Computes the Van Rossum Distance between two spike trains.
    
    Args:
    - spike_train1 (Tensor): First spike train of shape [batch, time, neuron].
    - spike_train2 (Tensor): Second spike train of shape [batch, time, neuron].
    - tau (float): Time constant for the exponential kernel.
    - dt (float): Time step (default is 1.0).
    
    Returns:
    - distance (Tensor): Van Rossum distance for each batch and neuron, shape [batch, neuron].
    """
    # Check the shape consistency
    assert spike_train1.shape == spike_train2.shape, "Both spike trains must have the same shape"
    
    # Shape info
    batch_size, time_steps, num_neurons = spike_train1.shape

    # Time vector
    time = torch.arange(time_steps, dtype=torch.float32, device=spike_train1.device) * dt
    
    # Exponential kernel: exp(-t/tau)
    kernel = torch.exp(-time / tau).view(1, 1, -1)  # Shape [1, 1, time_steps] for broadcasting
    kernel = kernel.expand(num_neurons, 1, -1)  # Expand to [num_neurons, 1, time_steps]

    # Convolve spike trains with the exponential kernel (using 1D convolution)
    filtered_spike_train1 = torch.nn.functional.conv1d(
        spike_train1.permute(0, 2, 1),  # Shape [batch, neuron, time]
        kernel,  # Kernel shape [num_neurons, 1, time_steps]
        padding=time_steps - 1,  # Ensure output length is maintained
        groups=num_neurons  # Apply convolution independently per neuron
    )
    
    filtered_spike_train2 = torch.nn.functional.conv1d(
        spike_train2.permute(0, 2, 1),  # Shape [batch, neuron, time]
        kernel,  # Kernel shape [num_neurons, 1, time_steps]
        padding=time_steps - 1,  # Ensure output length is maintained
        groups=num_neurons  # Apply convolution independently per neuron
    )
    
    # Compute the squared difference between the filtered signals
    squared_diff = (filtered_spike_train1 - filtered_spike_train2) ** 2
    
    # Sum over the time dimension to compute the Van Rossum distance
    distance = torch.sum(squared_diff, dim=2).sqrt()  # Sum over time dimension, shape [batch, neuron]

    return distance
def max_van_rossum_distance(time_steps, tau, dt=1.0):
    """
    Computes the maximum possible Van Rossum Distance for normalization.
    
    Args:
    - time_steps (int): Number of time steps in the spike train.
    - tau (float): Time constant for the exponential kernel.
    - dt (float): Time step (default is 1.0).
    
    Returns:
    - max_distance (float): Theoretical maximum Van Rossum distance.
    """
    time = torch.arange(time_steps, dtype=torch.float32) * dt
    kernel = torch.exp(-time / tau)

    # 최대로 멀리 있는 스파이크: 하나는 t=0, 하나는 t=T에 존재
    filtered_first_spike = kernel  # t=0에서 스파이크
    filtered_last_spike = torch.roll(kernel, shifts=time_steps-1)  # t=T에서 스파이크

    # Van Rossum 거리 계산
    squared_diff = (filtered_first_spike - filtered_last_spike) ** 2
    max_distance = torch.sum(squared_diff).sqrt().item()

    return max_distance

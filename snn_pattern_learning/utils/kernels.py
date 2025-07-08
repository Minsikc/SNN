import torch

def create_exponential_kernel(kernel_size, decay_constant):
    """
    Create an exponential kernel.
    Args:
        kernel_size (int): Size of the kernel.
        decay_rate (float): Decay rate of the kernel.
    Returns:
        torch.Tensor: Normalized exponential kernel of shape [1, 1, kernel_size].
    """
    t = torch.arange(kernel_size, dtype=torch.float32)
    kernel = torch.exp(-torch.flip(t, [0]) / decay_constant)
    kernel /= kernel.sum()  # Normalize
    return kernel.view(1, 1, kernel_size)

def create_gaussian_kernel(kernel_size, sigma):
    """
    Create a Gaussian kernel.
    Args:
        kernel_size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        torch.Tensor: Normalized Gaussian kernel of shape [1, 1, kernel_size].
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd for a symmetric Gaussian kernel.")
    mu = kernel_size // 2
    t = torch.arange(kernel_size, dtype=torch.float32) - mu
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    return kernel.view(1, 1, kernel_size)

def create_ones_kernel(kernel_size):
    """
    Create a kernel filled with ones.
    Args:
        kernel_size (int): Size of the kernel.
    Returns:
        torch.Tensor: Kernel of shape [1, 1, kernel_size].
    """
    kernel = torch.ones(kernel_size, dtype=torch.float32)
    return kernel.view(1, 1, kernel_size)

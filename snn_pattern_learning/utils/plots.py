import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_comparison_raster(target_spikes, output_spikes, ax, tensor_shape='btn', title="Comparison of Target and Output Spikes"):
    """Plot comparison of target and output spikes on the same axes, with spikes that occur at the same time in both target and output marked in a different color.
    
    Args:
        target_spikes (Tensor): Target spike data.
        output_spikes (Tensor): Output spike data.
        ax (matplotlib.axes.Axes): Axes to plot on.
        title (str): Plot title.
    """

    if tensor_shape == 'tbn':
        target_spikes = target_spikes.permute(1, 0, 2)
        output_spikes = output_spikes.permute(1, 0, 2)
    target_spikes = target_spikes.cpu()
    output_spikes = output_spikes.cpu()
    
    for neuron_idx in range(target_spikes.shape[2]):
        target_spike_times = target_spikes[0, :, neuron_idx].nonzero(as_tuple=True)[0]
        output_spike_times = output_spikes[0, :, neuron_idx].nonzero(as_tuple=True)[0]
        
        # Find common spikes
        common_spike_times = torch.tensor([time.item() for time in target_spike_times if time in output_spike_times], dtype=torch.long)
        
        # Plot target spikes
        ax.scatter(target_spike_times, [neuron_idx] * len(target_spike_times), color='green', marker='|', s=200, label='Target' if neuron_idx == 0 else "")
        
        # Plot output spikes
        ax.scatter(output_spike_times, [neuron_idx] * len(output_spike_times), color='red', marker='|', s=200, label='Output' if neuron_idx == 0 else "")
        
        # Plot common spikes in a different color (e.g., blue)
        ax.scatter(common_spike_times, [neuron_idx] * len(common_spike_times), color='blue', marker='|', s=200, label='Common' if neuron_idx == 0 else "")

    ax.set_xlabel('Time Step', fontdict={'fontsize': 16})
    ax.set_ylabel('Neuron Index', fontdict={'fontsize': 16})
    ax.set_title(title, fontsize=20)
    
    # Correcting the legend handling
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_indices = np.unique(labels, return_index=True)
    unique_handles = [handles[index] for index in unique_indices]
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=14)



def plot_raster(spike_data, ax, title="Spike Raster Plot"):
    """Generate a raster plot from the spike data on a given axes (subplot).
    
    Args:
        spike_data (Tensor): A binary tensor of shape [batch_size, num_steps, num_neurons]
                             indicating spike events.
        ax (matplotlib.axes.Axes): The axes on which to plot the raster.
        title (str): Title of the plot.
    """
    spike_data = spike_data.cpu()
    for neuron_idx in range(spike_data.shape[2]):
        spike_times = spike_data[0, :, neuron_idx].nonzero(as_tuple=True)[0]  # For batch index 0
        ax.scatter(spike_times, [neuron_idx] * len(spike_times), marker='|', s=200, color='k')
    
    ax.set_xlabel('Time Step', fontdict={'fontsize': 16})
    ax.set_ylabel('Neuron Index', fontdict={'fontsize': 16})
    ax.set_title(title, fontsize=20)


def plot_spike_trains(outputs_smoothed, targets_smoothed, epoch):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(outputs_smoothed[0].cpu().detach().numpy(), aspect='auto', origin='lower')
    plt.title(f'Output Spikes After {epoch} Epochs')
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')
    plt.subplot(1, 2, 2)
    plt.imshow(targets_smoothed[0].cpu().detach().numpy(), aspect='auto', origin='lower')
    plt.title('Target Spikes')
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.show()
import torch.nn as nn

def mse_acc_loss_over_time(acc_mem_out, targets, num_step):
    criterion = nn.MSELoss()
    total_loss = 0

    for i in range(num_step):
        # Calculate loss for each timestep
        loss = criterion(acc_mem_out[:, i, :], targets[:, i, :])
        total_loss += loss  # Accumulate loss over time

        # Summing across all timesteps for final prediction

    return total_loss
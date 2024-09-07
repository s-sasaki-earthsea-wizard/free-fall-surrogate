import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, epoch_max: int) -> float:
    """Train the model for one epoch."""
    for i, (motion, params) in enumerate(data_loader):
        # Fetch the first elements of the parameters tensor, because they are the same for all time steps
        unique_params = params[0].unsqueeze(0)

        # Fetch the model outputs
        outputs = model(unique_params)

        # Calculate the target values, i.e. the reaching distance and max height
        target_reaching_distance = motion[-1, 1]
        target_max_height = motion[:, 2].max()

        # Create target tensor
        target = torch.tensor([target_reaching_distance, target_max_height]).unsqueeze(0)

        # Calculate the loss
        loss = criterion(outputs, target)

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss and optimize the weights
        loss.backward()
        optimizer.step()

    return loss.item()
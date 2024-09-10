import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.parabolic_motion_utils import append_intermediate_height

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
    """Validate the model using the validation dataset."""
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for i, (motion, params) in enumerate(val_loader):
            # Fetch the first elements of the parameters tensor, because they are the same for all time steps
            unique_params = params[0].unsqueeze(0)

            # Fetch the model outputs from input parameters
            outputs = model(unique_params)

            # Calculate the target values, i.e. the reaching distance and max height
            target_reaching_distance = motion[-1, 1]
            target_max_height = motion[:, 2].max()

            # Create target tensor
            target = torch.tensor([target_reaching_distance, target_max_height]).unsqueeze(0)

            # Append intermediate points to the target tensor
            target = append_intermediate_height(motion, target, target_reaching_distance)

            # Calculate the loss
            loss = criterion(outputs, target)
            val_loss += loss.item()

    return val_loss / len(val_loader)
import torch.nn as nn
import torch.optim as optim

def configure_training(learning_rate: float, model: nn.Module) -> tuple:
    """Set up the loss function and optimizer."""
    criterion = nn.MSELoss()  # Evaluate loss by MSE.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimize model weights by Adam.
    return criterion, optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model_definitions.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config
from utils.validation_utils import verify_motion_against_params
from utils.data_utils import batch_shuffle

def initialize_model(input_size: int, hidden_size: int, output_size: int) -> nn.Module:
    """Initialize the ParabolicMotionModel with the given sizes."""
    model = ParabolicMotionModel(input_size, hidden_size, output_size)
    return model

def configure_training(learning_rate: float, model: nn.Module) -> tuple:
    """Set up the loss function and optimizer."""
    criterion = nn.MSELoss()  # Evaluate loss by MSE.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimize model weights by Adam.
    return criterion, optimizer

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

def save_model(model, file_path):
    """Save the PyTorch model to the specified file path."""
    torch.save(model.state_dict(), file_path)

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

            # Calculate the loss
            loss = criterion(outputs, target)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def train_model(train_dataset: Dataset, val_dataset: Dataset, batch_size: int) -> None:
    """Main training loop."""
    # Load the configuration file and extract the number of epochs and learning rate
    cfg = load_config('./cfg/cfg.yaml')
    epoch_max = cfg['training']['epoch_max']
    learning_rate = float(cfg['training']['learning_rate'])
    target_loss = float(cfg['training']['target_loss'])
    hidden_size = int(cfg['training']['hidden_size'])
    
    # Initialize the model
    model = initialize_model(input_size=2,
                             hidden_size=hidden_size,
                             output_size=2)

    # Set the loss function and optimizer
    criterion, optimizer = configure_training(learning_rate, model)
    
    # Loop for training
    for epoch in range(epoch_max):
        # Shuffle the dataset indices for the current epoch
        shuffled_indices = batch_shuffle(train_dataset, batch_size)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=shuffled_indices)
        
        # Train for one epoch
        train_one_epoch(model, data_loader, criterion, optimizer, epoch, epoch_max)

        # Calculate the validation loss and evaluate the model by it
        val_loss = validate_model(model, data_loader, criterion)
        print(f'Epoch [{epoch+1}/{epoch_max}], Validation Loss: {val_loss:.4f}')

        # Check if the loss has been reached less than the target loss
        if val_loss < target_loss:
            print(f"Target validation loss reached: {val_loss:.4f} < {target_loss:.4f} = target loss")
            break

        # Check if this is the last epoch
        if epoch == epoch_max - 1:
            print(f"Reached maximum number of epochs: {epoch+1}/{epoch_max}")
            print(f"Final validation loss: {val_loss:.4f}")

    # Save the trained model
    save_model(model, './trained_models/parabolic_motion_model.pth')
    print("Training complete!")
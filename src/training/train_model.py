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
        # Debug: Verify the motion against the parameters for this batch
        # assert verify_motion_against_params(motion, params), f"Data verification failed for batch {i}"

        # Debug: Print the current batch
        # print(f"Batch [{i+1}/{len(data_loader)}], Epoch [{epoch+1}/{epoch_max}]")
        # print(f"Motion shape: {motion.shape}, Params shape: {params.shape}")
        # print(f"Motion: {motion}, Params: {params}")

        # Fetch the model outputs
        outputs = model(params)

        # Calculate the loss
        loss = criterion(outputs, motion)

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss and optimize the weights
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epoch_max}], Loss: {loss.item():.4f}')
    return loss.item()

def save_model(model, file_path):
    """Save the PyTorch model to the specified file path."""
    torch.save(model.state_dict(), file_path)

def train_model(dataset: Dataset, batch_size: int) -> None:
    """Main training loop."""
    # Load the configuration file and extract the number of epochs and learning rate
    cfg = load_config('./cfg/cfg.yaml')
    epoch_max = cfg['training']['epoch_max']
    learning_rate = float(cfg['training']['learning_rate'])
    target_loss = float(cfg['training']['target_loss'])
    hidden_size = int(cfg['training']['hidden_size'])
    
    # Initialize the model
    model = initialize_model(input_size=2, hidden_size=hidden_size, output_size=3)

    # Set the loss function and optimizer
    criterion, optimizer = configure_training(learning_rate, model)
    
    # Loop for training
    for epoch in range(epoch_max):
        # Shuffle the dataset indices for the current epoch
        shuffled_indices = batch_shuffle(dataset, batch_size)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=shuffled_indices)
        
        # Train for one epoch and calculate the loss
        loss = train_one_epoch(model, data_loader, criterion, optimizer, epoch, epoch_max)

        # Check if the loss has been reached less than the target loss
        if loss < target_loss:
            print(f"Target loss reached: {loss:.4f} < {target_loss:.4f}")
            break

        # Check if this is the last epoch
        if epoch == epoch_max - 1:
            print(f"Reached maximum number of epochs: {epoch+1}/{epoch_max}")
            print(f"Final loss: {loss:.4f}")

    # Save the trained model
    save_model(model, './trained_models/parabolic_motion_model.pth')
    print("Training complete!")
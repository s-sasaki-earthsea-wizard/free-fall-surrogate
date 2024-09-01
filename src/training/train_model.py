import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model_definitions.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config
from utils.validation_utils import verify_motion_against_params
from utils.data_utils import batch_shuffle

def save_model(model, file_path):
    """Save the PyTorch model to the specified file path."""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved as '{file_path}'")

def train_model(dataset: Dataset, batch_size: int) -> None:
    # Load the configuration file and extract the number of epochs and learning rate
    cfg = load_config('./cfg/cfg.yaml')
    num_epochs = cfg['training']['num_epochs']
    learning_rate = cfg['training']['learning_rate']

    # initialize the model
    input_size = 2  # (initial velocity, angle)
    hidden_size = 64
    output_size = 3  # (time, x, y)
    model = ParabolicMotionModel(input_size, hidden_size, output_size)

    # Set the loss function and optimizer
    criterion = nn.MSELoss() # Evaluate loss by MSE.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimize model weights by Adam
    
    # Loop for training
    for epoch in range(num_epochs):
        # Shuffle the dataset indices for the current epoch
        shuffled_indices = batch_shuffle(dataset, batch_size)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=shuffled_indices)
        
        for i, (motion, params) in enumerate(data_loader):
            # Debug: Verify the motion against the parameters for this batch
            assert verify_motion_against_params(motion, params), f"Data verification failed for batch {i}"

            # Fetch the model outputs
            outputs = model(params)

            # Calculate the loss
            loss = criterion(outputs, motion)

            # Reset gradients
            optimizer.zero_grad()

            # Backpropagate the loss and optimize the weights
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    save_model(model, 'parabolic_motion_model.pth')
    print("Training complete!")
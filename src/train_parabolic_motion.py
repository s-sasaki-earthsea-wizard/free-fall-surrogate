import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.parabolic_motion_utils import load_training_data
from utils.data_utils import batch_shuffle
from utils.validation_utils import verify_motion_against_params

"""def verify_motion_against_params(motion, params, gravity=9.81, tolerance=1e-3):
    # Extract time, x, and y from motion tensor
    time = motion[:, 0].numpy()
    x_actual = motion[:, 1].numpy()
    y_actual = motion[:, 2].numpy()

    # Extract initial velocity and angle from params
    initial_velocity = params[0].item()
    angle_deg = params[1].item()
    angle_rad = np.radians(angle_deg)

    # Calculate theoretical motion
    x_theoretical = initial_velocity * np.cos(angle_rad) * time
    y_theoretical = initial_velocity * np.sin(angle_rad) * time - 0.5 * gravity * time**2

    # Compare actual and theoretical motion
    x_diff = np.abs(x_actual - x_theoretical)
    y_diff = np.abs(y_actual - y_theoretical)

    assert np.all(x_diff < tolerance), f"x values differ by more than {tolerance}: max diff = {x_diff.max()}"
    assert np.all(y_diff < tolerance), f"y values differ by more than {tolerance}: max diff = {y_diff.max()}"

    return True"""

# Define a simple neural network model
class ParabolicMotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParabolicMotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    # CSV file paths
    motion_data_path = './data/simulation/splits/train_motion_data.csv'
    params_data_path = './data/simulation/splits/train_params_data.csv'

    # Load training paraboloc motion and parameter data and convert them to PyTorch tensors
    motion_tensors, params_tensors = load_training_data(motion_data_path, params_data_path)

    # Prepare datasets for training using list comprehension
    datasets = [
        TensorDataset(motion, params.unsqueeze(0).expand(motion.size(0), -1))
        for motion, params in zip(motion_tensors, params_tensors)
    ]
    
    # Concatenate the datasets into a single large TensorDataset
    motion_concat = torch.cat([dataset.tensors[0] for dataset in datasets])
    params_concat = torch.cat([dataset.tensors[1] for dataset in datasets])

    # Create a combined TensorDataset
    dataset = TensorDataset(motion_concat, params_concat)

    # Set the batch size to the number of unique path_ids
    batch_size = len(motion_concat) // len(motion_tensors)  # This should equal the number of time steps per path_id
          
    # Create a DataLoader for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Shuffle the batches manually using the custom function
    shuffled_indices = batch_shuffle(dataset, batch_size)
    shuffled_loader = DataLoader(dataset, batch_size=batch_size, sampler=shuffled_indices)

if __name__ == "__main__":
    main()

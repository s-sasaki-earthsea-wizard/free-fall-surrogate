import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.parabolic_motion_utils import load_training_data

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

    # Load training data as separate lists of tensors
    motion_tensors, params_tensors = load_training_data(motion_data_path, params_data_path)

    # Prepare datasets for training
    datasets = [TensorDataset(motion, params.unsqueeze(0)) for motion, params in zip(motion_tensors, params_tensors)]
    
    # Create a DataLoader for training
    data_loader = DataLoader(datasets, batch_size=1, shuffle=True)

    # Verify the DataLoader by iterating through one batch
    for i, (motion, params) in enumerate(data_loader):
        print(f"Batch {i}:")
        print(f"  Motion Tensor: {motion}")
        print(f"  Params Tensor: {params}")

if __name__ == "__main__":
    main()

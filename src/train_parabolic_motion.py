import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.parabolic_motion_utils import load_training_data

def custom_collate_fn(batch):
    motion_batch = [item[0] for item in batch]
    params_batch = []
    
    # Stack the motion tensors to keep the time sequence intact
    motion_batch = torch.stack(motion_batch, dim=0)
    
    # Manually repeat params to match the shape of motion
    for motion, params in zip(motion_batch, [item[1] for item in batch]):
        # Repeat params tensor to match the number of time steps in the corresponding motion tensor
        expanded_params = params.unsqueeze(0).expand(motion.size(0), -1)
        params_batch.append(expanded_params)

    # Stack the manually created params_batch
    params_batch = torch.cat(params_batch, dim=0)  # Use cat instead of stack to avoid creating an additional dimension
    
    # Flatten the batch dimension to merge all the batches into one
    motion_batch = motion_batch.view(-1, motion_batch.size(-1))
    params_batch = params_batch.view(-1, params.size(-1))  # Use params.size(-1) to maintain the correct shape
    
    return motion_batch, params_batch

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
    print(f"Batch size: {batch_size}")

    # Create a DataLoader for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Debug: Verify the DataLoader by iterating through one batch
    for i, (motion, params) in enumerate(data_loader):
        print(f"Batch {i}:")
        print(f"  Motion Tensor shape: {motion.shape}")
        print(f"  Params Tensor shape: {params.shape}")
        print(f"  Motion Tensor: {motion}")
        print(f"  Params Tensor: {params}")

if __name__ == "__main__":
    main()

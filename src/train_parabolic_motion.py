import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.parabolic_motion_utils import load_training_data
from training.train_model import train_model

def main():
    # CSV file paths
    train_motion_data_path = './data/simulation/splits/train_motion_data.csv'
    train_params_data_path = './data/simulation/splits/train_params_data.csv'
    val_motion_data_path = './data/simulation/splits/val_motion_data.csv'
    val_params_data_path = './data/simulation/splits/val_params_data.csv'

    # Load training parabolic motion and parameter data and convert them to PyTorch tensors
    train_motion_tensors, train_params_tensors = load_training_data(train_motion_data_path, train_params_data_path)
    val_motion_tensors, val_params_tensors = load_training_data(val_motion_data_path, val_params_data_path)

    # Prepare datasets for training using list comprehension
    datasets = [
        TensorDataset(motion, params.unsqueeze(0).expand(motion.size(0), -1))
        for motion, params in zip(train_motion_tensors, train_params_tensors)
    ]

    val_datasets = [
        TensorDataset(val_motion, val_params.unsqueeze(0).expand(val_motion.size(0), -1))
        for val_motion, val_params in zip(val_motion_tensors, val_params_tensors)
    ]
    
    # Concatenate the datasets into a single large TensorDataset
    train_motion_concat = torch.cat([dataset.tensors[0] for dataset in datasets])
    train_params_concat = torch.cat([dataset.tensors[1] for dataset in datasets])

    val_motion_concat = torch.cat([dataset.tensors[0] for dataset in val_datasets])
    val_params_concat = torch.cat([dataset.tensors[1] for dataset in val_datasets])

    # Create combined TensorDatasets for training and validation
    dataset = TensorDataset(train_motion_concat, train_params_concat)
    val_dataset = TensorDataset(val_motion_concat, val_params_concat)

    # Set the batch size to the number of unique path_ids
    batch_size = len(train_motion_concat) // len(train_motion_tensors)  # This should equal the number of time steps per path_id
          
    # Run the training with validation
    train_model(dataset, val_dataset, batch_size)

if __name__ == "__main__":
    main()
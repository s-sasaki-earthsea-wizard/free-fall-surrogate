import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.parabolic_motion_utils import load_training_data
from utils.data_utils import batch_shuffle
from utils.validation_utils import verify_motion_against_params
from models.parabolic_motion_model import ParabolicMotionModel
from training.train_model import train_model

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
          
    # Run the training
    train_model(dataset, batch_size)

if __name__ == "__main__":
    main()

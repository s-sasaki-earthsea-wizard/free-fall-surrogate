import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.parabolic_motion_utils import load_training_data

def main():
    # CSV file paths
    motion_data_path = './data/simulation/splits/train_motion_data.csv'
    params_data_path = './data/simulation/splits/train_params_data.csv'

    # Load training data
    motion_data, params_data = load_training_data(motion_data_path, params_data_path)

if __name__ == "__main__":
    main()

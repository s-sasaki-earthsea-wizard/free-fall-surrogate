import pandas as pd
import numpy as np
import torch
from typing import List, Tuple

def load_training_data(motion_data_path: str, params_data_path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Load data from CSV files into pandas DataFrames
    motion_data = pd.read_csv(motion_data_path)
    params_data = pd.read_csv(params_data_path)

    # Group motion data by path_id
    grouped_motion = motion_data.groupby('path_id')

    motion_tensors = []
    params_tensors = []

    for path_id, group in grouped_motion:
        # Extract the corresponding parameters for the current path_id
        params = params_data[params_data['path_id'] == path_id].iloc[0]

        # Convert the motion group and params to tensors
        motion_tensor = torch.tensor(group[['time', 'x', 'y']].values, dtype=torch.float32)
        params_tensor = torch.tensor(params[['initial_velocity', 'angle (deg)']].values, dtype=torch.float32)
        params_tensor[1] = params_tensor[1] * (torch.tensor(np.pi / 180.0))  # Convert angle to radians

        # Append the tensors to their respective lists
        motion_tensors.append(motion_tensor)
        params_tensors.append(params_tensor)

    return motion_tensors, params_tensors

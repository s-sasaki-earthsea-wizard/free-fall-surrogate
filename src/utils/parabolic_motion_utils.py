import pandas as pd
import numpy as np
import torch
from torch import tensor, cat
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

def find_nearest_point(motion, target_x_value):
    """Find the index of the point in 'motion' closest to 'target_x_value'."""
    abs_diff = torch.abs(motion[:, 1] - target_x_value)
    nearest_index = torch.argmin(abs_diff).item()
    return nearest_index

def append_intermediate_height(motion, target, reaching_distance):
    """Find the height-values at x = max(x)/4 and 3*max(x)/4 and append them to the target tensor."""

    # Find the indices of the points closest to max(x)/4 and 3*max(x)/4
    index_1_4 = find_nearest_point(motion, reaching_distance / 4)
    index_3_4 = find_nearest_point(motion, 3 * reaching_distance / 4)

    # Get the corresponding y-values for these points
    target_height_1_4 = motion[index_1_4, 2]
    target_height_3_4 = motion[index_3_4, 2]

    # Append these to the target tensor
    updated_target = cat([target, tensor([[target_height_1_4, target_height_3_4]])], dim=1)

    return updated_target
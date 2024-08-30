import pandas as pd
import torch

def load_training_data(motion_data_path: str, params_data_path: str):
    """
    Load training data from the specified CSV files, group by path_id, and convert to tensors.

    Args:
        motion_data_path (str): Path to the motion data CSV file.
        params_data_path (str): Path to the params data CSV file.

    Returns:
        list of tuples: A list where each tuple contains motion tensor and params tensor.
    """
    # Load data from CSV files into pandas DataFrames
    motion_data = pd.read_csv(motion_data_path)
    params_data = pd.read_csv(params_data_path)

    # Group motion data by path_id
    grouped_motion = motion_data.groupby('path_id')

    data_tensors = []

    for path_id, group in grouped_motion:
        # Extract the corresponding parameters for the current path_id
        params = params_data[params_data['path_id'] == path_id].iloc[0]

        # Convert the motion group and params to tensors
        motion_tensor = torch.tensor(group[['time', 'x', 'y']].values, dtype=torch.float32)
        params_tensor = torch.tensor(params[['initial_velocity', 'angle (deg)']].values, dtype=torch.float32)

        # Append the pair of tensors to the list
        data_tensors.append((motion_tensor, params_tensor))

    return data_tensors

import pandas as pd

def load_training_data(motion_data_path: str, params_data_path: str):
    """
    Load training data from the specified CSV files.

    Args:
        motion_data_path (str): Path to the motion data CSV file.
        params_data_path (str): Path to the params data CSV file.

    Returns:
        tuple: Two pandas DataFrames, one for motion data and one for params data.
    """
    motion_data = pd.read_csv(motion_data_path)
    params_data = pd.read_csv(params_data_path)
    return motion_data, params_data

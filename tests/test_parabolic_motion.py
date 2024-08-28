import pandas as pd
import numpy as np
import os
import pytest
from utils import load_config

# Load the parabolic motion data
def load_motion_data(file_path):
    return pd.read_csv(file_path)

# Test to check if the y-values follow a parabolic curve
def test_parabolic_motion():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'cfg.yaml')
    cfg = load_config(config_path)
    gravity_acceleration = cfg['gravity_acceleration']
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation', 'parabolic_motion.csv')
    df = load_motion_data(file_path)

    # Group by path_id to test each trajectory separately
    for path_id, group in df.groupby('path_id'):
        # Extract time, x, and y values
        time = group['time'].values
        x = group['x'].values
        y = group['y'].values

        # Use the first few points to estimate initial velocity and angle
        initial_velocity = np.max(x) / np.max(time)

        # Calculate expected y-values using the parabolic formula
        expected_y = initial_velocity * np.sin(np.radians(45)) * time - 0.5 * gravity_acceleration * time**2
        
        # Allow a small margin of error
        assert np.allclose(y, expected_y, rtol=0.1), f"Path {path_id} does not follow a parabolic curve"

if __name__ == '__main__':
    pytest.main()

import numpy as np
from utils.config_utils import load_config

def verify_motion_against_params(motion, params, tolerance=1e-3):
    # Load the configuration file and gravity acceleration from it
    cfg = load_config('./cfg/cfg.yaml')
    gravity_acceleration = cfg['gravity_acceleration']

    # Extract time, x, and y from motion tensor
    time = motion[:, 0].numpy()
    x_actual = motion[:, 1].numpy()
    y_actual = motion[:, 2].numpy()

    # Extract initial velocity and angle from params
    initial_velocity = params[0].item()
    angle_deg = params[1].item()
    angle_rad = np.radians(angle_deg)

    # Calculate theoretical motion
    x_theoretical = initial_velocity * np.cos(angle_rad) * time
    y_theoretical = initial_velocity * np.sin(angle_rad) * time - 0.5 * gravity_acceleration * time**2

    # Compare actual and theoretical motion
    x_diff = np.abs(x_actual - x_theoretical)
    y_diff = np.abs(y_actual - y_theoretical)

    assert np.all(x_diff < tolerance), f"x values differ by more than {tolerance}: max diff = {x_diff.max()}"
    assert np.all(y_diff < tolerance), f"y values differ by more than {tolerance}: max diff = {y_diff.max()}"

    return True
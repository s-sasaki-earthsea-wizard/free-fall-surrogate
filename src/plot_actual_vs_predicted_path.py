import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from model_definitions.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config
from utils.parabolic_motion_utils import load_training_data, append_intermediate_height

def plot_predictions_vs_actual(model, motion_tensors, params_tensors, n_samples=5, save_dir='./plots/actual_vs_predicted_path'):
    """Plot actual parabolic motion curves and predicted values for max height, reaching distance, and intermediate points."""
    
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Pick n_samples random indices
    indices = random.sample(range(len(motion_tensors)), n_samples)
    
    for idx in indices:
        motion = motion_tensors[idx]
        params = params_tensors[idx]

        # Actual motion: x (distance) and y (height) values
        time = motion[:, 0].numpy()
        x_actual = motion[:, 1].numpy()
        y_actual = motion[:, 2].numpy()

        # Predicted values
        with torch.no_grad():
            predicted = model(params.unsqueeze(0))
        
        predicted_reaching_distance = predicted[0, 0].item()
        predicted_max_height = predicted[0, 1].item()
        predicted_height_1_4 = predicted[0, 2].item()
        predicted_height_3_4 = predicted[0, 3].item()

        # --- Actual values for intermediate points ---
        target_reaching_distance = motion[-1, 1]
        # Append intermediate points for actual data
        actual_target = torch.tensor([target_reaching_distance, motion[:, 2].max()]).unsqueeze(0)
        actual_target = append_intermediate_height(motion, actual_target, target_reaching_distance)

        # --- Plot ---
        # Plot actual motion
        plt.plot(x_actual, y_actual, label='Actual Trajectory')

        # Plot predicted max height (at half of predicted reaching distance)
        plt.scatter([predicted_reaching_distance / 2], [predicted_max_height], color='green', label='Predicted Max Height')

        # Plot predicted reaching distance
        plt.scatter([predicted_reaching_distance], [0], color='green', label='Predicted Reaching Distance')

        # Plot predicted intermediate points
        plt.scatter([predicted_reaching_distance / 4], [predicted_height_1_4], color='green', label='Predicted 1/4 Point')
        plt.scatter([3 * predicted_reaching_distance / 4], [predicted_height_3_4], color='green', label='Predicted 3/4 Point')

        # Plot the origin (0,0)
        plt.scatter([0], [0], color='red', label='Origin point')

        # Add labels and title
        plt.title(f'Path ID: {idx}')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.legend()

        # Save the plot as PNG
        plot_filename = os.path.join(save_dir, f'plot_path_{idx}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure after saving to avoid display
        print(f'Saved plot for Path ID {idx} to {plot_filename}')

def main():
    # Load configuration
    config_path = './cfg/cfg.yaml'
    cfg = load_config(config_path)

    # Load the trained model
    model_path = cfg['evaluation']['model_path']
    input_size = 2  # initial_velocity, angle
    hidden_size = cfg['training']['hidden_size']
    output_size = 4  # reaching distance, max height, and two intermediate points
    model = ParabolicMotionModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Load the test data
    test_motion_data_path = cfg['evaluation']['test_motion_data']
    test_params_data_path = cfg['evaluation']['test_params_data']
    motion_tensors, params_tensors = load_training_data(test_motion_data_path, test_params_data_path)

    # Plot the actual vs predicted motion
    plot_predictions_vs_actual(model, motion_tensors, params_tensors)

if __name__ == '__main__':
    main()

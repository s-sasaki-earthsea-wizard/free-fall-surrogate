import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from model_definitions.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config
from utils.parabolic_motion_utils import load_training_data, append_intermediate_height

def load_model(model_path, input_size, hidden_size, output_size):
    """Load the trained model from the specified path."""
    model = ParabolicMotionModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, motion_tensors, params_tensors):
    """Evaluate the model using test data and return the MSE."""
    predictions = []
    targets = []

    # Iterate over the test data
    for motion, params in zip(motion_tensors, params_tensors):
        # Run inference
        with torch.no_grad():
            predicted = model(params.unsqueeze(0))  # Expand the dimensions for batch processing
            predictions.append(predicted.numpy())

        # Extract the true reaching distance and max height
        true_reaching_distance = motion[-1, 1]
        true_max_height = motion[:, 2].max()

        # Create target tensor
        target = torch.tensor([true_reaching_distance, true_max_height]).unsqueeze(0)
        target = append_intermediate_height(motion, target, true_reaching_distance)
        targets.append(target.numpy())

    # Convert the torch tensors to numpy arrays
    predictions_arr = np.array(predictions)
    targets_arr = np.array(targets)

    # Compute Mean Squared Error (MSE) between predictions and targets
    predictions = torch.tensor(predictions_arr).squeeze()
    targets = torch.tensor(targets_arr).squeeze()
    mse = mean_squared_error(targets, predictions)
    
    return mse

def main():
    # Load configuration
    config_path = './cfg/cfg.yaml'
    cfg = load_config(config_path)

    # Load the trained model
    model_path = cfg['evaluation']['model_path']
    input_size = 2  # initial_velocity, angle
    hidden_size = cfg['training']['hidden_size']
    output_size = 4  # reaching distance, max height, and two intermediate points
    model = load_model(model_path, input_size, hidden_size, output_size)

    # Load the test data
    test_motion_data_path = cfg['evaluation']['test_motion_data']
    test_params_data_path = cfg['evaluation']['test_params_data']
    motion_tensors, params_tensors = load_training_data(test_motion_data_path, test_params_data_path)

    # Evaluate the model
    mse = evaluate_model(model, motion_tensors, params_tensors)
    print(f"Mean Squared Error (MSE) on test data: {mse:.4f}")

if __name__ == '__main__':
    main()

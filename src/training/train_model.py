import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config

def train_model(dataset, batch_size):
    # Load the configuration file and extract the number of epochs and learning rate
    cfg = load_config('./cfg/cfg.yaml')
    num_epochs = cfg['training']['num_epochs']
    learning_rate = cfg['training']['learning_rate']

    # initialize the model
    input_size = 2  # (initial velocity, angle)
    hidden_size = 64
    output_size = 3  # (time, x, y)
    model = ParabolicMotionModel(input_size, hidden_size, output_size)

    # Set the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loop for training
    for epoch in range(num_epochs):
        for i, (motion, params) in enumerate(data_loader):
            # Fetch the model outputs
            outputs = model(params)

            # Calculate the loss
            loss = criterion(outputs, motion)

            # Reset gradients
            optimizer.zero_grad()

            # Backpropagate the loss and optimize the weights
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")
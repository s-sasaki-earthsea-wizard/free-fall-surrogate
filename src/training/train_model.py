import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model_definitions.parabolic_motion_model import ParabolicMotionModel
from utils.config_utils import load_config
from utils.validation_utils import verify_motion_against_params
from utils.data_utils import batch_shuffle, save_model
from training.epoch_train import train_one_epoch
from training.inference_model_validation import validate_model
from training.training_setup import configure_training

def train_model(train_dataset: Dataset, val_dataset: Dataset, batch_size: int) -> None:
    """Main training loop."""
    # Load the configuration file and extract the number of epochs and learning rate
    cfg = load_config('./cfg/cfg.yaml')
    epoch_max = cfg['training']['epoch_max']
    init_learning_rate = float(cfg['training']['init_learning_rate'])
    sheduler_cycle_epochs = cfg['training']['sheduler_cycle_epochs']
    target_loss = float(cfg['training']['target_loss'])
    hidden_size = int(cfg['training']['hidden_size'])
    
    # Initialize the model
    model = ParabolicMotionModel(input_size=2,
                                 hidden_size=hidden_size,
                                 output_size=4)

    # Set the loss function and optimizer
    criterion, optimizer = configure_training(init_learning_rate, model)

    # Set up the CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sheduler_cycle_epochs)
    
    # Loop for training
    for epoch in range(epoch_max):
        # Shuffle the dataset indices for the current epoch
        shuffled_indices = batch_shuffle(train_dataset, batch_size)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=shuffled_indices)
        
        # Train for one epoch
        train_one_epoch(model, data_loader, criterion, optimizer, epoch, epoch_max)

        # Calculate the validation loss and evaluate the model by it
        val_loss = validate_model(model, data_loader, criterion)
        print(f'Epoch [{epoch+1}/{epoch_max}], Validation Loss: {val_loss:.4f}')

        # Update the learning rate using the scheduler
        scheduler.step()

        # Check if the loss has been reached less than the target loss
        if val_loss < target_loss:
            print(f"Target validation loss reached: {val_loss:.4f} < {target_loss:.4f} = target loss")
            break

        # Check if this is the last epoch
        if epoch == epoch_max - 1:
            print(f"Reached maximum number of epochs: {epoch+1}/{epoch_max}")
            print(f"Final validation loss: {val_loss:.4f}")

    # Save the trained model
    save_model(model, './trained_models/parabolic_motion_model.pth')
    print("Training complete!")
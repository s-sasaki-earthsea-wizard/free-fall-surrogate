import torch
from torch.utils.data import Dataset

# Custom function to split data into batches and shuffle them
def batch_shuffle(dataset: Dataset, batch_size: int) -> torch.Tensor:
    n_batches = len(dataset) // batch_size
    indices = torch.arange(len(dataset))

    # Reshape indices into batches
    indices = indices.view(n_batches, batch_size)
    
    # Shuffle the order of batches
    indices = indices[torch.randperm(n_batches)]
    return indices.view(-1)

def save_model(model, file_path):
    """Save the PyTorch model to the specified file path."""
    torch.save(model.state_dict(), file_path)
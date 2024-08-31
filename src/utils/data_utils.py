import torch

# Custom function to split data into batches and shuffle them
def batch_shuffle(dataset, batch_size):
    n_batches = len(dataset) // batch_size
    indices = torch.arange(len(dataset))

    # Reshape indices into batches
    indices = indices.view(n_batches, batch_size)
    
    # Shuffle the order of batches
    indices = indices[torch.randperm(n_batches)]
    return indices.view(-1)
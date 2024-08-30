import os
import pandas as pd
import pytest
import numpy as np
from split_parabolic_motion_data import DataSplitter

# Load the original motion data
def load_motion_data(file_path):
    return pd.read_csv(file_path)

# Load the split data (train, val, test)
def load_split_data(split_dir):
    train_motion_file = os.path.join(split_dir, 'train_motion_data.csv')
    val_motion_file = os.path.join(split_dir, 'val_motion_data.csv')
    test_motion_file = os.path.join(split_dir, 'test_motion_data.csv')
    
    train_data = pd.read_csv(train_motion_file)
    val_data = pd.read_csv(val_motion_file)
    test_data = pd.read_csv(test_motion_file)
    
    return train_data, val_data, test_data

# Test to check if the data splitting is done correctly
def test_data_splitting():
    # Define paths
    original_motion_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation', 'parabolic_motion.csv')
    original_params_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation', 'parabolic_params.csv')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'cfg.yaml')
    split_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation', 'splits')

    # Initialize DataSplitter and split the data
    data_splitter = DataSplitter(original_motion_file, original_params_file, config_path)
    data_splitter.split_data()
    data_splitter.save_data(split_dir)
    
    # Load original data
    original_motion_data = load_motion_data(original_motion_file)
    
    # Load split data
    train_data, val_data, test_data = load_split_data(split_dir)
    
    # Check for ID duplication across splits
    # Fetch all unique path_ids across splits
    all_ids = np.concatenate([pd.unique(train_data['path_id']),
                            pd.unique(val_data['path_id']),
                            pd.unique(test_data['path_id'])
                            ])

    # Check for duplicates
    duplicate_ids = all_ids[np.isin(all_ids, np.unique(all_ids), assume_unique=True) & (np.unique(all_ids, return_counts=True)[1] > 1)]

    if duplicate_ids.size > 0:
        print("Duplicate path_ids found across splits:", duplicate_ids)

    # Assert that no duplicates exist
    assert duplicate_ids.size == 0, "There are duplicate path_ids across the splits!"
    
    # Check that all path_id in the split datasets are in the original dataset
    for split_data in [train_data, val_data, test_data]:
        for path_id in split_data['path_id'].unique():
            # Ensure all rows for this path_id match between the original and split dataset
            original_rows = original_motion_data[original_motion_data['path_id'] == path_id]
            split_rows = split_data[split_data['path_id'] == path_id]
            
            pd.testing.assert_frame_equal(original_rows.reset_index(drop=True), 
                                          split_rows.reset_index(drop=True), 
                                          check_like=True,
                                          obj=f"Mismatch found in path_id {path_id} between original and split data")

if __name__ == '__main__':
    pytest.main()


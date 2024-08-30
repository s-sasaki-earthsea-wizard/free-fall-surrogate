import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from math import isclose
from utils.config_utils import load_config

# Split the parabolic motion data
class DataSplitter:
    def __init__(self, motion_file, params_file, config_path):
        # Load configuration
        self.config = load_config(config_path)
        
        # Load data
        self.motion_data = pd.read_csv(motion_file)
        self.params_data = pd.read_csv(params_file)
        
        # Extract settings
        self.random_state = self.config['random_state']
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_params = None
        self.val_params = None
        self.test_params = None

    def split_data(self):
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']

        # Log the data split ratios
        print(f"train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {test_ratio}")
        print(f"Sum: {train_ratio + val_ratio + test_ratio}")

        # Ensure the sum of the ratios is equal to 1
        assert isclose(train_ratio + val_ratio + test_ratio, 1.0), "The sum of train, validation, and test ratios must be 1."

        # Group the data by path_id
        grouped = self.motion_data.groupby('path_id')

        # Create a list of unique path ids
        path_ids = list(grouped.groups.keys())

        # Split the path ids into train, validation, and test sets
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        train_ids, remaining_ids = train_test_split(path_ids, test_size=(1 - train_ratio), random_state=self.random_state)
        val_ids, test_ids = train_test_split(remaining_ids, test_size=val_test_ratio, random_state=self.random_state)

        # Split the data based on the split path ids
        self.train_data = self.motion_data[self.motion_data['path_id'].isin(train_ids)]
        self.val_data = self.motion_data[self.motion_data['path_id'].isin(val_ids)]
        self.test_data = self.motion_data[self.motion_data['path_id'].isin(test_ids)]

        # Split the parameters data same as the motion data
        self.train_params = self.params_data[self.params_data['path_id'].isin(train_ids)]
        self.val_params = self.params_data[self.params_data['path_id'].isin(val_ids)]
        self.test_params = self.params_data[self.params_data['path_id'].isin(test_ids)]

    def save_data(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        train_motion_file = os.path.join(output_dir, 'train_motion_data.csv')
        val_motion_file = os.path.join(output_dir, 'val_motion_data.csv')
        test_motion_file = os.path.join(output_dir, 'test_motion_data.csv')
        train_params_file = os.path.join(output_dir, 'train_params_data.csv')
        val_params_file = os.path.join(output_dir, 'val_params_data.csv')
        test_params_file = os.path.join(output_dir, 'test_params_data.csv')

        # Save data to CSV files, sorted by 'path_id' and 'time'
        self.train_data.sort_values(by=['path_id', 'time']).to_csv(train_motion_file, index=False)
        self.val_data.sort_values(by=['path_id', 'time']).to_csv(val_motion_file, index=False)
        self.test_data.sort_values(by=['path_id', 'time']).to_csv(test_motion_file, index=False)
        self.train_params.sort_values(by='path_id').to_csv(train_params_file, index=False)
        self.val_params.sort_values(by='path_id').to_csv(val_params_file, index=False)
        self.test_params.sort_values(by='path_id').to_csv(test_params_file, index=False)

        print(f'Train motion data saved to {train_motion_file}')
        print(f'Validation motion data saved to {val_motion_file}')
        print(f'Test motion data saved to {test_motion_file}')
        print(f'Train params data saved to {train_params_file}')
        print(f'Validation params data saved to {val_params_file}')
        print(f'Test params data saved to {test_params_file}')

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'simulation')
    motion_file = os.path.join(data_dir, 'parabolic_motion.csv')
    params_file = os.path.join(data_dir, 'parabolic_params.csv')
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg', 'cfg.yaml')
    
    data_splitter = DataSplitter(motion_file, params_file, config_path)
    data_splitter.split_data()
    
    output_dir = os.path.join(data_dir, 'splits')
    data_splitter.save_data(output_dir)

if __name__ == "__main__":
    main()

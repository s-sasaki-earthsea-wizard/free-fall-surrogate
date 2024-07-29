import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# Add the path to the utils directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import load_config

# Split the parabolic motion data# Random seed for reproducibility
class DataSplitter:
    def __init__(self, motion_file, params_file, config_path):
        self.config = load_config(config_path)
        self.data = self.load_data(motion_file, params_file)
        self.random_state = self.config['random_state']
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self, motion_file, params_file):
        motion_data = pd.read_csv(motion_file)
        params_data = pd.read_csv(params_file)
        data = pd.merge(motion_data, params_data, on='path_id')
        return data
    
    def split_data(self):
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']

        print("total ratio: ", train_ratio + val_ratio + test_ratio)

        # The assert statement is used to check if the sum of the ratios is equal to 1
        assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0), "The sum of train, validation, and test ratios must be 1."

        # Split the data into the training data and the remaining data
        train_data, remaining_data = train_test_split(self.data, test_size=(1 - train_ratio), random_state=self.random_state)

        # Split the remaining data into validation and test data
        val_data, test_data = train_test_split(remaining_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=self.random_state)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    
    def save_data(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        train_file = os.path.join(output_dir, 'train_data.csv')
        val_file = os.path.join(output_dir, 'val_data.csv')
        test_file = os.path.join(output_dir, 'test_data.csv')
        self.train_data.sort_values(by='path_id').to_csv(train_file, index=False)
        self.val_data.sort_values(by='path_id').to_csv(val_file, index=False)
        self.test_data.sort_values(by='path_id').to_csv(test_file, index=False)
        print(f'Train data saved to {train_file}')
        print(f'Validation data saved to {val_file}')
        print(f'Test data saved to {test_file}')

        os.makedirs(output_dir, exist_ok=True)
        train_file = os.path.join(output_dir, 'train_data.csv')
        val_file = os.path.join(output_dir, 'val_data.csv')
        test_file = os.path.join(output_dir, 'test_data.csv')
        

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


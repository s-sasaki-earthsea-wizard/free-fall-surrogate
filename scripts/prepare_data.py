import numpy as np
import pandas as pd
import os
import sys

# Add the path to the utils directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils import load_config # import the load_config function from utils/utils.py

class ParabolicMotionDataGenerator:
    def __init__(self, config_path):
        cfg = load_config(config_path)

        # fetch the parameters from the configuration file
        self.gravity_acceleration = cfg['gravity_acceleration']
        self.num_samples = cfg['num_samples']
        self.num_points = cfg['num_points']
        self.initial_velocity_range = (cfg['initial_velocity_min'],
                                       cfg['initial_velocity_max'])
        self.angle_range = (cfg['angle_min'], cfg['angle_max'])

        # Create a directory to store the parabolic data
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'data', 'simulation')
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_parabolic_data(self):
        all_data = []
        all_params = []

        for sample_id in range(self.num_samples):
            initial_velocity = np.random.uniform(*self.initial_velocity_range)
            angle_deg = np.random.uniform(*self.angle_range)
            angle_rad = np.radians(angle_deg)

            # Calculate the time of flight, coordinates of the parabola curve
            time_of_flight = 2 * initial_velocity * np.sin(angle_rad) / self.gravity_acceleration
            time = np.linspace(0, time_of_flight, self.num_points)
            x = initial_velocity * np.cos(angle_rad) * time
            y = initial_velocity * np.sin(angle_rad) * time - 0.5 * self.gravity_acceleration * time**2

            # Store the path data in a dictionary
            data_temp = pd.DataFrame({'sample_id': sample_id, 'time': time, 'x': x, 'y': y})
            all_data.append(data_temp)

            # Store the initial boundary conditions in a dictionary
            params_temp = pd.DataFrame({
                'sample_id': [sample_id],
                'initial_velocity': [initial_velocity],
                'angle (deg)': [angle_deg]
            })
            all_params.append(params_temp)

        self.motion_data_df = pd.concat(all_data, ignore_index=True)
        self.initial_params_df = pd.concat(all_params, ignore_index=True)

    def save_data(self):
        # Save the data to a csv file
        output_motion_file = os.path.join(self.data_dir, 'parabolic_motion.csv')
        output_params_file = os.path.join(self.data_dir, 'parabolic_params.csv')
        self.motion_data_df.to_csv(output_motion_file, index=False)
        self.initial_params_df.to_csv(output_params_file, index=False)

        print(f'Motion data saved to {output_motion_file}')
        print(f'Initial velocities and angles saved to {output_params_file}')

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg', 'cfg.yaml')
    parabolic_data_gen = ParabolicMotionDataGenerator(config_path)
    parabolic_data_gen.generate_parabolic_data()
    parabolic_data_gen.save_data()

if __name__ == '__main__':
    main()

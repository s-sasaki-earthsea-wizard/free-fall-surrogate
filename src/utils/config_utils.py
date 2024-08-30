import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

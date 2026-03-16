import yaml
import os

def load_config(config_file='config.yaml'):
    if not os.path.isabs(config_file):
        base_dir = os.path.dirname(__file__)
        config_file = os.path.join(base_dir, config_file)
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
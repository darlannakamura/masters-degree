import os, yaml
from typing import Dict

def load_config(file: str) -> Dict[str, str]:
    from settings import BASE_DIR

    config = {}

    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['output_path'] = os.path.join(BASE_DIR, config.get('output', 'results'), config.get('name', 'default'))
    os.makedirs(config['output_path'], exist_ok=True)

    config['metadata_path'] = os.path.join(config['output_path'], ".metadata")
    os.makedirs(config['metadata_path'], exist_ok=True)

    return config
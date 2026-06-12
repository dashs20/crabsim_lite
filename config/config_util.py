import yaml
import os

class ConfigStruct(dict):
    """A dictionary that allows dot-notation access to its keys."""
    def __init__(self, data=None, base_dir=None):
        if data is None:
            data = {}
        super().__init__(data)
        self.base_dir = base_dir
        
        for key, value in self.items():
            # If the value is a dictionary, wrap it in a ConfigStruct
            if isinstance(value, dict):
                self[key] = ConfigStruct(value, base_dir=self.base_dir)
            
            # If the value is a list, wrap any dictionaries inside it
            elif isinstance(value, list):
                self[key] = [ConfigStruct(i, base_dir=self.base_dir) if isinstance(i, dict) else i for i in value]
            
            # If the value is a string ending in .yaml, try to load it recursively
            elif isinstance(value, str) and value.endswith('.yaml'):
                # Resolve path relative to the current yaml's directory
                if self.base_dir:
                    yaml_path = os.path.join(self.base_dir, value)
                else:
                    yaml_path = value
                
                if os.path.exists(yaml_path):
                    self[key] = load_generic_config(yaml_path)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value

def load_generic_config(yaml_path):
    """Loads any YAML configuration and returns a ConfigStruct, resolving nested YAMLs."""
    base_dir = os.path.dirname(yaml_path)
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return ConfigStruct(data, base_dir=base_dir)

def load_plant_config(yaml_path):
    """Loads plant YAML configuration."""
    return load_generic_config(yaml_path)

def load_gnc_config(yaml_path):
    """Loads GNC YAML configuration (and recursively loads referenced plant YAML)."""
    return load_generic_config(yaml_path)

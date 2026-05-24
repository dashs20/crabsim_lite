import yaml

with open('crabcopter.yaml', 'r') as file:
    data = yaml.safe_load(file)


data['rotor_length_mm']
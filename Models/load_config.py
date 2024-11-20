import json
import yaml
import xml.etree.ElementTree as ET

'''
Given predefined base config class, updates it with the data from a JSON, YAML, or XML file
- Dynamically adds new keys as attributes to the config class from data if not already present
- Only import dynamic_load_config
'''

def dynamic_load_config(filepath: str, config_class):
    if filepath.endswith('.xml'):
        return load_config_from_xml(filepath, config_class)
    elif filepath.endswith('.json') or filepath.endswith('.yaml'):
        return load_config_from_json_or_yaml(filepath, config_class)
    else:
        return config_class()

def load_config_from_json_or_yaml(filepath: str, config_class):
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            config_data = json.load(f) or {}
        else:
            config_data = yaml.safe_load(f) or {}

    # Create an instance of the `Config` class
    config = config_class()

    # Update config with YAML data, adding any new keys as attributes
    for key, value in config_data.items():
        # Get the type of the default value, or set to same as value if not present
        tp = type(getattr(config, key, value))
        setattr(config, key, tp(value))  # Convert to the same type as the default

    return config

def load_config_from_xml(filepath: str, config_class):
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Create an instance of the config class
    config = config_class()

    # Update config with XML data, adding any new tags as attributes
    for elem in root:
        data = "" if elem.text is None else elem.text  # Fixing NoneType error
        # Get the type of the default value, or set to same as value if not present
        tp = type(getattr(config, elem.tag, data))
        setattr(config, elem.tag, tp(data))  # Convert to the same type as the default

    return config

def update_config_with_factor(config):
    config.nhid *= config.factor
    config.dim_feedfwd *= config.factor
    return config

import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Configuration class for experiments"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config

def load_config(config_file: str) -> ExperimentConfig:
    """Load configuration from YAML file"""
    
    # If relative path, make it relative to configs directory
    if not os.path.isabs(config_file):
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(config_dir, config_file)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(config_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def save_config(config: ExperimentConfig, config_file: str):
    """Save configuration to YAML file"""
    
    # If relative path, make it relative to configs directory
    if not os.path.isabs(config_file):
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(config_dir, config_file)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, indent=2)

def get_available_configs() -> list:
    """Get list of available configuration files"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_files = []
    
    for file in os.listdir(config_dir):
        if file.endswith('.yaml') and file != 'default.yaml':
            config_files.append(file)
    
    return config_files

def merge_configs(base_config: ExperimentConfig, override_config: ExperimentConfig) -> ExperimentConfig:
    """Merge two configurations, with override taking precedence"""
    
    def merge_dict(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dict(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = merge_dict(base_config.to_dict(), override_config.to_dict())
    return ExperimentConfig(merged)
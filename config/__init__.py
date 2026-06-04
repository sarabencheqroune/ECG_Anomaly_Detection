"""
Configuration management for ECG Anomaly Detection project

Provides centralized configuration loading and access across the project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Default paths
DEFAULT_PATHS = {
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'data_processed': PROJECT_ROOT / 'data' / 'processed',
    'data_cache': PROJECT_ROOT / 'data' / 'cache',
    'models': PROJECT_ROOT / 'models',
    'logs': PROJECT_ROOT / 'logs',
    'checkpoints': PROJECT_ROOT / 'checkpoints',
    'reports': PROJECT_ROOT / 'reports',
    'notebooks': PROJECT_ROOT / 'notebooks',
}


class Config:
    """Central configuration manager"""
    
    def __init__(self):
        self._config = {}
        self._load_defaults()
        
    def _load_defaults(self):
        """Load default configuration values"""
        self._config = {
            'paths': DEFAULT_PATHS.copy(),
            'data': {
                'sampling_rate': 360,
                'beat_length': 187,
                'pre_window_ms': 100,
                'post_window_ms': 420,
                'num_classes': 5,
                'class_names': ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other'],
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'batch_size': 64,
                'num_workers': 4,
                'use_augmentation': True
            },
            'model': {
                'type': 'hybrid',
                'input_dim': 187,
                'num_classes': 5,
                'latent_dim': 32,
                'dropout': 0.2,
                'use_batch_norm': True
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adamw',
                'weight_decay': 0.0001,
                'scheduler': 'cosine',
                'early_stopping_patience': 20,
                'gradient_clip': 1.0,
                'mixed_precision': True
            },
            'inference': {
                'confidence_threshold': 0.7,
                'anomaly_threshold': 0.5,
                'batch_size': 32,
                'device': 'auto'
            },
            'hitl': {
                'enabled': True,
                'confidence_threshold': 0.7,
                'max_queue_size': 1000,
                'critical_timeout_seconds': 300,
                'high_timeout_seconds': 3600,
                'medium_timeout_seconds': 86400,
                'low_timeout_seconds': 604800
            },
            'features': {
                'use_rr_intervals': True,
                'use_wavelet': True,
                'use_morphological': True,
                'wavelet_type': 'db6',
                'wavelet_level': 4
            },
            'reproducibility': {
                'seed': 42,
                'deterministic': True,
                'cuda_deterministic': True
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_to_console': True,
                'log_dir': DEFAULT_PATHS['logs'],
                'tensorboard': True,
                'metrics_logger': True
            }
        }
        
    def load_yaml(self, yaml_path: str):
        """Load configuration from YAML file"""
        path = Path(yaml_path)
        if path.exists():
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self._merge_config(yaml_config)
                
    def _merge_config(self, new_config: Dict, base: Optional[Dict] = None):
        """Recursively merge configuration dictionaries"""
        if base is None:
            base = self._config
            
        for key, value in new_config.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(value, base[key])
            else:
                base[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-notation key"""
        keys = key.split('.')
        target = self._config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
            
        target[keys[-1]] = value
        
    def update(self, updates: Dict):
        """Update configuration with dictionary"""
        self._merge_config(updates)
        
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        import copy
        return copy.deepcopy(self._config)
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
            
    def create_directories(self):
        """Create all necessary directories from config"""
        paths = self.get('paths', {})
        for path in paths.values():
            if isinstance(path, (str, Path)):
                Path(path).mkdir(parents=True, exist_ok=True)
                
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required paths
        for path_name, path in self.get('paths', {}).items():
            if not isinstance(path, Path):
                errors.append(f"Path '{path_name}' is not a Path object")
                
        # Validate data config
        data_config = self.get('data', {})
        if data_config.get('sampling_rate', 0) <= 0:
            errors.append("sampling_rate must be positive")
            
        if not 0 < data_config.get('train_split', 0) + data_config.get('val_split', 0) + data_config.get('test_split', 0) <= 1:
            errors.append("Data splits must sum to <= 1.0")
            
        # Validate training config
        training_config = self.get('training', {})
        if training_config.get('epochs', 0) <= 0:
            errors.append("epochs must be positive")
            
        if training_config.get('learning_rate', 0) <= 0:
            errors.append("learning_rate must be positive")
            
        # Validate HITL config
        hitl_config = self.get('hitl', {})
        if hitl_config.get('max_queue_size', 0) <= 0:
            errors.append("max_queue_size must be positive")
            
        if not 0 <= hitl_config.get('confidence_threshold', 0) <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
            
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
            
        return True
    
    def __repr__(self) -> str:
        return f"Config({self._config.keys()})"


# Global configuration instance
_config_instance = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        # Load config files if they exist
        config_dir = PROJECT_ROOT / 'config'
        if (config_dir / 'model_config.yaml').exists():
            _config_instance.load_yaml(str(config_dir / 'model_config.yaml'))
        if (config_dir / 'logging_config.yaml').exists():
            _config_instance.load_yaml(str(config_dir / 'logging_config.yaml'))
    return _config_instance


def reload_config():
    """Reload configuration from defaults and files"""
    global _config_instance
    _config_instance = Config()
    _config_instance.load_yaml(str(PROJECT_ROOT / 'config' / 'model_config.yaml'))
    _config_instance.load_yaml(str(PROJECT_ROOT / 'config' / 'logging_config.yaml'))
    return _config_instance


# Convenience function for quick config access
def get(key: str, default: Any = None) -> Any:
    """Quick access to configuration values"""
    return get_config().get(key, default)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {config.get('paths.PROJECT_ROOT')}")
    print(f"Model type: {config.get('model.type')}")
    print(f"Batch size: {config.get('data.batch_size')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    
    # Create directories
    config.create_directories()
    print("\nDirectories created successfully!")
    
    # Validate
    config.validate()
    print("\nConfiguration validation passed!")
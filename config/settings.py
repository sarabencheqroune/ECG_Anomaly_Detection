"""
Central configuration settings for ECG Anomaly Detection project

This module provides a unified interface for all project settings,
including paths, hyperparameters, and runtime configuration.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class DeviceType(Enum):
    """Available device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"


class ModelType(Enum):
    """Available model types"""
    CNN = "cnn"
    AUTOENCODER = "autoencoder"
    HYBRID = "hybrid"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


class LossType(Enum):
    """Available loss functions"""
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    HYBRID = "hybrid"
    CONTRASTIVE = "contrastive"


class OptimizerType(Enum):
    """Available optimizers"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Available learning rate schedulers"""
    STEP = "step"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    NONE = "none"


@dataclass
class PathConfig:
    """Path configuration for the project"""
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_cache: Path = PROJECT_ROOT / "data" / "cache"
    models: Path = PROJECT_ROOT / "models"
    logs: Path = PROJECT_ROOT / "logs"
    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    reports: Path = PROJECT_ROOT / "reports"
    notebooks: Path = PROJECT_ROOT / "notebooks"
    config: Path = PROJECT_ROOT / "config"
    
    def __post_init__(self):
        """Create all directories"""
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, Path) and field_name != 'project_root':
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration"""
    sampling_rate: int = 360
    beat_length: int = 187
    pre_window_ms: int = 100
    post_window_ms: int = 420
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other'])
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 64
    num_workers: int = 4
    use_augmentation: bool = True
    balance_classes: bool = True
    max_beats_per_record: int = 500
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    notch_freq_hz: float = 50.0
    normalize_method: str = "zscore"  # zscore, minmax, robust
    r_peak_method: str = "neurokit2"  # neurokit2, biosppy, hamilton


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    type: ModelType = ModelType.HYBRID
    input_dim: int = 187
    num_classes: int = 5
    latent_dim: int = 32
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    # CNN specific
    cnn_conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 3])
    cnn_pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    
    # Autoencoder specific
    ae_encoder_layers: List[int] = field(default_factory=lambda: [187, 128, 64, 32])
    ae_decoder_layers: List[int] = field(default_factory=lambda: [32, 64, 128, 187])
    ae_activation: str = "relu"
    
    # LSTM specific
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # Transformer specific
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 512


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAMW
    weight_decay: float = 0.0001
    scheduler: SchedulerType = SchedulerType.COSINE
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0001
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    loss_type: LossType = LossType.HYBRID
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    classification_weight: float = 1.0
    reconstruction_weight: float = 0.5
    contrastive_weight: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class InferenceConfig:
    """Inference configuration"""
    confidence_threshold: float = 0.7
    anomaly_threshold: float = 0.5
    batch_size: int = 32
    device: DeviceType = DeviceType.AUTO
    use_amp: bool = True
    cache_predictions: bool = True
    max_cache_size: int = 1000
    calibration_temperature: float = 1.0


@dataclass
class HITLConfig:
    """Human-in-the-Loop configuration"""
    enabled: bool = True
    confidence_threshold: float = 0.7
    max_queue_size: int = 1000
    critical_timeout_seconds: int = 300
    high_timeout_seconds: int = 3600
    medium_timeout_seconds: int = 86400
    low_timeout_seconds: int = 604800
    enable_escalation: bool = True
    auto_skip_after_timeout: bool = False
    persistence_enabled: bool = True
    persistence_file: str = "data/review_queue.json"
    use_redis: bool = False
    redis_url: Optional[str] = None


@dataclass
class FeaturesConfig:
    """Feature extraction configuration"""
    use_rr_intervals: bool = True
    use_wavelet: bool = True
    use_morphological: bool = True
    wavelet_type: str = "db6"
    wavelet_level: int = 4
    fusion_method: str = "concat"  # concat, weighted, attention
    feature_selection: bool = True
    n_selected_features: int = 50
    dimensionality_reduction: Optional[str] = "pca"  # pca, umap, None
    n_components: int = 30


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration"""
    seed: int = 42
    deterministic: bool = True
    cuda_deterministic: bool = True
    torch_backend: str = "default"
    disable_cudnn_benchmark: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_dir: Path = PROJECT_ROOT / "logs"
    tensorboard: bool = True
    metrics_logger: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    export_torchscript: bool = True
    export_onnx: bool = True
    export_onnx_opset: int = 11
    quantize: bool = False
    prune: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_workers: int = 4
    api_timeout: int = 30


class Settings:
    """
    Central settings manager for the entire project
    """
    
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.hitl = HITLConfig()
        self.features = FeaturesConfig()
        self.reproducibility = ReproducibilityConfig()
        self.logging = LoggingConfig()
        self.deployment = DeploymentConfig()
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create Settings instance from dictionary"""
        settings = cls()
        
        for key, value in config_dict.items():
            if hasattr(settings, key):
                current = getattr(settings, key)
                if hasattr(current, '__dataclass_fields__'):
                    # Update dataclass fields
                    for field_name, field_value in value.items():
                        if hasattr(current, field_name):
                            setattr(current, field_name, field_value)
                else:
                    setattr(settings, key, value)
                    
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        result = {}
        
        for attr_name in dir(self):
            if not attr_name.startswith('_') and hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if hasattr(attr, '__dataclass_fields__'):
                    # Convert dataclass to dict
                    result[attr_name] = {
                        field_name: getattr(attr, field_name)
                        for field_name in attr.__dataclass_fields__
                        if hasattr(attr, field_name)
                    }
                elif not callable(attr):
                    result[attr_name] = attr
                    
        return result
    
    def save(self, filepath: str):
        """Save settings to YAML file"""
        import yaml
        
        def convert_enum(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, 
                     sort_keys=False, indent=2, default_representer=convert_enum)
            
    @classmethod
    def load(cls, filepath: str) -> 'Settings':
        """Load settings from YAML file"""
        import yaml
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)
    
    def get_device(self) -> str:
        """Get the appropriate device based on configuration"""
        if self.inference.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.inference.device.value
    
    def __repr__(self) -> str:
        return f"Settings(project_root={self.paths.project_root})"


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from config files"""
    global _settings
    _settings = Settings()
    
    # Try to load from YAML files
    yaml_path = PROJECT_ROOT / 'config' / 'model_config.yaml'
    if yaml_path.exists():
        _settings = Settings.load(str(yaml_path))
        
    return _settings


if __name__ == "__main__":
    # Test settings
    settings = get_settings()
    print("Settings loaded successfully!")
    print(f"Project root: {settings.paths.project_root}")
    print(f"Model type: {settings.model.type.value}")
    print(f"Batch size: {settings.data.batch_size}")
    print(f"Learning rate: {settings.training.learning_rate}")
    print(f"Device: {settings.get_device()}")
    
    # Save settings
    settings.save("config/settings_backup.yaml")
    print("\nSettings saved to config/settings_backup.yaml")
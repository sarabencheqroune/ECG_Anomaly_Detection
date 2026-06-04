"""
Reproducibility utilities for consistent results across runs
"""

import random
import numpy as np
import torch
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility"""
    seed: int = 42
    deterministic: bool = True
    cuda_deterministic: bool = True
    torch_backend: str = "default"  # 'default', 'mkldnn', 'cudnn'
    disable_cudnn_benchmark: bool = True
    python_hash_seed: Optional[int] = None


def set_seed(seed: int = 42,
            deterministic: bool = True,
            cuda_deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility
    
    Args:
        seed: Base seed value
        deterministic: Enable deterministic algorithms
        cuda_deterministic: Enable CUDA deterministic mode
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Set deterministic flags
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if cuda_deterministic:
            # For CUDA 10.2+
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
                
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
    logger.info(f"Random seeds set to {seed} (deterministic={deterministic})")


def get_reproducible_environment(config: Optional[ReproducibilityConfig] = None) -> Dict[str, Any]:
    """
    Get current environment settings for reproducibility
    
    Args:
        config: Reproducibility configuration
        
    Returns:
        Dictionary with environment settings
    """
    if config is None:
        config = ReproducibilityConfig()
        
    set_seed(config.seed, config.deterministic, config.cuda_deterministic)
    
    if config.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
        
    # Set PyTorch backend if specified
    if config.torch_backend == 'mkldnn':
        torch.backends.mkldnn.enabled = True
    elif config.torch_backend == 'cudnn':
        torch.backends.cudnn.enabled = True
        
    env_info = {
        'seed': config.seed,
        'deterministic': config.deterministic,
        'cuda_deterministic': config.cuda_deterministic,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else None,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'Not set')
    }
    
    return env_info


def deterministic_dataloader(dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """
    Convert dataloader to deterministic mode (fixes random shuffling)
    
    Args:
        dataloader: Original dataloader
        
    Returns:
        Dataloader with deterministic shuffling
    """
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'generator'):
        # Set fixed seed for sampler
        dataloader.sampler.generator = torch.Generator()
        dataloader.sampler.generator.manual_seed(42)
        
    return dataloader


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        PyTorch device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        
    return device


class ReproducibleWorker:
    """
    Worker class for reproducible multi-processing
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize reproducible worker
        
        Args:
            seed: Seed for this worker
        """
        self.seed = seed
        
    def __call__(self, worker_id: int) -> None:
        """
        Set worker seed based on base seed + worker_id
        
        Args:
            worker_id: Worker identifier
        """
        worker_seed = self.seed + worker_id
        set_seed(worker_seed)
        logger.debug(f"Worker {worker_id} initialized with seed {worker_seed}")


def set_global_determinism(config: Optional[ReproducibilityConfig] = None) -> Dict[str, Any]:
    """
    Set global determinism for the entire project
    
    Args:
        config: Reproducibility configuration
        
    Returns:
        Environment information dictionary
    """
    if config is None:
        config = ReproducibilityConfig()
        
    env_info = get_reproducible_environment(config)
    
    # Log environment info
    logger.info("=" * 50)
    logger.info("Reproducibility Environment:")
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    return env_info


def save_reproducibility_info(save_path: str, config: Optional[ReproducibilityConfig] = None) -> None:
    """
    Save reproducibility information to file
    
    Args:
        save_path: Path to save the information
        config: Reproducibility configuration
    """
    import json
    from datetime import datetime
    
    env_info = get_reproducible_environment(config)
    env_info['timestamp'] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(env_info, f, indent=2)
        
    logger.info(f"Reproducibility info saved to {save_path}")


# Convenience function for quick setup
def setup_reproducible_experiment(seed: int = 42,
                                  save_info_path: Optional[str] = None) -> torch.device:
    """
    Quick setup for reproducible experiment
    
    Args:
        seed: Random seed
        save_info_path: Path to save reproducibility info
        
    Returns:
        Device to use
    """
    set_seed(seed, deterministic=True, cuda_deterministic=True)
    device = get_device()
    
    if save_info_path:
        save_reproducibility_info(save_info_path, ReproducibilityConfig(seed=seed))
        
    return device


if __name__ == "__main__":
    # Test reproducibility setup
    config = ReproducibilityConfig(seed=123, deterministic=True)
    env_info = set_global_determinism(config)
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test random number generation
    print(f"Random number: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    # Second run should produce same numbers
    set_seed(123)
    print(f"\nAfter resetting seed:")
    print(f"Random number: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    print("\nReproducibility test completed")
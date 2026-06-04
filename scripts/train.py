#!/usr/bin/env python
"""
CLI training entrypoint for ECG models
Usage: python train.py --config configs/train_config.yaml --model cnn
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ECGBeatDataset, create_dataloaders
from src.models.cnn_classifier import ECG1DCNN
from src.models.autoencoder import ECGDenoisingAutoencoder
from src.models.hybrid_model import HybridECGModel
from src.training.trainer import Trainer
from src.training.metrics import MetricsCalculator
from src.utils.reproducibility import setup_reproducible_experiment
from src.utils.logger import setup_logger, MetricsLogger

logger = setup_logger('training')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ECG models')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'autoencoder', 'hybrid'],
                       help='Model type to train')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    return {}


def create_model(model_type: str, config: dict):
    """Create model based on type"""
    input_length = config.get('input_dim', 187)  # input_dim refers to beat length
    num_classes = config.get('num_classes', 5)
    
    if model_type == 'cnn':
        model = ECG1DCNN(input_channels=1, input_length=input_length, num_classes=num_classes)
        logger.info(f"Created CNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif model_type == 'autoencoder':
        latent_dim = config.get('latent_dim', 32)
        model = ECGDenoisingAutoencoder(input_channels=1, input_length=input_length, latent_dim=latent_dim)
        logger.info(f"Created Autoencoder with latent dim {latent_dim}")
        
    elif model_type == 'hybrid':
        latent_dim = config.get('latent_dim', 32)
        model = HybridECGModel(input_channels=1, input_length=input_length, num_classes=num_classes, latent_dim=latent_dim)
        logger.info(f"Created Hybrid model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup reproducibility
    device = setup_reproducible_experiment(seed=args.seed)
    
    # Override device if specified
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
        
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    # Override with CLI arguments
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_type': args.model,
        'num_classes': 5,
        'input_dim': 187
    })
    
    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.model}_{timestamp}"
        
    # Setup logging directories
    log_dir = Path(f"./logs/{experiment_name}")
    checkpoint_dir = Path(f"./checkpoints/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config['log_dir'] = str(log_dir)
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    # Create data loaders
    logger.info("Loading datasets...")
    
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        use_augmentation=True
    )
    
    logger.info(f"Train samples: {len(dataloaders['train'].dataset)}")
    logger.info(f"Val samples: {len(dataloaders['val'].dataset)}")
    logger.info(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    # Create model
    model = create_model(args.model, config)
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")
        
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save final checkpoint
    trainer.save_checkpoint(f"final_model.pth")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate_with_loader(dataloaders['test'])
    
    # Save test metrics
    metrics_path = log_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
        
    logger.info("Test metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
            
    # Save experiment config
    config_path = log_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    logger.info(f"\n✅ Training completed! Results saved to {log_dir}")
    logger.info(f"Best model saved to {checkpoint_dir}")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()
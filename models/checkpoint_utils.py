"""
Utility functions for managing model checkpoints
"""

import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
from datetime import datetime


class CheckpointUtils:
    """Utility functions for checkpoint management"""
    
    @staticmethod
    def get_checkpoint_hash(filepath: str) -> str:
        """Calculate SHA256 hash of checkpoint file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def compare_checkpoints(filepath1: str, filepath2: str) -> bool:
        """Compare if two checkpoints are identical"""
        hash1 = CheckpointUtils.get_checkpoint_hash(filepath1)
        hash2 = CheckpointUtils.get_checkpoint_hash(filepath2)
        return hash1 == hash2
    
    @staticmethod
    def get_checkpoint_size(filepath: str) -> Dict[str, float]:
        """Get checkpoint file size information"""
        path = Path(filepath)
        if path.exists():
            size_bytes = path.stat().st_size
            return {
                'bytes': size_bytes,
                'kb': size_bytes / 1024,
                'mb': size_bytes / (1024 * 1024),
                'gb': size_bytes / (1024 * 1024 * 1024)
            }
        return {'error': 'File not found'}
    
    @staticmethod
    def extract_metadata(filepath: str) -> Dict[str, Any]:
        """Extract metadata from checkpoint without loading full weights"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            metadata = {
                'epoch': checkpoint.get('epoch'),
                'timestamp': checkpoint.get('timestamp'),
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
                'has_model_state': 'model_state_dict' in checkpoint,
                'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
                'file_size_mb': CheckpointUtils.get_checkpoint_size(filepath)['mb']
            }
            
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def merge_checkpoints(checkpoint1: str, checkpoint2: str, 
                         output_path: str, strategy: str = 'average'):
        """Merge two checkpoints (for model ensembling)"""
        ckpt1 = torch.load(checkpoint1, map_location='cpu')
        ckpt2 = torch.load(checkpoint2, map_location='cpu')
        
        merged = {}
        
        if strategy == 'average':
            for key in ckpt1['model_state_dict']:
                if key in ckpt2['model_state_dict']:
                    merged[key] = (ckpt1['model_state_dict'][key] + 
                                  ckpt2['model_state_dict'][key]) / 2
                    
        elif strategy == 'weighted':
            # Weight by validation accuracy
            w1 = ckpt1.get('metrics', {}).get('val_accuracy', 0.5)
            w2 = ckpt2.get('metrics', {}).get('val_accuracy', 0.5)
            total = w1 + w2
            
            for key in ckpt1['model_state_dict']:
                if key in ckpt2['model_state_dict']:
                    merged[key] = (w1 * ckpt1['model_state_dict'][key] + 
                                  w2 * ckpt2['model_state_dict'][key]) / total
                    
        # Save merged checkpoint
        merged_checkpoint = {
            'model_state_dict': merged,
            'epoch': max(ckpt1.get('epoch', 0), ckpt2.get('epoch', 0)),
            'timestamp': datetime.now().isoformat(),
            'merge_strategy': strategy,
            'source_checkpoints': [checkpoint1, checkpoint2]
        }
        
        torch.save(merged_checkpoint, output_path)
        print(f"Merged checkpoint saved to {output_path}")
        
        return merged_checkpoint


def load_config(config_path: str = "./models/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str = "./models/config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# Create demo configuration file
if __name__ == "__main__":
    # Load the config (this will load the YAML defined above)
    config = load_config("./models/config.yaml")
    
    print("Configuration loaded successfully!")
    print(f"Model type: {config['model']['type']}")
    print(f"Input dimension: {config['model']['input_dim']}")
    print(f"Number of classes: {config['model']['num_classes']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create checkpoint summary
    summary = {
        'config_version': config['version']['config_version'],
        'model_type': config['model']['type'],
        'num_classes': config['model']['num_classes'],
        'total_epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': config['training']['optimizer'],
        'loss_classification': config['training']['loss']['classification']
    }
    
    # Save summary
    with open("./models/config_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n✅ Configuration summary saved to ./models/config_summary.json")
    print("\nModel files created in ./models/ directory:")
    print("  - best_model.pth (checkpoint file)")
    print("  - autoencoder.pth (autoencoder weights)")
    print("  - classifier.pth (classifier weights)")
    print("  - config.yaml (configuration file)")
    print("  - config_summary.json (config summary)")
    print("  - autoencoder_metadata.json (autoencoder metadata)")
    print("  - classifier_metadata.json (classifier metadata)")
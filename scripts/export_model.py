#!/usr/bin/env python
"""
Export trained model to production formats (TorchScript, ONNX)
Usage: python export_model.py --checkpoint ./checkpoints/best_model.pth --format torchscript
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.models.cnn_classifier import ECG1DCNN
from src.models.hybrid_model import HybridECGModel
from src.utils.logger import setup_logger

logger = setup_logger('export')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export model to production format')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    parser.add_argument('--output', type=str, default='./exported_model',
                       help='Output directory or file path')
    
    parser.add_argument('--format', type=str, required=True,
                       choices=['torchscript', 'onnx', 'both'],
                       help='Export format')
    
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'hybrid'],
                       help='Model type')
    
    parser.add_argument('--input-dim', type=int, default=187,
                       help='Input dimension')
    
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes')
    
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize model for deployment')
    
    return parser.parse_args()


class ModelExporter:
    """Handle model export to various formats"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def export_torchscript(self, output_path: str, example_input: torch.Tensor = None):
        """Export to TorchScript"""
        logger.info("Exporting to TorchScript...")
        
        if example_input is None:
            example_input = torch.randn(1, 1, 187).to(self.device)
            
        try:
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(output_path)
            logger.info(f"TorchScript model saved to {output_path}")
            
            # Verify
            test_input = torch.randn(1, 1, 187).to(self.device)
            with torch.no_grad():
                original_output = self.model(test_input)
                traced_output = traced_model(test_input)
                
            if isinstance(original_output, tuple):
                original_output = original_output[0]
                traced_output = traced_output[0]
                
            if torch.allclose(original_output, traced_output, atol=1e-6):
                logger.info("✓ TorchScript verification passed")
            else:
                logger.warning("⚠ TorchScript verification failed - outputs differ")
                
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
            
    def export_onnx(self, output_path: str, example_input: torch.Tensor = None,
                   opset_version: int = 11):
        """Export to ONNX"""
        logger.info("Exporting to ONNX...")
        
        if example_input is None:
            example_input = torch.randn(1, 1, 187).to(self.device)
            
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"ONNX model saved to {output_path}")
            
            # Verify ONNX model
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")
            
        except ImportError:
            logger.error("ONNX not installed. Install with: pip install onnx")
            raise
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
            
    def quantize_model(self, calibration_loader=None):
        """Quantize model for deployment"""
        logger.info("Quantizing model...")
        
        try:
            if calibration_loader:
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
                )
                logger.info("✓ Dynamic quantization completed")
                return quantized_model
            else:
                # Static quantization requires calibration
                logger.warning("Static quantization requires calibration data")
                return self.model
                
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return self.model


def create_model(model_type: str, input_dim: int, num_classes: int):
    """Create model instance"""
    if model_type == 'cnn':
        model = ECG1DCNN(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'hybrid':
        model = HybridECGModel(input_dim=input_dim, num_classes=num_classes, latent_dim=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def main():
    """Main export function"""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(args.model, args.input_dim, args.num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Create exporter
    exporter = ModelExporter(model, device)
    
    # Create example input
    example_input = torch.randn(1, 1, args.input_dim).to(device)
    
    # Export based on format
    if args.format in ['torchscript', 'both']:
        ts_path = output_path / 'model.pt'
        exporter.export_torchscript(str(ts_path), example_input)
        
    if args.format in ['onnx', 'both']:
        onnx_path = output_path / 'model.onnx'
        exporter.export_onnx(str(onnx_path), example_input, args.opset_version)
        
    # Quantize if requested
    if args.quantize:
        quantized_model = exporter.quantize_model()
        if quantized_model:
            quantized_path = output_path / 'model_quantized.pt'
            traced_quantized = torch.jit.trace(quantized_model, example_input)
            traced_quantized.save(str(quantized_path))
            logger.info(f"Quantized model saved to {quantized_path}")
            
    # Save model info
    info = {
        'model_type': args.model,
        'input_dim': args.input_dim,
        'num_classes': args.num_classes,
        'export_formats': args.format,
        'quantized': args.quantize,
        'source_checkpoint': str(args.checkpoint)
    }
    
    import json
    info_path = output_path / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
        
    logger.info(f"\n✅ Export completed! Files saved to {output_path}")


if __name__ == "__main__":
    main()
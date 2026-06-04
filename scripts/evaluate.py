#!/usr/bin/env python
"""
Run comprehensive evaluation suite for trained models
Usage: python evaluate.py --checkpoint ./checkpoints/best_model.pth --model cnn
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ECGBeatDataset, create_dataloaders
from src.models.cnn_classifier import ECG1DCNN
from src.models.autoencoder import ECGDenoisingAutoencoder
from src.models.hybrid_model import HybridECGModel
from src.training.metrics import MetricsCalculator
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_training_history,
    plot_reconstruction_errors
)
from src.utils.logger import setup_logger

logger = setup_logger('evaluation')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate ECG models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn', 'autoencoder', 'hybrid'],
                       help='Model type')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save detailed predictions')
    
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate HTML report')
    
    return parser.parse_args()


class Evaluator:
    """Comprehensive model evaluator"""
    
    def __init__(self, model, device, output_dir: str):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator(num_classes=5)
        self.class_names = ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other']
        
    def evaluate(self, dataloader: DataLoader, save_predictions: bool = False) -> Dict:
        """Run comprehensive evaluation"""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_reconstructions = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                labels = batch[1].cpu().numpy()
                
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    if len(outputs) > 1:
                        reconstructions = outputs[1]
                        all_reconstructions.extend(reconstructions.cpu().numpy())
                else:
                    logits = outputs
                    
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                
                all_labels.extend(labels)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self.metrics_calc.compute_all(all_labels, all_predictions, all_probabilities)
        
        # Additional metrics
        metrics['confusion_matrix'] = self.metrics_calc.compute_confusion_matrix(
            all_labels, all_predictions
        ).tolist()
        
        metrics['classification_report'] = self.metrics_calc.compute_classification_report(
            all_labels, all_predictions
        )
        
        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = (all_labels == i)
            if mask.sum() > 0:
                per_class[name] = {
                    'support': int(mask.sum()),
                    'accuracy': np.mean(all_predictions[mask] == all_labels[mask]),
                    'precision': metrics['classification_report'][name]['precision'],
                    'recall': metrics['classification_report'][name]['recall'],
                    'f1-score': metrics['classification_report'][name]['f1-score']
                }
        metrics['per_class'] = per_class
        
        # Anomaly detection metrics
        anomaly_metrics = self.metrics_calc.compute_anomaly_metrics(
            all_labels, all_predictions, anomaly_class=4
        )
        metrics['anomaly_detection'] = anomaly_metrics
        
        # Reconstruction metrics (for autoencoders)
        if all_reconstructions:
            all_reconstructions = np.array(all_reconstructions)
            # Need original inputs for reconstruction metrics
            metrics['reconstruction'] = {'available': True}
            
        # Save predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'true_label': all_labels,
                'predicted_label': all_predictions,
                'true_class': [self.class_names[l] for l in all_labels],
                'predicted_class': [self.class_names[p] for p in all_predictions],
                'confidence': np.max(all_probabilities, axis=1),
                **{f'prob_{self.class_names[i]}': all_probabilities[:, i] 
                   for i in range(len(self.class_names))}
            })
            predictions_df.to_csv(self.output_dir / 'predictions.csv', index=False)
            
        return metrics
    
    def generate_visualizations(self, metrics: Dict):
        """Generate evaluation visualizations"""
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            fig = plot_confusion_matrix(cm, self.class_names, 
                                        title='Confusion Matrix - Test Set')
            fig.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            
        # ROC curves would require probability predictions per class
        # This would need to be implemented with proper ROC computation
        
        logger.info(f"Visualizations saved to {self.output_dir}")
        
    def generate_report(self, metrics: Dict) -> str:
        """Generate HTML evaluation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ECG Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
                .container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; 
                        margin: 10px; flex: 1; min-width: 200px; }}
                .card-title {{ font-weight: bold; margin-bottom: 10px; color: #555; }}
                .card-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>ECG Anomaly Detection Model Evaluation</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <div class="container">
                <div class="card">
                    <div class="card-title">Overall Accuracy</div>
                    <div class="card-value">{metrics.get('accuracy', 0):.2%}</div>
                </div>
                <div class="card">
                    <div class="card-title">Weighted F1 Score</div>
                    <div class="card-value">{metrics.get('f1_weighted', 0):.2%}</div>
                </div>
                <div class="card">
                    <div class="card-title">Macro AUC-ROC</div>
                    <div class="card-value">{metrics.get('auc_roc_macro', 0):.2%}</div>
                </div>
                <div class="card">
                    <div class="card-title">MCC</div>
                    <div class="card-value">{metrics.get('mcc', 0):.3f}</div>
                </div>
            </div>
            
            <h2>Per-Class Performance</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
        """
        
        for class_name, class_metrics in metrics.get('per_class', {}).items():
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{class_metrics.get('precision', 0):.2%}</td>
                    <td>{class_metrics.get('recall', 0):.2%}</td>
                    <td>{class_metrics.get('f1-score', 0):.2%}</td>
                    <td>{class_metrics.get('support', 0)}</td>
                </tr>
            """
            
        html_content += """
            </table>
            
            <h2>Confusion Matrix</h2>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            
            <h2>Anomaly Detection Performance</h2>
        """
        
        anomaly_metrics = metrics.get('anomaly_detection', {})
        html_content += f"""
            <div class="container">
                <div class="card">
                    <div class="card-title">Anomaly Detection Rate</div>
                    <div class="card-value">{anomaly_metrics.get('anomaly_detection_rate', 0):.2%}</div>
                </div>
                <div class="card">
                    <div class="card-title">False Alarm Rate</div>
                    <div class="card-value">{anomaly_metrics.get('false_alarm_rate', 0):.2%}</div>
                </div>
                <div class="card">
                    <div class="card-title">Anomaly F1 Score</div>
                    <div class="card-value">{anomaly_metrics.get('f1_anomaly', 0):.2%}</div>
                </div>
            </div>
        """
        
        html_content += """
        </body>
        </html>
        """
        
        report_path = self.output_dir / 'evaluation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Report saved to {report_path}")
        return str(report_path)


def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'cnn':
        model = ECG1DCNN(input_channels=1, input_length=187, num_classes=5)
    elif model_type == 'autoencoder':
        model = ECGDenoisingAutoencoder(input_channels=1, input_length=187, latent_dim=32)
    elif model_type == 'hybrid':
        model = HybridECGModel(input_channels=1, input_length=187, num_classes=5, latent_dim=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.model, device)
    
    # Create test dataloader
    test_dataset = ECGBeatDataset(
        data_dir=args.data_dir,
        split='test',
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create evaluator
    evaluator = Evaluator(model, device, args.output_dir)
    
    # Run evaluation
    logger.info("Running evaluation...")
    metrics = evaluator.evaluate(test_loader, save_predictions=args.save_predictions)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Precision (weighted): {metrics.get('precision_weighted', 0):.4f}")
    print(f"Recall (weighted): {metrics.get('recall_weighted', 0):.4f}")
    print(f"F1 Score (weighted): {metrics.get('f1_weighted', 0):.4f}")
    print(f"MCC: {metrics.get('mcc', 0):.4f}")
    print(f"AUC-ROC (macro): {metrics.get('auc_roc_macro', 0):.4f}")
    
    # Generate visualizations
    evaluator.generate_visualizations(metrics)
    
    # Generate report
    if args.generate_report:
        report_path = evaluator.generate_report(metrics)
        print(f"\nHTML report generated: {report_path}")
        
    # Save metrics
    metrics_path = args.output_dir / 'metrics.json'
    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
        
    metrics_serializable = {k: convert_for_json(v) for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
        
    print(f"\n✅ Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
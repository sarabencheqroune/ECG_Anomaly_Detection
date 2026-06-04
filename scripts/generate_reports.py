#!/usr/bin/env python
"""
Report generation utilities for ECG Anomaly Detection project

Generates comprehensive analysis reports including:
- Model evaluation metrics
- Training history visualization
- HITL feedback analysis
- Error analysis and visualizations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate various reports for model analysis"""
    
    def __init__(self, report_dir: Path = Path('./reports')):
        """
        Initialize report generator
        
        Args:
            report_dir: Directory to save reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_evaluation_summary(self,
                                    metrics: Dict[str, Any],
                                    model_name: str,
                                    output_format: str = 'json') -> Path:
        """
        Generate evaluation summary report
        
        Args:
            metrics: Dictionary of computed metrics
            model_name: Name of the model being evaluated
            output_format: 'json' or 'html'
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics,
            'report_type': 'evaluation_summary'
        }
        
        if output_format == 'json':
            output_path = self.report_dir / f'evaluation_{model_name}_{timestamp}.json'
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            output_path = self.report_dir / f'evaluation_{model_name}_{timestamp}.html'
            html_content = self._generate_evaluation_html(report_data)
            with open(output_path, 'w') as f:
                f.write(html_content)
                
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    
    def generate_training_report(self,
                                 history: Dict[str, List[float]],
                                 model_name: str,
                                 config: Optional[Dict] = None) -> Path:
        """
        Generate training history report
        
        Args:
            history: Training history dict with loss, accuracy lists
            model_name: Name of trained model
            config: Training configuration
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'timestamp': timestamp,
            'model_name': model_name,
            'config': config or {},
            'history': history,
            'report_type': 'training_history',
            'best_epoch': min(range(len(history.get('val_loss', []))),
                             key=lambda i: history.get('val_loss', [])[i]) if history.get('val_loss') else None,
            'best_val_loss': min(history.get('val_loss', [float('inf')])),
            'best_val_accuracy': max(history.get('val_accuracy', [0]))
        }
        
        output_path = self.report_dir / f'training_{model_name}_{timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        logger.info(f"Training report saved to {output_path}")
        return output_path
    
    def generate_error_analysis(self,
                               errors: List[Dict],
                               model_name: str,
                               top_n: int = 20) -> Path:
        """
        Generate error analysis report
        
        Args:
            errors: List of error records with details
            model_name: Name of model analyzed
            top_n: Number of top errors to include
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sort errors by severity
        sorted_errors = sorted(errors, 
                             key=lambda x: x.get('confidence_diff', 0),
                             reverse=True)[:top_n]
        
        report_data = {
            'timestamp': timestamp,
            'model_name': model_name,
            'report_type': 'error_analysis',
            'total_errors': len(errors),
            'top_errors': sorted_errors,
            'error_categories': self._categorize_errors(errors)
        }
        
        output_path = self.report_dir / f'error_analysis_{model_name}_{timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        logger.info(f"Error analysis report saved to {output_path}")
        return output_path
    
    def generate_comparison_report(self,
                                   models_metrics: Dict[str, Dict[str, float]]) -> Path:
        """
        Generate model comparison report
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate rankings
        rankings = self._calculate_rankings(models_metrics)
        
        report_data = {
            'timestamp': timestamp,
            'report_type': 'model_comparison',
            'models': models_metrics,
            'rankings': rankings,
            'best_model': rankings.get('overall', {}).get('1st', 'N/A')
        }
        
        output_path = self.report_dir / f'model_comparison_{timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        logger.info(f"Comparison report saved to {output_path}")
        return output_path
    
    @staticmethod
    def _generate_evaluation_html(report_data: Dict) -> str:
        """Generate HTML evaluation report"""
        metrics = report_data.get('metrics', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ECG Model Evaluation - {report_data['model_name']}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                       margin: 20px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
                .header h1 {{ margin: 0; font-size: 28px; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .card {{ background: white; padding: 20px; margin: 15px 0;
                        border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                               gap: 15px; margin: 20px 0; }}
                .metric-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-box .label {{ font-size: 12px; opacity: 0.9; }}
                .metric-box .value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .generated {{ text-align: right; font-size: 12px; color: #999;
                             margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ECG Anomaly Detection - Model Evaluation Report</h1>
                <p>Model: <strong>{report_data['model_name']}</strong> | 
                   Generated: <strong>{report_data['timestamp']}</strong></p>
            </div>
            
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="label">Overall Accuracy</div>
                    <div class="value">{metrics.get('accuracy', 0):.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Weighted F1 Score</div>
                    <div class="value">{metrics.get('f1_weighted', 0):.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Macro AUC-ROC</div>
                    <div class="value">{metrics.get('auc_roc_macro', 0):.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Matthews Correlation Coefficient</div>
                    <div class="value">{metrics.get('mcc', 0):.3f}</div>
                </div>
            </div>
            
            <div class="card">
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
            html += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{class_metrics.get('precision', 0):.1%}</td>
                        <td>{class_metrics.get('recall', 0):.1%}</td>
                        <td>{class_metrics.get('f1-score', 0):.1%}</td>
                        <td>{class_metrics.get('support', 0)}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="card">
                <h2>Anomaly Detection Performance</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        anomaly = metrics.get('anomaly_detection', {})
        for key, value in anomaly.items():
            html += f"<tr><td>{key}</td><td>{value:.1%}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="generated">
                <p>This report was automatically generated by the ECG Anomaly Detection system.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def _categorize_errors(errors: List[Dict]) -> Dict[str, int]:
        """Categorize errors by type"""
        categories = {}
        for error in errors:
            category = error.get('error_type', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    @staticmethod
    def _calculate_rankings(models_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """Calculate model rankings based on metrics"""
        rankings = {}
        
        # Rank by overall accuracy
        sorted_models = sorted(models_metrics.items(),
                             key=lambda x: x[1].get('accuracy', 0),
                             reverse=True)
        
        rankings['overall'] = {str(i+1) + ('st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'): model
                             for i, (model, _) in enumerate(sorted_models)}
        
        return rankings


def main():
    """Generate reports for all recent models"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    generator = ReportGenerator()
    logger.info("Report generator initialized")


if __name__ == '__main__':
    main()

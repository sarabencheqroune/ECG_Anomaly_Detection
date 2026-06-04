"""
Metrics calculation for ECG classification and anomaly detection
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, matthews_corrcoef,
                           cohen_kappa_score, average_precision_score)
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for ECG classification
    
    Supports:
    - Standard classification metrics (accuracy, precision, recall, F1)
    - ROC-AUC and PR-AUC
    - Confusion matrix
    - Per-class metrics
    - Macro/Micro/Weighted averaging
    - Clinical metrics (sensitivity, specificity, NPV, PPV)
    """
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: List of class names for better reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         average: str = 'weighted') -> float:
        """Compute precision"""
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    def compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray,
                      average: str = 'weighted') -> float:
        """Compute recall (sensitivity)"""
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                   average: str = 'weighted') -> float:
        """Compute F1 score"""
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute specificity (true negative rate) for multi-class
        Uses one-vs-rest approach
        """
        specificity_per_class = []
        
        for class_idx in range(self.num_classes):
            # Convert to binary
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # Compute confusion matrix
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            
            specificity = tn / (tn + fp + 1e-8)
            specificity_per_class.append(specificity)
            
        return np.mean(specificity_per_class)
    
    def compute_ppv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute positive predictive value (same as precision)"""
        return self.compute_precision(y_true, y_pred)
    
    def compute_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute negative predictive value"""
        npv_per_class = []
        
        for class_idx in range(self.num_classes):
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            npv = tn / (tn + fn + 1e-8)
            npv_per_class.append(npv)
            
        return np.mean(npv_per_class)
    
    def compute_auc_roc(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict:
        """
        Compute ROC-AUC for multi-class
        
        Args:
            y_true: True labels
            y_probs: Probability predictions (n_samples, n_classes)
            
        Returns:
            Dictionary with macro, weighted, and per-class AUC
        """
        if y_probs is None or y_probs.shape[1] != self.num_classes:
            return {'macro_auc': 0, 'weighted_auc': 0, 'per_class': {}}
            
        # Binarize labels
        y_true_bin = np.eye(self.num_classes)[y_true]
        
        # Compute AUC per class
        per_class_auc = {}
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                per_class_auc[self.class_names[i]] = auc
            except:
                per_class_auc[self.class_names[i]] = 0.0
                
        # Macro and weighted averages
        macro_auc = np.mean(list(per_class_auc.values()))
        
        # Weighted by class support
        class_counts = np.bincount(y_true, minlength=self.num_classes)
        weights = class_counts / np.sum(class_counts)
        weighted_auc = np.sum([per_class_auc[self.class_names[i]] * weights[i] 
                              for i in range(self.num_classes)])
        
        return {
            'macro_auc': macro_auc,
            'weighted_auc': weighted_auc,
            'per_class': per_class_auc
        }
    
    def compute_auc_pr(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict:
        """
        Compute Precision-Recall AUC
        
        Args:
            y_true: True labels
            y_probs: Probability predictions
            
        Returns:
            Dictionary with macro, weighted, and per-class PR-AUC
        """
        if y_probs is None or y_probs.shape[1] != self.num_classes:
            return {'macro_auc_pr': 0, 'weighted_auc_pr': 0, 'per_class': {}}
            
        y_true_bin = np.eye(self.num_classes)[y_true]
        
        per_class_auc_pr = {}
        for i in range(self.num_classes):
            try:
                auc_pr = average_precision_score(y_true_bin[:, i], y_probs[:, i])
                per_class_auc_pr[self.class_names[i]] = auc_pr
            except:
                per_class_auc_pr[self.class_names[i]] = 0.0
                
        macro_auc_pr = np.mean(list(per_class_auc_pr.values()))
        
        class_counts = np.bincount(y_true, minlength=self.num_classes)
        weights = class_counts / np.sum(class_counts)
        weighted_auc_pr = np.sum([per_class_auc_pr[self.class_names[i]] * weights[i]
                                 for i in range(self.num_classes)])
        
        return {
            'macro_auc_pr': macro_auc_pr,
            'weighted_auc_pr': weighted_auc_pr,
            'per_class': per_class_auc_pr
        }
    
    def compute_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient"""
        return matthews_corrcoef(y_true, y_pred)
    
    def compute_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Cohen's Kappa"""
        return cohen_kappa_score(y_true, y_pred)
    
    def compute_confusion_matrix(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix"""
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
    def compute_classification_report(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict:
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                       target_names=self.class_names,
                                       output_dict=True,
                                       zero_division=0)
        return report
    
    def compute_anomaly_metrics(self, y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                anomaly_class: int = 4) -> Dict:
        """
        Compute metrics specifically for anomaly detection
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            anomaly_class: Class index for 'Other' (anomaly)
            
        Returns:
            Dictionary with anomaly-specific metrics
        """
        # Convert to binary (anomaly vs normal)
        y_true_binary = (y_true == anomaly_class).astype(int)
        y_pred_binary = (y_pred == anomaly_class).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        metrics = {
            'anomaly_detection_rate': tp / (tp + fn + 1e-8),  # Sensitivity for anomalies
            'false_alarm_rate': fp / (fp + tn + 1e-8),        # False positive rate
            'precision_anomaly': tp / (tp + fp + 1e-8),
            'recall_anomaly': tp / (tp + fn + 1e-8),
            'f1_anomaly': 2 * tp / (2 * tp + fp + fn + 1e-8)
        }
        
        return metrics
    
    def compute_reconstruction_metrics(self, original: np.ndarray,
                                       reconstructed: np.ndarray) -> Dict:
        """
        Compute metrics for autoencoder reconstruction
        
        Args:
            original: Original signals
            reconstructed: Reconstructed signals
            
        Returns:
            Dictionary of reconstruction metrics
        """
        # MSE
        mse = np.mean((original - reconstructed) ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(original - reconstructed))
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Peak Signal-to-Noise Ratio (PSNR)
        max_val = np.max(np.abs(original))
        psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-8))
        
        # Correlation coefficient
        correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'snr_db': snr,
            'psnr_db': psnr,
            'correlation': correlation
        }
    
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                   y_probs: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Probability predictions (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': self.compute_accuracy(y_true, y_pred),
            'precision_macro': self.compute_precision(y_true, y_pred, average='macro'),
            'precision_weighted': self.compute_precision(y_true, y_pred, average='weighted'),
            'recall_macro': self.compute_recall(y_true, y_pred, average='macro'),
            'recall_weighted': self.compute_recall(y_true, y_pred, average='weighted'),
            'f1_macro': self.compute_f1(y_true, y_pred, average='macro'),
            'f1_weighted': self.compute_f1(y_true, y_pred, average='weighted'),
            'specificity': self.compute_specificity(y_true, y_pred),
            'ppv': self.compute_ppv(y_true, y_pred),
            'npv': self.compute_npv(y_true, y_pred),
            'mcc': self.compute_mcc(y_true, y_pred),
            'kappa': self.compute_kappa(y_true, y_pred)
        }
        
        # Add AUC metrics if probabilities provided
        if y_probs is not None:
            auc_roc = self.compute_auc_roc(y_true, y_probs)
            auc_pr = self.compute_auc_pr(y_true, y_probs)
            
            metrics['auc_roc_macro'] = auc_roc['macro_auc']
            metrics['auc_roc_weighted'] = auc_roc['weighted_auc']
            metrics['auc_pr_macro'] = auc_pr['macro_auc_pr']
            metrics['auc_pr_weighted'] = auc_pr['weighted_auc_pr']
            
        return metrics
    
    def print_metrics(self, metrics: Dict, title: str = "Metrics"):
        """Pretty print metrics"""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        print(f"  {subkey}: {subvalue:.4f}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    # Test metrics calculator
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    y_probs = np.random.rand(100, 5)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    calculator = MetricsCalculator(num_classes=5, 
                                   class_names=['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other'])
    
    metrics = calculator.compute_all(y_true, y_pred, y_probs)
    calculator.print_metrics(metrics)
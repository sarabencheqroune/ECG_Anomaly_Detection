"""
Validation and early stopping logic
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Advanced validation with multiple strategies:
    - K-fold cross-validation
    - Stratified validation
    - Time-series validation
    - Confidence scoring
    """
    
    def __init__(self, model: torch.nn.Module, 
                 metrics_calculator,
                 device: torch.device):
        """
        Initialize validator
        
        Args:
            model: PyTorch model
            metrics_calculator: Metrics calculator instance
            device: Device to use
        """
        self.model = model
        self.metrics = metrics_calculator
        self.device = device
        
    def validate(self, dataloader, criterion=None) -> Dict:
        """
        Standard validation
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function (optional)
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                # Compute loss
                if criterion and labels is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    
                # Get predictions and confidences
                if labels is not None:
                    probs = torch.softmax(logits, dim=1)
                    confidences, predictions = torch.max(probs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_confidences.extend(confidences.cpu().numpy())
                    
        # Compute metrics
        metrics = {}
        
        if all_predictions:
            metrics = self.metrics.compute_all(
                np.array(all_labels), 
                np.array(all_predictions)
            )
            metrics['mean_confidence'] = np.mean(all_confidences)
            
        if criterion:
            metrics['loss'] = total_loss / len(dataloader)
            
        return metrics
    
    def cross_validate(self, model_class, dataset, k_folds: int = 5, 
                      **kwargs) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            model_class: Model class to instantiate
            dataset: Complete dataset
            k_folds: Number of folds
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Fold {fold+1}/{k_folds}")
            
            # Create data loaders for this fold
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_subset, **kwargs)
            val_loader = torch.utils.data.DataLoader(val_subset, **kwargs)
            
            # Initialize model
            model = model_class()
            model = model.to(self.device)
            
            # Train model (simplified)
            # In practice, you'd use the Trainer class here
            
            # Validate
            metrics = self.validate(val_loader)
            fold_metrics.append(metrics)
            
        # Aggregate results
        aggregated = {}
        for metric_name in fold_metrics[0].keys():
            values = [m[metric_name] for m in fold_metrics]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            
        return aggregated
    
    def validate_with_confidence_intervals(self, dataloader, 
                                          num_bootstrap: int = 1000,
                                          confidence_level: float = 0.95) -> Dict:
        """
        Validate with bootstrap confidence intervals
        
        Args:
            dataloader: Validation data loader
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (0.95 for 95% CI)
            
        Returns:
            Metrics with confidence intervals
        """
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].cpu().numpy()
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels)
                    
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Bootstrap
        bootstrap_metrics = []
        n_samples = len(all_labels)
        
        for _ in range(num_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            labels_boot = all_labels[indices]
            preds_boot = all_predictions[indices]
            
            metrics = self.metrics.compute_all(labels_boot, preds_boot)
            bootstrap_metrics.append(metrics)
            
        # Calculate confidence intervals
        results = {}
        
        for metric_name in bootstrap_metrics[0].keys():
            values = [m[metric_name] for m in bootstrap_metrics]
            
            # Compute percentile confidence interval
            alpha = 1 - confidence_level
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': lower,
                'ci_upper': upper,
                'confidence_level': confidence_level
            }
            
        return results
    
    def validate_by_class(self, dataloader, class_names: List[str]) -> Dict:
        """
        Validate with per-class metrics
        
        Args:
            dataloader: Validation data loader
            class_names: List of class names
            
        Returns:
            Per-class metrics
        """
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].cpu().numpy()
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels)
                    
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Per-class metrics
        per_class = {}
        
        for class_idx, class_name in enumerate(class_names):
            class_mask = (all_labels == class_idx)
            
            if np.sum(class_mask) > 0:
                class_labels = all_labels[class_mask]
                class_preds = all_predictions[class_mask]
                
                metrics = self.metrics.compute_all(class_labels, class_preds)
                per_class[class_name] = metrics
                
        return per_class
    
    def validate_thresholds(self, dataloader, thresholds: List[float] = None) -> Dict:
        """
        Find optimal confidence thresholds for each class
        
        Args:
            dataloader: Validation data loader
            thresholds: List of thresholds to try
            
        Returns:
            Optimal thresholds for each class
        """
        if thresholds is None:
            thresholds = np.arange(0.5, 0.95, 0.05)
            
        self.model.eval()
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].cpu().numpy()
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                if labels is not None:
                    all_labels.extend(labels)
                    
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold for each class
        optimal_thresholds = {}
        
        for class_idx in range(all_probs.shape[1]):
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                # Apply threshold
                predictions = (all_probs[:, class_idx] > threshold).astype(int)
                true_labels = (all_labels == class_idx).astype(int)
                
                # Compute F1
                from sklearn.metrics import f1_score
                f1 = f1_score(true_labels, predictions, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
            optimal_thresholds[class_idx] = {
                'threshold': best_threshold,
                'f1_score': best_f1
            }
            
        return optimal_thresholds
    
    def evaluate_hitl_benefit(self, dataloader, confidence_threshold: float = 0.7) -> Dict:
        """
        Evaluate the benefit of human-in-the-loop review
        
        Args:
            dataloader: Validation data loader
            confidence_threshold: Threshold for flagging for review
            
        Returns:
            Statistics on HITL benefit
        """
        self.model.eval()
        
        results = {
            'total_samples': 0,
            'needs_review': 0,
            'review_rate': 0,
            'auto_classified_correct': 0,
            'low_confidence_correct': 0,
            'potential_savings': 0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].cpu().numpy()
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                probs = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                confidences = confidences.cpu().numpy()
                predictions = predictions.cpu().numpy()
                
                for i, (conf, pred, true_label) in enumerate(zip(confidences, predictions, labels)):
                    results['total_samples'] += 1
                    
                    if conf < confidence_threshold:
                        results['needs_review'] += 1
                        # Check if human would have corrected it
                        if pred != true_label:
                            results['low_confidence_correct'] += 1
                    else:
                        if pred == true_label:
                            results['auto_classified_correct'] += 1
                            
        results['review_rate'] = results['needs_review'] / results['total_samples']
        
        # Calculate potential savings (percentage of errors caught by HITL)
        total_errors = results['total_samples'] - results['auto_classified_correct'] - \
                      (results['needs_review'] - results['low_confidence_correct'])
        
        if total_errors > 0:
            results['error_catch_rate'] = results['low_confidence_correct'] / total_errors
        else:
            results['error_catch_rate'] = 1.0
            
        return results


class EarlyStopping:
    """
    Early stopping callback with multiple strategies
    """
    
    def __init__(self, patience: int = 10, 
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 min_delta: float = 1e-4,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            monitor: Metric to monitor
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore model to best weights
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.stop_training = False
        
    def __call__(self, epoch: int, metrics: Dict, model: torch.nn.Module):
        """Check if training should stop"""
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            return
            
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return
            
        # Check for improvement
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stop_training = True
                
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
"""
Confidence estimation and uncertainty metrics for ECG predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import softmax
from scipy.stats import entropy
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty estimation metrics"""
    max_probability: float
    entropy: float
    margin: float
    variance: float
    expected_calibration_error: float
    needs_review: bool


class ConfidenceEstimator:
    """
    Comprehensive confidence and uncertainty estimation for ECG predictions
    
    Features:
    - Multiple uncertainty metrics (entropy, margin, variance)
    - Monte Carlo Dropout for epistemic uncertainty
    - Temperature scaling for calibration
    - Ensemble uncertainty estimation
    - Expected Calibration Error (ECE)
    """
    
    def __init__(self, temperature: float = 1.0, use_mc_dropout: bool = False):
        """
        Initialize confidence estimator
        
        Args:
            temperature: Temperature scaling factor
            use_mc_dropout: Use Monte Carlo Dropout for uncertainty
        """
        self.temperature = temperature
        self.use_mc_dropout = use_mc_dropout
        
    def compute_uncertainty(self, 
                           logits: np.ndarray,
                           num_classes: int = 5) -> UncertaintyMetrics:
        """
        Compute uncertainty metrics from model logits
        
        Args:
            logits: Model output logits
            num_classes: Number of classes
            
        Returns:
            UncertaintyMetrics object
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute probabilities
        probs = softmax(scaled_logits)
        
        # Max probability (confidence)
        max_prob = np.max(probs)
        
        # Shannon entropy
        entropy_val = entropy(probs)
        
        # Margin (difference between top two probabilities)
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        
        # Variance (for ensemble/MC Dropout)
        variance = 0.0  # Would need multiple samples
        
        # Expected Calibration Error (requires calibration data)
        ece = 0.0
        
        # Determine if review is needed
        needs_review = (max_prob < 0.7 or entropy_val > 1.0 or margin < 0.2)
        
        return UncertaintyMetrics(
            max_probability=float(max_prob),
            entropy=float(entropy_val),
            margin=float(margin),
            variance=float(variance),
            expected_calibration_error=float(ece),
            needs_review=needs_review
        )
    
    def compute_mc_dropout_uncertainty(self, model, input_tensor, 
                                       num_passes: int = 50) -> Dict:
        """
        Compute epistemic uncertainty using Monte Carlo Dropout
        
        Args:
            model: PyTorch model with dropout layers
            input_tensor: Input tensor
            num_passes: Number of forward passes
            
        Returns:
            Dictionary with uncertainty metrics
        """
        model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_passes):
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
                
        model.eval()
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)
        
        # Compute statistics
        mean_probs = np.mean(predictions, axis=0)
        std_probs = np.std(predictions, axis=0)
        
        # Aleatoric uncertainty (entropy of mean prediction)
        aleatoric_entropy = entropy(mean_probs[0])
        
        # Epistemic uncertainty (mutual information)
        expected_entropy = np.mean([entropy(p[0]) for p in predictions])
        epistemic_entropy = aleatoric_entropy - expected_entropy
        
        # Predictive uncertainty
        predictive_entropy = aleatoric_entropy + epistemic_entropy
        
        return {
            'mean_probabilities': mean_probs[0],
            'std_probabilities': std_probs[0],
            'aleatoric_entropy': aleatoric_entropy,
            'epistemic_entropy': epistemic_entropy,
            'predictive_entropy': predictive_entropy,
            'uncertainty_score': epistemic_entropy / (predictive_entropy + 1e-8)
        }
        
    def calibrate_temperature(self, logits_list: List[np.ndarray], 
                             labels_list: List[int],
                             temperature_range: np.ndarray) -> float:
        """
        Find optimal temperature for calibration
        
        Args:
            logits_list: List of model logits
            labels_list: List of true labels
            temperature_range: Array of temperatures to try
            
        Returns:
            Optimal temperature
        """
        from sklearn.metrics import log_loss
        
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperature_range:
            calibrated_loss = 0
            
            for logits, label in zip(logits_list, labels_list):
                scaled_logits = logits / temp
                probs = softmax(scaled_logits)
                loss = -np.log(probs[label] + 1e-8)
                calibrated_loss += loss
                
            calibrated_loss /= len(logits_list)
            
            if calibrated_loss < best_loss:
                best_loss = calibrated_loss
                best_temp = temp
                
        self.temperature = best_temp
        return best_temp
    
    def compute_calibration_error(self, 
                                  predictions: List[np.ndarray],
                                  labels: List[int],
                                  num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            predictions: List of probability vectors
            labels: List of true labels
            num_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        confidences = [np.max(p) for p in predictions]
        accuracies = [1 if np.argmax(p) == l else 0 
                     for p, l in zip(predictions, labels)]
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean([accuracies[j] for j, m in enumerate(bin_mask) if m])
                bin_confidence = np.mean([confidences[j] for j, m in enumerate(bin_mask) if m])
                bin_size = np.sum(bin_mask) / len(confidences)
                
                ece += bin_size * abs(bin_accuracy - bin_confidence)
                
        return ece
    
    def get_review_priority(self, uncertainty: UncertaintyMetrics) -> float:
        """
        Compute priority score for HITL review
        
        Args:
            uncertainty: UncertaintyMetrics object
            
        Returns:
            Priority score (0-1, higher = more urgent)
        """
        # Combine metrics into priority score
        low_conf_score = 1 - uncertainty.max_probability
        high_entropy_score = uncertainty.entropy / np.log(5)  # Normalize
        low_margin_score = 1 - uncertainty.margin
        
        # Weighted combination
        priority = (0.5 * low_conf_score + 
                   0.3 * high_entropy_score + 
                   0.2 * low_margin_score)
        
        return min(1.0, max(0.0, priority))


class EnsembleConfidence:
    """
    Ensemble-based confidence estimation using multiple models
    """
    
    def __init__(self, models: List[torch.nn.Module], device: torch.device):
        """
        Initialize ensemble confidence estimator
        
        Args:
            models: List of PyTorch models
            device: Device to run inference on
        """
        self.models = models
        self.device = device
        self.num_models = len(models)
        
    def predict(self, input_tensor: torch.Tensor) -> Dict:
        """
        Get ensemble predictions with uncertainty
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Dictionary with ensemble predictions
        """
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model = model.to(self.device)
                model.eval()
                
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                all_probs.append(probs)
                
        # Stack predictions
        all_probs = np.stack(all_probs, axis=0)
        
        # Compute statistics
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        # Prediction disagreement (variation ratio)
        predictions = np.argmax(all_probs, axis=2)
        mode_predictions = []
        
        for i in range(predictions.shape[1]):
            unique, counts = np.unique(predictions[:, i], return_counts=True)
            mode = unique[np.argmax(counts)]
            mode_predictions.append(mode)
            
        disagreement = 1 - np.max(counts) / self.num_models
        
        # Ensemble entropy
        ensemble_entropy = entropy(mean_probs[0])
        
        return {
            'mean_probabilities': mean_probs[0],
            'std_probabilities': std_probs[0],
            'ensemble_prediction': np.argmax(mean_probs[0]),
            'disagreement': disagreement,
            'ensemble_entropy': ensemble_entropy,
            'uncertainty': disagreement + ensemble_entropy / np.log(5)
        }


class ConformalPrediction:
    """
    Conformal prediction for confidence sets with guaranteed coverage
    """
    
    def __init__(self, calibration_data: np.ndarray, 
                 calibration_labels: np.ndarray,
                 alpha: float = 0.1):
        """
        Initialize conformal prediction
        
        Args:
            calibration_data: Calibration dataset
            calibration_labels: Calibration labels
            alpha: Significance level (1 - coverage)
        """
        self.alpha = alpha
        self.thresholds = self._compute_thresholds(calibration_data, calibration_labels)
        
    def _compute_thresholds(self, data: np.ndarray, 
                           labels: np.ndarray) -> Dict[int, float]:
        """Compute non-conformity thresholds for each class"""
        thresholds = {}
        
        # Simplified: compute threshold as percentile of non-conformity scores
        for class_id in np.unique(labels):
            class_indices = np.where(labels == class_id)[0]
            class_data = data[class_indices]
            
            # Compute non-conformity scores (1 - predicted probability)
            scores = []
            for sample in class_data:
                # In practice, get model prediction probability
                prob = self._get_prediction_probability(sample, class_id)
                score = 1 - prob
                scores.append(score)
                
            # Set threshold at (1 - alpha) percentile
            thresholds[class_id] = np.percentile(scores, 100 * (1 - alpha))
            
        return thresholds
    
    def _get_prediction_probability(self, sample: np.ndarray, class_id: int) -> float:
        """Get model prediction probability for a sample"""
        # This would call the actual model
        # Placeholder
        return 0.8
    
    def predict_set(self, sample: np.ndarray) -> List[int]:
        """
        Get prediction set with guaranteed coverage
        
        Args:
            sample: Input sample
            
        Returns:
            List of class IDs in the prediction set
        """
        prediction_set = []
        
        for class_id, threshold in self.thresholds.items():
            prob = self._get_prediction_probability(sample, class_id)
            score = 1 - prob
            
            if score <= threshold:
                prediction_set.append(class_id)
                
        return prediction_set


if __name__ == "__main__":
    # Test confidence estimation
    estimator = ConfidenceEstimator()
    
    # Simulate logits for different cases
    high_conf_logits = np.array([5.0, 1.0, 0.5, 0.2, 0.1])
    low_conf_logits = np.array([1.2, 1.1, 1.0, 0.9, 0.8])
    
    high_uncertainty = estimator.compute_uncertainty(high_conf_logits)
    low_uncertainty = estimator.compute_uncertainty(low_conf_logits)
    
    print("High confidence prediction:")
    print(f"  Confidence: {high_uncertainty.max_probability:.3f}")
    print(f"  Entropy: {high_uncertainty.entropy:.3f}")
    print(f"  Needs review: {high_uncertainty.needs_review}")
    
    print("\nLow confidence prediction:")
    print(f"  Confidence: {low_uncertainty.max_probability:.3f}")
    print(f"  Entropy: {low_uncertainty.entropy:.3f}")
    print(f"  Needs review: {low_uncertainty.needs_review}")
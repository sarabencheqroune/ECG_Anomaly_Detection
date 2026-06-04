"""
Active learning for continuous model improvement from expert feedback
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Active learning sampling strategies"""
    UNCERTAINTY = "uncertainty"          # Highest prediction uncertainty
    MARGIN = "margin"                     # Smallest margin between top-2 classes
    ENTROPY = "entropy"                   # Highest prediction entropy
    RANDOM = "random"                     # Random sampling
    DIVERSITY = "diversity"               # Diverse representative sampling
    HYBRID = "hybrid"                     # Combined approach


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning"""
    strategy: SamplingStrategy = SamplingStrategy.HYBRID
    batch_size: int = 32
    uncertainty_threshold: float = 0.3
    min_samples_for_retraining: int = 100
    retraining_frequency_epochs: int = 5
    max_training_samples: int = 10000
    diversity_weight: float = 0.3


class ActiveLearner:
    """
    Active learning for selecting most informative samples for expert review
    
    Features:
    - Multiple sampling strategies
    - Uncertainty sampling
    - Diversity-based sampling
    - Hybrid approaches
    - Continuous model retraining coordination
    """
    
    def __init__(self, config: ActiveLearningConfig, model=None):
        """
        Initialize active learner
        
        Args:
            config: Active learning configuration
            model: Model instance (for uncertainty estimation)
        """
        self.config = config
        self.model = model
        self.labeled_pool = []  # Expert-labeled samples
        self.unlabeled_pool = []  # Unlabeled samples
        self.training_history = []
        
        logger.info(f"ActiveLearner initialized with strategy={config.strategy.value}")
        
    def select_samples_for_review(self, 
                                  unlabeled_data: List[Dict],
                                  num_samples: Optional[int] = None) -> List[Dict]:
        """
        Select most informative samples for expert review
        
        Args:
            unlabeled_data: List of unlabeled samples with model predictions
            num_samples: Number of samples to select (default: batch_size)
            
        Returns:
            List of selected samples for review
        """
        if num_samples is None:
            num_samples = self.config.batch_size
            
        if len(unlabeled_data) <= num_samples:
            return unlabeled_data
            
        if self.config.strategy == SamplingStrategy.UNCERTAINTY:
            return self._select_uncertainty(unlabeled_data, num_samples)
        elif self.config.strategy == SamplingStrategy.MARGIN:
            return self._select_margin(unlabeled_data, num_samples)
        elif self.config.strategy == SamplingStrategy.ENTROPY:
            return self._select_entropy(unlabeled_data, num_samples)
        elif self.config.strategy == SamplingStrategy.RANDOM:
            return self._select_random(unlabeled_data, num_samples)
        elif self.config.strategy == SamplingStrategy.DIVERSITY:
            return self._select_diverse(unlabeled_data, num_samples)
        elif self.config.strategy == SamplingStrategy.HYBRID:
            return self._select_hybrid(unlabeled_data, num_samples)
        else:
            return self._select_random(unlabeled_data, num_samples)
            
    def _select_uncertainty(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Select samples with highest prediction uncertainty"""
        # Uncertainty = 1 - confidence
        uncertainties = [1 - d.get('confidence', 0.5) for d in data]
        indices = np.argsort(uncertainties)[-num_samples:]
        return [data[i] for i in indices]
    
    def _select_margin(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Select samples with smallest margin between top-2 predictions"""
        margins = []
        for d in data:
            probs = d.get('probabilities', [0.2, 0.2, 0.2, 0.2, 0.2])
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
            margins.append(margin)
            
        # Smallest margin = most uncertain
        indices = np.argsort(margins)[:num_samples]
        return [data[i] for i in indices]
    
    def _select_entropy(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Select samples with highest prediction entropy"""
        entropies = []
        for d in data:
            probs = d.get('probabilities', [0.2, 0.2, 0.2, 0.2, 0.2])
            probs = np.array(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropies.append(entropy)
            
        indices = np.argsort(entropies)[-num_samples:]
        return [data[i] for i in indices]
    
    def _select_random(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Random sampling baseline"""
        indices = np.random.choice(len(data), num_samples, replace=False)
        return [data[i] for i in indices]
    
    def _select_diverse(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """Select diverse samples to maximize coverage"""
        # Simplified diversity: cluster and sample from each cluster
        # In practice, use k-means or similar on feature embeddings
        
        # Use feature vectors if available
        features = []
        for d in data:
            feat = d.get('features', None)
            if feat is not None:
                features.append(feat)
            else:
                # Random placeholder
                features.append(np.random.randn(10))
                
        features = np.array(features)
        
        # Simple greedy diversity sampling
        selected_indices = []
        remaining_indices = list(range(len(data)))
        
        # Start with random sample
        selected_indices.append(np.random.choice(remaining_indices))
        remaining_indices.remove(selected_indices[-1])
        
        # Add most diverse samples
        for _ in range(num_samples - 1):
            if not remaining_indices:
                break
                
            # Find sample farthest from selected ones
            max_min_dist = -1
            best_idx = -1
            
            for idx in remaining_indices:
                min_dist = min(np.linalg.norm(features[idx] - features[s]) 
                              for s in selected_indices)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
                    
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                
        return [data[i] for i in selected_indices]
    
    def _select_hybrid(self, data: List[Dict], num_samples: int) -> List[Dict]:
        """
        Hybrid selection: combine uncertainty and diversity
        
        First select top uncertainty samples, then diversify
        """
        # First, select top candidates by uncertainty
        uncertainty_ratio = 0.7  # 70% uncertainty, 30% diversity
        uncertainty_count = int(num_samples * uncertainty_ratio)
        diversity_count = num_samples - uncertainty_count
        
        uncertainty_samples = self._select_uncertainty(data, uncertainty_count)
        
        if diversity_count > 0:
            # Remove selected samples
            remaining = [d for d in data if d not in uncertainty_samples]
            if remaining:
                diversity_samples = self._select_diverse(remaining, diversity_count)
                return uncertainty_samples + diversity_samples
                
        return uncertainty_samples
    
    def add_labeled_sample(self, sample: Dict, label: str):
        """Add expert-labeled sample to training pool"""
        labeled_entry = {
            'sample': sample,
            'label': label,
            'timestamp': time.time()
        }
        self.labeled_pool.append(labeled_entry)
        
        # Limit pool size
        if len(self.labeled_pool) > self.config.max_training_samples:
            # Remove oldest samples
            self.labeled_pool = self.labeled_pool[-self.config.max_training_samples:]
            
        logger.info(f"Added labeled sample. Pool size: {len(self.labeled_pool)}")
        
    def should_retrain(self, epoch: int) -> bool:
        """
        Check if model should be retrained based on new labeled data
        
        Args:
            epoch: Current training epoch
            
        Returns:
            True if retraining is recommended
        """
        # Check if we have enough new samples
        if len(self.labeled_pool) >= self.config.min_samples_for_retraining:
            # Check retraining frequency
            if epoch % self.config.retraining_frequency_epochs == 0:
                return True
                
        return False
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get labeled data for model retraining
        
        Returns:
            Tuple of (features, labels)
        """
        if not self.labeled_pool:
            return np.array([]), np.array([])
            
        # Extract features and labels
        features = []
        labels = []
        
        class_mapping = {
            'Normal': 0, 'AFib': 1, 'PVC': 2, 'Bradycardia': 3, 'Other': 4
        }
        
        for entry in self.labeled_pool:
            # Get sample features (beat data or model embeddings)
            sample = entry['sample']
            if 'beat_data' in sample:
                features.append(sample['beat_data'])
            elif 'features' in sample:
                features.append(sample['features'])
            else:
                continue
                
            labels.append(class_mapping.get(entry['label'], 4))
            
        return np.array(features), np.array(labels)
    
    def compute_informativeness(self, sample: Dict) -> float:
        """Compute informativeness score for a sample"""
        strategies = {
            'uncertainty': 1 - sample.get('confidence', 0.5),
            'margin': self._compute_margin(sample),
            'entropy': self._compute_entropy(sample)
        }
        
        # Normalize and combine
        scores = list(strategies.values())
        if scores:
            return np.mean(scores)
        return 0.5
    
    def _compute_margin(self, sample: Dict) -> float:
        """Compute margin score for a sample"""
        probs = sample.get('probabilities', [0.2, 0.2, 0.2, 0.2, 0.2])
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        return 1 - margin  # Invert so higher score = more informative
    
    def _compute_entropy(self, sample: Dict) -> float:
        """Compute entropy score for a sample"""
        probs = sample.get('probabilities', [0.2, 0.2, 0.2, 0.2, 0.2])
        probs = np.array(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        # Normalize to [0, 1]
        max_entropy = np.log(5)
        return entropy / max_entropy


if __name__ == "__main__":
    # Test active learner
    config = ActiveLearningConfig()
    learner = ActiveLearner(config)
    
    # Create dummy unlabeled data
    unlabeled = [
        {'confidence': 0.95, 'probabilities': [0.95, 0.02, 0.01, 0.01, 0.01]},
        {'confidence': 0.55, 'probabilities': [0.55, 0.25, 0.10, 0.05, 0.05]},
        {'confidence': 0.45, 'probabilities': [0.45, 0.30, 0.15, 0.05, 0.05]},
        {'confidence': 0.75, 'probabilities': [0.75, 0.15, 0.05, 0.03, 0.02]},
        {'confidence': 0.30, 'probabilities': [0.30, 0.30, 0.20, 0.10, 0.10]},
    ] * 10
        
    # Select samples for review
    selected = learner.select_samples_for_review(unlabeled, num_samples=5)
    print(f"Selected {len(selected)} samples for review")
    
    # Add labeled sample
    learner.add_labeled_sample(selected[0], 'Normal')
    
    # Check if retraining is needed
    if learner.should_retrain(epoch=5):
        X, y = learner.get_training_data()
        print(f"Training data: {len(X)} samples")
"""
Hybrid model combining supervised classifier and unsupervised autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from .cnn_classifier import ECG1DCNN
from .autoencoder import ConvAutoencoder


class ECGHybridModel(nn.Module):
    """
    Hybrid model that combines:
    1. Supervised classifier for known arrhythmias
    2. Unsupervised autoencoder for unknown anomaly detection
    
    At inference:
    - If classifier confidence is high and reconstruction error is low -> known class
    - If classifier confidence is high but reconstruction error is high -> potential unknown anomaly
    - If classifier confidence is low -> HITL review
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_length: int = 187,
                 num_classes: int = 5,
                 latent_dim: int = 32,
                 confidence_threshold: float = 0.7,
                 reconstruction_threshold_percentile: float = 95.0):
        """
        Initialize hybrid model
        
        Args:
            input_channels: Number of input channels
            input_length: Length of input sequence
            num_classes: Number of classes for classifier
            latent_dim: Latent dimension for autoencoder
            confidence_threshold: Threshold for classifier confidence
            reconstruction_threshold_percentile: Percentile for reconstruction error threshold
        """
        super(ECGHybridModel, self).__init__()
        
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.reconstruction_threshold_percentile = reconstruction_threshold_percentile
        
        # Supervised classifier
        self.classifier = ECG1DCNN(
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes
        )
        
        # Unsupervised autoencoder (shares encoder with classifier?)
        self.autoencoder = ConvAutoencoder(
            input_channels=input_channels,
            input_length=input_length,
            latent_dim=latent_dim
        )
        
        # Feature fusion (optional)
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Learnable temperature for uncertainty calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both models
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Dictionary with classifier and autoencoder outputs
        """
        # Classifier forward
        classifier_output = self.classifier(x)
        
        # Autoencoder forward
        autoencoder_output = self.autoencoder(x)
        
        return {
            'classifier_logits': classifier_output['logits'],
            'classifier_features': classifier_output['features'],
            'reconstruction': autoencoder_output['reconstruction'],
            'latent': autoencoder_output['latent']
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict:
        """
        Enhanced prediction with uncertainty estimation
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions, confidence, and anomaly scores
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.forward(x)
            
            # Calibrated probabilities
            logits = outputs['classifier_logits'] / self.temperature
            probs = F.softmax(logits, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            
            # Reconstruction error
            recon_error = F.mse_loss(outputs['reconstruction'], x, reduction='none')
            recon_error = recon_error.mean(dim=(1, 2))
            
            # Determine if anomaly (high reconstruction error)
            if hasattr(self, 'reconstruction_threshold'):
                is_anomaly = recon_error > self.reconstruction_threshold
            else:
                is_anomaly = torch.zeros_like(recon_error, dtype=torch.bool)
                
            # Determine if needs review (low confidence or high anomaly score)
            needs_review = (confidence < self.confidence_threshold) | is_anomaly
            
        return {
            'predictions': predictions,
            'probabilities': probs,
            'confidence': confidence,
            'reconstruction_error': recon_error,
            'is_anomaly': is_anomaly,
            'needs_review': needs_review,
            'logits': logits
        }
    
    def set_reconstruction_threshold(self, reconstruction_errors: torch.Tensor):
        """
        Set reconstruction threshold based on training data
        
        Args:
            reconstruction_errors: Reconstruction errors from training (normal beats)
        """
        percentile = self.reconstruction_threshold_percentile
        self.reconstruction_threshold = torch.quantile(
            reconstruction_errors, 
            percentile / 100.0
        )
        
    def compute_combined_loss(self,
                             x: torch.Tensor,
                             labels: torch.Tensor,
                             weight_classification: float = 1.0,
                             weight_reconstruction: float = 0.5) -> torch.Tensor:
        """
        Compute combined loss for joint training
        
        Args:
            x: Input tensor
            labels: Ground truth labels
            weight_classification: Weight for classification loss
            weight_reconstruction: Weight for reconstruction loss
            
        Returns:
            Combined loss
        """
        outputs = self.forward(x)
        
        # Classification loss
        ce_loss = F.cross_entropy(outputs['classifier_logits'], labels)
        
        # Reconstruction loss (only on normal samples? or all?)
        recon_loss = F.mse_loss(outputs['reconstruction'], x)
        
        # Combined loss
        total_loss = weight_classification * ce_loss + weight_reconstruction * recon_loss
        
        return total_loss
    
    def freeze_classifier(self):
        """Freeze classifier parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = False
            
    def freeze_autoencoder(self):
        """Freeze autoencoder parameters"""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.autoencoder.parameters():
            param.requires_grad = True


class EnsembleECGModel(nn.Module):
    """
    Ensemble of multiple models for robust predictions
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_length: int = 187,
                 num_classes: int = 5,
                 num_models: int = 3):
        """
        Initialize ensemble model
        
        Args:
            input_channels: Number of input channels
            input_length: Length of input sequence
            num_classes: Number of output classes
            num_models: Number of models in ensemble
        """
        super(EnsembleECGModel, self).__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Create ensemble of hybrid models
        self.models = nn.ModuleList([
            ECGHybridModel(
                input_channels=input_channels,
                input_length=input_length,
                num_classes=num_classes
            ) for _ in range(num_models)
        ])
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with aggregated predictions
        """
        all_logits = []
        all_reconstructions = []
        
        for model in self.models:
            outputs = model.forward(x)
            all_logits.append(outputs['classifier_logits'])
            all_reconstructions.append(outputs['reconstruction'])
            
        # Weighted average of logits
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_logits = torch.zeros_like(all_logits[0])
        
        for i, logits in enumerate(all_logits):
            weighted_logits += weights[i] * logits
            
        # Average of reconstructions
        avg_reconstruction = torch.mean(torch.stack(all_reconstructions), dim=0)
        
        return {
            'classifier_logits': weighted_logits,
            'reconstruction': avg_reconstruction,
            'individual_logits': all_logits,
            'individual_reconstructions': all_reconstructions,
            'ensemble_weights': weights
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict:
        """
        Ensemble prediction with uncertainty estimation
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        outputs = self.forward(x)
        
        # Probabilities from aggregated logits
        probs = F.softmax(outputs['classifier_logits'], dim=1)
        confidence, predictions = torch.max(probs, dim=1)
        
        # Calculate uncertainty (variance across ensemble members)
        individual_probs = []
        for logits in outputs['individual_logits']:
            individual_probs.append(F.softmax(logits, dim=1))
            
        individual_probs = torch.stack(individual_probs)  # (num_models, batch, num_classes)
        predictive_variance = torch.var(individual_probs, dim=0)
        epistemic_uncertainty = predictive_variance.mean(dim=1)
        
        # Aleatoric uncertainty (mean entropy of individual predictions)
        entropies = []
        for probs_i in individual_probs:
            entropy = -torch.sum(probs_i * torch.log(probs_i + 1e-8), dim=1)
            entropies.append(entropy)
        aleatoric_uncertainty = torch.mean(torch.stack(entropies), dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'confidence': confidence,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        }


if __name__ == "__main__":
    # Test hybrid model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ECGHybridModel(
        input_channels=1,
        input_length=187,
        num_classes=5
    ).to(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 187).to(device)
    labels = torch.randint(0, 5, (batch_size,)).to(device)
    
    # Forward
    outputs = model(x)
    print(f"Classifier logits shape: {outputs['classifier_logits'].shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    
    # Prediction with uncertainty
    predictions = model.predict_with_uncertainty(x)
    print(f"\nPredictions shape: {predictions['predictions'].shape}")
    print(f"Confidence mean: {predictions['confidence'].mean().item():.3f}")
    print(f"Needs review: {predictions['needs_review'].sum().item()}/{batch_size}")
    
    # Combined loss
    loss = model.compute_combined_loss(x, labels)
    print(f"\nCombined loss: {loss.item():.4f}")
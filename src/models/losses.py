"""
Loss functions for ECG anomaly detection models

Includes:
- Standard classification losses (CrossEntropy, Focal Loss)
- Reconstruction losses for autoencoders (MSE, MAE, Huber)
- Contrastive losses (Triplet Loss, Contrastive Loss)
- Combination losses for hybrid models
- Regularization losses (sparsity, orthogonality)
- ECG-specific losses (morphology-aware, temporal consistency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Focal Loss focuses training on hard examples by down-weighting easy examples.
    Especially useful for imbalanced ECG datasets where normal beats dominate.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for classes (can be tensor or float)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Target labels (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with automatic class weight calculation
    
    Args:
        class_weights: Manual class weights (tensor of shape num_classes)
        weight_method: 'inverse_frequency', 'balanced', or None
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 weight_method: Optional[str] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.class_weights = class_weights
        self.weight_method = weight_method
        self.reduction = reduction
        
    def set_class_weights(self, class_counts: torch.Tensor):
        """
        Compute class weights based on class frequencies
        
        Args:
            class_counts: Number of samples per class (tensor)
        """
        if self.weight_method == 'inverse_frequency':
            # Inverse frequency weighting
            weights = 1.0 / (class_counts.float() + 1e-8)
            self.class_weights = weights / weights.sum()
            
        elif self.weight_method == 'balanced':
            # Balanced weighting (inverse sqrt)
            weights = 1.0 / torch.sqrt(class_counts.float() + 1e-8)
            self.class_weights = weights / weights.sum()
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(inputs.device)
            
        return F.cross_entropy(inputs, targets, 
                              weight=self.class_weights, 
                              reduction=self.reduction)


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for autoencoders
    
    Supports multiple loss types and combination strategies
    """
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 reduction: str = 'mean',
                 weighted: bool = True):
        """
        Args:
            loss_type: 'mse', 'mae', 'huber', or 'combined'
            reduction: 'mean', 'sum', or 'none'
            weighted: Apply weighted loss (emphasize QRS complex)
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.weighted = weighted
        
        # Huber loss delta
        self.huber_delta = 1.0
        
    def forward(self, recon: torch.Tensor, 
                target: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            recon: Reconstructed signal (batch_size, channels, sequence_len)
            target: Original signal (batch_size, channels, sequence_len)
            weights: Optional per-sample weights for weighted loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(recon, target, reduction='none')
            
        elif self.loss_type == 'mae':
            loss = F.l1_loss(recon, target, reduction='none')
            
        elif self.loss_type == 'huber':
            diff = recon - target
            abs_diff = torch.abs(diff)
            quadratic = torch.clamp(abs_diff, max=self.huber_delta)
            linear = abs_diff - quadratic
            loss = 0.5 * quadratic ** 2 + self.huber_delta * linear
            
        elif self.loss_type == 'combined':
            # Combined MSE + MAE (like in VAE)
            mse = F.mse_loss(recon, target, reduction='none')
            mae = F.l1_loss(recon, target, reduction='none')
            loss = mse + 0.5 * mae
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        # Apply per-sample weights (e.g., emphasize QRS region)
        if self.weighted and weights is not None:
            loss = loss * weights
            
        # Reduce across dimensions
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning embeddings where similar samples are close
    and dissimilar samples are far apart.
    
    Used in Siamese networks for anomaly detection.
    
    Args:
        margin: Margin for dissimilar pairs (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive sample embeddings (same class)
            negative: Negative sample embeddings (different class)
        """
        # Euclidean distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Contrastive loss
        loss = pos_distance + torch.clamp(self.margin - neg_distance, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning
    
    Ensures that anchor-positive distance < anchor-negative distance by a margin.
    Useful for learning discriminative beat embeddings.
    
    Args:
        margin: Margin between positive and negative pairs
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive sample embeddings (same class)
            negative: Negative sample embeddings (different class)
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) for multi-class classification
    
    Pulls together samples from same class and pushes apart samples from
    different classes in embedding space.
    
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Embedding features (batch_size, embedding_dim)
            labels: Class labels (batch_size,)
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)
        
        # Compute log softmax
        exp_sim = torch.exp(similarity)
        
        # Sum over positive pairs
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        
        # Sum over all pairs (including negatives)
        all_sum = torch.sum(exp_sim, dim=1)
        
        # Compute loss
        loss = -torch.log(pos_sum / (all_sum + 1e-8))
        
        # Mask out samples with no positive pairs
        loss = loss * (pos_sum > 0).float()
        
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VAELoss(nn.Module):
    """
    Variational Autoencoder Loss (ELBO)
    
    Combines reconstruction loss with KL divergence regularization.
    """
    
    def __init__(self, 
                 beta: float = 1.0,
                 reconstruction_loss: str = 'mse',
                 anneal_epochs: int = 10):
        """
        Args:
            beta: Weight for KL divergence (β-VAE)
            reconstruction_loss: Type of reconstruction loss
            anneal_epochs: Number of epochs to anneal beta from 0 to full
        """
        super().__init__()
        self.beta = beta
        self.anneal_epochs = anneal_epochs
        self.recon_loss_fn = ReconstructionLoss(loss_type=reconstruction_loss)
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for beta annealing"""
        self.current_epoch = epoch
        
    def forward(self, recon: torch.Tensor, 
                target: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            recon: Reconstructed signal
            target: Original target signal
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(recon, target)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / target.shape[0]  # Normalize by batch size
        
        # Anneal beta
        if self.current_epoch < self.anneal_epochs:
            current_beta = self.beta * (self.current_epoch / self.anneal_epochs)
        else:
            current_beta = self.beta
            
        total_loss = recon_loss + current_beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class HybridLoss(nn.Module):
    """
    Combined loss for hybrid model (classifier + autoencoder)
    
    Balances classification accuracy with reconstruction quality
    and anomaly detection capabilities.
    """
    
    def __init__(self,
                 classification_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 contrastive_weight: float = 0.0,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0):
        """
        Args:
            classification_weight: Weight for classification loss
            reconstruction_weight: Weight for reconstruction loss
            contrastive_weight: Weight for contrastive loss
            use_focal_loss: Use focal loss for classification
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.class_weight = classification_weight
        self.recon_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        
        # Initialize losses
        if use_focal_loss:
            self.class_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.class_loss = nn.CrossEntropyLoss()
            
        self.recon_loss = ReconstructionLoss(loss_type='mse')
        
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss()
        else:
            self.contrastive_loss = None
            
    def forward(self,
                class_logits: torch.Tensor,
                recon: torch.Tensor,
                original: torch.Tensor,
                labels: torch.Tensor,
                anchor_embeddings: Optional[torch.Tensor] = None,
                positive_embeddings: Optional[torch.Tensor] = None,
                negative_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            class_logits: Classification logits
            recon: Reconstructed signal
            original: Original signal
            labels: Class labels
            anchor_embeddings: Anchor embeddings (for contrastive)
            positive_embeddings: Positive sample embeddings
            negative_embeddings: Negative sample embeddings
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Classification loss
        if class_logits is not None:
            losses['classification'] = self.class_loss(class_logits, labels)
            
        # Reconstruction loss
        if recon is not None and original is not None:
            losses['reconstruction'] = self.recon_loss(recon, original)
            
        # Contrastive loss
        if (self.contrastive_weight > 0 and 
            anchor_embeddings is not None and
            positive_embeddings is not None and
            negative_embeddings is not None):
            losses['contrastive'] = self.contrastive_loss(
                anchor_embeddings, positive_embeddings, negative_embeddings
            )
            
        # Total loss
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if 'classification' in losses:
            total_loss = total_loss + self.class_weight * losses['classification']
        if 'reconstruction' in losses:
            total_loss = total_loss + self.recon_weight * losses['reconstruction']
        if 'contrastive' in losses:
            total_loss = total_loss + self.contrastive_weight * losses['contrastive']
            
        losses['total'] = total_loss
        
        return losses


class MorphologyAwareLoss(nn.Module):
    """
    ECG-specific morphology-aware loss
    
    Emphasizes clinically important features like:
    - QRS complex morphology
    - ST segment deviations
    - P and T wave preservation
    """
    
    def __init__(self,
                 sampling_rate: int = 360,
                 qrs_weight: float = 2.0,
                 st_weight: float = 1.5,
                 pt_weight: float = 1.0,
                 baseline_weight: float = 0.5):
        """
        Args:
            sampling_rate: Signal sampling rate in Hz
            qrs_weight: Weight for QRS region
            st_weight: Weight for ST segment
            pt_weight: Weight for P and T waves
            baseline_weight: Weight for baseline regions
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.qrs_weight = qrs_weight
        self.st_weight = st_weight
        self.pt_weight = pt_weight
        self.baseline_weight = baseline_weight
        
    def _get_region_weights(self, signal_length: int, r_peak_idx: int) -> torch.Tensor:
        """
        Generate per-sample weights based on ECG morphology regions
        
        Args:
            signal_length: Length of the signal
            r_peak_idx: Index of R-peak (center of QRS)
            
        Returns:
            Weight tensor of shape (signal_length,)
        """
        weights = torch.ones(signal_length)
        
        # Convert sample indices to time (ms)
        time_axis = torch.arange(signal_length) / self.sampling_rate * 1000
        
        # QRS region (±40ms around R-peak)
        qrs_start = max(0, r_peak_idx - int(0.04 * self.sampling_rate))
        qrs_end = min(signal_length, r_peak_idx + int(0.04 * self.sampling_rate))
        weights[qrs_start:qrs_end] = self.qrs_weight
        
        # ST segment (R-peak + 60ms to R-peak + 200ms)
        st_start = min(signal_length, r_peak_idx + int(0.06 * self.sampling_rate))
        st_end = min(signal_length, r_peak_idx + int(0.20 * self.sampling_rate))
        weights[st_start:st_end] = self.st_weight
        
        # P and T wave regions
        # P wave: ~200-60ms before R-peak
        p_start = max(0, r_peak_idx - int(0.20 * self.sampling_rate))
        p_end = max(0, r_peak_idx - int(0.06 * self.sampling_rate))
        weights[p_start:p_end] = self.pt_weight
        
        # T wave: ~150-400ms after R-peak
        t_start = min(signal_length, r_peak_idx + int(0.15 * self.sampling_rate))
        t_end = min(signal_length, r_peak_idx + int(0.40 * self.sampling_rate))
        weights[t_start:t_end] = self.pt_weight
        
        # Baseline regions (PR segment, TP segment)
        baseline_mask = (weights == 1.0)
        weights[baseline_mask] = self.baseline_weight
        
        return weights
    
    def forward(self, recon: torch.Tensor, 
                target: torch.Tensor,
                r_peak_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            recon: Reconstructed signal (batch_size, channels, sequence_len)
            target: Original signal (batch_size, channels, sequence_len)
            r_peak_indices: R-peak positions within each signal (batch_size,)
            
        Returns:
            Weighted reconstruction loss
        """
        batch_size = recon.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            weights = self._get_region_weights(recon.shape[-1], r_peak_indices[i].item())
            weights = weights.to(recon.device)
            
            # Compute weighted MSE
            diff = recon[i, 0] - target[i, 0]
            weighted_loss = torch.mean(weights * (diff ** 2))
            total_loss += weighted_loss
            
        return total_loss / batch_size


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization loss
    
    Encourages smooth reconstructions by penalizing high-frequency components.
    Useful for denoising autoencoders.
    """
    
    def __init__(self, order: int = 1, reduction: str = 'mean'):
        """
        Args:
            order: Derivative order (1 = first derivative, 2 = second derivative)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.order = order
        self.reduction = reduction
        
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: Input signal (batch_size, channels, sequence_len)
            
        Returns:
            Smoothness loss
        """
        diff = signal
        
        for _ in range(self.order):
            diff = diff[:, :, 1:] - diff[:, :, :-1]
            
        loss = torch.mean(diff ** 2, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SparsityLoss(nn.Module):
    """
    Sparsity regularization for autoencoder latent space
    
    Encourages sparse latent representations, which can help with
    feature disentanglement and anomaly detection.
    """
    
    def __init__(self, sparsity_target: float = 0.05, 
                 sparsity_weight: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            sparsity_target: Target sparsity level (e.g., 0.05 = 5% active neurons)
            sparsity_weight: Weight for sparsity penalty
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.reduction = reduction
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent activations (batch_size, latent_dim)
            
        Returns:
            Sparsity loss (KL divergence from target sparsity)
        """
        # Average activation across batch
        rho_hat = torch.mean(latent, dim=0)
        
        # KL divergence between target sparsity and actual sparsity
        rho = torch.full_like(rho_hat, self.sparsity_target)
        
        kl_div = rho * torch.log(rho / (rho_hat + 1e-8)) + \
                 (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
                 
        loss = self.sparsity_weight * torch.sum(kl_div)
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality regularization for learned representations
    
    Encourages different latent dimensions to be orthogonal,
    promoting disentangled representations.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature representations (batch_size, feature_dim)
            
        Returns:
            Orthogonality loss
        """
        # Normalize features
        features_norm = F.normalize(features, dim=1)
        
        # Compute correlation matrix
        corr = torch.mm(features_norm, features_norm.T)
        
        # Penalize off-diagonal elements (except identity)
        identity = torch.eye(features.shape[0], device=features.device)
        off_diagonal = corr - identity
        
        loss = torch.sum(off_diagonal ** 2)
        
        if self.reduction == 'mean':
            return loss / (features.shape[0] ** 2)
        else:
            return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for sequential predictions
    
    Encourages smooth transitions between consecutive predictions.
    Useful for LSTM/Transformer models.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Sequence predictions (batch_size, seq_len, ...)
            
        Returns:
            Temporal consistency loss
        """
        # First-order difference penalty
        diff = predictions[:, 1:] - predictions[:, :-1]
        loss = torch.mean(diff ** 2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedRegularizationLoss(nn.Module):
    """
    Combine multiple regularization losses
    """
    
    def __init__(self,
                 smoothness_weight: float = 0.0,
                 sparsity_weight: float = 0.0,
                 orthogonality_weight: float = 0.0,
                 temporal_weight: float = 0.0,
                 smoothness_order: int = 1,
                 sparsity_target: float = 0.05):
        """
        Args:
            smoothness_weight: Weight for smoothness loss
            sparsity_weight: Weight for sparsity loss
            orthogonality_weight: Weight for orthogonality loss
            temporal_weight: Weight for temporal consistency loss
            smoothness_order: Order for smoothness loss
            sparsity_target: Target sparsity for sparsity loss
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.orthogonality_weight = orthogonality_weight
        self.temporal_weight = temporal_weight
        
        self.smoothness_loss = SmoothnessLoss(order=smoothness_order)
        self.sparsity_loss = SparsityLoss(sparsity_target=sparsity_target)
        self.orthogonality_loss = OrthogonalityLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
    def forward(self, 
                signal: torch.Tensor,
                latent: Optional[torch.Tensor] = None,
                predictions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            signal: Input signal (for smoothness)
            latent: Latent representations (for sparsity & orthogonality)
            predictions: Sequential predictions (for temporal consistency)
            
        Returns:
            Combined regularization loss
        """
        total_loss = torch.tensor(0.0, device=signal.device)
        
        if self.smoothness_weight > 0:
            total_loss += self.smoothness_weight * self.smoothness_loss(signal)
            
        if self.sparsity_weight > 0 and latent is not None:
            total_loss += self.sparsity_weight * self.sparsity_loss(latent)
            
        if self.orthogonality_weight > 0 and latent is not None:
            total_loss += self.orthogonality_weight * self.orthogonality_loss(latent)
            
        if self.temporal_weight > 0 and predictions is not None:
            total_loss += self.temporal_weight * self.temporal_loss(predictions)
            
        return total_loss


class AdaptiveAnomalyLoss(nn.Module):
    """
    Adaptive loss for anomaly detection
    
    Automatically adjusts threshold for reconstruction error based
    on training statistics.
    """
    
    def __init__(self, 
                 percentile: float = 95.0,
                 margin: float = 0.1,
                 reduction: str = 'mean'):
        """
        Args:
            percentile: Percentile for threshold calculation
            margin: Margin above normal reconstruction error
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.percentile = percentile
        self.margin = margin
        self.reduction = reduction
        self.register_buffer('threshold', torch.tensor(0.0))
        
    def update_threshold(self, recon_errors: torch.Tensor):
        """
        Update anomaly threshold based on reconstruction errors
        """
        self.threshold = torch.quantile(recon_errors, self.percentile / 100.0)
        
    def forward(self, recon: torch.Tensor, 
                target: torch.Tensor,
                is_anomaly: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            recon: Reconstructed signal
            target: Original signal
            is_anomaly: Binary mask indicating anomalies (1=anomaly, 0=normal)
            
        Returns:
            Adaptive loss
        """
        # Compute reconstruction error
        recon_error = torch.mean((recon - target) ** 2, dim=(1, 2))
        
        if is_anomaly is None:
            # Unsupervised: penalize errors above threshold
            loss = torch.clamp(recon_error - self.threshold + self.margin, min=0)
        else:
            # Supervised: different loss for normal vs anomaly
            normal_mask = (is_anomaly == 0)
            anomaly_mask = (is_anomaly == 1)
            
            loss = torch.zeros_like(recon_error)
            
            # Normal samples: keep reconstruction error low
            if normal_mask.any():
                loss[normal_mask] = recon_error[normal_mask]
                
            # Anomaly samples: ensure error is above threshold
            if anomaly_mask.any():
                loss[anomaly_mask] = torch.clamp(
                    self.threshold - recon_error[anomaly_mask] + self.margin, 
                    min=0
                )
                
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Factory function to get loss function by name
def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss module instance
    """
    losses = {
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': nn.HuberLoss,
        'reconstruction': ReconstructionLoss,
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'supcon': SupervisedContrastiveLoss,
        'vae': VAELoss,
        'hybrid': HybridLoss,
        'morphology': MorphologyAwareLoss,
        'smoothness': SmoothnessLoss,
        'sparsity': SparsityLoss,
        'orthogonality': OrthogonalityLoss,
        'temporal': TemporalConsistencyLoss,
        'anomaly_adaptive': AdaptiveAnomalyLoss
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
        
    return losses[loss_name](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    batch_size = 32
    num_classes = 5
    seq_len = 187
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    recon = torch.randn(batch_size, 1, seq_len)
    original = torch.randn(batch_size, 1, seq_len)
    
    # Test classification losses
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, labels)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test reconstruction losses
    recon_loss = ReconstructionLoss(loss_type='combined')
    loss = recon_loss(recon, original)
    print(f"Reconstruction Loss: {loss.item():.4f}")
    
    # Test hybrid loss
    hybrid_loss = HybridLoss(classification_weight=1.0, reconstruction_weight=0.5)
    losses_dict = hybrid_loss(logits, recon, original, labels)
    print(f"Hybrid Loss - Total: {losses_dict['total'].item():.4f}")
    
    # Test morphology-aware loss
    r_peaks = torch.randint(50, 150, (batch_size,))
    morph_loss = MorphologyAwareLoss()
    loss = morph_loss(recon, original, r_peaks)
    print(f"Morphology-Aware Loss: {loss.item():.4f}")
    
    print("\nAll loss functions initialized successfully!")
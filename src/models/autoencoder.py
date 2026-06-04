"""
Autoencoder models for unsupervised anomaly detection in ECG signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for ECG anomaly detection
    
    Encoder: Convolutional layers with pooling
    Decoder: Transposed convolutions for reconstruction
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_length: int = 187,
                 latent_dim: int = 32,
                 hidden_dims: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [5, 5, 3],
                 dropout_rate: float = 0.1):
        """
        Initialize convolutional autoencoder
        
        Args:
            input_channels: Number of input channels
            input_length: Length of input sequence
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for each layer
            kernel_sizes: Kernel sizes for each conv layer
            dropout_rate: Dropout probability
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_modules = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            padding = kernel_size // 2
            
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, stride=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
            )
            in_channels = out_channels
            
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Calculate encoder output size
        self.encoded_length = self._calculate_encoded_length(input_length, kernel_sizes)
        
        # Latent space mapping
        self.fc_encoder = nn.Sequential(
            nn.Linear(hidden_dims[-1] * self.encoded_length, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1] * self.encoded_length),
            nn.BatchNorm1d(hidden_dims[-1] * self.encoded_length),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions (transposed)
        decoder_modules = []
        in_channels = hidden_dims[-1]
        
        for i, (out_channels, kernel_size) in enumerate(zip(
            reversed(hidden_dims[:-1]), reversed(kernel_sizes[:-1])
        )):
            padding = kernel_size // 2
            
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                      stride=2, padding=padding, output_padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
            )
            in_channels = out_channels
            
        # Final decoder layer to reconstruct input
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose1d(in_channels, input_channels, 
                                  kernel_size=kernel_sizes[0], stride=2,
                                  padding=kernel_sizes[0]//2, output_padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def _calculate_encoded_length(self, input_length: int, kernel_sizes: List[int]) -> int:
        """Calculate length after convolutions with stride=2"""
        length = input_length
        for _ in kernel_sizes:
            length = length // 2
        return max(1, length)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        # Convolutional encoding
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Latent space
        z = self.fc_encoder(x)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        # FC decoding
        x = self.fc_decoder(z)
        
        # Reshape for conv decoder
        x = x.view(x.size(0), self.hidden_dims[-1], self.encoded_length)
        
        # Convolutional decoding
        reconstruction = self.decoder(x)
        
        # Trim if needed
        if reconstruction.size(2) != self.input_length:
            reconstruction = reconstruction[:, :, :self.input_length]
            
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Dictionary with 'reconstruction' and 'latent' keys
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'latent': z
        }
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error as anomaly score
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        with torch.no_grad():
            output = self.forward(x)
            reconstruction = output['reconstruction']
            
            # MSE reconstruction error
            mse = F.mse_loss(reconstruction, x, reduction='none')
            anomaly_score = mse.mean(dim=(1, 2))
            
        return anomaly_score


class ECGAnomalyAutoencoder(ConvAutoencoder):
    """
    Enhanced autoencoder specifically for ECG anomaly detection
    with additional features like:
    - Multi-scale reconstruction loss
    - Peak-aware loss (focus on QRS complex)
    """
    
    def __init__(self, **kwargs):
        super(ECGAnomalyAutoencoder, self).__init__(**kwargs)
        
        # Learnable threshold for anomaly detection
        self.anomaly_threshold = nn.Parameter(torch.tensor(0.5))
        
    def compute_peak_weighted_loss(self, 
                                   reconstruction: torch.Tensor, 
                                   original: torch.Tensor,
                                   peak_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss with higher weight on QRS peaks
        
        Args:
            reconstruction: Reconstructed signal
            original: Original signal
            peak_indices: Indices of peaks to weight higher
            
        Returns:
            Weighted MSE loss
        """
        mse = (reconstruction - original) ** 2
        
        if peak_indices is not None:
            # Create weight mask
            weights = torch.ones_like(mse)
            
            # Higher weight around peaks
            for batch_idx in range(peak_indices.shape[0]):
                for peak in peak_indices[batch_idx]:
                    if 0 <= peak < mse.shape[2]:
                        start = max(0, peak - 10)
                        end = min(mse.shape[2], peak + 10)
                        weights[batch_idx, :, start:end] *= 3.0
                        
            weighted_mse = mse * weights
            loss = weighted_mse.mean()
        else:
            loss = mse.mean()
            
        return loss
    
    def compute_multi_scale_loss(self, 
                                 reconstruction: torch.Tensor, 
                                 original: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale reconstruction loss
        
        Args:
            reconstruction: Reconstructed signal
            original: Original signal
            
        Returns:
            Multi-scale loss
        """
        loss = F.mse_loss(reconstruction, original)
        
        # Downsample and compute loss at different scales
        for scale in [2, 4]:
            recon_down = F.avg_pool1d(reconstruction, kernel_size=scale, stride=scale)
            orig_down = F.avg_pool1d(original, kernel_size=scale, stride=scale)
            loss += F.mse_loss(recon_down, orig_down) / scale
            
        return loss
    
    def forward(self, 
                x: torch.Tensor, 
                peak_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional peak-weighted loss
        
        Args:
            x: Input tensor
            peak_indices: Optional peak indices for weighting
            
        Returns:
            Dictionary with reconstruction and losses
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        
        # Compute losses
        mse_loss = F.mse_loss(reconstruction, x)
        
        if peak_indices is not None:
            peak_loss = self.compute_peak_weighted_loss(reconstruction, x, peak_indices)
        else:
            peak_loss = mse_loss
            
        multi_scale_loss = self.compute_multi_scale_loss(reconstruction, x)
        
        total_loss = mse_loss + 0.5 * peak_loss + 0.3 * multi_scale_loss
        
        return {
            'reconstruction': reconstruction,
            'latent': z,
            'mse_loss': mse_loss,
            'peak_loss': peak_loss,
            'multi_scale_loss': multi_scale_loss,
            'total_loss': total_loss
        }


class VariationalAutoencoder(ConvAutoencoder):
    """
    Variational Autoencoder (VAE) for ECG anomaly detection
    
    Adds KL divergence loss for regularized latent space
    """
    
    def __init__(self, beta: float = 1.0, **kwargs):
        """
        Initialize VAE
        
        Args:
            beta: Weight for KL divergence loss (beta-VAE)
            **kwargs: Arguments for ConvAutoencoder
        """
        super(VariationalAutoencoder, self).__init__(**kwargs)
        
        self.beta = beta
        
        # Modify latent space for VAE (mean and log variance)
        self.fc_mean = nn.Linear(self.hidden_dims[-1] * self.encoded_length, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1] * self.encoded_length, self.latent_dim)
        
        # Remove original fc_encoder
        self.fc_encoder = None
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (z, mu, logvar)
        """
        # Convolutional encoding
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get distribution parameters
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with VAE losses
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with reconstruction and losses
        """
        # Encode
        z, mu, logvar = self.encode(x)
        
        # Decode
        reconstruction = self.decode(z)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss (beta-VAE)
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'reconstruction': reconstruction,
            'latent': z,
            'mu': mu,
            'logvar': logvar,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score combining reconstruction error and latent likelihood
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly score
        """
        with torch.no_grad():
            output = self.forward(x)
            
            # Reconstruction error
            recon_error = F.mse_loss(output['reconstruction'], x, reduction='none')
            recon_error = recon_error.mean(dim=(1, 2))
            
            # Latent likelihood (negative log probability)
            mu = output['mu']
            logvar = output['logvar']
            latent_ll = 0.5 * torch.sum(logvar.exp() + mu**2 - logvar - 1, dim=1)
            
            # Combined score
            anomaly_score = recon_error + 0.1 * latent_ll
            
        return anomaly_score


if __name__ == "__main__":
    # Test autoencoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    ae = ConvAutoencoder(input_channels=1, input_length=187, latent_dim=32).to(device)
    vae = VariationalAutoencoder(input_channels=1, input_length=187, latent_dim=32).to(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 187).to(device)
    
    # Regular AE
    with torch.no_grad():
        output_ae = ae(x)
        
    print(f"AE Reconstruction shape: {output_ae['reconstruction'].shape}")
    print(f"AE Latent shape: {output_ae['latent'].shape}")
    
    # VAE
    with torch.no_grad():
        output_vae = vae(x)
        
    print(f"\nVAE Reconstruction shape: {output_vae['reconstruction'].shape}")
    print(f"VAE Latent shape: {output_vae['latent'].shape}")
    print(f"VAE Mu shape: {output_vae['mu'].shape}")
    
    # Anomaly scores
    anomaly_scores = ae.compute_anomaly_score(x)
    print(f"\nAnomaly scores shape: {anomaly_scores.shape}")
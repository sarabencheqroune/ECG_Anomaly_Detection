"""
Unit tests for autoencoder model
"""

import pytest
import torch
import numpy as np

from src.models.autoencoder import ECGDenoisingAutoencoder, ConvAutoencoder


class TestECGDenoisingAutoencoder:
    """Test suite for ECG denoising autoencoder"""
    
    @pytest.fixture
    def autoencoder(self):
        """Create autoencoder instance"""
        return ECGDenoisingAutoencoder(input_dim=187, latent_dim=32)
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch of ECG beats"""
        clean = torch.randn(32, 1, 187)
        noisy = clean + 0.1 * torch.randn(32, 1, 187)
        return clean, noisy
    
    def test_initialization(self, autoencoder):
        """Test autoencoder initialization"""
        assert autoencoder is not None
        assert hasattr(autoencoder, 'encoder')
        assert hasattr(autoencoder, 'decoder')
        assert hasattr(autoencoder, 'encode')
        assert hasattr(autoencoder, 'decode')
        
    def test_forward_pass(self, autoencoder, sample_batch):
        """Test forward pass"""
        clean, noisy = sample_batch
        
        with torch.no_grad():
            reconstructed = autoencoder(noisy)
            
        assert reconstructed.shape == noisy.shape
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
        
    def test_encode_decode(self, autoencoder, sample_batch):
        """Test encoding and decoding separately"""
        clean, noisy = sample_batch
        
        with torch.no_grad():
            latent = autoencoder.encode(noisy)
            reconstructed = autoencoder.decode(latent)
            
        assert latent.shape == (32, 32)
        assert reconstructed.shape == noisy.shape
        
    def test_latent_space(self, autoencoder, sample_batch):
        """Test latent space properties"""
        clean, noisy = sample_batch
        
        with torch.no_grad():
            latent = autoencoder.encode(noisy)
            
        # Check latent space statistics
        assert latent.mean().abs() < 1.0
        assert latent.std() < 2.0
        
    def test_reconstruction_loss(self, autoencoder, sample_batch):
        """Test reconstruction loss computation"""
        clean, noisy = sample_batch
        
        reconstructed = autoencoder(noisy)
        loss = torch.nn.functional.mse_loss(reconstructed, clean)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_gradient_flow(self, autoencoder, sample_batch):
        """Test gradient flow through autoencoder"""
        clean, noisy = sample_batch
        
        # Forward pass
        reconstructed = autoencoder(noisy)
        loss = torch.nn.functional.mse_loss(reconstructed, clean)
        
        # Backward pass
        loss.backward()
        
        # Check gradients for encoder and decoder
        for name, param in autoencoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            
    def test_model_save_load(self, autoencoder, tmp_path):
        """Test autoencoder saving and loading"""
        # Save model
        save_path = tmp_path / "test_autoencoder.pth"
        torch.save(autoencoder.state_dict(), save_path)
        
        # Load model
        new_autoencoder = ECGDenoisingAutoencoder(input_dim=187, latent_dim=32)
        new_autoencoder.load_state_dict(torch.load(save_path))
        new_autoencoder.eval()
        
        # Compare outputs
        test_input = torch.randn(16, 1, 187)
        with torch.no_grad():
            original_output = autoencoder(test_input)
            loaded_output = new_autoencoder(test_input)
            
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        
    def test_latent_dimension(self):
        """Test different latent dimensions"""
        for latent_dim in [16, 32, 64, 128]:
            autoencoder = ECGDenoisingAutoencoder(input_dim=187, latent_dim=latent_dim)
            test_input = torch.randn(16, 1, 187)
            
            with torch.no_grad():
                latent = autoencoder.encode(test_input)
                
            assert latent.shape[1] == latent_dim


class TestConvAutoencoder:
    """Test suite for convolutional autoencoder"""
    
    @pytest.fixture
    def conv_autoencoder(self):
        """Create convolutional autoencoder"""
        return ConvAutoencoder(input_dim=187, latent_dim=64)
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch"""
        return torch.randn(32, 1, 187)
    
    def test_initialization(self, conv_autoencoder):
        """Test initialization"""
        assert conv_autoencoder is not None
        
    def test_forward_pass(self, conv_autoencoder, sample_batch):
        """Test forward pass"""
        with torch.no_grad():
            reconstructed = conv_autoencoder(sample_batch)
            
        assert reconstructed.shape == sample_batch.shape
        
    def test_latent_shape(self, conv_autoencoder, sample_batch):
        """Test latent space shape"""
        with torch.no_grad():
            latent = conv_autoencoder.encode(sample_batch)
            
        # Check latent shape is reasonable
        assert len(latent.shape) in [2, 3]
        
    def test_parameter_count(self, conv_autoencoder):
        """Test parameter count is reasonable"""
        total_params = sum(p.numel() for p in conv_autoencoder.parameters())
        
        # Should be reasonable size
        assert 100000 < total_params < 2000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
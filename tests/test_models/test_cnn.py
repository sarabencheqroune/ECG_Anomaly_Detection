"""
Unit tests for CNN classifier model
"""

import pytest
import torch
import numpy as np

from src.models.cnn_classifier import ECG1DCNN, ECG1DCNNLight


class TestECG1DCNN:
    """Test suite for ECG1DCNN model"""
    
    @pytest.fixture
    def model(self):
        """Create CNN model instance"""
        return ECG1DCNN(input_dim=187, num_classes=5)
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch of ECG beats"""
        return torch.randn(32, 1, 187)
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert hasattr(model, 'conv_block1')
        assert hasattr(model, 'conv_block2')
        assert hasattr(model, 'conv_block3')
        assert hasattr(model, 'global_avg_pool')
        assert hasattr(model, 'classifier')
        
    def test_forward_pass(self, model, sample_batch):
        """Test forward pass through model"""
        with torch.no_grad():
            output = model(sample_batch)
            
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_output_probabilities(self, model, sample_batch):
        """Test that outputs can be converted to probabilities"""
        with torch.no_grad():
            logits = model(sample_batch)
            probs = torch.softmax(logits, dim=1)
            
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32))
        
    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients flow through the model"""
        # Create dummy labels
        labels = torch.randint(0, 5, (32,))
        
        # Forward pass
        logits = model(sample_batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            
    def test_model_parameters_count(self, model):
        """Test number of parameters is reasonable"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have reasonable size (not too large, not too small)
        assert 10000 < total_params < 500000
        assert trainable_params == total_params
        
    def test_different_input_lengths(self):
        """Test model handles different input lengths"""
        model = ECG1DCNN(input_dim=200, num_classes=5)
        
        # Test with different sequence lengths
        test_lengths = [150, 187, 200, 250]
        
        for length in test_lengths:
            x = torch.randn(16, 1, length)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (16, 5)
            
    def test_model_save_load(self, model, sample_batch, tmp_path):
        """Test model saving and loading"""
        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        new_model = ECG1DCNN(input_dim=187, num_classes=5)
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()
        
        # Compare outputs
        with torch.no_grad():
            original_output = model(sample_batch)
            loaded_output = new_model(sample_batch)
            
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        
    def test_model_device_compatibility(self, model):
        """Test model can be moved between devices"""
        # Test CPU
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'
            
    def test_batch_normalization(self, model, sample_batch):
        """Test batch normalization layers are present and working"""
        # Check that BN layers are in eval mode by default
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert not module.training
                
        # Switch to train mode
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert module.training
                
    def test_dropout(self, model, sample_batch):
        """Test dropout layers are present"""
        dropout_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_count += 1
                
        assert dropout_count >= 3  # At least 3 dropout layers


class TestECG1DCNNLight:
    """Test suite for lightweight CNN model"""
    
    @pytest.fixture
    def model(self):
        """Create lightweight CNN model"""
        return ECG1DCNNLight(input_dim=187, num_classes=5)
    
    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch"""
        return torch.randn(32, 1, 187)
    
    def test_model_initialization(self, model):
        """Test lightweight model initialization"""
        assert model is not None
        
    def test_forward_pass(self, model, sample_batch):
        """Test forward pass"""
        with torch.no_grad():
            output = model(sample_batch)
            
        assert output.shape == (32, 5)
        
    def test_parameter_count(self, model):
        """Test lightweight model has fewer parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Lightweight model should have < 100k parameters
        assert total_params < 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
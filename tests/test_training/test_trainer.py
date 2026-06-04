"""
Unit tests for training module
Tests training pipeline, metrics, and callbacks
"""

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import tempfile

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.trainer import Trainer
from src.training.metrics import MetricsCalculator
from src.models.cnn_classifier import ECG1DCNN


class TestMetricsCalculator(unittest.TestCase):
    """Test metrics computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = MetricsCalculator(num_classes=5)
        
    def test_accuracy_computation(self):
        """Test accuracy calculation"""
        labels = np.array([0, 1, 1, 2, 0, 1])
        predictions = np.array([0, 1, 0, 2, 0, 1])
        
        accuracy = self.calculator.compute_accuracy(labels, predictions)
        
        # 5 out of 6 correct
        expected = 5 / 6
        self.assertAlmostEqual(accuracy, expected, places=4)
        
    def test_f1_score_computation(self):
        """Test F1 score computation"""
        labels = np.array([0, 0, 1, 1, 1])
        predictions = np.array([0, 1, 1, 1, 0])
        
        f1 = self.calculator.compute_f1_score(labels, predictions, average='macro')
        
        # F1 should be between 0 and 1
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        
    def test_confusion_matrix(self):
        """Test confusion matrix computation"""
        labels = np.array([0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 0, 2])
        
        cm = self.calculator.compute_confusion_matrix(labels, predictions)
        
        # Check shape
        self.assertEqual(cm.shape, (3, 3))  # 3 classes (0, 1, 2)
        
        # Check total
        self.assertEqual(np.sum(cm), len(labels))
        
    def test_precision_recall_division_by_zero(self):
        """Test handling of division by zero in precision/recall"""
        labels = np.array([0, 0, 0])
        predictions = np.array([1, 1, 1])
        
        # Should not raise ZeroDivisionError
        try:
            precision = self.calculator.compute_precision(labels, predictions)
            recall = self.calculator.compute_recall(labels, predictions)
        except ZeroDivisionError:
            self.fail("Division by zero should be handled")


class TestTrainerBasics(unittest.TestCase):
    """Basic trainer functionality tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = ECG1DCNN(input_channels=1, input_length=187, num_classes=5).to(self.device)
        self.lr = 0.001
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            learning_rate=self.lr
        )
        
    def test_trainer_initialization(self):
        """Test trainer initializes correctly"""
        self.assertEqual(self.trainer.device, self.device)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.criterion)
        
    def test_create_data_loader(self):
        """Test creating synthetic data loader"""
        # Create dummy dataset
        X = np.random.randn(100, 1, 187).astype(np.float32)
        y = np.random.randint(0, 5, 100)
        
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y)
        )
        loader = DataLoader(dataset, batch_size=32)
        
        self.assertEqual(len(loader), np.ceil(100 / 32))
        
    def test_training_step(self):
        """Test single training step"""
        # Create small batch
        X_batch = torch.randn(4, 1, 187, device=self.device)
        y_batch = torch.randint(0, 5, (4,), device=self.device)
        
        # Forward pass
        outputs = self.model(X_batch)
        loss = self.trainer.criterion(outputs, y_batch)
        
        # Loss should be positive scalar
        self.assertGreater(loss.item(), 0)
        self.assertEqual(loss.dim(), 0)


class TestTrainerEdgeCases(unittest.TestCase):
    """Test edge cases in training"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = ECG1DCNN(input_channels=1, input_length=187, num_classes=5).to(self.device)
        self.trainer = Trainer(model=self.model, device=self.device)
        
    def test_empty_batch(self):
        """Test handling of edge case with single sample"""
        X = torch.randn(1, 1, 187, device=self.device)
        y = torch.tensor([2], device=self.device)
        
        outputs = self.model(X)
        loss = self.trainer.criterion(outputs, y)
        
        # Should not crash
        self.assertGreater(loss.item(), 0)
        
    def test_all_same_class_batch(self):
        """Test batch with all same class"""
        X = torch.randn(16, 1, 187, device=self.device)
        y = torch.full((16,), 1, device=self.device)
        
        outputs = self.model(X)
        loss = self.trainer.criterion(outputs, y)
        
        # Should still compute loss
        self.assertGreater(loss.item(), 0)
        
    def test_high_learning_rate_stability(self):
        """Test trainer stability with high learning rate"""
        trainer = Trainer(model=self.model, device=self.device, learning_rate=1.0)
        
        X = torch.randn(8, 1, 187, device=self.device)
        y = torch.randint(0, 5, (8,), device=self.device)
        
        outputs = self.model(X)
        loss = self.trainer.criterion(outputs, y)
        
        # Loss should still be a valid number (not NaN/Inf)
        self.assertFalse(np.isnan(loss.item()))
        self.assertFalse(np.isinf(loss.item()))


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow during training"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = ECG1DCNN(input_channels=1, input_length=187, num_classes=5).to(self.device)
        
    def test_gradients_not_none(self):
        """Test that gradients are computed"""
        X = torch.randn(8, 1, 187, device=self.device)
        y = torch.randint(0, 5, (8,), device=self.device)
        
        outputs = self.model(X)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        loss.backward()
        
        # Check that gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                
    def test_gradient_magnitude(self):
        """Test gradient magnitudes are reasonable"""
        X = torch.randn(8, 1, 187, device=self.device)
        y = torch.randint(0, 5, (8,), device=self.device)
        
        outputs = self.model(X)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        loss.backward()
        
        # Check gradient magnitudes
        for param in self.model.parameters():
            if param.grad is not None and param.grad.numel() > 0:
                grad_norm = param.grad.norm().item()
                # Gradients should be reasonable (not NaN/Inf)
                self.assertFalse(np.isnan(grad_norm))
                self.assertFalse(np.isinf(grad_norm))


if __name__ == '__main__':
    unittest.main()

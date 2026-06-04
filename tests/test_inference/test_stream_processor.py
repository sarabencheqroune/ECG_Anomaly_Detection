"""
Unit tests for ECG stream processor module
Tests real-time ECG signal processing and inference
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.stream_processor import StreamProcessor
from src.inference.beat_detector import BeatDetector


class TestStreamProcessor(unittest.TestCase):
    """Test ECG stream processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sampling_rate = 360
        self.buffer_size = 1024
        self.processor = StreamProcessor(
            sampling_rate=self.sampling_rate,
            buffer_size=self.buffer_size
        )
        
    def test_initialization(self):
        """Test stream processor initialization"""
        self.assertEqual(self.processor.sampling_rate, self.sampling_rate)
        self.assertFalse(self.processor.is_processing)
        
    def test_add_sample(self):
        """Test adding samples to buffer"""
        sample = 0.5
        self.processor.add_sample(sample)
        
        # Check buffer was updated
        self.assertGreater(len(self.processor.buffer), 0)
        
    def test_add_batch(self):
        """Test adding batch of samples"""
        batch = np.random.randn(100)
        self.processor.add_batch(batch)
        
        # Check buffer size
        self.assertGreaterEqual(len(self.processor.buffer), 100)
        
    def test_start_stop_processing(self):
        """Test starting and stopping processing"""
        self.processor.start_processing()
        self.assertTrue(self.processor.is_processing)
        
        self.processor.stop_processing()
        self.assertFalse(self.processor.is_processing)
        
    def test_invalid_sample_handling(self):
        """Test handling of invalid samples (NaN, Inf)"""
        # Should not raise exception
        self.processor.add_sample(np.nan)
        self.processor.add_sample(np.inf)
        
        # Buffer should still be valid
        self.assertGreater(len(self.processor.buffer), 0)
        
    def test_buffer_overflow(self):
        """Test buffer handling when exceeding max size"""
        # Add more samples than buffer size
        large_batch = np.random.randn(self.buffer_size * 2)
        self.processor.add_batch(large_batch)
        
        # Buffer should not exceed max size
        self.assertLessEqual(len(self.processor.buffer), self.buffer_size)
        
    def test_reset(self):
        """Test processor reset"""
        self.processor.add_batch(np.random.randn(100))
        self.assertGreater(len(self.processor.buffer), 0)
        
        self.processor.reset()
        self.assertEqual(len(self.processor.buffer), 0)


class TestStreamProcessorIntegration(unittest.TestCase):
    """Integration tests for stream processor with beat detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = StreamProcessor(sampling_rate=360)
        
    @patch('src.inference.beat_detector.BeatDetector')
    def test_beat_detection_integration(self, mock_detector):
        """Test integration with beat detector"""
        mock_detector_instance = MagicMock()
        mock_detector.return_value = mock_detector_instance
        
        # Generate synthetic ECG signal
        t = np.linspace(0, 10, 3600)
        signal = np.sin(2 * np.pi * 1 * t)  # 1 Hz sine wave
        
        # Add signal to processor
        self.processor.add_batch(signal)
        
        # Processor should buffer the signal
        self.assertGreater(len(self.processor.buffer), 0)
        
    def test_performance_metrics(self):
        """Test performance tracking"""
        # Add known number of samples
        samples = np.random.randn(3600)
        for sample in samples:
            self.processor.add_sample(sample)
            
        # Should track statistics
        self.assertGreater(self.processor.total_samples_processed, 0)


class TestStreamProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = StreamProcessor(sampling_rate=360)
        
    def test_empty_buffer_processing(self):
        """Test processing with empty buffer"""
        # Should not raise exception
        try:
            self.processor.process()
        except IndexError:
            self.fail("Empty buffer should not raise IndexError")
            
    def test_single_sample_processing(self):
        """Test processing with single sample"""
        self.processor.add_sample(1.0)
        
        # Should handle gracefully
        self.assertEqual(len(self.processor.buffer), 1)
        
    def test_all_zero_signal(self):
        """Test processing all-zero signal"""
        signal = np.zeros(1000)
        self.processor.add_batch(signal)
        
        # Should not crash
        self.assertGreater(len(self.processor.buffer), 0)
        
    def test_extreme_values(self):
        """Test processing extreme values"""
        extreme_signal = np.array([1e6, -1e6, 1e-6, -1e-6])
        self.processor.add_batch(extreme_signal)
        
        # Should handle without crashing
        self.assertGreater(len(self.processor.buffer), 0)


if __name__ == '__main__':
    unittest.main()

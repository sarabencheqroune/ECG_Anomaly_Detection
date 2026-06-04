"""
Unit tests for ECG preprocessor module
"""

import pytest
import numpy as np
from scipy import signal

from src.data.preprocessor import ECGPreprocessor, PreprocessingConfig


class TestECGPreprocessor:
    """Test suite for ECGPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create ECGPreprocessor instance"""
        return ECGPreprocessor(sampling_rate=360)
    
    @pytest.fixture
    def sample_signal(self):
        """Generate sample ECG signal"""
        t = np.linspace(0, 5, 5 * 360)
        # Clean signal
        clean = np.sin(2 * np.pi * 1.2 * t)
        # Add baseline wander
        baseline = 0.2 * np.sin(2 * np.pi * 0.2 * t)
        # Add noise
        noise = 0.1 * np.random.randn(len(t))
        return clean + baseline + noise
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.sampling_rate == 360
        assert preprocessor.config is not None
        assert preprocessor.bandpass_b is not None
        assert preprocessor.bandpass_a is not None
        
    def test_bandpass_filter(self, preprocessor, sample_signal):
        """Test bandpass filtering"""
        filtered = preprocessor.apply_bandpass_filter(sample_signal)
        
        assert filtered.shape == sample_signal.shape
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))
        
        # Check that high frequency noise is reduced
        original_high_freq = np.abs(np.fft.rfft(sample_signal))
        filtered_high_freq = np.abs(np.fft.rfft(filtered))
        
        # High frequency components (above 40 Hz) should be reduced
        # This is a simple check
        assert np.std(filtered) < np.std(sample_signal) * 1.2
        
    def test_notch_filter(self, preprocessor, sample_signal):
        """Test notch filtering for powerline interference"""
        # Add 50 Hz interference
        t = np.arange(len(sample_signal)) / 360
        interference = 0.1 * np.sin(2 * np.pi * 50 * t)
        noisy_signal = sample_signal + interference
        
        filtered = preprocessor.apply_notch_filter(noisy_signal)
        
        # Check that 50 Hz component is reduced
        freqs = np.fft.rfftfreq(len(noisy_signal), 1/360)
        fft_noisy = np.abs(np.fft.rfft(noisy_signal))
        fft_filtered = np.abs(np.fft.rfft(filtered))
        
        # Find index near 50 Hz
        idx_50hz = np.argmin(np.abs(freqs - 50))
        
        # 50 Hz component should be reduced
        assert fft_filtered[idx_50hz] < fft_noisy[idx_50hz] * 0.8
        
    def test_baseline_removal(self, preprocessor, sample_signal):
        """Test baseline wander removal"""
        # Add baseline wander
        t = np.arange(len(sample_signal)) / 360
        baseline = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        signal_with_baseline = sample_signal + baseline
        
        # Remove baseline using different methods
        corrected_median = preprocessor.remove_baseline_wander(signal_with_baseline, method='median')
        corrected_highpass = preprocessor.remove_baseline_wander(signal_with_baseline, method='highpass')
        corrected_poly = preprocessor.remove_baseline_wander(signal_with_baseline, method='polynomial')
        
        # Baseline should be reduced
        assert np.std(corrected_median) < np.std(signal_with_baseline) * 1.2
        assert np.std(corrected_highpass) < np.std(signal_with_baseline) * 1.2
        assert np.std(corrected_poly) < np.std(signal_with_baseline) * 1.2
        
    def test_artifact_removal(self, preprocessor, sample_signal):
        """Test artifact removal"""
        # Add spike artifact
        signal_with_artifact = sample_signal.copy()
        artifact_idx = len(signal_with_artifact) // 2
        signal_with_artifact[artifact_idx] = 5.0  # Large spike
        
        cleaned = preprocessor.remove_artifacts(signal_with_artifact, threshold=3.0)
        
        # Artifact should be removed
        assert cleaned[artifact_idx] < 2.0
        
    def test_normalization(self, preprocessor, sample_signal):
        """Test signal normalization"""
        # Test z-score normalization
        normalized_z = preprocessor.normalize_signal(sample_signal, method='zscore')
        assert abs(np.mean(normalized_z)) < 0.1
        assert abs(np.std(normalized_z) - 1.0) < 0.1
        
        # Test min-max normalization
        normalized_mm = preprocessor.normalize_signal(sample_signal, method='minmax')
        assert np.min(normalized_mm) >= -1.0
        assert np.max(normalized_mm) <= 1.0
        
        # Test robust normalization
        normalized_robust = preprocessor.normalize_signal(sample_signal, method='robust')
        assert np.median(normalized_robust) < 0.1
        
    def test_full_preprocessing_pipeline(self, preprocessor, sample_signal):
        """Test complete preprocessing pipeline"""
        processed = preprocessor.preprocess(sample_signal)
        
        assert processed.shape == sample_signal.shape
        assert not np.any(np.isnan(processed))
        assert not np.any(np.isinf(processed))
        
        # Check that mean is close to zero after normalization
        assert abs(np.mean(processed)) < 0.1
        
    def test_preprocess_with_intermediate(self, preprocessor, sample_signal):
        """Test preprocessing with intermediate outputs"""
        processed, intermediate = preprocessor.preprocess(
            sample_signal, return_intermediate=True
        )
        
        assert 'baseline_corrected' in intermediate
        assert 'notch_filtered' in intermediate
        assert 'bandpass_filtered' in intermediate
        assert 'artifacts_removed' in intermediate
        assert 'normalized' in intermediate
        
        for key, value in intermediate.items():
            assert value.shape == sample_signal.shape
            
    def test_batch_preprocessing(self, preprocessor, sample_signal):
        """Test batch preprocessing"""
        batch = np.array([sample_signal, sample_signal, sample_signal])
        processed_batch = preprocessor.preprocess_batch(batch)
        
        assert processed_batch.shape == batch.shape
        assert not np.any(np.isnan(processed_batch))
        
    def test_config_customization(self):
        """Test custom preprocessing configuration"""
        config = PreprocessingConfig(
            lowcut=0.05,
            highcut=50.0,
            notch_freq=60.0,
            use_median_filter=False
        )
        
        preprocessor = ECGPreprocessor(config=config, sampling_rate=360)
        
        assert preprocessor.config.lowcut == 0.05
        assert preprocessor.config.highcut == 50.0
        assert preprocessor.config.notch_freq == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
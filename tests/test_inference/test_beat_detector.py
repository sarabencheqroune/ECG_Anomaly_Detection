"""
Unit tests for online beat detector
"""

import pytest
import numpy as np
from scipy import signal

from src.inference.beat_detector import (
    OnlineBeatDetector, BeatDetectorConfig, 
    MultiLeadBeatDetector, RealTimeQRSDetector
)


class TestOnlineBeatDetector:
    """Test suite for OnlineBeatDetector"""
    
    @pytest.fixture
    def config(self):
        """Create beat detector configuration"""
        return BeatDetectorConfig(sampling_rate=360)
    
    @pytest.fixture
    def detector(self, config):
        """Create beat detector instance"""
        return OnlineBeatDetector(config)
    
    @pytest.fixture
    def synthetic_ecg(self):
        """Generate synthetic ECG with R-peaks"""
        sampling_rate = 360
        duration = 10
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Base signal
        ecg = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        ecg += 0.05 * np.random.randn(len(t))
        
        # Add R-peaks at 75 BPM (every 0.8 seconds)
        for i in range(0, len(t), int(0.8 * sampling_rate)):
            if i + 20 < len(t):
                ecg[i:i+20] += np.hanning(20) * 0.8
                
        return ecg
    
    def test_initialization(self, detector, config):
        """Test detector initialization"""
        assert detector.config == config
        assert detector.sampling_rate == 360
        assert len(detector.buffer) == 0
        assert len(detector.r_peaks) == 0
        
    def test_process_sample(self, detector, synthetic_ecg):
        """Test processing samples one by one"""
        detected_peaks = []
        
        for i, sample in enumerate(synthetic_ecg):
            peak = detector.process_sample(sample)
            if peak is not None:
                detected_peaks.append(i)
                
        # Should detect approximately 12-13 peaks in 10 seconds (75 BPM)
        assert 10 <= len(detected_peaks) <= 15
        
    def test_heart_rate_estimation(self, detector, synthetic_ecg):
        """Test heart rate estimation"""
        # Process enough samples
        for sample in synthetic_ecg[:3600]:  # 10 seconds
            detector.process_sample(sample)
            
        heart_rate = detector.get_heart_rate()
        
        # Heart rate should be around 75 BPM
        assert 60 <= heart_rate <= 90
        
    def test_rr_intervals(self, detector, synthetic_ecg):
        """Test RR interval calculation"""
        for sample in synthetic_ecg:
            detector.process_sample(sample)
            
        rr_intervals = detector.get_rr_intervals()
        
        if len(rr_intervals) > 0:
            # RR intervals should be around 0.8 seconds (75 BPM)
            assert all(0.6 < rr < 1.0 for rr in rr_intervals)
            
    def test_refractory_period(self, detector):
        """Test refractory period prevents double detection"""
        # Create a signal with two very close peaks
        test_signal = np.zeros(1000)
        test_signal[100] = 1.0
        test_signal[120] = 0.8  # Very close to first peak
        
        detected = []
        for sample in test_signal:
            peak = detector.process_sample(sample)
            if peak:
                detected.append(peak)
                
        # Should only detect first peak due to refractory period
        assert len(detected) <= 1
        
    def test_adaptive_threshold(self, detector):
        """Test adaptive threshold adjustment"""
        # Low noise signal
        clean_signal = np.zeros(1000)
        clean_signal[500] = 1.0
        
        for sample in clean_signal:
            detector.process_sample(sample)
            
        # Threshold should adapt to noise level
        assert detector.threshold > 0
        
    def test_get_latest_beat(self, detector, synthetic_ecg):
        """Test retrieving latest detected beat"""
        for sample in synthetic_ecg[:2000]:
            detector.process_sample(sample)
            
        latest_beat = detector.get_latest_beat()
        
        if latest_beat is not None:
            assert len(latest_beat) > 0
            
    def test_reset(self, detector, synthetic_ecg):
        """Test detector reset"""
        # Process some data
        for sample in synthetic_ecg[:1000]:
            detector.process_sample(sample)
            
        assert len(detector.r_peaks) > 0
        
        # Reset
        detector.reset()
        
        assert len(detector.buffer) == 0
        assert len(detector.r_peaks) == 0
        assert detector.heart_rate == 75.0
        
    def test_config_parameters(self):
        """Test different configuration parameters"""
        # Higher sampling rate
        config_high = BeatDetectorConfig(sampling_rate=1000)
        detector_high = OnlineBeatDetector(config_high)
        
        assert detector_high.sampling_rate == 1000
        assert detector_high.integration_window > 0
        
        # Different thresholds
        config_custom = BeatDetectorConfig(
            sampling_rate=360,
            threshold_initial=0.8,
            threshold_adjustment=0.1
        )
        detector_custom = OnlineBeatDetector(config_custom)
        
        assert detector_custom.threshold == 0.8


class TestMultiLeadBeatDetector:
    """Test suite for MultiLeadBeatDetector"""
    
    @pytest.fixture
    def config(self):
        return BeatDetectorConfig(sampling_rate=360)
    
    @pytest.fixture
    def detector(self, config):
        return MultiLeadBeatDetector(num_leads=3, config=config)
    
    def test_initialization(self, detector):
        """Test multi-lead detector initialization"""
        assert detector.num_leads == 3
        assert len(detector.detectors) == 3
        
    def test_process_sample(self, detector):
        """Test processing multi-lead samples"""
        # Simulate samples from 3 leads
        samples = np.array([0.5, 0.6, 0.4])
        
        # Process many samples (will likely not detect peaks without proper signal)
        for _ in range(1000):
            detector.process_sample(samples + np.random.randn(3) * 0.1)
            
        # Should work without errors
        assert True
        
    def test_get_heart_rate(self, detector):
        """Test getting average heart rate"""
        heart_rate = detector.get_heart_rate()
        assert 0 <= heart_rate <= 200


class TestRealTimeQRSDetector:
    """Test suite for RealTimeQRSDetector"""
    
    @pytest.fixture
    def detector(self):
        return RealTimeQRSDetector(sampling_rate=360)
    
    @pytest.fixture
    def synthetic_ecg(self):
        """Generate synthetic ECG"""
        t = np.linspace(0, 5, 1800)
        ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        
        # Add QRS complexes
        for i in range(0, len(t), 360):
            if i + 20 < len(t):
                ecg[i:i+20] += 1.0
                
        return ecg
    
    def test_initialization(self, detector):
        """Test initialization"""
        assert detector.sampling_rate == 360
        assert detector.qrs_threshold == 0.5
        
    def test_process(self, detector, synthetic_ecg):
        """Test QRS detection"""
        qrs_scores = []
        
        for sample in synthetic_ecg:
            score = detector.process(sample)
            if score is not None:
                qrs_scores.append(score)
                
        # Should detect QRS complexes
        assert len(qrs_scores) > 5
        
    def test_heart_rate(self, detector, synthetic_ecg):
        """Test heart rate from QRS detector"""
        for sample in synthetic_ecg:
            detector.process(sample)
            
        heart_rate = detector.get_heart_rate()
        assert 60 <= heart_rate <= 100
        
    def test_reset(self, detector, synthetic_ecg):
        """Test reset functionality"""
        # Process some data
        for sample in synthetic_ecg[:1000]:
            detector.process(sample)
            
        assert len(detector.qrs_peaks) > 0
        
        # Reset
        detector.reset()
        
        assert len(detector.ecg_buffer) == 0
        assert len(detector.qrs_peaks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
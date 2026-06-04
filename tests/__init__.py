"""
Unit tests for ECG anomaly detection project
"""

import pytest
import numpy as np
import torch

# Common test fixtures that can be shared across tests
@pytest.fixture
def sample_ecg_signal():
    """Generate a synthetic ECG signal for testing"""
    sampling_rate = 360
    duration = 5  # seconds
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Base signal with some noise
    signal = 0.1 * np.sin(2 * np.pi * 0.2 * t)  # Baseline wander
    signal += 0.05 * np.random.randn(len(t))
    
    # Add synthetic R-peaks
    for i in range(0, len(t), int(0.8 * sampling_rate)):
        if i + 30 < len(t):
            signal[i:i+30] += np.hanning(30) * 0.8
            
    return signal.astype(np.float32)

@pytest.fixture
def sample_ecg_beat():
    """Generate a synthetic ECG beat for testing"""
    sampling_rate = 360
    t = np.linspace(-0.2, 0.5, int(0.7 * sampling_rate))
    
    beat = np.zeros_like(t)
    
    # P wave
    p_idx = np.argmin(np.abs(t - (-0.15)))
    if p_idx + 15 < len(beat):
        beat[p_idx:p_idx+15] = 0.15 * np.hanning(15)
    
    # QRS complex
    r_idx = np.argmin(np.abs(t - 0))
    if r_idx - 5 >= 0 and r_idx + 15 < len(beat):
        beat[r_idx-5:r_idx+15] = 1.0 * np.hanning(20)
    
    # S wave
    s_idx = np.argmin(np.abs(t - 0.04))
    if s_idx + 10 < len(beat):
        beat[s_idx:s_idx+10] = -0.3 * np.hanning(10)
    
    # T wave
    t_idx = np.argmin(np.abs(t - 0.3))
    if t_idx + 25 < len(beat):
        beat[t_idx:t_idx+25] = 0.25 * np.hanning(25)
    
    # Add small noise
    beat += 0.02 * np.random.randn(len(beat))
    
    return beat.astype(np.float32), r_idx

@pytest.fixture
def sample_batch_beats():
    """Generate a batch of synthetic beats for testing"""
    beats = []
    r_peaks = []
    
    for i in range(32):
        beat, r_idx = sample_ecg_beat()
        # Slight variations
        beat += 0.01 * np.random.randn(len(beat))
        beats.append(beat)
        r_peaks.append(r_idx)
        
    return np.array(beats), np.array(r_peaks)

@pytest.fixture
def sample_labels():
    """Generate sample labels for testing"""
    return np.random.randint(0, 5, 100)

@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing"""
    return np.random.randint(0, 5, 100)

@pytest.fixture
def sample_probabilities():
    """Generate sample class probabilities for testing"""
    probs = np.random.rand(100, 5)
    return probs / probs.sum(axis=1, keepdims=True)

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinifline("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinifline("markers", "gpu: marks tests that require GPU")
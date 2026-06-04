"""
Data augmentation for ECG signals to improve model robustness
"""

import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Optional, Dict, List, Callable
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGAugmenter:
    """
    ECG-specific data augmentation techniques
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize augmenter
        
        Args:
            sampling_rate: Signal sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def add_gaussian_noise(self, signal: np.ndarray, 
                          snr_db: float = 20.0) -> np.ndarray:
        """
        Add Gaussian noise with specified SNR
        
        Args:
            signal: Input ECG signal
            snr_db: Signal-to-noise ratio in decibels
            
        Returns:
            Noisy signal
        """
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        
        return signal + noise
    
    def add_baseline_wander(self, signal: np.ndarray, 
                           max_amplitude: float = 0.3,
                           frequency_range: Tuple[float, float] = (0.1, 0.5)) -> np.ndarray:
        """
        Add synthetic baseline wander
        
        Args:
            signal: Input ECG signal
            max_amplitude: Maximum amplitude of baseline wander (relative to signal)
            frequency_range: Range of wander frequencies (Hz)
            
        Returns:
            Signal with baseline wander
        """
        t = np.arange(len(signal)) / self.sampling_rate
        
        # Generate random sinusoidal baseline
        freq = np.random.uniform(frequency_range[0], frequency_range[1])
        phase = np.random.uniform(0, 2 * np.pi)
        baseline = max_amplitude * np.std(signal) * np.sin(2 * np.pi * freq * t + phase)
        
        return signal + baseline
    
    def add_powerline_noise(self, signal: np.ndarray, 
                           freq: float = 50.0,
                           max_amplitude: float = 0.1) -> np.ndarray:
        """
        Add powerline interference
        
        Args:
            signal: Input ECG signal
            freq: Powerline frequency (50 or 60 Hz)
            max_amplitude: Maximum amplitude of interference
            
        Returns:
            Signal with powerline noise
        """
        t = np.arange(len(signal)) / self.sampling_rate
        interference = max_amplitude * np.std(signal) * np.sin(2 * np.pi * freq * t)
        
        # Add harmonics
        for harmonic in [2, 3]:
            interference += (max_amplitude / harmonic) * np.std(signal) * \
                           np.sin(2 * np.pi * freq * harmonic * t)
                           
        return signal + interference
    
    def time_warp(self, signal: np.ndarray, 
                 sigma: float = 0.2,
                 knot_distance: int = 100) -> np.ndarray:
        """
        Apply time warping (smooth temporal distortion)
        
        Args:
            signal: Input ECG signal
            sigma: Distortion strength
            knot_distance: Distance between control points
            
        Returns:
            Time-warped signal
        """
        original_indices = np.arange(len(signal))
        
        # Generate warping path
        num_knots = len(signal) // knot_distance + 2
        knots = np.linspace(0, len(signal) - 1, num_knots)
        
        # Random displacements at knots
        displacements = np.random.normal(0, sigma, num_knots) * knot_distance
        
        # Cumulative displacements
        cum_displacements = np.cumsum(displacements)
        
        # Interpolate displacement for all points
        f = interpolate.interp1d(knots, cum_displacements, kind='cubic', 
                                 fill_value='extrapolate')
        displacements_all = f(original_indices)
        
        # Apply warping
        warped_indices = original_indices + displacements_all
        warped_indices = np.clip(warped_indices, 0, len(signal) - 1)
        
        # Interpolate signal
        f = interpolate.interp1d(original_indices, signal, kind='linear')
        warped_signal = f(warped_indices)
        
        return warped_signal
    
    def amplitude_scale(self, signal: np.ndarray, 
                       sigma: float = 0.1) -> np.ndarray:
        """
        Random amplitude scaling
        
        Args:
            signal: Input ECG signal
            sigma: Scaling factor standard deviation
            
        Returns:
            Amplitude-scaled signal
        """
        scale_factor = np.random.normal(1.0, sigma)
        return signal * scale_factor
    
    def electrode_shift(self, signal: np.ndarray, 
                       max_shift: float = 0.1) -> np.ndarray:
        """
        Simulate electrode shift (adds a DC offset that decays exponentially)
        
        Args:
            signal: Input ECG signal
            max_shift: Maximum relative offset
            
        Returns:
            Signal with simulated electrode shift
        """
        t = np.arange(len(signal)) / self.sampling_rate
        shift_amplitude = max_shift * np.std(signal)
        
        # Exponential decay offset
        decay_rate = np.random.uniform(0.1, 2.0)
        offset = shift_amplitude * np.exp(-decay_rate * t)
        
        # Add random polarity
        if np.random.random() > 0.5:
            offset = -offset
            
        return signal + offset
    
    def muscle_artifact(self, signal: np.ndarray, 
                       amplitude: float = 0.15,
                       duration_ms: float = 200.0) -> np.ndarray:
        """
        Add simulated muscle artifact (high frequency burst)
        
        Args:
            signal: Input ECG signal
            amplitude: Relative amplitude of artifact
            duration_ms: Duration of artifact in milliseconds
            
        Returns:
            Signal with muscle artifact
        """
        artifact_signal = signal.copy()
        
        # Random position for artifact
        duration_samples = int(duration_ms * self.sampling_rate / 1000)
        start_idx = np.random.randint(0, len(signal) - duration_samples)
        end_idx = start_idx + duration_samples
        
        # High frequency noise
        noise = np.random.normal(0, amplitude * np.std(signal), duration_samples)
        
        # Bandpass filter to simulate muscle noise (20-100 Hz)
        nyquist = 0.5 * self.sampling_rate
        b, a = signal.butter(4, [20/nyquist, 100/nyquist], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        artifact_signal[start_idx:end_idx] += filtered_noise
        
        return artifact_signal
    
    def dropout_lead(self, signal: np.ndarray,
                    dropout_duration_ms: float = 100.0,
                    dropout_prob: float = 0.1) -> np.ndarray:
        """
        Simulate lead dropout (signal goes to zero briefly)
        
        Args:
            signal: Input ECG signal
            dropout_duration_ms: Duration of dropout in milliseconds
            dropout_prob: Probability of dropout occurrence
            
        Returns:
            Signal with simulated dropout
        """
        if np.random.random() > dropout_prob:
            return signal
            
        dropout_signal = signal.copy()
        duration_samples = int(dropout_duration_ms * self.sampling_rate / 1000)
        start_idx = np.random.randint(0, len(signal) - duration_samples)
        end_idx = start_idx + duration_samples
        
        dropout_signal[start_idx:end_idx] = 0
        
        return dropout_signal
    
    def resample(self, signal: np.ndarray, 
                orig_rate: int,
                target_rate: int) -> np.ndarray:
        """
        Resample signal to simulate different sampling rates
        
        Args:
            signal: Input ECG signal
            orig_rate: Original sampling rate
            target_rate: Target sampling rate
            
        Returns:
            Resampled signal
        """
        num_samples = int(len(signal) * target_rate / orig_rate)
        resampled = signal.resample(signal, num_samples)
        
        # Resample back to original rate for consistent shape
        if target_rate != self.sampling_rate:
            resampled = signal.resample(resampled, len(signal))
            
        return resampled
    
    def augment(self, signal: np.ndarray, 
               augmentations: Optional[List[str]] = None,
               intensity: str = 'medium') -> np.ndarray:
        """
        Apply a random selection of augmentations
        
        Args:
            signal: Input ECG signal
            augmentations: List of augmentation types to apply (None = random)
            intensity: 'light', 'medium', or 'heavy'
            
        Returns:
            Augmented signal
        """
        # Define augmentation parameters based on intensity
        intensity_params = {
            'light': {
                'noise_snr': 25,
                'baseline_amplitude': 0.1,
                'warp_sigma': 0.1,
                'scale_sigma': 0.05,
                'muscle_amplitude': 0.1
            },
            'medium': {
                'noise_snr': 20,
                'baseline_amplitude': 0.2,
                'warp_sigma': 0.2,
                'scale_sigma': 0.1,
                'muscle_amplitude': 0.15
            },
            'heavy': {
                'noise_snr': 15,
                'baseline_amplitude': 0.4,
                'warp_sigma': 0.3,
                'scale_sigma': 0.2,
                'muscle_amplitude': 0.25
            }
        }
        
        params = intensity_params[intensity]
        
        # Available augmentations
        aug_methods = {
            'noise': lambda x: self.add_gaussian_noise(x, snr_db=params['noise_snr']),
            'baseline': lambda x: self.add_baseline_wander(x, max_amplitude=params['baseline_amplitude']),
            'powerline': lambda x: self.add_powerline_noise(x),
            'time_warp': lambda x: self.time_warp(x, sigma=params['warp_sigma']),
            'amplitude': lambda x: self.amplitude_scale(x, sigma=params['scale_sigma']),
            'electrode': lambda x: self.electrode_shift(x),
            'muscle': lambda x: self.muscle_artifact(x, amplitude=params['muscle_amplitude']),
            'dropout': lambda x: self.dropout_lead(x)
        }
        
        # Select augmentations
        if augmentations is None:
            # Randomly select 2-4 augmentations
            num_augs = np.random.randint(2, 5)
            selected_augs = np.random.choice(list(aug_methods.keys()), 
                                            size=num_augs, replace=False)
        else:
            selected_augs = augmentations
            
        # Apply augmentations
        augmented = signal.copy()
        
        for aug_name in selected_augs:
            if aug_name in aug_methods:
                augmented = aug_methods[aug_name](augmented)
                
        return augmented
    
    def augment_batch(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        """
        Augment a batch of signals
        
        Args:
            signals: Batch of signals (batch_size, samples)
            **kwargs: Arguments for augment()
            
        Returns:
            Augmented batch
        """
        augmented_batch = []
        
        for signal in signals:
            augmented = self.augment(signal, **kwargs)
            augmented_batch.append(augmented)
            
        return np.array(augmented_batch)


if __name__ == "__main__":
    # Test augmenter
    import matplotlib.pyplot as plt
    
    # Generate synthetic signal
    t = np.linspace(0, 1, 360)
    synthetic = np.sin(2 * np.pi * 2 * t) + 0.2 * np.random.randn(len(t))
    
    # Apply augmentations
    augmenter = ECGAugmenter(sampling_rate=360)
    
    augmented_noise = augmenter.add_gaussian_noise(synthetic, snr_db=15)
    augmented_baseline = augmenter.add_baseline_wander(synthetic, max_amplitude=0.3)
    augmented_warp = augmenter.time_warp(synthetic, sigma=0.2)
    
    print(f"Original shape: {synthetic.shape}")
    print(f"Augmented noise shape: {augmented_noise.shape}")
    print(f"Augmented baseline shape: {augmented_baseline.shape}")
    print(f"Augmented warp shape: {augmented_warp.shape}")
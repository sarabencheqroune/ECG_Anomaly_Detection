"""
ECG signal preprocessing: filtering, noise removal, baseline wander correction
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
from typing import Tuple, Optional, Dict, Callable
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    lowcut: float = 0.5      # Low frequency cutoff (Hz)
    highcut: float = 40.0    # High frequency cutoff (Hz)
    order: int = 4           # Filter order
    notch_freq: float = 50.0 # Notch filter frequency (Hz, 60Hz for US)
    use_median_filter: bool = True
    median_window: int = 5   # Median filter window size (samples)
    remove_baseline: bool = True
    baseline_window: int = 500  # Window for baseline estimation (ms)


class ECGPreprocessor:
    """
    ECG signal preprocessing with multiple filtering options
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None,
                 sampling_rate: int = 360):
        """
        Initialize preprocessor
        
        Args:
            config: Preprocessing configuration
            sampling_rate: Signal sampling rate in Hz
        """
        self.config = config or PreprocessingConfig()
        self.sampling_rate = sampling_rate
        
        # Design filters
        self.bandpass_b, self.bandpass_a = self._design_bandpass_filter()
        self.notch_b, self.notch_a = self._design_notch_filter()
        
    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter"""
        nyquist = 0.5 * self.sampling_rate
        
        low = self.config.lowcut / nyquist
        high = self.config.highcut / nyquist
        
        b, a = butter(self.config.order, [low, high], btype='band')
        return b, a
    
    def _design_notch_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design notch filter for powerline interference"""
        nyquist = 0.5 * self.sampling_rate
        notch_freq_norm = self.config.notch_freq / nyquist
        
        # Quality factor (sharpness of notch)
        Q = 30
        b, a = signal.iirnotch(notch_freq_norm, Q)
        return b, a
    
    def apply_bandpass_filter(self, signal: np.ndarray, 
                             zero_phase: bool = True) -> np.ndarray:
        """
        Apply bandpass filter to remove low and high frequency noise
        
        Args:
            signal: Input ECG signal
            zero_phase: Use zero-phase filtering (filtfilt)
            
        Returns:
            Filtered signal
        """
        if zero_phase:
            filtered = filtfilt(self.bandpass_b, self.bandpass_a, signal)
        else:
            filtered = signal.lfilter(self.bandpass_b, self.bandpass_a, signal)
            
        return filtered
    
    def apply_notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove powerline interference
        
        Args:
            signal: Input ECG signal
            
        Returns:
            Filtered signal
        """
        return filtfilt(self.notch_b, self.notch_a, signal)
    
    def remove_baseline_wander(self, signal: np.ndarray, 
                               method: str = 'median') -> np.ndarray:
        """
        Remove baseline wander from ECG signal
        
        Args:
            signal: Input ECG signal
            method: 'median' or 'highpass' or 'polynomial'
            
        Returns:
            Signal with baseline removed
        """
        if method == 'median':
            # Median filter for baseline estimation
            window_size = int(self.config.baseline_window * self.sampling_rate / 1000)
            if window_size % 2 == 0:
                window_size += 1
            
            baseline = medfilt(signal, kernel_size=window_size)
            corrected = signal - baseline
            
        elif method == 'highpass':
            # Highpass filter (0.5 Hz cutoff)
            nyquist = 0.5 * self.sampling_rate
            cutoff = 0.5 / nyquist
            b, a = butter(2, cutoff, btype='high')
            corrected = filtfilt(b, a, signal)
            
        elif method == 'polynomial':
            # Polynomial fit
            x = np.arange(len(signal))
            coeffs = np.polyfit(x, signal, deg=2)
            baseline = np.polyval(coeffs, x)
            corrected = signal - baseline
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return corrected
    
    def remove_artifacts(self, signal: np.ndarray, 
                        threshold: float = 3.0) -> np.ndarray:
        """
        Remove outlier artifacts using moving Z-score
        
        Args:
            signal: Input ECG signal
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Signal with artifacts replaced by interpolation
        """
        # Calculate moving statistics
        window = int(1.5 * self.sampling_rate)  # 1.5 second window
        half_window = window // 2
        
        cleaned = signal.copy()
        
        for i in range(half_window, len(signal) - half_window):
            window_data = signal[i - half_window:i + half_window]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                z_score = abs(signal[i] - mean) / std
                if z_score > threshold:
                    # Replace with median of neighbors
                    neighbors = np.concatenate([
                        signal[i - half_window:i - half_window + 10],
                        signal[i + half_window - 10:i + half_window]
                    ])
                    cleaned[i] = np.median(neighbors)
                    
        return cleaned
    
    def normalize_signal(self, signal: np.ndarray, 
                        method: str = 'zscore') -> np.ndarray:
        """
        Normalize ECG signal
        
        Args:
            signal: Input ECG signal
            method: 'zscore', 'minmax', or 'robust'
            
        Returns:
            Normalized signal
        """
        if method == 'zscore':
            mean = np.mean(signal)
            std = np.std(signal)
            normalized = (signal - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            min_val = np.min(signal)
            max_val = np.max(signal)
            normalized = (signal - min_val) / (max_val - min_val + 1e-8)
            normalized = normalized * 2 - 1  # Range [-1, 1]
            
        elif method == 'robust':
            median = np.median(signal)
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            normalized = (signal - median) / (iqr + 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized
    
    def preprocess(self, signal: np.ndarray, 
                  apply_normalization: bool = True,
                  return_intermediate: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            signal: Raw ECG signal
            apply_normalization: Apply z-score normalization
            return_intermediate: Return intermediate stages
            
        Returns:
            Preprocessed signal
        """
        intermediate = {} if return_intermediate else None
        
        # Step 1: Remove baseline wander
        signal_clean = self.remove_baseline_wander(signal)
        if return_intermediate:
            intermediate['baseline_corrected'] = signal_clean.copy()
            
        # Step 2: Apply notch filter (powerline)
        signal_clean = self.apply_notch_filter(signal_clean)
        if return_intermediate:
            intermediate['notch_filtered'] = signal_clean.copy()
            
        # Step 3: Apply bandpass filter
        signal_clean = self.apply_bandpass_filter(signal_clean)
        if return_intermediate:
            intermediate['bandpass_filtered'] = signal_clean.copy()
            
        # Step 4: Remove artifacts
        signal_clean = self.remove_artifacts(signal_clean)
        if return_intermediate:
            intermediate['artifacts_removed'] = signal_clean.copy()
            
        # Step 5: Normalize
        if apply_normalization:
            signal_clean = self.normalize_signal(signal_clean)
            if return_intermediate:
                intermediate['normalized'] = signal_clean.copy()
                
        if return_intermediate:
            return signal_clean, intermediate
            
        return signal_clean
    
    def preprocess_batch(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess a batch of signals
        
        Args:
            signals: Batch of signals (batch_size, samples)
            **kwargs: Arguments for preprocess()
            
        Returns:
            Preprocessed batch
        """
        processed = []
        
        for signal in signals:
            processed_signal = self.preprocess(signal, **kwargs)
            processed.append(processed_signal)
            
        return np.array(processed)


if __name__ == "__main__":
    # Test preprocessor
    import matplotlib.pyplot as plt
    
    # Generate synthetic ECG (simplified)
    t = np.linspace(0, 10, 3600)  # 10 seconds at 360 Hz
    synthetic_ecg = np.sin(2 * np.pi * 1 * t) + 0.5 * np.random.randn(len(t))
    
    # Add baseline wander
    baseline = 0.2 * np.sin(2 * np.pi * 0.2 * t)
    noisy_ecg = synthetic_ecg + baseline
    
    # Preprocess
    preprocessor = ECGPreprocessor(sampling_rate=360)
    cleaned = preprocessor.preprocess(noisy_ecg)
    
    print(f"Original shape: {noisy_ecg.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Original mean: {np.mean(noisy_ecg):.3f}, std: {np.std(noisy_ecg):.3f}")
    print(f"Cleaned mean: {np.mean(cleaned):.3f}, std: {np.std(cleaned):.3f}")
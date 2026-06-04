"""
Online R-peak detection for real-time ECG processing
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict
from scipy.signal import butter, filtfilt, find_peaks
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BeatDetectorConfig:
    """Configuration for online beat detection"""
    sampling_rate: int = 360
    lowcut: float = 8.0  # Hz
    highcut: float = 20.0  # Hz
    integration_window_ms: int = 150  # ms
    refractory_ms: int = 250  # ms (absolute refractory period)
    search_back_ms: int = 150  # ms
    threshold_initial: float = 0.5
    threshold_adjustment: float = 0.2
    min_heart_rate: float = 30  # BPM
    max_heart_rate: float = 200  # BPM


class OnlineBeatDetector:
    """
    Real-time R-peak detection for streaming ECG
    
    Implements an adaptive threshold algorithm for online
    R-peak detection with minimal latency.
    
    Features:
    - Adaptive threshold based on signal statistics
    - Refractory period to avoid double detection
    - Search back for missed beats
    - Real-time heart rate estimation
    """
    
    def __init__(self, config: BeatDetectorConfig):
        """
        Initialize online beat detector
        
        Args:
            config: Beat detector configuration
        """
        self.config = config
        self.sampling_rate = config.sampling_rate
        
        # Filter coefficients
        self._init_filters()
        
        # State variables
        self.buffer = deque(maxlen=int(5 * sampling_rate))  # 5 second buffer
        self.filtered_buffer = deque(maxlen=int(5 * sampling_rate))
        self.integrated_buffer = deque(maxlen=int(5 * sampling_rate))
        
        self.r_peaks = []
        self.last_peak_time = -float('inf')
        self.threshold = config.threshold_initial
        self.signal_peak = 0.0
        self.noise_peak = 0.0
        
        # Heart rate estimation
        self.rr_intervals = deque(maxlen=10)  # Last 10 RR intervals
        self.heart_rate = 75.0  # Initial estimate
        
        # For search back
        self.missed_beats_buffer = []
        
        logger.info(f"OnlineBeatDetector initialized: SR={sampling_rate}Hz")
        
    def _init_filters(self):
        """Initialize bandpass filters"""
        nyquist = 0.5 * self.sampling_rate
        low = self.config.lowcut / nyquist
        high = self.config.highcut / nyquist
        
        self.b, self.a = butter(2, [low, high], btype='band')
        
        # Integration window
        self.integration_window = int(
            self.config.integration_window_ms * self.sampling_rate / 1000
        )
        self.integration_kernel = np.ones(self.integration_window) / self.integration_window
        
    def process_sample(self, sample: float) -> Optional[int]:
        """
        Process a single ECG sample and detect R-peaks
        
        Args:
            sample: Raw ECG sample
            
        Returns:
            R-peak index if detected, None otherwise
        """
        self.buffer.append(sample)
        
        # Need enough samples for filtering
        if len(self.buffer) < self.integration_window + 10:
            return None
            
        # Convert buffer to array
        signal = np.array(self.buffer)
        
        # Apply bandpass filter
        filtered = filtfilt(self.b, self.a, signal)
        self.filtered_buffer.append(filtered[-1])
        
        # Square and integrate
        squared = filtered ** 2
        integrated = np.convolve(squared, self.integration_kernel, mode='same')
        self.integrated_buffer.append(integrated[-1])
        
        # Update adaptive threshold
        self._update_threshold(integrated[-1])
        
        # Check for peak
        if len(self.integrated_buffer) >= 3:
            current = self.integrated_buffer[-1]
            previous = self.integrated_buffer[-2]
            next_val = self.integrated_buffer[-3] if len(self.integrated_buffer) >= 3 else 0
            
            # Peak detection
            if (current > previous and current > next_val and 
                current > self.threshold and
                self._check_refractory()):
                
                # Peak found
                peak_time = len(self.buffer)
                self.r_peaks.append(peak_time)
                self.last_peak_time = peak_time
                
                # Update RR interval and heart rate
                self._update_heart_rate(peak_time)
                
                return peak_time
                
        return None
    
    def _update_threshold(self, value: float):
        """Update adaptive threshold based on signal statistics"""
        # Update signal peak (running average)
        if value > self.signal_peak:
            self.signal_peak = 0.95 * self.signal_peak + 0.05 * value
        else:
            self.signal_peak = 0.99 * self.signal_peak + 0.01 * value
            
        # Update noise peak
        if value < self.noise_peak:
            self.noise_peak = 0.95 * self.noise_peak + 0.05 * value
        else:
            self.noise_peak = 0.99 * self.noise_peak + 0.01 * value
            
        # Compute threshold
        self.threshold = self.noise_peak + self.config.threshold_adjustment * (
            self.signal_peak - self.noise_peak
        )
        
    def _check_refractory(self) -> bool:
        """Check if enough time has passed since last peak"""
        current_time = len(self.buffer)
        refractory_samples = int(self.config.refractory_ms * self.sampling_rate / 1000)
        
        if current_time - self.last_peak_time < refractory_samples:
            return False
            
        return True
    
    def _update_heart_rate(self, peak_time: int):
        """Update heart rate estimate based on RR intervals"""
        if len(self.r_peaks) >= 2:
            rr_interval = (self.r_peaks[-1] - self.r_peaks[-2]) / self.sampling_rate
            self.rr_intervals.append(rr_interval)
            
            # Filter invalid intervals
            valid_intervals = [
                r for r in self.rr_intervals 
                if 60/self.config.max_heart_rate <= r <= 60/self.config.min_heart_rate
            ]
            
            if valid_intervals:
                mean_rr = np.mean(valid_intervals)
                self.heart_rate = 60 / mean_rr
                
    def get_heart_rate(self) -> float:
        """Get current heart rate estimate"""
        return self.heart_rate
    
    def get_rr_intervals(self) -> List[float]:
        """Get recent RR intervals"""
        return list(self.rr_intervals)
    
    def get_latest_beat(self) -> Optional[np.ndarray]:
        """Get the most recent detected beat segment"""
        if len(self.r_peaks) < 1:
            return None
            
        # Define beat window (100ms before, 400ms after R-peak)
        pre_window = int(0.1 * self.sampling_rate)
        post_window = int(0.4 * self.sampling_rate)
        
        r_peak = self.r_peaks[-1]
        start = max(0, r_peak - pre_window)
        end = min(len(self.buffer), r_peak + post_window)
        
        return np.array(list(self.buffer))[start:end]
    
    def reset(self):
        """Reset detector state"""
        self.buffer.clear()
        self.filtered_buffer.clear()
        self.integrated_buffer.clear()
        self.r_peaks.clear()
        self.rr_intervals.clear()
        self.last_peak_time = -float('inf')
        self.threshold = self.config.threshold_initial
        self.signal_peak = 0.0
        self.noise_peak = 0.0
        self.heart_rate = 75.0
        
        logger.info("Beat detector reset")


class MultiLeadBeatDetector:
    """
    R-peak detection using multiple leads for improved accuracy
    """
    
    def __init__(self, num_leads: int, config: BeatDetectorConfig):
        """
        Initialize multi-lead beat detector
        
        Args:
            num_leads: Number of ECG leads
            config: Beat detector configuration
        """
        self.num_leads = num_leads
        self.detectors = [OnlineBeatDetector(config) for _ in range(num_leads)]
        
    def process_sample(self, samples: np.ndarray) -> Optional[int]:
        """
        Process samples from multiple leads
        
        Args:
            samples: Array of samples from each lead (num_leads,)
            
        Returns:
            R-peak index if detected (voting across leads)
        """
        detections = []
        
        for i, detector in enumerate(self.detectors):
            peak = detector.process_sample(samples[i])
            if peak is not None:
                detections.append(i)
                
        # Return if majority of leads detected a peak
        if len(detections) > self.num_leads // 2:
            # Use first detector's peak time
            return self.detectors[detections[0]].r_peaks[-1]
            
        return None
    
    def get_heart_rate(self) -> float:
        """Get average heart rate across leads"""
        rates = [d.get_heart_rate() for d in self.detectors]
        return np.mean(rates)
    
    def get_consensus_beat(self) -> Optional[np.ndarray]:
        """Get beat segment from lead with best signal quality"""
        # Select lead with highest signal-to-noise ratio
        best_lead = np.argmax([
            np.std(d.filtered_buffer) / (np.std(d.buffer) + 1e-8)
            for d in self.detectors
        ])
        
        return self.detectors[best_lead].get_latest_beat()


class RealTimeQRSDetector:
    """
    Real-time QRS detector with advanced features
    
    Uses Hamilton-Tompkins algorithm adapted for real-time processing
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize real-time QRS detector
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.ecg_buffer = []
        self.qrs_peaks = []
        
        # Filter coefficients
        self._init_filters()
        
        # State variables
        self.qrs_threshold = 0.5
        self.spki = 0.0  # Signal peak estimate
        self.npki = 0.0  # Noise peak estimate
        self.rr_avg = 0.8  # Average RR interval (seconds)
        self.rr_avg2 = 0.8
        self.rr_est = 0.8
        self.qrs_detected = False
        
    def _init_filters(self):
        """Initialize bandpass and derivative filters"""
        # Bandpass filter (5-15 Hz)
        nyquist = 0.5 * self.sampling_rate
        b, a = butter(2, [5/nyquist, 15/nyquist], btype='band')
        self.bp_b, self.bp_a = b, a
        
        # Derivative filter coefficients (5-point derivative)
        self.deriv_coeff = [-0.2, -0.1, 0, 0.1, 0.2]
        
    def process(self, sample: float) -> Optional[float]:
        """
        Process a single sample and detect QRS
        
        Args:
            sample: Raw ECG sample
            
        Returns:
            QRS score (0-1) if detected, None otherwise
        """
        self.ecg_buffer.append(sample)
        
        if len(self.ecg_buffer) < 50:
            return None
            
        # Apply bandpass filter
        filtered = filtfilt(self.bp_b, self.bp_a, np.array(self.ecg_buffer))
        
        # Derivative
        derivative = np.convolve(filtered, self.deriv_coeff, mode='same')
        
        # Square
        squared = derivative ** 2
        
        # Moving window integration
        window = int(0.15 * self.sampling_rate)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Update thresholds
        current_value = integrated[-1]
        
        if current_value > self.spki:
            self.spki = 0.95 * self.spki + 0.05 * current_value
        else:
            self.spki = 0.99 * self.spki + 0.01 * current_value
            
        if current_value < self.npki:
            self.npki = 0.95 * self.npki + 0.05 * current_value
        else:
            self.npki = 0.99 * self.npki + 0.01 * current_value
            
        # Compute threshold
        self.qrs_threshold = self.npki + 0.3 * (self.spki - self.npki)
        
        # Detect QRS
        if current_value > self.qrs_threshold and not self.qrs_detected:
            self.qrs_detected = True
            self.qrs_peaks.append(len(self.ecg_buffer))
            
            # Update RR estimates
            if len(self.qrs_peaks) >= 2:
                rr = (self.qrs_peaks[-1] - self.qrs_peaks[-2]) / self.sampling_rate
                self.rr_avg = 0.875 * self.rr_avg + 0.125 * rr
                self.rr_avg2 = 0.75 * self.rr_avg2 + 0.25 * rr
                self.rr_est = 0.75 * self.rr_avg + 0.25 * self.rr_avg2
                
            return current_value / (self.spki + 1e-8)
            
        # Reset detection after refractory period
        if self.qrs_detected and len(self.ecg_buffer) - self.qrs_peaks[-1] > 0.3 * self.sampling_rate:
            self.qrs_detected = False
            
        return None
    
    def get_heart_rate(self) -> float:
        """Get current heart rate"""
        if len(self.qrs_peaks) >= 2:
            last_rr = (self.qrs_peaks[-1] - self.qrs_peaks[-2]) / self.sampling_rate
            return 60 / max(0.3, min(2.0, last_rr))
        return 75.0
    
    def reset(self):
        """Reset detector"""
        self.ecg_buffer.clear()
        self.qrs_peaks.clear()
        self.spki = 0.0
        self.npki = 0.0
        self.qrs_detected = False


if __name__ == "__main__":
    # Test beat detector
    config = BeatDetectorConfig()
    detector = OnlineBeatDetector(config)
    
    # Simulate ECG signal
    t = np.linspace(0, 10, 3600)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    
    # Add synthetic R-peaks
    for i in range(0, 3600, 360):
        ecg[i:i+10] += 1.0
    
    # Process
    peaks = []
    for i, sample in enumerate(ecg):
        peak = detector.process_sample(sample)
        if peak:
            peaks.append(i)
            
    print(f"Detected {len(peaks)} peaks (expected ~10)")
    print(f"Heart rate: {detector.get_heart_rate():.1f} BPM")
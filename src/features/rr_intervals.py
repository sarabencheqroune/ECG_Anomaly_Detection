"""
RR interval and heart rate variability (HRV) feature extraction
"""

import numpy as np
from scipy import signal, stats
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


class RRIntervalFeatures:
    """
    Extract RR interval and HRV features from ECG R-peak timings
    
    Features include:
    - Time domain HRV metrics (SDNN, RMSSD, pNN50, etc.)
    - Frequency domain HRV metrics (VLF, LF, HF power)
    - Non-linear dynamics (Poincaré plot, entropy)
    - Heart rate statistics
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize RR interval feature extractor
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def extract_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Extract RR intervals from R-peak positions
        
        Args:
            r_peaks: Array of R-peak indices
            
        Returns:
            Array of RR intervals in seconds
        """
        if len(r_peaks) < 2:
            return np.array([])
            
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        
        # Remove outliers (physiologically impossible intervals)
        rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
        
        return rr_intervals
    
    def extract_time_domain_features(self, rr_intervals: np.ndarray) -> Dict:
        """
        Extract time domain HRV features
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            
        Returns:
            Dictionary of time domain features
        """
        if len(rr_intervals) < 2:
            return self._empty_time_features()
            
        features = {}
        
        # Basic statistics
        features['mean_rr'] = np.mean(rr_intervals) * 1000  # ms
        features['median_rr'] = np.median(rr_intervals) * 1000  # ms
        features['std_rr'] = np.std(rr_intervals) * 1000  # ms
        features['min_rr'] = np.min(rr_intervals) * 1000  # ms
        features['max_rr'] = np.max(rr_intervals) * 1000  # ms
        
        # Heart rate (BPM)
        features['mean_hr'] = 60.0 / features['mean_rr'] * 1000
        features['max_hr'] = 60.0 / features['min_rr'] * 1000
        features['min_hr'] = 60.0 / features['max_rr'] * 1000
        features['std_hr'] = 60.0 * features['std_rr'] / (features['mean_rr'] ** 2) * 1000
        
        # SDNN (Standard deviation of NN intervals)
        features['sdnn'] = features['std_rr']
        
        # RMSSD (Root mean square of successive differences)
        successive_diff = np.diff(rr_intervals) * 1000
        features['rmssd'] = np.sqrt(np.mean(successive_diff ** 2))
        
        # pNN50 (Percentage of successive differences > 50ms)
        features['pnn50'] = 100.0 * np.sum(np.abs(successive_diff) > 50) / len(successive_diff)
        
        # pNN20 (Percentage of successive differences > 20ms)
        features['pnn20'] = 100.0 * np.sum(np.abs(successive_diff) > 20) / len(successive_diff)
        
        # HRV triangular index
        histogram, bin_edges = np.histogram(rr_intervals * 1000, bins=50)
        features['triangular_index'] = len(rr_intervals) / np.max(histogram)
        
        # TINN (Triangular interpolation of NN interval histogram)
        # Simplified calculation
        peak_bin = np.argmax(histogram)
        left_base = 0
        right_base = len(histogram) - 1
        for i in range(peak_bin, -1, -1):
            if histogram[i] < histogram[peak_bin] / 10:
                left_base = i
                break
        for i in range(peak_bin, len(histogram)):
            if histogram[i] < histogram[peak_bin] / 10:
                right_base = i
                break
        features['tinn'] = (bin_edges[right_base] - bin_edges[left_base])
        
        return features
    
    def extract_frequency_domain_features(self, rr_intervals: np.ndarray, 
                                          fs: float = 4.0) -> Dict:
        """
        Extract frequency domain HRV features using Welch's method
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            fs: Resampling frequency for HRV spectrum (Hz)
            
        Returns:
            Dictionary of frequency domain features
        """
        if len(rr_intervals) < 5:
            return self._empty_freq_features()
            
        features = {}
        
        # Interpolate RR intervals to uniform sampling
        time = np.cumsum(rr_intervals)
        time = np.insert(time, 0, 0)
        
        # Resample at desired frequency
        duration = time[-1]
        num_samples = int(duration * fs)
        
        if num_samples < 10:
            return self._empty_freq_features()
            
        t_interp = np.linspace(0, duration, num_samples)
        
        # Create interpolation function
        f = interp1d(time, np.insert(rr_intervals, 0, rr_intervals[0]), 
                    kind='cubic', fill_value='extrapolate')
        rr_interpolated = f(t_interp)
        
        # Remove mean and detrend
        rr_interpolated = signal.detrend(rr_interpolated)
        
        # Compute power spectral density using Welch's method
        nperseg = min(256, len(rr_interpolated) // 2)
        if nperseg < 4:
            return self._empty_freq_features()
            
        freqs, psd = signal.welch(rr_interpolated, fs=fs, nperseg=nperseg)
        
        # Define frequency bands
        vlf_band = (0.0033, 0.04)  # Very Low Frequency
        lf_band = (0.04, 0.15)     # Low Frequency
        hf_band = (0.15, 0.4)      # High Frequency
        
        # Calculate power in each band
        vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
        lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
        
        features['vlf_power'] = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
        features['lf_power'] = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        features['hf_power'] = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        
        features['total_power'] = features['vlf_power'] + features['lf_power'] + features['hf_power']
        
        # Ratios
        if features['hf_power'] > 0:
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
        else:
            features['lf_hf_ratio'] = 0
            
        if features['total_power'] > 0:
            features['lf_norm'] = features['lf_power'] / features['total_power'] * 100
            features['hf_norm'] = features['hf_power'] / features['total_power'] * 100
        else:
            features['lf_norm'] = 0
            features['hf_norm'] = 0
            
        # Peak frequencies
        if np.any(lf_mask):
            lf_peak_idx = np.argmax(psd[lf_mask])
            features['lf_peak_freq'] = freqs[lf_mask][lf_peak_idx]
        else:
            features['lf_peak_freq'] = 0
            
        if np.any(hf_mask):
            hf_peak_idx = np.argmax(psd[hf_mask])
            features['hf_peak_freq'] = freqs[hf_mask][hf_peak_idx]
        else:
            features['hf_peak_freq'] = 0
            
        return features
    
    def extract_nonlinear_features(self, rr_intervals: np.ndarray) -> Dict:
        """
        Extract non-linear dynamics features from RR intervals
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            
        Returns:
            Dictionary of non-linear features
        """
        if len(rr_intervals) < 10:
            return self._empty_nonlinear_features()
            
        features = {}
        
        # Poincaré plot features (SD1, SD2)
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        # SD1 (short-term variability) - standard deviation perpendicular to identity line
        sd1 = np.std((rr_n - rr_n1) / np.sqrt(2)) * 1000  # ms
        
        # SD2 (long-term variability) - standard deviation along identity line
        sd2 = np.std((rr_n + rr_n1) / np.sqrt(2)) * 1000  # ms
        
        features['sd1'] = sd1
        features['sd2'] = sd2
        features['sd1_sd2_ratio'] = sd1 / sd2 if sd2 > 0 else 0
        
        # Sample entropy (complexity measure)
        features['sample_entropy'] = self._calculate_sample_entropy(rr_intervals)
        
        # Approximate entropy
        features['approx_entropy'] = self._calculate_approx_entropy(rr_intervals)
        
        # Detrended fluctuation analysis (DFA) alpha
        features['dfa_alpha1'], features['dfa_alpha2'] = self._calculate_dfa(rr_intervals)
        
        return features
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate sample entropy of time series
        
        Args:
            data: Input time series
            m: Embedding dimension
            r: Tolerance (as fraction of standard deviation)
            
        Returns:
            Sample entropy value
        """
        if len(data) < m + 2:
            return 0
            
        # Normalize by standard deviation
        data = data / np.std(data)
        r = r * np.std(data)
        
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
            
        def _phi(m):
            x = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
            C = 0
            for i in range(len(x)):
                count = np.sum([_maxdist(x[i], x[j]) <= r for j in range(len(x)) if j != i])
                C += count / (len(x) - 1)
            return C / len(x)
            
        if _phi(m) == 0 or _phi(m + 1) == 0:
            return 0
            
        return -np.log(_phi(m + 1) / _phi(m))
    
    def _calculate_approx_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate approximate entropy of time series
        
        Args:
            data: Input time series
            m: Embedding dimension
            r: Tolerance (as fraction of standard deviation)
            
        Returns:
            Approximate entropy value
        """
        if len(data) < m + 1:
            return 0
            
        # Normalize
        data = data / np.std(data)
        r = r * np.std(data)
        
        def _phi(m):
            x = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
            C = 0
            for i in range(len(x)):
                distances = np.max(np.abs(x - x[i]), axis=1)
                count = np.sum(distances <= r)
                C += np.log(count / (len(x)))
            return C / len(x)
            
        return _phi(m) - _phi(m + 1)
    
    def _calculate_dfa(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Detrended Fluctuation Analysis exponents
        
        Args:
            data: Input time series
            
        Returns:
            Tuple of (alpha1 for short-term, alpha2 for long-term)
        """
        if len(data) < 100:
            return 0, 0
            
        # Cumulative sum
        y = np.cumsum(data - np.mean(data))
        
        # Define window sizes
        scales = np.logspace(np.log10(4), np.log10(len(y) // 4), 20).astype(int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        for scale in scales:
            if scale < 4:
                continue
                
            # Split into non-overlapping windows
            n_windows = len(y) // scale
            if n_windows < 2:
                continue
                
            # Detrend each window
            local_fluctuations = []
            for i in range(n_windows):
                window = y[i * scale:(i + 1) * scale]
                x = np.arange(len(window))
                
                # Linear detrending
                coeffs = np.polyfit(x, window, 1)
                trend = np.polyval(coeffs, x)
                detrended = window - trend
                
                # RMS fluctuation
                local_fluctuations.append(np.sqrt(np.mean(detrended ** 2)))
                
            fluctuations.append(np.mean(local_fluctuations))
            
        # Fit power law
        if len(fluctuations) < 4:
            return 0, 0
            
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        
        # Separate short-term and long-term scales
        split_idx = len(log_scales) // 2
        
        # Short-term alpha (scales 4-11)
        if split_idx > 4:
            coeffs_short = np.polyfit(log_scales[:split_idx], log_fluctuations[:split_idx], 1)
            alpha1 = coeffs_short[0]
        else:
            alpha1 = 0
            
        # Long-term alpha
        if len(log_scales) > split_idx + 4:
            coeffs_long = np.polyfit(log_scales[split_idx:], log_fluctuations[split_idx:], 1)
            alpha2 = coeffs_long[0]
        else:
            alpha2 = 0
            
        return alpha1, alpha2
    
    def _empty_time_features(self) -> Dict:
        """Return empty time domain features"""
        return {
            'mean_rr': 0, 'median_rr': 0, 'std_rr': 0, 'min_rr': 0, 'max_rr': 0,
            'mean_hr': 0, 'max_hr': 0, 'min_hr': 0, 'std_hr': 0,
            'sdnn': 0, 'rmssd': 0, 'pnn50': 0, 'pnn20': 0,
            'triangular_index': 0, 'tinn': 0
        }
    
    def _empty_freq_features(self) -> Dict:
        """Return empty frequency domain features"""
        return {
            'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 'total_power': 0,
            'lf_hf_ratio': 0, 'lf_norm': 0, 'hf_norm': 0,
            'lf_peak_freq': 0, 'hf_peak_freq': 0
        }
    
    def _empty_nonlinear_features(self) -> Dict:
        """Return empty non-linear features"""
        return {
            'sd1': 0, 'sd2': 0, 'sd1_sd2_ratio': 0,
            'sample_entropy': 0, 'approx_entropy': 0,
            'dfa_alpha1': 0, 'dfa_alpha2': 0
        }
    
    def extract_all_features(self, r_peaks: np.ndarray) -> Dict:
        """
        Extract all RR interval features
        
        Args:
            r_peaks: Array of R-peak indices
            
        Returns:
            Dictionary containing all RR interval features
        """
        rr_intervals = self.extract_rr_intervals(r_peaks)
        
        if len(rr_intervals) < 2:
            return {
                **self._empty_time_features(),
                **self._empty_freq_features(),
                **self._empty_nonlinear_features(),
                'num_beats': len(r_peaks),
                'num_valid_rr': len(rr_intervals)
            }
            
        features = {
            'num_beats': len(r_peaks),
            'num_valid_rr': len(rr_intervals)
        }
        
        # Extract all feature groups
        features.update(self.extract_time_domain_features(rr_intervals))
        features.update(self.extract_frequency_domain_features(rr_intervals))
        features.update(self.extract_nonlinear_features(rr_intervals))
        
        return features


if __name__ == "__main__":
    # Test RR interval features
    extractor = RRIntervalFeatures(sampling_rate=360)
    
    # Generate synthetic R-peaks
    rr_intervals = 0.8 + 0.05 * np.random.randn(100)  # ~75 BPM
    r_peaks = np.cumsum(rr_intervals * 360).astype(int)
    
    # Extract features
    features = extractor.extract_all_features(r_peaks)
    
    print("RR Interval Features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.3f}")
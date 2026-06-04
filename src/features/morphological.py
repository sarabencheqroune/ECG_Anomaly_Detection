"""
Morphological feature extraction from ECG waveform (PQRST complex)
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from typing import Tuple, List, Dict, Optional
from scipy.interpolate import interp1d


class MorphologicalFeatures:
    """
    Extract morphological features from ECG beats (PQRST complex)
    
    Features include:
    - P wave: amplitude, duration, area
    - QRS complex: width, amplitude, slopes
    - ST segment: elevation/depression
    - T wave: amplitude, symmetry, morphology
    - QT interval and corrected QT
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize morphological feature extractor
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def detect_pqrst_peaks(self, beat: np.ndarray, 
                          r_peak_idx: int) -> Dict:
        """
        Detect P, Q, R, S, T wave peaks in a single beat
        
        Args:
            beat: Single ECG beat segment
            r_peak_idx: Index of R-peak within the beat
            
        Returns:
            Dictionary with peak indices and amplitudes
        """
        peaks = {
            'p_idx': -1, 'q_idx': -1, 'r_idx': r_peak_idx,
            's_idx': -1, 't_idx': -1, 't_end_idx': -1
        }
        amplitudes = {
            'p_amp': 0, 'q_amp': 0, 'r_amp': beat[r_peak_idx],
            's_amp': 0, 't_amp': 0
        }
        
        # Search windows (in samples, assuming 360 Hz)
        # These windows need to be adjusted for different sampling rates
        scale_factor = self.sampling_rate / 360
        
        # Q wave: 20-60 ms before R peak
        q_start = max(0, r_peak_idx - int(0.06 * self.sampling_rate))
        q_end = max(0, r_peak_idx - int(0.01 * self.sampling_rate))
        
        if q_start < q_end:
            q_region = beat[q_start:q_end]
            if len(q_region) > 0:
                q_idx_rel = np.argmin(q_region)
                peaks['q_idx'] = q_start + q_idx_rel
                amplitudes['q_amp'] = beat[peaks['q_idx']]
                
        # S wave: 20-60 ms after R peak
        s_start = min(len(beat) - 1, r_peak_idx + int(0.01 * self.sampling_rate))
        s_end = min(len(beat) - 1, r_peak_idx + int(0.06 * self.sampling_rate))
        
        if s_start < s_end:
            s_region = beat[s_start:s_end]
            if len(s_region) > 0:
                s_idx_rel = np.argmin(s_region)
                peaks['s_idx'] = s_start + s_idx_rel
                amplitudes['s_amp'] = beat[peaks['s_idx']]
                
        # P wave: 200-400 ms before R peak
        p_start = max(0, r_peak_idx - int(0.45 * self.sampling_rate))
        p_end = max(0, r_peak_idx - int(0.20 * self.sampling_rate))
        
        if p_start < p_end:
            p_region = beat[p_start:p_end]
            if len(p_region) > 0:
                p_idx_rel = np.argmax(p_region)
                peaks['p_idx'] = p_start + p_idx_rel
                amplitudes['p_amp'] = beat[peaks['p_idx']]
                
        # T wave: 150-400 ms after R peak
        t_start = min(len(beat) - 1, r_peak_idx + int(0.15 * self.sampling_rate))
        t_end = min(len(beat) - 1, r_peak_idx + int(0.45 * self.sampling_rate))
        
        if t_start < t_end:
            t_region = beat[t_start:t_end]
            if len(t_region) > 0:
                t_idx_rel = np.argmax(t_region)
                peaks['t_idx'] = t_start + t_idx_rel
                amplitudes['t_amp'] = beat[peaks['t_idx']]
                
        # Find T wave end (where signal returns to baseline after T peak)
        if peaks['t_idx'] > 0:
            t_end_candidates = []
            for i in range(peaks['t_idx'], min(len(beat), peaks['t_idx'] + int(0.2 * self.sampling_rate))):
                if abs(beat[i]) < 0.05 * amplitudes['t_amp']:
                    t_end_candidates.append(i)
            if t_end_candidates:
                peaks['t_end_idx'] = t_end_candidates[0]
                
        return {'peaks': peaks, 'amplitudes': amplitudes}
    
    def extract_interval_features(self, peaks: Dict) -> Dict:
        """
        Extract interval-based features (durations)
        
        Args:
            peaks: Dictionary of peak indices from detect_pqrst_peaks
            
        Returns:
            Dictionary of interval features in milliseconds
        """
        features = {}
        
        # Convert sample indices to time (ms)
        def to_ms(samples):
            return samples / self.sampling_rate * 1000
            
        # PR interval (P wave onset to QRS onset)
        if peaks['p_idx'] > 0 and peaks['q_idx'] > 0:
            features['pr_interval_ms'] = to_ms(peaks['q_idx'] - peaks['p_idx'])
        else:
            features['pr_interval_ms'] = 0
            
        # QRS duration (Q wave to S wave)
        if peaks['q_idx'] > 0 and peaks['s_idx'] > 0:
            features['qrs_duration_ms'] = to_ms(peaks['s_idx'] - peaks['q_idx'])
        else:
            features['qrs_duration_ms'] = 0
            
        # QT interval (Q wave to T wave end)
        if peaks['q_idx'] > 0 and peaks['t_end_idx'] > 0:
            features['qt_interval_ms'] = to_ms(peaks['t_end_idx'] - peaks['q_idx'])
        else:
            features['qt_interval_ms'] = 0
            
        # QTc (corrected QT using Bazett's formula)
        # Note: Requires heart rate from RR intervals
        features['qtc_ms'] = 0  # Will be set externally
            
        # ST segment (J point to T wave onset)
        if peaks['s_idx'] > 0 and peaks['t_idx'] > 0:
            j_point = peaks['s_idx'] + int(0.02 * self.sampling_rate)  # 20 ms after S wave
            features['st_duration_ms'] = to_ms(peaks['t_idx'] - j_point)
        else:
            features['st_duration_ms'] = 0
            
        return features
    
    def extract_amplitude_features(self, beat: np.ndarray, 
                                   peaks: Dict, amplitudes: Dict) -> Dict:
        """
        Extract amplitude-based features
        
        Args:
            beat: ECG beat signal
            peaks: Peak indices
            amplitudes: Peak amplitudes
            
        Returns:
            Dictionary of amplitude features
        """
        features = {}
        
        # Basic amplitudes (mV, assuming normalized)
        features['r_amplitude'] = amplitudes['r_amp']
        features['q_amplitude'] = amplitudes['q_amp']
        features['s_amplitude'] = amplitudes['s_amp']
        features['p_amplitude'] = amplitudes['p_amp']
        features['t_amplitude'] = amplitudes['t_amp']
        
        # Amplitude ratios
        if features['r_amplitude'] != 0:
            features['q_r_ratio'] = abs(features['q_amplitude'] / features['r_amplitude'])
            features['s_r_ratio'] = abs(features['s_amplitude'] / features['r_amplitude'])
            features['t_r_ratio'] = abs(features['t_amplitude'] / features['r_amplitude'])
        else:
            features['q_r_ratio'] = 0
            features['s_r_ratio'] = 0
            features['t_r_ratio'] = 0
            
        # ST segment elevation/depression
        if peaks['s_idx'] > 0:
            # J point (end of QRS)
            j_point = min(len(beat) - 1, peaks['s_idx'] + int(0.02 * self.sampling_rate))
            # ST point (80 ms after J point)
            st_point = min(len(beat) - 1, j_point + int(0.08 * self.sampling_rate))
            
            features['st_elevation'] = beat[st_point] - beat[j_point]
        else:
            features['st_elevation'] = 0
            
        # T wave symmetry
        if peaks['t_idx'] > 0:
            # Check T wave symmetry by comparing upslope and downslope
            t_peak = peaks['t_idx']
            t_start = max(0, t_peak - int(0.1 * self.sampling_rate))
            t_end = min(len(beat) - 1, t_peak + int(0.1 * self.sampling_rate))
            
            if t_start < t_peak < t_end:
                upslope = beat[t_peak] - beat[t_start]
                downslope = beat[t_peak] - beat[t_end]
                features['t_wave_symmetry'] = upslope / (downslope + 1e-8)
            else:
                features['t_wave_symmetry'] = 1.0
        else:
            features['t_wave_symmetry'] = 1.0
            
        return features
    
    def extract_area_features(self, beat: np.ndarray, peaks: Dict) -> Dict:
        """
        Extract area under curve features
        
        Args:
            beat: ECG beat signal
            peaks: Peak indices
            
        Returns:
            Dictionary of area features
        """
        features = {}
        
        # Normalize beat for area calculation
        beat_norm = beat - np.mean(beat)
        
        # QRS area
        if peaks['q_idx'] > 0 and peaks['s_idx'] > 0:
            qrs_indices = slice(peaks['q_idx'], peaks['s_idx'])
            features['qrs_area'] = np.trapz(np.abs(beat_norm[qrs_indices]))
        else:
            features['qrs_area'] = 0
            
        # P wave area
        if peaks['p_idx'] > 0:
            p_start = max(0, peaks['p_idx'] - int(0.05 * self.sampling_rate))
            p_end = min(len(beat) - 1, peaks['p_idx'] + int(0.05 * self.sampling_rate))
            features['p_area'] = np.trapz(np.abs(beat_norm[p_start:p_end]))
        else:
            features['p_area'] = 0
            
        # T wave area
        if peaks['t_idx'] > 0:
            t_start = max(0, peaks['t_idx'] - int(0.1 * self.sampling_rate))
            t_end = min(len(beat) - 1, peaks['t_idx'] + int(0.1 * self.sampling_rate))
            features['t_area'] = np.trapz(np.abs(beat_norm[t_start:t_end]))
        else:
            features['t_area'] = 0
            
        # Total beat area
        features['total_area'] = np.trapz(np.abs(beat_norm))
        
        # Area ratios
        if features['total_area'] > 0:
            features['qrs_area_ratio'] = features['qrs_area'] / features['total_area']
            features['p_area_ratio'] = features['p_area'] / features['total_area']
            features['t_area_ratio'] = features['t_area'] / features['total_area']
        else:
            features['qrs_area_ratio'] = 0
            features['p_area_ratio'] = 0
            features['t_area_ratio'] = 0
            
        return features
    
    def extract_slope_features(self, beat: np.ndarray, peaks: Dict) -> Dict:
        """
        Extract slope/derivative features
        
        Args:
            beat: ECG beat signal
            peaks: Peak indices
            
        Returns:
            Dictionary of slope features
        """
        features = {}
        
        # Calculate first derivative
        derivative = np.gradient(beat)
        
        # Maximum upslope and downslope in QRS
        if peaks['q_idx'] > 0 and peaks['s_idx'] > 0:
            qrs_deriv = derivative[peaks['q_idx']:peaks['s_idx']]
            if len(qrs_deriv) > 0:
                features['max_qrs_upslope'] = np.max(qrs_deriv)
                features['max_qrs_downslope'] = np.min(qrs_deriv)
            else:
                features['max_qrs_upslope'] = 0
                features['max_qrs_downslope'] = 0
        else:
            features['max_qrs_upslope'] = 0
            features['max_qrs_downslope'] = 0
            
        # Maximum slope in ST segment
        if peaks['s_idx'] > 0:
            st_start = peaks['s_idx'] + int(0.02 * self.sampling_rate)
            st_end = min(len(beat) - 1, st_start + int(0.1 * self.sampling_rate))
            if st_start < st_end:
                st_deriv = derivative[st_start:st_end]
                features['max_st_slope'] = np.max(np.abs(st_deriv)) if len(st_deriv) > 0 else 0
            else:
                features['max_st_slope'] = 0
        else:
            features['max_st_slope'] = 0
            
        # Zero crossing rate (indicator of noise)
        zero_crossings = np.sum(np.diff(np.sign(beat)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(beat)
        
        # Maximum derivative overall
        features['max_derivative'] = np.max(np.abs(derivative))
        
        return features
    
    def extract_morphology_score(self, beat: np.ndarray, 
                                 peaks: Dict, amplitudes: Dict) -> float:
        """
        Calculate a morphology quality score
        
        Args:
            beat: ECG beat signal
            peaks: Peak indices
            amplitudes: Peak amplitudes
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Check if major peaks are present
        if peaks['r_idx'] < 0:
            return 0.0
            
        # Penalize if P or T waves are missing
        if peaks['p_idx'] < 0 or amplitudes['p_amp'] < 0.05 * amplitudes['r_amp']:
            score -= 0.2
            
        if peaks['t_idx'] < 0 or amplitudes['t_amp'] < 0.1 * amplitudes['r_amp']:
            score -= 0.2
            
        # Check QRS duration (should be 60-120ms)
        if peaks['q_idx'] > 0 and peaks['s_idx'] > 0:
            qrs_duration = (peaks['s_idx'] - peaks['q_idx']) / self.sampling_rate * 1000
            if qrs_duration < 60 or qrs_duration > 120:
                score -= 0.2
                
        # Check for clipping
        if np.max(np.abs(beat)) > 0.95:  # Assuming normalized signal
            score -= 0.2
            
        # Check signal-to-noise ratio in flat regions
        pr_segment = beat[max(0, peaks['p_idx'] - 20):max(0, peaks['p_idx'])] if peaks['p_idx'] > 0 else beat[:50]
        if len(pr_segment) > 0:
            noise_level = np.std(pr_segment)
            if noise_level > 0.1:
                score -= noise_level
                
        return max(0, min(1, score))
    
    def extract_all_features(self, beat: np.ndarray, 
                            r_peak_idx: int,
                            heart_rate: Optional[float] = None) -> Dict:
        """
        Extract all morphological features from an ECG beat
        
        Args:
            beat: Single ECG beat segment
            r_peak_idx: Index of R-peak within the beat
            heart_rate: Heart rate in BPM (for QTc calculation)
            
        Returns:
            Dictionary of all morphological features
        """
        # Detect PQRST peaks
        detection = self.detect_pqrst_peaks(beat, r_peak_idx)
        peaks = detection['peaks']
        amplitudes = detection['amplitudes']
        
        # Extract feature groups
        features = {}
        features.update(self.extract_interval_features(peaks))
        features.update(self.extract_amplitude_features(beat, peaks, amplitudes))
        features.update(self.extract_area_features(beat, peaks))
        features.update(self.extract_slope_features(beat, peaks))
        
        # Calculate QTc if heart rate provided
        if heart_rate and heart_rate > 0 and features['qt_interval_ms'] > 0:
            # Bazett's formula: QTc = QT / sqrt(RR)
            rr_interval = 60000 / heart_rate  # RR in ms
            features['qtc_ms'] = features['qt_interval_ms'] / np.sqrt(rr_interval / 1000)
        else:
            features['qtc_ms'] = 0
            
        # Morphology quality score
        features['morphology_quality'] = self.extract_morphology_score(beat, peaks, amplitudes)
        
        return features


if __name__ == "__main__":
    # Test morphological features
    extractor = MorphologicalFeatures(sampling_rate=360)
    
    # Generate synthetic beat (simplified)
    t = np.linspace(-0.3, 0.5, int(0.8 * 360))
    beat = np.zeros_like(t)
    
    # R peak
    r_idx = np.argmin(np.abs(t - 0))
    beat[r_idx] = 1.0
    
    # Q wave
    q_idx = np.argmin(np.abs(t + 0.03))
    beat[q_idx] = -0.15
    
    # S wave
    s_idx = np.argmin(np.abs(t + 0.05))
    beat[s_idx] = -0.3
    
    # T wave
    t_idx = np.argmin(np.abs(t + 0.25))
    beat[t_idx] = 0.3
    
    # Extract features
    features = extractor.extract_all_features(beat, r_idx, heart_rate=75)
    
    print("Morphological Features:")
    for key, value in list(features.items())[:15]:
        print(f"  {key}: {value:.3f}")
"""
ECG beat segmentation using R-peak detection
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Dict
import neurokit2 as nk
from biosppy.signals import ecg
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BeatSegment:
    """Container for a segmented beat"""
    beat_id: int
    signal: np.ndarray  # Segmented beat waveform
    r_peak_index: int   # Position of R-peak within segment
    start_index: int    # Start index in original signal
    end_index: int      # End index in original signal
    label: Optional[str] = None
    quality_score: float = 1.0


class ECGSegmenter:
    """
    ECG beat segmentation with multiple R-peak detection algorithms
    """
    
    def __init__(self, sampling_rate: int = 360,
                 pre_window_ms: int = 100,  # ms before R-peak
                 post_window_ms: int = 400):  # ms after R-peak
        """
        Initialize segmenter
        
        Args:
            sampling_rate: Signal sampling rate in Hz
            pre_window_ms: Window before R-peak in milliseconds
            post_window_ms: Window after R-peak in milliseconds
        """
        self.sampling_rate = sampling_rate
        self.pre_window = int(pre_window_ms * sampling_rate / 1000)
        self.post_window = int(post_window_ms * sampling_rate / 1000)
        self.segment_length = self.pre_window + self.post_window + 1
        
    def detect_r_peaks_neurokit(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect R-peaks using NeuroKit2's algorithm
        
        Args:
            signal: Preprocessed ECG signal
            
        Returns:
            Tuple of (r_peak_indices, detected_peaks_info)
        """
        try:
            # Clean signal first
            cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)
            
            # Detect peaks
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=self.sampling_rate)
            r_peaks = rpeaks['ECG_R_Peaks']
            
            # Remove NaN values
            r_peaks = r_peaks[~np.isnan(r_peaks)].astype(int)
            
            return r_peaks, {'method': 'neurokit2', 'num_peaks': len(r_peaks)}
            
        except Exception as e:
            logger.error(f"NeuroKit2 peak detection failed: {e}")
            return np.array([]), {'error': str(e)}
    
    def detect_r_peaks_biosppy(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect R-peaks using BioSPy's algorithm (Pan-Tompkins variant)
        
        Args:
            signal: Preprocessed ECG signal
            
        Returns:
            Tuple of (r_peak_indices, detected_peaks_info)
        """
        try:
            # Process with BioSPy
            processed = ecg.ecg(signal=signal, sampling_rate=self.sampling_rate, show=False)
            r_peaks = processed['rpeaks']
            
            return r_peaks, {'method': 'biosppy', 'num_peaks': len(r_peaks)}
            
        except Exception as e:
            logger.error(f"BioSPy peak detection failed: {e}")
            return np.array([]), {'error': str(e)}
    
    def detect_r_peaks_hamilton(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect R-peaks using Hamilton's algorithm (simplified)
        
        Args:
            signal: Preprocessed ECG signal
            
        Returns:
            Tuple of (r_peak_indices, detected_peaks_info)
        """
        # Simplified Hamilton algorithm
        # Step 1: Bandpass filter around QRS complex (8-20 Hz)
        from scipy.signal import butter, filtfilt
        
        nyquist = 0.5 * self.sampling_rate
        low = 8 / nyquist
        high = 20 / nyquist
        
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Step 2: Square the signal
        squared = filtered ** 2
        
        # Step 3: Moving window integration
        window = int(0.15 * self.sampling_rate)  # 150ms window
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Step 4: Find peaks
        peaks, properties = find_peaks(integrated, 
                                      distance=int(0.3 * self.sampling_rate),
                                      height=np.mean(integrated) + np.std(integrated))
        
        # Step 5: Refine peaks to original signal
        refined_peaks = []
        for peak in peaks:
            search_start = max(0, peak - window//2)
            search_end = min(len(signal), peak + window//2)
            local_max = search_start + np.argmax(np.abs(signal[search_start:search_end]))
            refined_peaks.append(local_max)
            
        return np.array(refined_peaks), {'method': 'hamilton', 'num_peaks': len(refined_peaks)}
    
    def detect_r_peaks(self, signal: np.ndarray, 
                      method: str = 'neurokit2') -> Tuple[np.ndarray, Dict]:
        """
        Detect R-peaks using specified method
        
        Args:
            signal: Preprocessed ECG signal
            method: 'neurokit2', 'biosppy', or 'hamilton'
            
        Returns:
            Tuple of (r_peak_indices, metadata)
        """
        methods = {
            'neurokit2': self.detect_r_peaks_neurokit,
            'biosppy': self.detect_r_peaks_biosppy,
            'hamilton': self.detect_r_peaks_hamilton
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
            
        r_peaks, metadata = methods[method](signal)
        
        # Calculate heart rate
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)
            metadata['heart_rate_bpm'] = heart_rate
            metadata['rr_interval_mean_ms'] = np.mean(rr_intervals) * 1000
            metadata['rr_interval_std_ms'] = np.std(rr_intervals) * 1000
            
        return r_peaks, metadata
    
    def segment_beats(self, signal: np.ndarray, 
                     r_peaks: np.ndarray,
                     min_beat_distance: float = 0.3) -> List[BeatSegment]:
        """
        Segment individual heartbeats around detected R-peaks
        
        Args:
            signal: Preprocessed ECG signal
            r_peaks: Array of R-peak indices
            min_beat_distance: Minimum distance between beats in seconds
            
        Returns:
            List of BeatSegment objects
        """
        beats = []
        min_distance_samples = int(min_beat_distance * self.sampling_rate)
        
        for i, r_peak in enumerate(r_peaks):
            # Calculate segment boundaries
            start = max(0, r_peak - self.pre_window)
            end = min(len(signal), r_peak + self.post_window + 1)
            
            # Check if segment is valid length
            if end - start < self.segment_length * 0.8:
                continue
                
            # Extract beat
            beat_signal = signal[start:end]
            
            # Pad if necessary
            if len(beat_signal) < self.segment_length:
                pad_left = (self.segment_length - len(beat_signal)) // 2
                pad_right = self.segment_length - len(beat_signal) - pad_left
                beat_signal = np.pad(beat_signal, (pad_left, pad_right), mode='constant')
            elif len(beat_signal) > self.segment_length:
                # Trim to desired length
                center = self.pre_window
                half = self.segment_length // 2
                start_trim = max(0, center - half)
                beat_signal = beat_signal[start_trim:start_trim + self.segment_length]
            
            # Calculate beat quality (simple metric: SNR within beat)
            noise_floor = np.std(beat_signal[:self.pre_window//4])
            peak_amplitude = np.max(np.abs(beat_signal))
            quality = peak_amplitude / (noise_floor + 1e-8)
            quality = min(1.0, quality / 5.0)  # Normalize
            
            # Create BeatSegment
            beat = BeatSegment(
                beat_id=i,
                signal=beat_signal,
                r_peak_index=self.pre_window,  # R-peak centered in segment
                start_index=start,
                end_index=end,
                quality_score=quality
            )
            
            beats.append(beat)
            
            # Check for too-close beats
            if i > 0 and (r_peak - r_peaks[i-1]) < min_distance_samples:
                logger.warning(f"Beats {i-1} and {i} are very close ({r_peak - r_peaks[i-1]} samples)")
                
        logger.info(f"Segmented {len(beats)} beats from {len(r_peaks)} R-peaks")
        return beats
    
    def resample_beats(self, beats: List[BeatSegment], 
                      target_length: int = 187) -> List[BeatSegment]:
        """
        Resample all beats to fixed length
        
        Args:
            beats: List of BeatSegment objects
            target_length: Desired length for all beats
            
        Returns:
            List of BeatSegment objects with resampled signals
        """
        from scipy import interpolate
        
        resampled_beats = []
        
        for beat in beats:
            current_length = len(beat.signal)
            
            if current_length == target_length:
                resampled_beats.append(beat)
                continue
                
            # Create interpolation function
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            
            f = interpolate.interp1d(x_old, beat.signal, kind='cubic')
            resampled_signal = f(x_new)
            
            # Update R-peak position proportionally
            new_r_peak = int(beat.r_peak_index * target_length / current_length)
            
            # Create new beat segment
            resampled_beat = BeatSegment(
                beat_id=beat.beat_id,
                signal=resampled_signal,
                r_peak_index=new_r_peak,
                start_index=beat.start_index,
                end_index=beat.end_index,
                label=beat.label,
                quality_score=beat.quality_score
            )
            
            resampled_beats.append(resampled_beat)
            
        return resampled_beats
    
    def get_beat_features(self, beat: BeatSegment) -> Dict:
        """
        Extract features from a single beat
        
        Args:
            beat: BeatSegment object
            
        Returns:
            Dictionary of beat features
        """
        signal = beat.signal
        
        # Find PQRST peaks within beat
        from scipy.signal import find_peaks
        
        # R-peak is at known position
        r_idx = beat.r_peak_index
        
        # Find Q-wave (local min before R)
        search_start = max(0, r_idx - int(0.03 * self.sampling_rate))
        search_end = max(0, r_idx - int(0.01 * self.sampling_rate))
        if search_start < search_end:
            q_idx = search_start + np.argmin(signal[search_start:search_end])
        else:
            q_idx = r_idx
            
        # Find S-wave (local min after R)
        search_start = min(len(signal)-1, r_idx + int(0.01 * self.sampling_rate))
        search_end = min(len(signal)-1, r_idx + int(0.06 * self.sampling_rate))
        if search_start < search_end:
            s_idx = search_start + np.argmin(signal[search_start:search_end])
        else:
            s_idx = r_idx
            
        # Find P-wave (before Q)
        search_start = max(0, r_idx - int(0.2 * self.sampling_rate))
        search_end = max(0, r_idx - int(0.05 * self.sampling_rate))
        if search_start < search_end:
            p_idx = search_start + np.argmax(signal[search_start:search_end])
            p_amplitude = signal[p_idx]
        else:
            p_idx = -1
            p_amplitude = 0
            
        # Find T-wave (after S)
        search_start = min(len(signal)-1, r_idx + int(0.15 * self.sampling_rate))
        search_end = min(len(signal)-1, r_idx + int(0.4 * self.sampling_rate))
        if search_start < search_end:
            t_idx = search_start + np.argmax(signal[search_start:search_end])
            t_amplitude = signal[t_idx]
        else:
            t_idx = -1
            t_amplitude = 0
            
        features = {
            'r_amplitude': signal[r_idx],
            'q_amplitude': signal[q_idx] if q_idx != r_idx else 0,
            's_amplitude': signal[s_idx] if s_idx != r_idx else 0,
            'p_amplitude': p_amplitude,
            't_amplitude': t_amplitude,
            'qrs_amplitude': signal[r_idx] - min(signal[q_idx], signal[s_idx]),
            'qr_interval': abs(r_idx - q_idx) / self.sampling_rate * 1000,
            'rs_interval': abs(s_idx - r_idx) / self.sampling_rate * 1000,
            'qt_interval': (t_idx - q_idx) / self.sampling_rate * 1000 if t_idx > q_idx else 0,
            'heart_rate': 60 / ((beat.end_index - beat.start_index) / self.sampling_rate)
        }
        
        return features


if __name__ == "__main__":
    # Test segmenter
    import matplotlib.pyplot as plt
    
    # Generate synthetic signal
    t = np.linspace(0, 10, 3600)
    synthetic = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
    
    # Segment
    segmenter = ECGSegmenter(sampling_rate=360)
    r_peaks, meta = segmenter.detect_r_peaks(synthetic, method='neurokit2')
    beats = segmenter.segment_beats(synthetic, r_peaks)
    
    print(f"Detected {len(r_peaks)} R-peaks")
    print(f"Heart rate: {meta.get('heart_rate_bpm', 'N/A'):.1f} BPM")
    print(f"Segmented {len(beats)} beats")
    
    if beats:
        features = segmenter.get_beat_features(beats[0])
        print(f"Beat features: {list(features.keys())[:5]}...")
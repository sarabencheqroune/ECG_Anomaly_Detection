"""
Wavelet decomposition features for ECG analysis

This module implements various wavelet-based feature extraction techniques
for ECG signal processing, including:
- Multi-resolution analysis (MRA)
- Wavelet coefficient statistics
- Energy and entropy measures
- Wavelet-based denoising
- Scalogram generation
- Wavelet packet decomposition
"""

import numpy as np
import pywt
from scipy import stats, signal
from scipy.signal import find_peaks, find_peaks_cwt
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class WaveletFeatures:
    """
    Extract wavelet-based features from ECG signals
    
    Features include:
    - Wavelet coefficients statistics (mean, std, energy, entropy)
    - Multi-resolution analysis features
    - Energy distribution across scales
    - Wavelet-based QRS detection
    - Wavelet denoising capabilities
    
    Common wavelets for ECG:
    - 'db4' or 'db6' (Daubechies) - Best for QRS detection
    - 'sym5' (Symlet) - Good for morphology analysis
    - 'coif3' (Coiflet) - Better for P and T waves
    - 'bior3.5' (Biorthogonal) - Good for reconstruction
    """
    
    # Standard wavelet configurations for ECG
    WAVELET_CONFIGS = {
        'qrs_detection': {'wavelet': 'db4', 'level': 4, 'scale_range': (3, 12)},
        'morphology': {'wavelet': 'sym5', 'level': 5, 'scale_range': (4, 15)},
        'denoising': {'wavelet': 'db6', 'level': 6, 'scale_range': (2, 8)},
        'compression': {'wavelet': 'bior3.5', 'level': 3, 'scale_range': (3, 10)}
    }
    
    def __init__(self, wavelet: str = 'db6', level: int = 4, sampling_rate: int = 360):
        """
        Initialize wavelet feature extractor
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'db6', 'sym5', 'coif3', 'bior3.5')
            level: Decomposition level
            sampling_rate: Signal sampling rate in Hz
        """
        self.wavelet = wavelet
        self.level = level
        self.sampling_rate = sampling_rate
        
        # Verify wavelet exists
        available_wavelets = pywt.wavelist()
        if wavelet not in available_wavelets:
            raise ValueError(f"Wavelet {wavelet} not found. Available: {available_wavelets[:20]}...")
            
        # Get wavelet object for properties
        self.wavelet_obj = pywt.Wavelet(wavelet)
        
    def decompose_signal(self, signal: np.ndarray, 
                        mode: str = 'symmetric') -> List[np.ndarray]:
        """
        Perform wavelet decomposition of the ECG signal
        
        Args:
            signal: Input ECG signal (1D array)
            mode: Extension mode ('symmetric', 'periodic', 'constant')
            
        Returns:
            List of coefficients: [cD1, cD2, ..., cDn, cAn]
        """
        # Pad signal to appropriate length if needed
        original_len = len(signal)
        padded_signal = self._pad_signal(signal)
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(padded_signal, self.wavelet, level=self.level, mode=mode)
        
        # Trim coefficients to original signal length
        coeffs = self._trim_coeffs(coeffs, original_len)
        
        return coeffs
    
    def _pad_signal(self, signal: np.ndarray) -> np.ndarray:
        """Pad signal to length divisible by 2^level"""
        min_len = 2 ** self.level
        if len(signal) < min_len:
            pad_length = min_len - len(signal)
            signal = np.pad(signal, (0, pad_length), mode='constant')
        return signal
    
    def _trim_coeffs(self, coeffs: List[np.ndarray], original_len: int) -> List[np.ndarray]:
        """Trim wavelet coefficients to original signal length"""
        # Calculate approximate length of approximation coefficients
        approx_len = original_len // (2 ** (len(coeffs) - 1))
        
        # Ensure we don't trim too much
        if approx_len < 1:
            approx_len = 1
            
        # Trim approximation coefficients
        coeffs[-1] = coeffs[-1][:approx_len]
        
        # Detail coefficients are fine as they are (they're already shorter)
        return coeffs
    
    def reconstruct_signal(self, coeffs: List[np.ndarray], 
                          original_len: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients
        
        Args:
            coeffs: Wavelet coefficients from decompose_signal
            original_len: Original signal length (for trimming)
            
        Returns:
            Reconstructed signal
        """
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        
        if original_len is not None:
            reconstructed = reconstructed[:original_len]
            
        return reconstructed
    
    def extract_coefficient_statistics(self, coeffs: List[np.ndarray]) -> Dict:
        """
        Extract statistical features from wavelet coefficients at each level
        
        Args:
            coeffs: Wavelet coefficients from decompose_signal
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        for i, coeff in enumerate(coeffs):
            if len(coeff) == 0:
                continue
                
            # Level name
            if i == len(coeffs) - 1:
                level_name = f'level_{self.level}_approx'
            else:
                level_name = f'level_{i+1}_detail'
                
            # Basic statistics
            features[f'{level_name}_mean'] = float(np.mean(coeff))
            features[f'{level_name}_std'] = float(np.std(coeff))
            features[f'{level_name}_var'] = float(np.var(coeff))
            features[f'{level_name}_max'] = float(np.max(coeff))
            features[f'{level_name}_min'] = float(np.min(coeff))
            features[f'{level_name}_range'] = float(np.max(coeff) - np.min(coeff))
            features[f'{level_name}_median'] = float(np.median(coeff))
            features[f'{level_name}_mad'] = float(np.median(np.abs(coeff - np.median(coeff))))
            
            # Higher order moments
            if len(coeff) > 3:
                features[f'{level_name}_skewness'] = float(stats.skew(coeff))
                features[f'{level_name}_kurtosis'] = float(stats.kurtosis(coeff))
            else:
                features[f'{level_name}_skewness'] = 0.0
                features[f'{level_name}_kurtosis'] = 0.0
                
            # Energy (L2 norm)
            features[f'{level_name}_energy'] = float(np.sum(coeff ** 2))
            
            # L1 norm
            features[f'{level_name}_l1_norm'] = float(np.sum(np.abs(coeff)))
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(coeff)) != 0)
            features[f'{level_name}_zero_crossing_rate'] = float(zero_crossings / len(coeff))
            
            # Sparsity measure (ratio of L1 to L2)
            l2_norm = np.sqrt(np.sum(coeff ** 2))
            if l2_norm > 0:
                features[f'{level_name}_sparsity'] = float(np.sum(np.abs(coeff)) / l2_norm)
            else:
                features[f'{level_name}_sparsity'] = 0.0
                
        return features
    
    def extract_energy_features(self, coeffs: List[np.ndarray]) -> Dict:
        """
        Extract energy distribution features across scales
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Dictionary of energy-related features
        """
        features = {}
        
        # Calculate energy at each level
        energies = []
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff ** 2)
            energies.append(energy)
            
            if i == len(coeffs) - 1:
                features[f'approx_energy'] = float(energy)
            else:
                features[f'detail_{i+1}_energy'] = float(energy)
                
        # Total energy
        total_energy = sum(energies)
        features['total_wavelet_energy'] = float(total_energy)
        
        # Energy ratios
        if total_energy > 0:
            for i, energy in enumerate(energies):
                ratio = energy / total_energy
                if i == len(coeffs) - 1:
                    features[f'approx_energy_ratio'] = float(ratio)
                else:
                    features[f'detail_{i+1}_energy_ratio'] = float(ratio)
                    
            # Energy entropy (Shannon)
            energy_ratios = [e / total_energy for e in energies if e > 0]
            if energy_ratios:
                energy_entropy = -np.sum(energy_ratios * np.log(energy_ratios))
                features['wavelet_energy_entropy'] = float(energy_entropy)
            else:
                features['wavelet_energy_entropy'] = 0.0
                
        # Cumulative energy
        cum_energy = np.cumsum(energies)
        features['energy_concentration'] = float(cum_energy[-1] / total_energy) if total_energy > 0 else 0
        
        return features
    
    def extract_entropy_features(self, coeffs: List[np.ndarray]) -> Dict:
        """
        Extract various entropy measures from wavelet coefficients
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Dictionary of entropy features
        """
        features = {}
        
        for i, coeff in enumerate(coeffs):
            if len(coeff) < 5:
                continue
                
            if i == len(coeffs) - 1:
                level_name = 'approx'
            else:
                level_name = f'detail_{i+1}'
                
            # Normalized coefficients for probability distribution
            coeff_abs = np.abs(coeff)
            prob = coeff_abs / (np.sum(coeff_abs) + 1e-10)
            
            # Shannon entropy
            shannon_entropy = -np.sum(prob * np.log(prob + 1e-10))
            features[f'{level_name}_shannon_entropy'] = float(shannon_entropy)
            
            # Log energy entropy
            log_energy = np.sum(np.log(coeff ** 2 + 1e-10))
            features[f'{level_name}_log_energy'] = float(log_energy)
            
            # Renyi entropy (order 2)
            renyi_entropy = -np.log(np.sum(prob ** 2) + 1e-10)
            features[f'{level_name}_renyi_entropy'] = float(renyi_entropy)
            
            # Sure entropy
            threshold = np.std(coeff)
            sure_entropy = np.sum(np.minimum(coeff ** 2, threshold ** 2))
            features[f'{level_name}_sure_entropy'] = float(sure_entropy)
            
        return features
    
    def extract_qrs_wavelet_features(self, coeffs: List[np.ndarray]) -> Dict:
        """
        Extract QRS-specific features from wavelet coefficients
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Dictionary of QRS wavelet features
        """
        features = {}
        
        # QRS complex is typically best captured in detail coefficients
        # at levels 3-5 (depending on sampling rate)
        
        # Determine which detail levels to analyze based on sampling rate
        if self.sampling_rate <= 250:
            qrs_levels = [2, 3]  # Lower sampling rate -> lower levels
        elif self.sampling_rate <= 500:
            qrs_levels = [3, 4]  # Normal ECG sampling rates
        else:
            qrs_levels = [4, 5]  # High sampling rate
            
        for level in qrs_levels:
            if level <= len(coeffs) - 1:
                detail_coeff = coeffs[level - 1]
                
                # Find peaks in detail coefficients
                # Adjust distance based on expected heart rate
                min_distance = int(self.sampling_rate * 0.3)  # Minimum 300ms between R waves
                
                try:
                    peaks, properties = find_peaks(np.abs(detail_coeff), 
                                                  distance=min_distance,
                                                  height=np.std(detail_coeff))
                    
                    features[f'qrs_peaks_level_{level}'] = len(peaks)
                    
                    if len(peaks) > 0:
                        features[f'mean_peak_height_level_{level}'] = float(np.mean(properties['peak_heights']))
                        features[f'std_peak_height_level_{level}'] = float(np.std(properties['peak_heights']))
                        features[f'max_peak_height_level_{level}'] = float(np.max(properties['peak_heights']))
                        
                        # Peak spacing
                        if len(peaks) > 1:
                            peak_spacing = np.diff(peaks) / self.sampling_rate
                            features[f'mean_peak_spacing_level_{level}'] = float(np.mean(peak_spacing))
                            features[f'std_peak_spacing_level_{level}'] = float(np.std(peak_spacing))
                    else:
                        features[f'mean_peak_height_level_{level}'] = 0.0
                        features[f'std_peak_height_level_{level}'] = 0.0
                        features[f'max_peak_height_level_{level}'] = 0.0
                        features[f'mean_peak_spacing_level_{level}'] = 0.0
                        features[f'std_peak_spacing_level_{level}'] = 0.0
                        
                except Exception as e:
                    features[f'qrs_peaks_level_{level}'] = 0
                    features[f'mean_peak_height_level_{level}'] = 0.0
                    features[f'std_peak_height_level_{level}'] = 0.0
                    features[f'max_peak_height_level_{level}'] = 0.0
                    features[f'mean_peak_spacing_level_{level}'] = 0.0
                    features[f'std_peak_spacing_level_{level}'] = 0.0
                    
        return features
    
    def extract_multiscale_features(self, signal: np.ndarray) -> Dict:
        """
        Extract multiscale features using continuous wavelet transform (CWT)
        
        Args:
            signal: Input ECG signal
            
        Returns:
            Dictionary of multiscale features
        """
        features = {}
        
        # Define scales for CWT (1-30 Hz range approximately)
        scales = np.arange(1, min(30, len(signal) // 4))
        
        # Compute continuous wavelet transform
        try:
            coefficients, frequencies = pywt.cwt(signal, scales, self.wavelet, 
                                                 sampling_period=1/self.sampling_rate)
            
            features['cwt_scales'] = len(scales)
            features['cwt_coeff_mean'] = float(np.mean(np.abs(coefficients)))
            features['cwt_coeff_std'] = float(np.std(np.abs(coefficients)))
            features['cwt_coeff_max'] = float(np.max(np.abs(coefficients)))
            
            # Energy at different scales
            scale_energies = np.sum(coefficients ** 2, axis=1)
            features['cwt_scale_energy_mean'] = float(np.mean(scale_energies))
            features['cwt_scale_energy_std'] = float(np.std(scale_energies))
            
            # Scale of maximum energy
            max_energy_scale = np.argmax(scale_energies)
            features['cwt_max_energy_scale'] = float(max_energy_scale)
            
            # Wavelet coherence (scale correlation)
            if coefficients.shape[0] > 1:
                correlation = np.corrcoef(coefficients[0], coefficients[1])[0, 1]
                features['cwt_scale_correlation'] = float(correlation)
            else:
                features['cwt_scale_correlation'] = 0.0
                
        except Exception as e:
            features['cwt_coeff_mean'] = 0.0
            features['cwt_coeff_std'] = 0.0
            features['cwt_coeff_max'] = 0.0
            features['cwt_scale_energy_mean'] = 0.0
            features['cwt_scale_energy_std'] = 0.0
            features['cwt_max_energy_scale'] = 0.0
            features['cwt_scale_correlation'] = 0.0
            
        return features
    
    def extract_wavelet_packet_features(self, signal: np.ndarray, max_level: int = 3) -> Dict:
        """
        Extract features using wavelet packet decomposition (more detailed than DWT)
        
        Args:
            signal: Input ECG signal
            max_level: Maximum decomposition level
            
        Returns:
            Dictionary of wavelet packet features
        """
        features = {}
        
        try:
            # Perform wavelet packet decomposition
            wp = pywt.WaveletPacket(signal, self.wavelet, maxlevel=max_level)
            
            # Extract features from each node
            node_energies = {}
            
            for node in wp.get_level(max_level):
                coeffs = node.data
                if len(coeffs) > 0:
                    node_name = node.path
                    node_energy = np.sum(coeffs ** 2)
                    node_energies[node_name] = node_energy
                    
            # Total energy
            total_energy = sum(node_energies.values())
            features['wp_total_energy'] = float(total_energy)
            
            # Energy ratios and entropy
            if total_energy > 0:
                energy_ratios = [e / total_energy for e in node_energies.values()]
                wp_entropy = -np.sum(energy_ratios * np.log(energy_ratios + 1e-10))
                features['wp_entropy'] = float(wp_entropy)
                
                # Dominant node (node with max energy)
                dominant_node = max(node_energies, key=node_energies.get)
                features['wp_dominant_node'] = dominant_node
                features['wp_dominant_energy_ratio'] = float(node_energies[dominant_node] / total_energy)
                
            # Node statistics
            features['wp_num_nodes'] = len(node_energies)
            features['wp_mean_node_energy'] = float(np.mean(list(node_energies.values())))
            features['wp_std_node_energy'] = float(np.std(list(node_energies.values())))
            
        except Exception as e:
            features['wp_total_energy'] = 0.0
            features['wp_entropy'] = 0.0
            features['wp_dominant_node'] = 'none'
            features['wp_dominant_energy_ratio'] = 0.0
            features['wp_num_nodes'] = 0
            features['wp_mean_node_energy'] = 0.0
            features['wp_std_node_energy'] = 0.0
            
        return features
    
    def denoise_signal(self, signal: np.ndarray, 
                      method: str = 'soft',
                      threshold_scale: float = 1.0) -> np.ndarray:
        """
        Denoise ECG signal using wavelet thresholding
        
        Args:
            signal: Noisy ECG signal
            method: Thresholding method ('soft', 'hard', 'garrote')
            threshold_scale: Scale factor for threshold (higher = more aggressive)
            
        Returns:
            Denoised signal
        """
        # Decompose signal
        coeffs = self.decompose_signal(signal)
        
        # Estimate noise level from finest detail coefficients
        sigma = np.median(np.abs(coeffs[0])) / 0.6745
        
        # Universal threshold
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients (keep approximation)
        denoised_coeffs = [coeffs[0]]  # Keep finest detail? Actually we want to threshold it
        
        for i in range(len(coeffs) - 1):  # Don't threshold approximation
            if method == 'soft':
                thresholded = pywt.threshold(coeffs[i], threshold, mode='soft')
            elif method == 'hard':
                thresholded = pywt.threshold(coeffs[i], threshold, mode='hard')
            elif method == 'garrote':
                # Garrote thresholding
                mask = np.abs(coeffs[i]) > threshold
                thresholded = coeffs[i].copy()
                thresholded[mask] = coeffs[i][mask] - (threshold ** 2) / coeffs[i][mask]
            else:
                raise ValueError(f"Unknown method: {method}")
                
            denoised_coeffs.append(thresholded)
            
        # Add approximation coefficient
        denoised_coeffs.append(coeffs[-1])
        
        # Reconstruct signal
        denoised = self.reconstruct_signal(denoised_coeffs, original_len=len(signal))
        
        return denoised
    
    def extract_scalogram(self, signal: np.ndarray, 
                          scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scalogram (time-frequency representation) using CWT
        
        Args:
            signal: Input ECG signal
            scales: Scale array (if None, automatically determined)
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            # Determine scales based on signal length and sampling rate
            max_scale = min(64, len(signal) // 4)
            scales = np.arange(1, max_scale)
            
        coefficients, frequencies = pywt.cwt(signal, scales, self.wavelet, 
                                            sampling_period=1/self.sampling_rate)
        
        return coefficients, frequencies
    
    def extract_scalogram_features(self, signal: np.ndarray) -> Dict:
        """
        Extract features from scalogram (time-frequency representation)
        
        Args:
            signal: Input ECG signal
            
        Returns:
            Dictionary of scalogram-based features
        """
        features = {}
        
        try:
            coefficients, frequencies = self.extract_scalogram(signal)
            
            # Global statistics
            features['scalogram_mean'] = float(np.mean(np.abs(coefficients)))
            features['scalogram_std'] = float(np.std(np.abs(coefficients)))
            features['scalogram_max'] = float(np.max(np.abs(coefficients)))
            features['scalogram_min'] = float(np.min(np.abs(coefficients)))
            
            # Energy in different frequency bands
            # Define frequency bands based on ECG components
            # These bands depend on sampling rate and scales
            if len(frequencies) > 0:
                # Low frequencies (P wave, baseline)
                low_idx = frequencies < 5
                # Mid frequencies (QRS complex)
                mid_idx = (frequencies >= 5) & (frequencies < 15)
                # High frequencies (noise, muscle artifact)
                high_idx = frequencies >= 15
                
                features['scalogram_low_energy'] = float(np.sum(coefficients[low_idx] ** 2)) if np.any(low_idx) else 0
                features['scalogram_mid_energy'] = float(np.sum(coefficients[mid_idx] ** 2)) if np.any(mid_idx) else 0
                features['scalogram_high_energy'] = float(np.sum(coefficients[high_idx] ** 2)) if np.any(high_idx) else 0
                
                total_energy = features['scalogram_low_energy'] + features['scalogram_mid_energy'] + features['scalogram_high_energy']
                if total_energy > 0:
                    features['scalogram_low_ratio'] = features['scalogram_low_energy'] / total_energy
                    features['scalogram_mid_ratio'] = features['scalogram_mid_energy'] / total_energy
                    features['scalogram_high_ratio'] = features['scalogram_high_energy'] / total_energy
                else:
                    features['scalogram_low_ratio'] = 0
                    features['scalogram_mid_ratio'] = 0
                    features['scalogram_high_ratio'] = 0
                    
        except Exception as e:
            features['scalogram_mean'] = 0.0
            features['scalogram_std'] = 0.0
            features['scalogram_max'] = 0.0
            features['scalogram_min'] = 0.0
            features['scalogram_low_energy'] = 0.0
            features['scalogram_mid_energy'] = 0.0
            features['scalogram_high_energy'] = 0.0
            features['scalogram_low_ratio'] = 0.0
            features['scalogram_mid_ratio'] = 0.0
            features['scalogram_high_ratio'] = 0.0
            
        return features
    
    def detect_r_peaks_wavelet(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using wavelet transform (more robust than time-domain)
        
        Args:
            signal: Preprocessed ECG signal
            
        Returns:
            Array of R-peak indices
        """
        # Decompose signal
        coeffs = self.decompose_signal(signal)
        
        # Use detail coefficients at appropriate level for QRS detection
        if len(coeffs) >= 4:
            qrs_coeff = coeffs[3]  # Level 4 detail coefficients (adjust as needed)
        elif len(coeffs) >= 3:
            qrs_coeff = coeffs[2]  # Level 3 detail
        else:
            qrs_coeff = coeffs[0]  # Fallback
            
        # Square the coefficients to enhance QRS complexes
        squared = qrs_coeff ** 2
        
        # Apply moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms window
        kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, kernel, mode='same')
        
        # Find peaks in integrated signal
        min_distance = int(0.3 * self.sampling_rate)  # Minimum 300ms between beats
        peaks, properties = find_peaks(integrated, 
                                      distance=min_distance,
                                      height=np.mean(integrated) + np.std(integrated))
        
        # Refine peaks to actual R-peak locations
        refined_peaks = []
        for peak in peaks:
            # Search ±50ms around detected peak
            search_start = max(0, peak - int(0.05 * self.sampling_rate))
            search_end = min(len(signal), peak + int(0.05 * self.sampling_rate))
            
            # Find maximum in original signal
            local_max = search_start + np.argmax(np.abs(signal[search_start:search_end]))
            refined_peaks.append(local_max)
            
        return np.array(refined_peaks)
    
    def extract_all_features(self, signal: np.ndarray) -> Dict:
        """
        Extract all wavelet features from ECG signal
        
        Args:
            signal: Input ECG signal
            
        Returns:
            Dictionary containing all wavelet features
        """
        # Decompose signal
        coeffs = self.decompose_signal(signal)
        
        # Extract various feature groups
        features = {}
        
        # Coefficient statistics
        coeff_stats = self.extract_coefficient_statistics(coeffs)
        features.update(coeff_stats)
        
        # Energy features
        energy_features = self.extract_energy_features(coeffs)
        features.update(energy_features)
        
        # Entropy features
        entropy_features = self.extract_entropy_features(coeffs)
        features.update(entropy_features)
        
        # QRS features
        qrs_features = self.extract_qrs_wavelet_features(coeffs)
        features.update(qrs_features)
        
        # Multiscale features (CWT)
        multiscale_features = self.extract_multiscale_features(signal)
        features.update(multiscale_features)
        
        # Scalogram features
        scalogram_features = self.extract_scalogram_features(signal)
        features.update(scalogram_features)
        
        # Wavelet packet features (for detailed analysis)
        if len(signal) >= 64:  # Only if signal is long enough
            packet_features = self.extract_wavelet_packet_features(signal)
            features.update(packet_features)
            
        return features
    
    def get_optimal_wavelet(self, signal: np.ndarray, 
                           wavelets_to_test: List[str] = None) -> Dict:
        """
        Find the optimal wavelet for ECG analysis based on energy-to-shannon entropy ratio
        
        Args:
            signal: ECG signal
            wavelets_to_test: List of wavelets to test (default: common ECG wavelets)
            
        Returns:
            Dictionary with optimal wavelet and metrics
        """
        if wavelets_to_test is None:
            wavelets_to_test = ['db2', 'db4', 'db6', 'db8', 'sym4', 'sym6', 'coif3', 'bior3.5']
            
        results = {}
        
        for wavelet_name in wavelets_to_test:
            try:
                # Create temporary extractor
                temp_extractor = WaveletFeatures(wavelet=wavelet_name, level=4)
                
                # Decompose
                coeffs = temp_extractor.decompose_signal(signal)
                
                # Calculate energy and entropy
                total_energy = sum(np.sum(c ** 2) for c in coeffs)
                
                # Calculate Shannon entropy of coefficients
                all_coeffs = np.concatenate([c.flatten() for c in coeffs])
                coeff_abs = np.abs(all_coeffs)
                prob = coeff_abs / (np.sum(coeff_abs) + 1e-10)
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                
                # Energy-to-entropy ratio (higher is better for compression/feature extraction)
                energy_entropy_ratio = total_energy / (entropy + 1e-10)
                
                results[wavelet_name] = {
                    'energy': total_energy,
                    'entropy': entropy,
                    'energy_entropy_ratio': energy_entropy_ratio
                }
                
            except Exception as e:
                results[wavelet_name] = {'error': str(e)}
                
        # Find optimal (max energy-entropy ratio)
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            optimal_wavelet = max(valid_results, key=lambda x: valid_results[x]['energy_entropy_ratio'])
            results['optimal'] = optimal_wavelet
            
        return results
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names (for documentation)
        
        Returns:
            List of feature name patterns
        """
        feature_patterns = [
            'level_X_detail_mean', 'level_X_detail_std', 'level_X_detail_energy',
            'level_X_detail_entropy', 'detail_X_energy_ratio',
            'approx_energy', 'total_wavelet_energy', 'wavelet_energy_entropy',
            'qrs_peaks_level_X', 'mean_peak_height_level_X',
            'cwt_coeff_mean', 'cwt_coeff_std', 'scalogram_mean', 'scalogram_std',
            'wp_total_energy', 'wp_entropy', 'wp_dominant_node'
        ]
        return feature_patterns


def create_wavelet_feature_pipeline(signal: np.ndarray, 
                                   wavelet_type: str = 'db6',
                                   level: int = 4) -> np.ndarray:
    """
    Convenience function to create a fixed-length feature vector from wavelet decomposition
    
    Args:
        signal: ECG signal
        wavelet_type: Type of wavelet to use
        level: Decomposition level
        
    Returns:
        Feature vector as numpy array
    """
    extractor = WaveletFeatures(wavelet=wavelet_type, level=level)
    features = extractor.extract_all_features(signal)
    
    # Convert to vector (sort keys for consistent order)
    sorted_keys = sorted(features.keys())
    feature_vector = np.array([features[k] for k in sorted_keys if isinstance(features[k], (int, float))])
    
    return feature_vector


if __name__ == "__main__":
    # Test wavelet features
    import matplotlib.pyplot as plt
    
    # Generate synthetic ECG-like signal
    t = np.linspace(0, 2, int(2 * 360))  # 2 seconds at 360 Hz
    synthetic = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
    
    # Add QRS-like spikes
    for i in range(0, len(t), int(0.8 * 360)):
        if i + 20 < len(t):
            synthetic[i:i+20] += 0.5 * np.hanning(20)
            
    # Extract wavelet features
    extractor = WaveletFeatures(wavelet='db6', level=4, sampling_rate=360)
    
    # Decompose
    coeffs = extractor.decompose_signal(synthetic)
    
    # Extract all features
    features = extractor.extract_all_features(synthetic)
    
    print("Wavelet Features Extracted:")
    print(f"  Total features: {len(features)}")
    print(f"  Feature names: {list(features.keys())[:10]}...")
    
    # Find optimal wavelet
    optimal = extractor.get_optimal_wavelet(synthetic)
    print(f"\nOptimal wavelet: {optimal.get('optimal', 'not found')}")
    
    # Denoise example
    noisy = synthetic + 0.2 * np.random.randn(len(synthetic))
    denoised = extractor.denoise_signal(noisy, method='soft', threshold_scale=1.2)
    
    print(f"\nDenoising improvement:")
    print(f"  Original SNR: {10*np.log10(np.var(synthetic)/np.var(noisy-synthetic)):.2f} dB")
    print(f"  Denoised SNR: {10*np.log10(np.var(synthetic)/np.var(denoised-synthetic)):.2f} dB")
    
    # Detect R-peaks
    r_peaks = extractor.detect_r_peaks_wavelet(synthetic)
    print(f"\nDetected {len(r_peaks)} R-peaks")
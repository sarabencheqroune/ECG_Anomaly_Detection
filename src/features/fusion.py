"""
Feature fusion and selection for ECG classification
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


class FeatureFusion:
    """
    Combine and optimize features from multiple sources
    
    Supports:
    - Feature concatenation
    - Feature selection (univariate, mutual information)
    - Dimensionality reduction (PCA)
    - Feature scaling and normalization
    - Correlation-based feature removal
    """
    
    def __init__(self, 
                 use_rr_features: bool = True,
                 use_wavelet_features: bool = True,
                 use_morphological_features: bool = True,
                 scaler_type: str = 'standard'):
        """
        Initialize feature fusion module
        
        Args:
            use_rr_features: Include RR interval features
            use_wavelet_features: Include wavelet features
            use_morphological_features: Include morphological features
            scaler_type: 'standard', 'minmax', or None
        """
        self.use_rr_features = use_rr_features
        self.use_wavelet_features = use_wavelet_features
        self.use_morphological_features = use_morphological_features
        self.scaler_type = scaler_type
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        self.pca = None
        self.feature_selector = None
        self.selected_feature_indices = None
        self.feature_names = []
        
    def extract_all_features(self, 
                            r_peaks: np.ndarray,
                            beat: np.ndarray,
                            r_peak_idx: int,
                            raw_signal: Optional[np.ndarray] = None,
                            heart_rate: Optional[float] = None) -> Dict:
        """
        Extract and combine all features from different modules
        
        Args:
            r_peaks: Array of R-peak indices (for RR features)
            beat: Single ECG beat segment
            r_peak_idx: Index of R-peak within beat
            raw_signal: Raw ECG signal (for wavelet features)
            heart_rate: Heart rate in BPM
            
        Returns:
            Dictionary of all extracted features
        """
        from .rr_intervals import RRIntervalFeatures
        from .wavelet import WaveletFeatures
        from .morphological import MorphologicalFeatures
        
        all_features = {}
        
        # Extract RR interval features
        if self.use_rr_features:
            rr_extractor = RRIntervalFeatures()
            rr_features = rr_extractor.extract_all_features(r_peaks)
            all_features.update(rr_features)
            
        # Extract wavelet features
        if self.use_wavelet_features:
            wavelet_extractor = WaveletFeatures()
            
            # Use raw signal if provided, otherwise use beat
            signal_to_use = raw_signal if raw_signal is not None else beat
            
            # Ensure minimum length for wavelet decomposition
            min_len = 2 ** 4  # 16 samples minimum for level 4
            if len(signal_to_use) >= min_len:
                wavelet_features = wavelet_extractor.extract_all_features(signal_to_use)
                # Add prefix to avoid name conflicts
                wavelet_features = {f'wavelet_{k}': v for k, v in wavelet_features.items()}
                all_features.update(wavelet_features)
                
        # Extract morphological features
        if self.use_morphological_features:
            morph_extractor = MorphologicalFeatures()
            morph_features = morph_extractor.extract_all_features(beat, r_peak_idx, heart_rate)
            all_features.update(morph_features)
            
        return all_features
    
    def features_to_vector(self, features_dict: Dict) -> np.ndarray:
        """
        Convert feature dictionary to numpy array
        
        Args:
            features_dict: Dictionary of feature name-value pairs
            
        Returns:
            Feature vector as numpy array
        """
        # Sort features for consistent ordering
        sorted_keys = sorted(features_dict.keys())
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = sorted_keys
            
        # Extract values in consistent order
        feature_vector = np.array([features_dict[key] for key in sorted_keys])
        
        return feature_vector
    
    def extract_feature_batch(self, 
                             r_peaks_list: List[np.ndarray],
                             beats: np.ndarray,
                             r_peak_indices: List[int],
                             heart_rates: Optional[List[float]] = None) -> np.ndarray:
        """
        Extract features for a batch of beats
        
        Args:
            r_peaks_list: List of R-peak arrays for each beat
            beats: Batch of beat signals (n_beats, signal_length)
            r_peak_indices: List of R-peak indices for each beat
            heart_rates: Optional list of heart rates
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_vectors = []
        
        for i, beat in enumerate(beats):
            hr = heart_rates[i] if heart_rates else None
            
            features = self.extract_all_features(
                r_peaks=r_peaks_list[i],
                beat=beat,
                r_peak_idx=r_peak_indices[i],
                heart_rate=hr
            )
            
            feature_vector = self.features_to_vector(features)
            feature_vectors.append(feature_vector)
            
        return np.array(feature_vectors)
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features to zero mean and unit variance
        
        Args:
            X: Feature matrix
            fit: Fit scaler (True for training, False for inference)
            
        Returns:
            Scaled feature matrix
        """
        if self.scaler is None:
            return X
            
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
    
    def select_features_anova(self, X: np.ndarray, y: np.ndarray, 
                             k: int = 50) -> np.ndarray:
        """
        Select top K features using ANOVA F-test
        
        Args:
            X: Feature matrix
            y: Labels
            k: Number of features to select
            
        Returns:
            Selected feature matrix
        """
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Store selected feature indices
        self.selected_feature_indices = selector.get_support(indices=True)
        self.feature_selector = selector
        
        return X_selected
    
    def select_features_mutual_info(self, X: np.ndarray, y: np.ndarray,
                                   k: int = 50) -> np.ndarray:
        """
        Select top K features using mutual information
        
        Args:
            X: Feature matrix
            y: Labels
            k: Number of features to select
            
        Returns:
            Selected feature matrix
        """
        scores = mutual_info_classif(X, y)
        top_k_indices = np.argsort(scores)[-k:]
        
        self.selected_feature_indices = top_k_indices
        X_selected = X[:, top_k_indices]
        
        return X_selected
    
    def reduce_dimensions_pca(self, X: np.ndarray, 
                             n_components: int = 30,
                             fit: bool = True) -> np.ndarray:
        """
        Reduce dimensionality using PCA
        
        Args:
            X: Feature matrix
            n_components: Number of principal components
            fit: Fit PCA (True for training, False for inference)
            
        Returns:
            Reduced feature matrix
        """
        if fit:
            self.pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
            X_reduced = self.pca.fit_transform(X)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call fit=True first.")
            X_reduced = self.pca.transform(X)
            
        return X_reduced
    
    def remove_correlated_features(self, X: np.ndarray, 
                                   threshold: float = 0.95) -> Tuple[np.ndarray, List[int]]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold (features above this are removed)
            
        Returns:
            Tuple of (reduced feature matrix, indices to keep)
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find correlated features
        to_keep = list(range(X.shape[1]))
        to_remove = set()
        
        for i in range(len(to_keep)):
            if to_keep[i] in to_remove:
                continue
                
            for j in range(i + 1, len(to_keep)):
                if to_keep[j] in to_remove:
                    continue
                    
                if abs(corr_matrix[to_keep[i], to_keep[j]]) > threshold:
                    to_remove.add(to_keep[j])
                    
        # Keep uncorrelated features
        keep_indices = [i for i in to_keep if i not in to_remove]
        X_reduced = X[:, keep_indices]
        
        return X_reduced, keep_indices
    
    def fuse_features(self, 
                     X_rr: Optional[np.ndarray],
                     X_wavelet: Optional[np.ndarray],
                     X_morph: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse features from different modalities by concatenation
        
        Args:
            X_rr: RR interval features
            X_wavelet: Wavelet features
            X_morph: Morphological features
            
        Returns:
            Fused feature matrix
        """
        feature_list = []
        
        if self.use_rr_features and X_rr is not None:
            feature_list.append(X_rr)
            
        if self.use_wavelet_features and X_wavelet is not None:
            feature_list.append(X_wavelet)
            
        if self.use_morphological_features and X_morph is not None:
            feature_list.append(X_morph)
            
        if not feature_list:
            raise ValueError("No features to fuse")
            
        X_fused = np.hstack(feature_list)
        
        return X_fused
    
    def get_feature_importance(self, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using mutual information
        
        Args:
            y: Labels (only needed if not already computed)
            
        Returns:
            Dictionary of feature name to importance score
        """
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted. Run select_features first.")
            
        if hasattr(self.feature_selector, 'scores_'):
            scores = self.feature_selector.scores_
            importance = {}
            for idx, name in enumerate(self.feature_names[:len(scores)]):
                importance[name] = scores[idx]
                
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
        else:
            return {}
    
    def create_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                       use_feature_selection: bool = True,
                       use_pca: bool = False,
                       n_features: int = 50,
                       n_pca_components: int = 30) -> 'FeatureFusion':
        """
        Create complete feature processing pipeline
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            use_feature_selection: Apply feature selection
            use_pca: Apply PCA dimensionality reduction
            n_features: Number of features to select
            n_pca_components: Number of PCA components
            
        Returns:
            Fitted feature fusion object
        """
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        
        # Remove correlated features
        X_train_uncorr, keep_indices = self.remove_correlated_features(X_train_scaled)
        
        # Update feature names
        if self.feature_names:
            self.feature_names = [self.feature_names[i] for i in keep_indices]
            
        # Feature selection
        if use_feature_selection:
            X_train_selected = self.select_features_anova(X_train_uncorr, y_train, k=n_features)
        else:
            X_train_selected = X_train_uncorr
            
        # PCA
        if use_pca:
            X_train_final = self.reduce_dimensions_pca(X_train_selected, 
                                                       n_components=n_pca_components,
                                                       fit=True)
        else:
            X_train_final = X_train_selected
            
        # Store training data shape for validation
        self.train_shape = X_train_final.shape
        
        return X_train_final
    
    def transform_pipeline(self, X_test: np.ndarray) -> np.ndarray:
        """
        Apply trained pipeline to test data
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Transformed test features
        """
        # Scale
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        # Remove correlated features (using stored indices)
        if hasattr(self, 'selected_feature_indices') and self.selected_feature_indices is not None:
            X_test_corr = X_test_scaled[:, self.selected_feature_indices]
        else:
            X_test_corr = X_test_scaled
            
        # Feature selection
        if self.feature_selector is not None:
            X_test_selected = self.feature_selector.transform(X_test_corr)
        else:
            X_test_selected = X_test_corr
            
        # PCA
        if self.pca is not None:
            X_test_final = self.pca.transform(X_test_selected)
        else:
            X_test_final = X_test_selected
            
        return X_test_final
    
    def get_feature_statistics(self, X: np.ndarray) -> Dict:
        """
        Get statistics about feature set
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'feature_dtype': str(X.dtype),
            'has_nan': np.any(np.isnan(X)),
            'has_inf': np.any(np.isinf(X)),
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X))
        }
        
        # Feature-wise statistics
        feature_stats = {
            'feature_means': np.mean(X, axis=0),
            'feature_stds': np.std(X, axis=0),
            'feature_mins': np.min(X, axis=0),
            'feature_maxs': np.max(X, axis=0)
        }
        
        stats.update(feature_stats)
        
        return stats


if __name__ == "__main__":
    # Test feature fusion
    fusion = FeatureFusion(use_rr_features=True, 
                          use_wavelet_features=True,
                          use_morphological_features=True)
    
    # Generate dummy features
    np.random.seed(42)
    n_samples = 100
    
    X_rr = np.random.randn(n_samples, 20)
    X_wavelet = np.random.randn(n_samples, 30)
    X_morph = np.random.randn(n_samples, 15)
    y = np.random.randint(0, 5, n_samples)
    
    # Fuse features
    X_fused = fusion.fuse_features(X_rr, X_wavelet, X_morph)
    print(f"Fused feature shape: {X_fused.shape}")
    
    # Process pipeline
    X_processed = fusion.create_pipeline(X_fused, y, 
                                        use_feature_selection=True,
                                        use_pca=False,
                                        n_features=30)
    print(f"Processed feature shape: {X_processed.shape}")
    
    # Get feature importance
    importance = fusion.get_feature_importance(y)
    if importance:
        print("\nTop 5 features:")
        for i, (name, score) in enumerate(list(importance.items())[:5]):
            print(f"  {i+1}. {name}: {score:.4f}")
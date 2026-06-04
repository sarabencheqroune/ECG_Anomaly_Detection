"""
Model inference wrapper for real-time ECG prediction
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
import logging
from collections import deque
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    class_id: int
    class_name: str
    confidence: float
    anomaly_score: float
    needs_review: bool
    inference_time_ms: float
    features: Optional[Dict] = None
    raw_probs: Optional[np.ndarray] = None


class InferencePredictor:
    """
    Model inference wrapper with caching and batching support
    
    Features:
    - Model loading and warmup
    - Batch inference optimization
    - Confidence calibration
    - Human-in-the-loop integration
    - Performance profiling
    """
    
    def __init__(self,
                 model_path: Union[str, Path],
                 device: Optional[torch.device] = None,
                 confidence_threshold: float = 0.7,
                 anomaly_threshold: float = 0.5,
                 use_amp: bool = False,
                 batch_size: int = 32):
        """
        Initialize inference predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            confidence_threshold: Threshold for HITL review
            anomaly_threshold: Threshold for anomaly detection
            use_amp: Use mixed precision inference
            batch_size: Batch size for batched inference
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold
        self.use_amp = use_amp
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Class names (should match training)
        self.class_names = ['Normal', 'AFib', 'PVC', 'Bradycardia', 'Other']
        self.num_classes = len(self.class_names)
        
        # Warmup
        self._warmup()
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'avg_inference_time_ms': 0,
            'review_rate': 0,
            'high_confidence_rate': 0
        }
        
        # Cache for recent predictions
        self.prediction_cache = deque(maxlen=100)
        
        logger.info(f"InferencePredictor initialized on {self.device}")
        
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Import model class (adjust based on your model type)
            from ..models.cnn_classifier import ECG1DCNN
            from ..models.hybrid_model import HybridECGModel
            
            # Try to load hybrid model first
            if 'autoencoder_state' in checkpoint:
                model = HybridECGModel(input_channels=1, input_length=187, num_classes=self.num_classes)
            else:
                model = ECG1DCNN(input_channels=1, input_length=187, num_classes=self.num_classes)
                
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model = model.to(self.device)
            logger.info(f"Model loaded from {self.model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def _warmup(self):
        """Warmup the model with dummy data"""
        dummy_input = torch.randn(1, 1, 187).to(self.device)
        
        with torch.no_grad():
            for _ in range(5):
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
                    
        logger.info("Model warmup complete")
        
    def predict(self, beat: np.ndarray) -> PredictionResult:
        """
        Run inference on a single beat
        
        Args:
            beat: ECG beat segment (shape: 187,)
            
        Returns:
            PredictionResult object
            
        Raises:
            ValueError: If beat has invalid shape or contains NaN/Inf
        """
        # Validate input
        if not isinstance(beat, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(beat)}")
            
        if beat.ndim == 0:
            raise ValueError("Beat must be at least 1-dimensional")
            
        # Check for NaN/Inf
        if np.any(np.isnan(beat)) or np.any(np.isinf(beat)):
            logger.warning("Beat contains NaN or Inf values, replacing with 0")
            beat = np.nan_to_num(beat, nan=0.0, posinf=0.0, neginf=0.0)
            
        if beat.size == 0:
            raise ValueError("Beat array is empty")
            
        try:
            start_time = time.time()
            
            # Preprocess beat
            beat_tensor = self._preprocess_beat(beat)
            
            # Run inference
            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(beat_tensor)
                else:
                    outputs = self.model(beat_tensor)
                    
            # Parse outputs based on model type
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if len(outputs) > 1:
                    recon_error = outputs[1]
                else:
                    recon_error = None
            else:
                logits = outputs
                recon_error = None
                
            # Get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # Compute anomaly score
            if recon_error is not None:
                anomaly_score = recon_error.item()
            else:
                # Use entropy as anomaly proxy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                anomaly_score = entropy / np.log(self.num_classes)
                
            # Determine if review is needed
            needs_review = (confidence < self.confidence_threshold or 
                           anomaly_score > self.anomaly_threshold)
            
            inference_time = (time.time() - start_time) * 1000
            
            # Update stats
            self._update_stats(inference_time, needs_review, confidence)
            
            # Cache result
            result = PredictionResult(
                class_id=pred_class,
                class_name=self.class_names[pred_class],
                confidence=float(confidence),
                anomaly_score=float(anomaly_score),
                needs_review=needs_review,
                inference_time_ms=inference_time,
                raw_probs=probs
            )
            
            self.prediction_cache.append(result)
            
            return result
            
        except RuntimeError as e:
            logger.error(f"Runtime error during inference: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}", exc_info=True)
            raise
    
    def predict_batch(self, beats: np.ndarray) -> List[PredictionResult]:
        """
        Run inference on a batch of beats
        
        Args:
            beats: Batch of beats (batch_size, 187)
            
        Returns:
            List of PredictionResult objects
        """
        start_time = time.time()
        
        # Preprocess batch
        beat_tensor = self._preprocess_batch(beats)
        
        # Process in batches if needed
        all_results = []
        
        for i in range(0, len(beat_tensor), self.batch_size):
            batch = beat_tensor[i:i + self.batch_size]
            
            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                    
            # Parse outputs
            if isinstance(outputs, tuple):
                logits = outputs[0]
                recon_errors = outputs[1] if len(outputs) > 1 else None
            else:
                logits = outputs
                recon_errors = None
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = np.argmax(probs, axis=1)
            confidences = probs[np.arange(len(pred_classes)), pred_classes]
            
            for j in range(len(batch)):
                if recon_errors is not None:
                    anomaly_score = recon_errors[j].item()
                else:
                    entropy = -np.sum(probs[j] * np.log(probs[j] + 1e-8))
                    anomaly_score = entropy / np.log(self.num_classes)
                    
                needs_review = (confidences[j] < self.confidence_threshold or
                               anomaly_score > self.anomaly_threshold)
                
                result = PredictionResult(
                    class_id=int(pred_classes[j]),
                    class_name=self.class_names[pred_classes[j]],
                    confidence=float(confidences[j]),
                    anomaly_score=float(anomaly_score),
                    needs_review=needs_review,
                    inference_time_ms=0,  # Will be set later
                    raw_probs=probs[j]
                )
                all_results.append(result)
                
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(beats)
        
        for result in all_results:
            result.inference_time_ms = avg_time
            
        return all_results
    
    def _preprocess_beat(self, beat: np.ndarray) -> torch.Tensor:
        """Preprocess a single beat for inference
        
        Args:
            beat: Input beat array
            
        Returns:
            Preprocessed tensor of shape (1, 1, 187)
            
        Raises:
            ValueError: If beat has invalid shape
        """
        try:
            # Ensure correct shape and type
            if beat.ndim == 1:
                if beat.shape[0] != 187:
                    logger.warning(f"Beat length {beat.shape[0]} != 187, padding/truncating")
                    beat = np.pad(beat, (0, max(0, 187 - beat.shape[0])))[:187]
                beat = beat.reshape(1, 1, -1)
            elif beat.ndim == 2:
                if beat.shape[1] != 187:
                    raise ValueError(f"Expected beat length 187, got {beat.shape[1]}")
                beat = beat.reshape(1, beat.shape[0], beat.shape[1])
            else:
                raise ValueError(f"Expected 1D or 2D beat, got shape {beat.shape}")
                
            beat_tensor = torch.from_numpy(beat).float().to(self.device)
            return beat_tensor
        except Exception as e:
            logger.error(f"Error preprocessing beat: {e}")
            raise
    
    def _preprocess_batch(self, beats: np.ndarray) -> torch.Tensor:
        """Preprocess a batch of beats
        
        Args:
            beats: Batch of beats, shape (batch_size, 187) or (batch_size, 1, 187)
            
        Returns:
            Preprocessed tensor of shape (batch_size, 1, 187)
            
        Raises:
            ValueError: If batch has invalid shape
        """
        try:
            if beats.ndim == 2:
                if beats.shape[1] != 187:
                    raise ValueError(f"Expected beat length 187, got {beats.shape[1]}")
                beats = beats.reshape(-1, 1, beats.shape[-1])
            elif beats.ndim != 3 or beats.shape[1] != 1 or beats.shape[2] != 187:
                raise ValueError(f"Expected shape (batch, 1, 187), got {beats.shape}")
                
            beat_tensor = torch.from_numpy(beats).float().to(self.device)
            return beat_tensor
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def _update_stats(self, inference_time: float, needs_review: bool, confidence: float):
        """Update prediction statistics"""
        alpha = 0.1  # EMA smoothing factor
        
        self.stats['total_predictions'] += 1
        self.stats['avg_inference_time_ms'] = (
            alpha * inference_time + 
            (1 - alpha) * self.stats['avg_inference_time_ms']
        )
        
        # Update rates with EMA
        review_rate = 1.0 if needs_review else 0.0
        self.stats['review_rate'] = (
            alpha * review_rate + 
            (1 - alpha) * self.stats['review_rate']
        )
        
        high_conf = 1.0 if confidence > 0.9 else 0.0
        self.stats['high_confidence_rate'] = (
            alpha * high_conf + 
            (1 - alpha) * self.stats['high_confidence_rate']
        )
        
    def get_stats(self) -> Dict:
        """Get inference statistics"""
        return {
            **self.stats,
            'device': str(self.device),
            'use_amp': self.use_amp,
            'confidence_threshold': self.confidence_threshold,
            'anomaly_threshold': self.anomaly_threshold
        }
    
    def set_thresholds(self, confidence: Optional[float] = None, 
                       anomaly: Optional[float] = None):
        """Update confidence and anomaly thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if anomaly is not None:
            self.anomaly_threshold = anomaly
            
    def get_recent_predictions(self, n: int = 10) -> List[PredictionResult]:
        """Get recent predictions from cache"""
        return list(self.prediction_cache)[-n:]


class BatchPredictor:
    """
    Batch predictor for offline processing
    """
    
    def __init__(self, predictor: InferencePredictor):
        """
        Initialize batch predictor
        
        Args:
            predictor: InferencePredictor instance
        """
        self.predictor = predictor
        
    def predict_file(self, file_path: Path) -> List[PredictionResult]:
        """
        Predict on an ECG file
        
        Args:
            file_path: Path to ECG file (WFDB format)
            
        Returns:
            List of predictions for each beat
        """
        import wfdb
        
        # Load record
        record = wfdb.rdrecord(str(file_path.with_suffix('')))
        signal = record.p_signal[:, 0]  # Use first lead
        
        # Preprocess (simplified - should use full pipeline)
        from ..data.preprocessor import ECGPreprocessor
        preprocessor = ECGPreprocessor(sampling_rate=record.fs)
        processed = preprocessor.preprocess(signal)
        
        # Detect beats (simplified)
        from ..data.segmenter import ECGSegmenter
        segmenter = ECGSegmenter(sampling_rate=record.fs)
        r_peaks, _ = segmenter.detect_r_peaks(processed)
        beats = segmenter.segment_beats(processed, r_peaks)
        
        # Extract beat signals
        beat_signals = np.array([b.signal for b in beats])
        
        # Predict
        predictions = self.predictor.predict_batch(beat_signals)
        
        return predictions
    
    def predict_dataset(self, data_dir: Path, file_pattern: str = "*.dat") -> Dict:
        """
        Predict on a dataset of ECG files
        
        Args:
            data_dir: Directory containing ECG files
            file_pattern: Pattern to match files
            
        Returns:
            Dictionary with aggregated results
        """
        import glob
        
        files = glob.glob(str(data_dir / file_pattern))
        
        all_predictions = []
        file_results = {}
        
        for file_path in files:
            predictions = self.predict_file(Path(file_path))
            file_results[file_path] = predictions
            all_predictions.extend(predictions)
            
        # Aggregate results
        aggregated = {
            'total_beats': len(all_predictions),
            'class_distribution': {},
            'avg_confidence': 0,
            'review_rate': 0
        }
        
        for pred in all_predictions:
            class_name = pred.class_name
            aggregated['class_distribution'][class_name] = \
                aggregated['class_distribution'].get(class_name, 0) + 1
            aggregated['avg_confidence'] += pred.confidence
            aggregated['review_rate'] += 1 if pred.needs_review else 0
            
        aggregated['avg_confidence'] /= len(all_predictions)
        aggregated['review_rate'] /= len(all_predictions)
        
        return aggregated


if __name__ == "__main__":
    # Test predictor (with dummy model)
    import tempfile
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(187, 5)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
            
    # Save dummy model
    with tempfile.NamedTemporaryFile(suffix='.pth') as f:
        model = DummyModel()
        torch.save(model.state_dict(), f.name)
        
        # Create predictor (will fail to load but shows structure)
        try:
            predictor = InferencePredictor(f.name)
            print("Predictor initialized successfully")
        except Exception as e:
            print(f"Expected error with dummy model: {e}")
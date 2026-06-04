"""
Real-time ECG stream processing with sliding window and buffering
"""

import numpy as np
import threading
import queue
import time
from collections import deque
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream processor states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class StreamingConfig:
    """Configuration for stream processing"""
    sampling_rate: int = 360  # Hz
    window_size_seconds: float = 5.0  # Window duration in seconds
    step_size_seconds: float = 1.0  # Step duration in seconds
    buffer_size_seconds: float = 30.0  # Buffer size for history
    overlap_ratio: float = 0.5  # Overlap ratio between windows
    enable_online_detection: bool = True
    inference_timeout_ms: int = 100  # Max inference time in ms
    queue_maxsize: int = 1000  # Max queue size


class StreamProcessor:
    """
    Real-time ECG stream processor with sliding window and async inference
    
    Features:
    - Real-time sliding window processing
    - Async inference with queue
    - Signal preprocessing on the fly
    - Buffer management for historical data
    - Thread-safe operation
    """
    
    def __init__(self, 
                 config: StreamingConfig,
                 preprocessor=None,
                 callback: Optional[Callable] = None):
        """
        Initialize stream processor
        
        Args:
            config: Streaming configuration
            preprocessor: ECG preprocessor instance
            callback: Callback function for predictions
        """
        self.config = config
        self.preprocessor = preprocessor
        self.callback = callback
        
        # Calculate window sizes
        self.window_size = int(config.window_size_seconds * config.sampling_rate)
        self.step_size = int(config.step_size_seconds * config.sampling_rate)
        self.buffer_size = int(config.buffer_size_seconds * config.sampling_rate)
        
        # Initialize buffers
        self.raw_buffer = deque(maxlen=self.buffer_size)
        self.processed_buffer = deque(maxlen=self.buffer_size)
        
        # Queue for async processing
        self.inference_queue = queue.Queue(maxsize=config.queue_maxsize)
        self.result_queue = queue.Queue(maxsize=config.queue_maxsize)
        
        # Threading
        self.thread = None
        self.state = StreamState.IDLE
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_samples_processed': 0,
            'total_windows_processed': 0,
            'avg_inference_time_ms': 0,
            'dropped_windows': 0,
            'last_heart_rate': 0
        }
        
        logger.info(f"StreamProcessor initialized: window={self.window_size}, "
                   f"step={self.step_size}, buffer={self.buffer_size}")
        
    def start(self):
        """Start the stream processor"""
        if self.state == StreamState.RUNNING:
            logger.warning("Stream processor already running")
            return
            
        self.state = StreamState.RUNNING
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        # Start inference worker
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
        logger.info("Stream processor started")
        
    def stop(self):
        """Stop the stream processor"""
        self.state = StreamState.STOPPED
        
        if self.thread:
            self.thread.join(timeout=5.0)
            
        logger.info("Stream processor stopped")
        
    def feed_data(self, samples: np.ndarray):
        """
        Feed new ECG samples into the processor
        
        Args:
            samples: New ECG samples (1D array)
        """
        if self.state != StreamState.RUNNING:
            logger.warning(f"Stream not running (state={self.state})")
            return
            
        with self.lock:
            # Add to raw buffer
            for sample in samples:
                self.raw_buffer.append(sample)
                
            self.stats['total_samples_processed'] += len(samples)
            
            # Trigger window processing if enough data
            self._check_and_queue_windows()
            
    def _check_and_queue_windows(self):
        """Check if we have enough data for new windows and queue them"""
        with self.lock:
            current_size = len(self.raw_buffer)
            
            # Calculate number of complete windows available
            if current_size >= self.window_size:
                # Get the most recent complete window
                start_idx = max(0, current_size - self.window_size - self.step_size)
                
                for offset in range(0, current_size - self.window_size + 1, self.step_size):
                    if offset >= len(self.raw_buffer):
                        break
                        
                    window = list(self.raw_buffer)[offset:offset + self.window_size]
                    window = np.array(window)
                    
                    # Queue for inference
                    try:
                        self.inference_queue.put_nowait({
                            'window': window,
                            'timestamp': time.time(),
                            'window_id': self.stats['total_windows_processed']
                        })
                        self.stats['total_windows_processed'] += 1
                    except queue.Full:
                        self.stats['dropped_windows'] += 1
                        logger.warning("Inference queue full, dropping window")
                        
    def _process_loop(self):
        """Main processing loop"""
        while self.state == StreamState.RUNNING:
            try:
                # Get results from inference
                result = self.result_queue.get(timeout=0.1)
                
                # Update stats
                if 'inference_time_ms' in result:
                    # Exponential moving average
                    alpha = 0.1
                    self.stats['avg_inference_time_ms'] = (
                        alpha * result['inference_time_ms'] + 
                        (1 - alpha) * self.stats['avg_inference_time_ms']
                    )
                    
                # Call callback if provided
                if self.callback:
                    self.callback(result)
                    
            except queue.Empty:
                continue
                
    def _inference_worker(self):
        """Worker thread for running inference"""
        # Import here to avoid circular imports
        from .predictor import InferencePredictor
        
        # This would be initialized with actual model
        # For now, placeholder
        while self.state == StreamState.RUNNING:
            try:
                item = self.inference_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process window
                processed = self._preprocess_window(item['window'])
                
                # Run inference (placeholder - would call actual model)
                prediction = self._run_inference(processed)
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                result = {
                    'window_id': item['window_id'],
                    'timestamp': item['timestamp'],
                    'prediction': prediction,
                    'inference_time_ms': inference_time,
                    'heart_rate': self._estimate_heart_rate(processed)
                }
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
                
    def _preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Preprocess a window of ECG data"""
        if self.preprocessor:
            return self.preprocessor.preprocess(window)
        return window
    
    def _run_inference(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a window (placeholder)
        In production, this would call the actual model
        """
        # Placeholder - replace with actual model inference
        return {
            'class': 'Normal',
            'confidence': 0.85,
            'anomaly_score': 0.12,
            'needs_review': False
        }
    
    def _estimate_heart_rate(self, signal: np.ndarray) -> float:
        """Estimate heart rate from signal window"""
        try:
            from scipy.signal import find_peaks
            
            # Simple peak detection
            peaks, _ = find_peaks(signal, distance=self.config.sampling_rate * 0.3)
            
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / self.config.sampling_rate
                mean_rr = np.mean(rr_intervals)
                heart_rate = 60 / mean_rr
                self.stats['last_heart_rate'] = heart_rate
                return heart_rate
                
        except Exception as e:
            logger.debug(f"Heart rate estimation failed: {e}")
            
        return self.stats['last_heart_rate']
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """Get the most recent complete window"""
        with self.lock:
            if len(self.raw_buffer) >= self.window_size:
                return np.array(list(self.raw_buffer)[-self.window_size:])
        return None
    
    def get_live_heart_rate(self) -> float:
        """Get current heart rate estimate"""
        return self.stats['last_heart_rate']
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            **self.stats,
            'state': self.state.value,
            'buffer_size': len(self.raw_buffer),
            'queue_size': self.inference_queue.qsize()
        }
    
    def reset(self):
        """Reset the processor state"""
        with self.lock:
            self.raw_buffer.clear()
            self.processed_buffer.clear()
            
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
                
        self.stats = {k: 0 for k in self.stats}
        self.stats['avg_inference_time_ms'] = 0
        self.state = StreamState.IDLE
        
        logger.info("Stream processor reset")


class SimulatedECGStream:
    """
    Simulate real-time ECG stream for testing
    
    Generates synthetic ECG-like signal with configurable
    heart rate and arrhythmia patterns.
    """
    
    def __init__(self, sampling_rate: int = 360, heart_rate: float = 75):
        """
        Initialize simulated stream
        
        Args:
            sampling_rate: Sampling rate in Hz
            heart_rate: Target heart rate in BPM
        """
        self.sampling_rate = sampling_rate
        self.heart_rate = heart_rate
        self.time = 0
        self.beat_pattern = self._generate_beat_pattern()
        
    def _generate_beat_pattern(self) -> np.ndarray:
        """Generate a synthetic ECG beat pattern"""
        beat_duration = 60 / self.heart_rate
        beat_length = int(beat_duration * self.sampling_rate)
        
        t = np.linspace(0, beat_duration, beat_length)
        
        # Simplified ECG waveform components
        # P wave
        p_wave = 0.1 * np.exp(-((t - 0.1) / 0.05) ** 2)
        # QRS complex
        qrs = -0.2 * np.exp(-((t - 0.2) / 0.01) ** 2) + \
               1.0 * np.exp(-((t - 0.22) / 0.005) ** 2) + \
              -0.3 * np.exp(-((t - 0.24) / 0.01) ** 2)
        # T wave
        t_wave = 0.2 * np.exp(-((t - 0.35) / 0.08) ** 2)
        
        beat = p_wave + qrs + t_wave
        
        # Add small noise
        beat += 0.02 * np.random.randn(len(beat))
        
        return beat
        
    def generate_samples(self, duration_seconds: float) -> np.ndarray:
        """
        Generate samples for a duration
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Array of ECG samples
        """
        num_samples = int(duration_seconds * self.sampling_rate)
        samples = []
        
        for i in range(num_samples):
            # Repeat beat pattern
            beat_index = int(self.time * self.heart_rate / 60) % len(self.beat_pattern)
            sample = self.beat_pattern[beat_index]
            
            # Add baseline drift
            sample += 0.05 * np.sin(2 * np.pi * 0.2 * self.time)
            
            samples.append(sample)
            self.time += 1 / self.sampling_rate
            
        return np.array(samples)
    
    def set_heart_rate(self, heart_rate: float):
        """Change heart rate"""
        self.heart_rate = heart_rate
        self.beat_pattern = self._generate_beat_pattern()


if __name__ == "__main__":
    # Test stream processor
    config = StreamingConfig()
    processor = StreamProcessor(config)
    
    # Generate simulated stream
    stream = SimulatedECGStream()
    
    # Start processor
    processor.start()
    
    # Feed data in chunks
    for _ in range(10):
        samples = stream.generate_samples(1.0)  # 1 second of data
        processor.feed_data(samples)
        time.sleep(0.1)
        
    # Get stats
    stats = processor.get_stats()
    print(f"Stream stats: {stats}")
    
    processor.stop()
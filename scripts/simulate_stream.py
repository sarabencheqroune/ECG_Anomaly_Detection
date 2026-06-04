#!/usr/bin/env python
"""
Simulate real-time ECG stream for testing
Usage: python simulate_stream.py --duration 60 --heart-rate 75 --output ./stream_output
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import signal
from datetime import datetime
import json
from collections import deque
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.stream_processor import StreamProcessor, StreamingConfig
from src.inference.beat_detector import OnlineBeatDetector, BeatDetectorConfig
from src.inference.predictor import InferencePredictor
from src.utils.logger import setup_logger

logger = setup_logger('simulate_stream')


class ECGStreamSimulator:
    """Simulate real-time ECG stream with configurable parameters"""
    
    def __init__(self, sampling_rate: int = 360, heart_rate: float = 75,
                 arrhythmia_type: str = 'normal'):
        """
        Initialize stream simulator
        
        Args:
            sampling_rate: Sampling rate in Hz
            heart_rate: Target heart rate in BPM
            arrhythmia_type: 'normal', 'afib', 'pvc', 'bradycardia'
        """
        self.sampling_rate = sampling_rate
        self.heart_rate = heart_rate
        self.arrhythmia_type = arrhythmia_type
        self.time = 0
        self.beat_counter = 0
        
        # Generate beat templates
        self.normal_beat = self._generate_beat_template()
        self.afib_beat = self._generate_afib_beat()
        self.pvc_beat = self._generate_pvc_beat()
        
    def _generate_beat_template(self) -> np.ndarray:
        """Generate a normal ECG beat template"""
        beat_duration = 60 / self.heart_rate
        beat_length = int(beat_duration * self.sampling_rate)
        
        t = np.linspace(0, beat_duration, beat_length)
        beat = np.zeros_like(t)
        
        # P wave
        p_idx = np.argmin(np.abs(t - 0.1))
        if p_idx + 15 < len(beat):
            beat[p_idx:p_idx+15] = 0.15 * np.hanning(15)
        
        # QRS complex
        r_idx = np.argmin(np.abs(t - 0.2))
        if r_idx - 5 >= 0 and r_idx + 15 < len(beat):
            beat[r_idx-5:r_idx+15] = 1.0 * np.hanning(20)
        
        # S wave
        s_idx = np.argmin(np.abs(t - 0.24))
        if s_idx + 10 < len(beat):
            beat[s_idx:s_idx+10] = -0.3 * np.hanning(10)
        
        # T wave
        t_idx = np.argmin(np.abs(t - 0.35))
        if t_idx + 25 < len(beat):
            beat[t_idx:t_idx+25] = 0.25 * np.hanning(25)
        
        return beat
    
    def _generate_afib_beat(self) -> np.ndarray:
        """Generate AFib beat (irregular, missing P wave)"""
        beat = self._generate_beat_template()
        
        # Remove P wave (characteristic of AFib)
        p_region = slice(0, int(0.12 * self.sampling_rate))
        beat[p_region] = 0
        
        # Add fibrillatory waves
        fibril = 0.05 * np.random.randn(len(beat))
        fibril = np.convolve(fibril, np.ones(5)/5, mode='same')
        beat += fibril
        
        return beat
    
    def _generate_pvc_beat(self) -> np.ndarray:
        """Generate PVC beat (wide QRS, no preceding P wave)"""
        beat = self._generate_beat_template()
        
        # Wide QRS (double duration)
        r_idx = np.argmin(np.abs(np.linspace(0, 0.6, len(beat)) - 0.2))
        if r_idx - 10 >= 0 and r_idx + 30 < len(beat):
            beat[r_idx-10:r_idx+30] = 1.2 * np.hanning(40)
        
        # Remove P wave
        p_region = slice(0, int(0.12 * self.sampling_rate))
        beat[p_region] = 0
        
        return beat
    
    def generate_samples(self, duration_seconds: float) -> np.ndarray:
        """Generate ECG samples for given duration"""
        num_samples = int(duration_seconds * self.sampling_rate)
        samples = []
        
        for i in range(num_samples):
            # Determine beat interval based on arrhythmia
            if self.arrhythmia_type == 'afib':
                # Irregular intervals for AFib
                current_interval = 60 / self.heart_rate
                current_interval += np.random.randn() * 0.1
                beat_interval_samples = int(current_interval * self.sampling_rate)
            elif self.arrhythmia_type == 'bradycardia':
                # Longer intervals for bradycardia
                beat_interval_samples = int(60 / 50 * self.sampling_rate)  # 50 BPM
            else:
                beat_interval_samples = int(60 / self.heart_rate * self.sampling_rate)
            
            # Get current beat template
            beat_position = self.beat_counter % beat_interval_samples
            
            if beat_position == 0:
                # New beat starts
                if self.arrhythmia_type == 'afib':
                    beat = self.afib_beat
                elif self.arrhythmia_type == 'pvc':
                    # Occasional PVCs (every 8th beat)
                    if self.beat_counter % 8 == 0:
                        beat = self.pvc_beat
                    else:
                        beat = self.normal_beat
                else:
                    beat = self.normal_beat
                    
                self.beat_counter += 1
                current_beat = beat.copy()
                
            # Get sample from current beat
            if beat_position < len(current_beat):
                sample = current_beat[beat_position]
            else:
                sample = 0
                
            # Add noise and baseline drift
            sample += 0.05 * np.random.randn()
            sample += 0.1 * np.sin(2 * np.pi * 0.2 * self.time)
            
            samples.append(sample)
            self.time += 1 / self.sampling_rate
            
        return np.array(samples)
    
    def set_heart_rate(self, heart_rate: float):
        """Change heart rate during simulation"""
        self.heart_rate = heart_rate
        self.normal_beat = self._generate_beat_template()
        
    def set_arrhythmia(self, arrhythmia_type: str):
        """Change arrhythmia type during simulation"""
        self.arrhythmia_type = arrhythmia_type


class StreamTester:
    """Test stream processing pipeline with simulated data"""
    
    def __init__(self, model_path: str = None, output_dir: str = "./stream_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config = StreamingConfig()
        self.processor = StreamProcessor(self.config)
        
        if model_path and Path(model_path).exists():
            self.predictor = InferencePredictor(model_path)
        else:
            self.predictor = None
            
        self.detector = OnlineBeatDetector(BeatDetectorConfig())
        
        # Data storage
        self.signal_buffer = deque(maxlen=3600)  # 10 seconds
        self.predictions = []
        self.detected_peaks = []
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_predictions': 0,
            'avg_inference_time_ms': 0,
            'detected_beats': 0,
            'heart_rate_history': []
        }
        
    def callback(self, result: dict):
        """Callback for stream processor results"""
        self.predictions.append(result)
        self.stats['total_predictions'] += 1
        
        if 'inference_time_ms' in result:
            alpha = 0.1
            self.stats['avg_inference_time_ms'] = (
                alpha * result['inference_time_ms'] + 
                (1 - alpha) * self.stats['avg_inference_time_ms']
            )
            
        logger.debug(f"Prediction: {result.get('prediction', {})}")
        
    def run(self, duration_seconds: float, simulator: ECGStreamSimulator):
        """Run stream test"""
        logger.info(f"Starting stream test for {duration_seconds} seconds")
        logger.info(f"Arrhythmia type: {simulator.arrhythmia_type}")
        logger.info(f"Heart rate: {simulator.heart_rate} BPM")
        
        # Start processor
        self.processor.callback = self.callback
        self.processor.start()
        
        # Simulate stream
        chunk_duration = 0.5  # 500ms chunks
        num_chunks = int(duration_seconds / chunk_duration)
        
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            # Generate chunk
            samples = simulator.generate_samples(chunk_duration)
            
            # Feed to processor
            self.processor.feed_data(samples)
            
            # Also run beat detector for comparison
            for sample in samples:
                peak = self.detector.process_sample(sample)
                if peak is not None:
                    self.detected_peaks.append(peak)
                    
            # Update signal buffer
            self.signal_buffer.extend(samples)
            
            # Update stats
            self.stats['total_samples'] += len(samples)
            self.stats['detected_beats'] = len(self.detected_peaks)
            self.stats['heart_rate_history'].append(self.detector.get_heart_rate())
            
            # Progress indicator
            if (chunk_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                progress = (chunk_idx + 1) / num_chunks * 100
                logger.info(f"Progress: {progress:.1f}% | HR: {self.detector.get_heart_rate():.1f} BPM | "
                           f"Beats: {self.stats['detected_beats']}")
                
            # Small delay to simulate real-time
            time.sleep(0.01)
            
        # Stop processor
        self.processor.stop()
        
        elapsed = time.time() - start_time
        logger.info(f"\nStream test completed in {elapsed:.2f} seconds")
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Get test results"""
        results = {
            'duration_seconds': len(self.signal_buffer) / 360,
            'total_samples': self.stats['total_samples'],
            'total_predictions': self.stats['total_predictions'],
            'detected_beats': self.stats['detected_beats'],
            'avg_inference_time_ms': self.stats['avg_inference_time_ms'],
            'avg_heart_rate': np.mean(self.stats['heart_rate_history']) if self.stats['heart_rate_history'] else 0,
            'heart_rate_std': np.std(self.stats['heart_rate_history']) if self.stats['heart_rate_history'] else 0
        }
        
        return results
    
    def save_results(self, simulator: ECGStreamSimulator):
        """Save test results to disk"""
        import matplotlib.pyplot as plt
        
        # Save signal
        signal_array = np.array(self.signal_buffer)
        np.save(self.output_dir / 'recorded_signal.npy', signal_array)
        
        # Save predictions
        with open(self.output_dir / 'predictions.json', 'w') as f:
            json.dump(self.predictions, f, indent=2)
            
        # Save peaks
        np.save(self.output_dir / 'detected_peaks.npy', np.array(self.detected_peaks))
        
        # Plot signal
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Signal with peaks
        time_axis = np.arange(len(signal_array)) / 360
        axes[0].plot(time_axis, signal_array, 'b-', linewidth=1)
        if self.detected_peaks:
            peak_times = np.array(self.detected_peaks) / 360
            peak_values = signal_array[self.detected_peaks]
            axes[0].scatter(peak_times, peak_values, c='red', s=30, marker='^')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].set_title(f'ECG Signal - {simulator.arrhythmia_type.upper()} '
                         f'(HR: {simulator.heart_rate:.0f} BPM)')
        axes[0].grid(True, alpha=0.3)
        
        # Heart rate over time
        if self.stats['heart_rate_history']:
            axes[1].plot(self.stats['heart_rate_history'], 'g-', linewidth=1.5)
            axes[1].axhline(simulator.heart_rate, color='r', linestyle='--', 
                           label=f'Target: {simulator.heart_rate:.0f} BPM')
            axes[1].set_ylabel('Heart Rate (BPM)')
            axes[1].set_title('Estimated Heart Rate')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        # Inference time
        if self.predictions:
            inference_times = [p.get('inference_time_ms', 0) for p in self.predictions]
            axes[2].plot(inference_times, 'b-', linewidth=1)
            axes[2].axhline(self.stats['avg_inference_time_ms'], color='r', 
                           linestyle='--', label=f'Avg: {self.stats["avg_inference_time_ms"]:.1f} ms')
            axes[2].set_xlabel('Prediction Number')
            axes[2].set_ylabel('Inference Time (ms)')
            axes[2].set_title('Inference Latency')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'simulation_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save results JSON
        results = self.get_results()
        results['arrhythmia_type'] = simulator.arrhythmia_type
        results['target_heart_rate'] = simulator.heart_rate
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {self.output_dir}")


def run_scenarios(tester: StreamTester, output_dir: Path):
    """Run multiple test scenarios"""
    scenarios = [
        {'name': 'normal', 'arrhythmia': 'normal', 'heart_rate': 75},
        {'name': 'tachycardia', 'arrhythmia': 'normal', 'heart_rate': 120},
        {'name': 'bradycardia', 'arrhythmia': 'bradycardia', 'heart_rate': 45},
        {'name': 'afib', 'arrhythmia': 'afib', 'heart_rate': 85},
        {'name': 'pvc', 'arrhythmia': 'pvc', 'heart_rate': 75}
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scenario: {scenario['name']}")
        logger.info(f"{'='*60}")
        
        # Create simulator for this scenario
        simulator = ECGStreamSimulator(
            sampling_rate=360,
            heart_rate=scenario['heart_rate'],
            arrhythmia_type=scenario['arrhythmia']
        )
        
        # Create scenario-specific tester
        scenario_tester = StreamTester(output_dir=str(output_dir / scenario['name']))
        
        # Run test
        results = scenario_tester.run(duration_seconds=30, simulator=simulator)
        scenario_tester.save_results(simulator)
        
        all_results[scenario['name']] = results
        
        # Print summary
        print(f"\nResults for {scenario['name']}:")
        print(f"  Detected beats: {results['detected_beats']}")
        print(f"  Avg heart rate: {results['avg_heart_rate']:.1f} BPM")
        print(f"  Avg inference time: {results['avg_inference_time_ms']:.1f} ms")
        
    return all_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Simulate real-time ECG stream')
    
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration of simulation in seconds')
    
    parser.add_argument('--heart-rate', type=int, default=75,
                       help='Target heart rate in BPM')
    
    parser.add_argument('--arrhythmia', type=str, default='normal',
                       choices=['normal', 'afib', 'pvc', 'bradycardia'],
                       help='Type of arrhythmia to simulate')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for inference (optional)')
    
    parser.add_argument('--output', type=str, default='./stream_results',
                       help='Output directory for results')
    
    parser.add_argument('--scenarios', action='store_true',
                       help='Run all test scenarios')
    
    return parser.parse_args()


def main():
    """Main simulation function"""
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.scenarios:
        # Run all scenarios
        tester = StreamTester(model_path=args.model, output_dir=str(output_dir))
        all_results = run_scenarios(tester, output_dir)
        
        # Save combined results
        with open(output_dir / 'all_scenarios_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
            
    else:
        # Run single scenario
        simulator = ECGStreamSimulator(
            sampling_rate=360,
            heart_rate=args.heart_rate,
            arrhythmia_type=args.arrhythmia
        )
        
        tester = StreamTester(model_path=args.model, output_dir=str(output_dir))
        
        logger.info("=" * 60)
        logger.info("ECG Stream Simulation")
        logger.info("=" * 60)
        logger.info(f"Duration: {args.duration} seconds")
        logger.info(f"Heart rate: {args.heart_rate} BPM")
        logger.info(f"Arrhythmia: {args.arrhythmia}")
        logger.info("=" * 60)
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            logger.info("\nStopping simulation...")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Run test
        results = tester.run(duration_seconds=args.duration, simulator=simulator)
        tester.save_results(simulator)
        
        # Print results
        print("\n" + "=" * 60)
        print("Simulation Results")
        print("=" * 60)
        print(f"Total samples processed: {results['total_samples']}")
        print(f"Detected beats: {results['detected_beats']}")
        print(f"Average heart rate: {results['avg_heart_rate']:.1f} BPM")
        print(f"Heart rate variability: ±{results['heart_rate_std']:.1f} BPM")
        print(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms")
        print("=" * 60)
        
    logger.info(f"\n✅ Simulation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
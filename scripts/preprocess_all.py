#!/usr/bin/env python
"""
Batch preprocessing of ECG datasets
Usage: python preprocess_all.py --data-dir ./data --output-dir ./data/processed
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import pickle
from typing import List, Dict
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import ECGLoader
from src.data.preprocessor import ECGPreprocessor
from src.data.segmenter import ECGSegmenter, BeatSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch preprocessing of ECG datasets')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root data directory')
    
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    
    parser.add_argument('--datasets', nargs='+', 
                       default=['mit-bih', 'ptb-xl'],
                       help='Datasets to preprocess')
    
    parser.add_argument('--max-beats-per-record', type=int, default=500,
                       help='Maximum beats to extract per record')
    
    parser.add_argument('--sampling-rate', type=int, default=360,
                       help='Target sampling rate')
    
    parser.add_argument('--leads', type=int, nargs='+', default=[0],
                       help='Leads to use (0=first lead)')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save sample plots')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show sample visualizations')
    
    return parser.parse_args()


class BatchPreprocessor:
    """Batch preprocessing pipeline"""
    
    def __init__(self, data_dir: str, output_dir: str, sampling_rate: int = 360):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sampling_rate = sampling_rate
        
        # Create output directories
        self.beats_dir = self.output_dir / 'beats'
        self.metadata_dir = self.output_dir / 'metadata'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.beats_dir, self.metadata_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize processors
        self.preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)
        self.segmenter = ECGSegmenter(sampling_rate=sampling_rate)
        self.loader = ECGLoader(data_dir=str(data_dir))
        
        self.stats = {
            'total_records': 0,
            'total_beats': 0,
            'total_annotations': 0,
            'beats_per_class': {},
            'records_processed': []
        }
        
    def preprocess_mitbih(self, max_beats_per_record: int = 500,
                         leads: List[int] = None) -> Dict:
        """Preprocess MIT-BIH dataset"""
        logger.info("Processing MIT-BIH dataset...")
        
        # MIT-BIH record list
        records = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116,
                   117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 208,
                   209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228,
                   230, 231, 232, 233, 234]
        
        dataset_stats = {
            'records': [],
            'total_beats': 0,
            'beats_per_class': {}
        }
        
        for record_name in tqdm(records, desc="Processing MIT-BIH records"):
            try:
                # Load record
                record = self.loader.load_mit_bih_record(
                    record_name, 
                    leads=leads or [0]
                )
                
                # Preprocess signal
                signal = self.preprocessor.preprocess(record.signal.flatten())
                
                # Detect R-peaks
                r_peaks, _ = self.segmenter.detect_r_peaks(signal, method='neurokit2')
                
                # Segment beats
                beats = self.segmenter.segment_beats(signal, r_peaks)
                
                # Limit beats per record
                if len(beats) > max_beats_per_record:
                    beats = beats[:max_beats_per_record]
                    
                if len(beats) == 0:
                    continue
                    
                # Extract beat signals and labels
                beat_signals = []
                beat_labels = []
                beat_metadata = []
                
                for i, beat in enumerate(beats):
                    # Get label from closest annotation
                    if record.annotations is not None:
                        closest_idx = np.argmin(np.abs(record.annotations - beat.start_index - beat.r_peak_index))
                        if abs(record.annotations[closest_idx] - (beat.start_index + beat.r_peak_index)) < 50:
                            label = record.annotation_symbols[closest_idx]
                        else:
                            label = 'Normal'
                    else:
                        label = 'Normal'
                        
                    # Map to class
                    class_label = self._map_label(label)
                    
                    beat_signals.append(beat.signal)
                    beat_labels.append(class_label)
                    beat_metadata.append({
                        'record_id': record.record_id,
                        'beat_id': i,
                        'r_peak_position': beat.r_peak_index,
                        'quality': beat.quality_score
                    })
                    
                    dataset_stats['beats_per_class'][class_label] = \
                        dataset_stats['beats_per_class'].get(class_label, 0) + 1
                        
                # Save to file
                self._save_record_data(
                    dataset='mitbih',
                    record_name=str(record_name),
                    beats=np.array(beat_signals),
                    labels=np.array(beat_labels),
                    metadata=beat_metadata
                )
                
                dataset_stats['records'].append(str(record_name))
                dataset_stats['total_beats'] += len(beats)
                
                # Save sample plot if requested
                if hasattr(self, 'plots_dir') and len(beats) > 0:
                    self._save_sample_plot(beats[0], record_name)
                    
            except Exception as e:
                logger.warning(f"Failed to process record {record_name}: {e}")
                
        return dataset_stats
    
    def preprocess_ptbxl(self, max_beats_per_record: int = 500,
                        leads: List[int] = None) -> Dict:
        """Preprocess PTB-XL dataset"""
        logger.info("Processing PTB-XL dataset...")
        
        import pandas as pd
        
        ptbxl_dir = self.data_dir / 'raw' / 'ptb-xl'
        
        if not ptbxl_dir.exists():
            logger.warning(f"PTB-XL directory not found: {ptbxl_dir}")
            return {}
            
        # Load database CSV
        try:
            df = pd.read_csv(ptbxl_dir / 'ptbxl_database.csv')
        except Exception as e:
            logger.error(f"Failed to load PTB-XL database: {e}")
            return {}
            
        dataset_stats = {
            'records': [],
            'total_beats': 0,
            'beats_per_class': {}
        }
        
        # Process first N records for demonstration
        records_to_process = df.head(100) if 'test' in str(self.output_dir) else df
        
        for idx, row in tqdm(records_to_process.iterrows(), 
                            total=len(records_to_process),
                            desc="Processing PTB-XL records"):
            try:
                record_name = f"{row['ecg_id']:05d}_hr"
                
                record = self.loader.load_ptbxl_record(record_name, leads=leads or [0])
                
                # Process signal
                signal = self.preprocessor.preprocess(record.signal.flatten())
                
                # Detect R-peaks
                r_peaks, _ = self.segmenter.detect_r_peaks(signal, method='neurokit2')
                
                # Segment beats
                beats = self.segmenter.segment_beats(signal, r_peaks)
                
                if len(beats) == 0:
                    continue
                    
                # Get label
                label = record.annotation_symbols[0] if record.annotation_symbols else 'Other'
                class_label = self._map_label(label)
                
                # Limit beats
                if len(beats) > max_beats_per_record:
                    beats = beats[:max_beats_per_record]
                    
                beat_signals = [b.signal for b in beats]
                beat_labels = [class_label] * len(beats)
                beat_metadata = [{
                    'record_id': record.record_id,
                    'beat_id': i,
                    'r_peak_position': b.r_peak_index,
                    'quality': b.quality_score
                } for i, b in enumerate(beats)]
                
                # Save to file
                self._save_record_data(
                    dataset='ptbxl',
                    record_name=record_name,
                    beats=np.array(beat_signals),
                    labels=np.array(beat_labels),
                    metadata=beat_metadata
                )
                
                dataset_stats['records'].append(record_name)
                dataset_stats['total_beats'] += len(beats)
                dataset_stats['beats_per_class'][class_label] = \
                    dataset_stats['beats_per_class'].get(class_label, 0) + len(beats)
                    
            except Exception as e:
                logger.debug(f"Failed to process record {record_name}: {e}")
                
        return dataset_stats
    
    def _map_label(self, label: str) -> str:
        """Map label to standard classes"""
        label_mapping = {
            'Normal': 'Normal',
            'PVC (Premature Ventricular Contraction)': 'PVC',
            'AFib (Atrial Fibrillation)': 'AFib',
            'Bradycardia': 'Bradycardia'
        }
        
        for key, value in label_mapping.items():
            if key.lower() in label.lower():
                return value
                
        return 'Other'
    
    def _save_record_data(self, dataset: str, record_name: str,
                         beats: np.ndarray, labels: np.ndarray,
                         metadata: List[Dict]):
        """Save processed record data"""
        # Save beats
        beats_path = self.beats_dir / f"{dataset}_{record_name}_beats.npy"
        np.save(beats_path, beats)
        
        # Save labels
        labels_path = self.beats_dir / f"{dataset}_{record_name}_labels.npy"
        np.save(labels_path, labels)
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{dataset}_{record_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _save_sample_plot(self, beat: BeatSegment, record_name: str):
        """Save sample plot of a beat"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 4))
            time = np.arange(len(beat.signal)) / self.sampling_rate * 1000
            
            ax.plot(time, beat.signal, 'b-', linewidth=1.5)
            ax.axvline(time[beat.r_peak_index], color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (mV)')
            ax.set_title(f'Sample Beat - Record {record_name}')
            ax.grid(True, alpha=0.3)
            
            plot_path = self.plots_dir / f"{record_name}_sample_beat.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.debug(f"Could not save plot: {e}")
            
    def run(self, datasets: List[str], max_beats_per_record: int = 500,
            leads: List[int] = None):
        """Run full preprocessing pipeline"""
        logger.info("=" * 60)
        logger.info("Batch Preprocessing Started")
        logger.info("=" * 60)
        
        all_stats = {}
        
        for dataset in datasets:
            if dataset == 'mit-bih':
                stats = self.preprocess_mitbih(max_beats_per_record, leads)
                all_stats['mitbih'] = stats
                self.stats['total_records'] += len(stats['records'])
                self.stats['total_beats'] += stats['total_beats']
                for cls, count in stats['beats_per_class'].items():
                    self.stats['beats_per_class'][cls] = \
                        self.stats['beats_per_class'].get(cls, 0) + count
                        
            elif dataset == 'ptb-xl':
                stats = self.preprocess_ptbxl(max_beats_per_record, leads)
                all_stats['ptbxl'] = stats
                self.stats['total_records'] += len(stats['records'])
                self.stats['total_beats'] += stats['total_beats']
                for cls, count in stats['beats_per_class'].items():
                    self.stats['beats_per_class'][cls] = \
                        self.stats['beats_per_class'].get(cls, 0) + count
                        
        # Save overall stats
        stats_path = self.output_dir / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        # Print summary
        self._print_summary()
        
        return all_stats
    
    def _print_summary(self):
        """Print preprocessing summary"""
        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("=" * 60)
        print(f"Total records processed: {self.stats['total_records']}")
        print(f"Total beats extracted: {self.stats['total_beats']}")
        print("\nClass distribution:")
        for cls, count in sorted(self.stats['beats_per_class'].items()):
            percentage = 100 * count / self.stats['total_beats']
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        print(f"\nOutput directory: {self.output_dir}")
        print("=" * 60)


def main():
    """Main entry point"""
    args = parse_args()
    
    preprocessor = BatchPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate
    )
    
    # Set save plots flag
    if args.save_plots:
        preprocessor.plots_dir.mkdir(parents=True, exist_ok=True)
        
    preprocessor.run(
        datasets=args.datasets,
        max_beats_per_record=args.max_beats_per_record,
        leads=args.leads
    )
    
    print("\n✅ Preprocessing completed!")


if __name__ == "__main__":
    main()
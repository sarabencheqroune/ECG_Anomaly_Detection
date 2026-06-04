"""
PyTorch Dataset classes for ECG beat classification
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import json

from .loader import ECGLoader, ECGRecord
from .preprocessor import ECGPreprocessor
from .segmenter import ECGSegmenter, BeatSegment
from .augmenter import ECGAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGBeatDataset(Dataset):
    """
    PyTorch Dataset for ECG beat classification
    
    Handles loading, preprocessing, and augmentation of ECG beats
    """
    
    # Class mapping
    CLASS_MAPPING = {
        'Normal': 0,
        'AFib': 1,
        'PVC': 2,
        'Bradycardia': 3,
        'Other': 4
    }
    
    REVERSE_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    
    def __init__(self, 
                 data_dir: str = "./data",
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 augment: bool = False,
                 max_beats_per_record: Optional[int] = None,
                 use_ptbxl: bool = True,
                 use_mitbih: bool = True,
                 balance_classes: bool = False):
        """
        Initialize ECG Beat Dataset
        
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            augment: Apply data augmentation
            max_beats_per_record: Maximum beats per record (for balance)
            use_ptbxl: Include PTB-XL dataset
            use_mitbih: Include MIT-BIH dataset
            balance_classes: Balance class distribution
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augment = augment
        self.max_beats_per_record = max_beats_per_record
        
        # Initialize processors
        self.preprocessor = ECGPreprocessor(sampling_rate=360)
        self.segmenter = ECGSegmenter(sampling_rate=360)
        self.augmenter = ECGAugmenter(sampling_rate=360) if augment else None
        
        # Load data
        self.beats = []
        self.labels = []
        self.metadata = []
        
        if use_mitbih:
            self._load_mitbih_data()
        if use_ptbxl:
            self._load_ptbxl_data()
            
        # Apply class balancing if requested
        if balance_classes and split == 'train':
            self._balance_classes()
            
        # Convert to numpy arrays
        self.beats = np.array(self.beats, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        logger.info(f"Loaded {len(self.beats)} beats for {split} set")
        self._log_class_distribution()
        
    def _load_mitbih_data(self):
        """Load and process MIT-BIH dataset"""
        mitbih_dir = self.data_dir / "raw" / "mit-bih"
        
        if not mitbih_dir.exists():
            logger.warning(f"MIT-BIH directory not found: {mitbih_dir}")
            return
            
        # Records to use for each split (based on patient IDs)
        split_records = {
            'train': [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 
                      117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 208],
            'val': [209, 210, 212, 213, 214, 215, 217, 219, 220, 221],
            'test': [222, 223, 228, 230, 231, 232, 233, 234]
        }
        
        records_to_load = split_records.get(self.split, [])
        
        loader = ECGLoader(data_dir=str(self.data_dir / "raw"))
        
        for record_name in records_to_load:
            try:
                record = loader.load_mit_bih_record(record_name, leads=[0])
                
                # Preprocess signal
                signal = self.preprocessor.preprocess(record.signal.flatten())
                
                # Detect R-peaks
                r_peaks, _ = self.segmenter.detect_r_peaks(signal, method='neurokit2')
                
                # Segment beats
                beats = self.segmenter.segment_beats(signal, r_peaks)
                
                # Assign labels based on annotations
                for i, beat in enumerate(beats):
                    if record.annotations is not None:
                        # Find closest annotation
                        closest_ann = self._find_closest_annotation(
                            beat.start_index + beat.r_peak_index, 
                            record.annotations, 
                            record.annotation_symbols
                        )
                        label = self._map_mitbih_label(closest_ann)
                    else:
                        label = 'Other'
                        
                    # Limit beats per record
                    if self.max_beats_per_record and i >= self.max_beats_per_record:
                        break
                        
                    self.beats.append(beat.signal)
                    self.labels.append(self.CLASS_MAPPING[label])
                    self.metadata.append({
                        'dataset': 'mitbih',
                        'record': record_name,
                        'beat_id': i,
                        'quality': beat.quality_score
                    })
                    
                logger.info(f"Loaded {len(beats)} beats from MIT-BIH record {record_name}")
                
            except Exception as e:
                logger.error(f"Failed to load MIT-BIH record {record_name}: {e}")
                
    def _load_ptbxl_data(self):
        """Load and process PTB-XL dataset"""
        ptbxl_dir = self.data_dir / "raw" / "ptb-xl"
        
        if not ptbxl_dir.exists():
            logger.warning(f"PTB-XL directory not found: {ptbxl_dir}")
            return
            
        # Load PTB-XL database CSV
        try:
            df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
            
            # Filter by split
            # Note: You'll need proper patient-wise split here
            # This is simplified for demonstration
            
            # Limit number of records for development
            records = df.head(100) if self.split == 'train' else df.head(20)
            
            loader = ECGLoader(data_dir=str(self.data_dir / "raw"))
            
            for idx, row in records.iterrows():
                record_name = f"{row['ecg_id']:05d}_hr"
                
                try:
                    record = loader.load_ptbxl_record(record_name)
                    
                    # Process each lead separately
                    for lead_idx in range(min(record.signal.shape[1], 3)):  # Use up to 3 leads
                        signal = self.preprocessor.preprocess(record.signal[:, lead_idx])
                        
                        # Detect R-peaks
                        r_peaks, _ = self.segmenter.detect_r_peaks(signal, method='neurokit2')
                        
                        # Segment beats
                        beats = self.segmenter.segment_beats(signal, r_peaks)
                        
                        # Get label
                        label = record.annotation_symbols[0] if record.annotation_symbols else 'Other'
                        label_idx = self.CLASS_MAPPING.get(label, 4)  # Default to 'Other'
                        
                        for i, beat in enumerate(beats):
                            if self.max_beats_per_record and i >= self.max_beats_per_record:
                                break
                                
                            self.beats.append(beat.signal)
                            self.labels.append(label_idx)
                            self.metadata.append({
                                'dataset': 'ptbxl',
                                'record': record_name,
                                'lead': lead_idx,
                                'beat_id': i,
                                'quality': beat.quality_score
                            })
                            
                except Exception as e:
                    logger.error(f"Failed to load PTB-XL record {record_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load PTB-XL database: {e}")
            
    def _find_closest_annotation(self, beat_center: int, 
                                 annotations: np.ndarray,
                                 symbols: List[str]) -> str:
        """Find closest annotation to beat center"""
        if len(annotations) == 0:
            return 'Normal'
            
        # Find closest annotation index
        closest_idx = np.argmin(np.abs(annotations - beat_center))
        
        if abs(annotations[closest_idx] - beat_center) < 50:  # Within 50 samples
            return symbols[closest_idx]
        else:
            return 'Normal'
            
    def _map_mitbih_label(self, label: str) -> str:
        """Map MIT-BIH label to our classes"""
        mapping = {
            'Normal': 'Normal',
            'PVC (Premature Ventricular Contraction)': 'PVC',
            'AFib (Atrial Fibrillation)': 'AFib',
            'Bradycardia': 'Bradycardia',
            'Tachycardia': 'Other'
        }
        
        # Check for AFib pattern
        if 'AFib' in label or 'A-FIB' in label:
            return 'AFib'
            
        return mapping.get(label, 'Other')
        
    def _balance_classes(self):
        """Balance class distribution by undersampling"""
        from collections import Counter
        
        label_counts = Counter(self.labels)
        min_count = min(label_counts.values())
        
        balanced_indices = []
        
        for label in range(len(self.CLASS_MAPPING)):
            label_indices = [i for i, l in enumerate(self.labels) if l == label]
            
            if len(label_indices) > min_count:
                # Random undersample
                selected = np.random.choice(label_indices, min_count, replace=False)
            else:
                selected = label_indices
                
            balanced_indices.extend(selected)
            
        # Keep balanced subset
        self.beats = [self.beats[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]
        self.metadata = [self.metadata[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset: {len(self.beats)} beats")
        
    def _log_class_distribution(self):
        """Log class distribution"""
        from collections import Counter
        
        label_counts = Counter(self.labels)
        
        logger.info("Class distribution:")
        for label_idx, count in label_counts.items():
            label_name = self.REVERSE_MAPPING[label_idx]
            percentage = 100 * count / len(self.labels)
            logger.info(f"  {label_name}: {count} ({percentage:.1f}%)")
            
    def __len__(self) -> int:
        return len(self.beats)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            Tuple of (beat_tensor, label_tensor)
        """
        beat = self.beats[idx]
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and self.augmenter:
            beat = self.augmenter.augment(beat, intensity='medium')
            
        # Apply transform
        if self.transform:
            beat = self.transform(beat)
            
        # Convert to tensor
        beat_tensor = torch.from_numpy(beat).float()
        
        # Add channel dimension: (samples,) -> (1, samples)
        if len(beat_tensor.shape) == 1:
            beat_tensor = beat_tensor.unsqueeze(0)
            
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return beat_tensor, label_tensor


class ECGStreamingDataset(Dataset):
    """
    Dataset for streaming ECG data (sliding window approach)
    """
    
    def __init__(self, 
                 signal: np.ndarray,
                 window_size_seconds: float = 5.0,
                 step_size_seconds: float = 1.0,
                 sampling_rate: int = 360,
                 preprocessor: Optional[ECGPreprocessor] = None):
        """
        Initialize streaming dataset
        
        Args:
            signal: Raw ECG signal
            window_size_seconds: Size of sliding window in seconds
            step_size_seconds: Step size between windows in seconds
            sampling_rate: Signal sampling rate
            preprocessor: ECG preprocessor
        """
        self.signal = signal
        self.window_size = int(window_size_seconds * sampling_rate)
        self.step_size = int(step_size_seconds * sampling_rate)
        self.sampling_rate = sampling_rate
        self.preprocessor = preprocessor or ECGPreprocessor(sampling_rate=sampling_rate)
        
        # Create windows
        self.windows = []
        self.start_indices = []
        
        for start in range(0, len(signal) - self.window_size, self.step_size):
            window = signal[start:start + self.window_size]
            processed = self.preprocessor.preprocess(window)
            self.windows.append(processed)
            self.start_indices.append(start)
            
        self.windows = np.array(self.windows, dtype=np.float32)
        
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window = self.windows[idx]
        start_idx = self.start_indices[idx]
        
        window_tensor = torch.from_numpy(window).float()
        window_tensor = window_tensor.unsqueeze(0)  # Add channel dimension
        
        return window_tensor, start_idx


def create_dataloaders(data_dir: str = "./data",
                       batch_size: int = 64,
                       num_workers: int = 4,
                       use_augmentation: bool = True) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        use_augmentation: Apply data augmentation for training
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create datasets
    train_dataset = ECGBeatDataset(
        data_dir=data_dir,
        split='train',
        augment=use_augmentation,
        balance_classes=True,
        max_beats_per_record=500  # Limit for memory
    )
    
    val_dataset = ECGBeatDataset(
        data_dir=data_dir,
        split='val',
        augment=False,
        balance_classes=False
    )
    
    test_dataset = ECGBeatDataset(
        data_dir=data_dir,
        split='test',
        augment=False,
        balance_classes=False
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


if __name__ == "__main__":
    # Test dataset
    print("Testing ECGBeatDataset...")
    
    dataset = ECGBeatDataset(
        data_dir="./data",
        split='train',
        augment=False,
        max_beats_per_record=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        beat, label = dataset[0]
        print(f"Beat shape: {beat.shape}")
        print(f"Label: {label} ({dataset.REVERSE_MAPPING[label.item()]})")
        
    # Test dataloaders
    print("\nTesting dataloaders...")
    dataloaders = create_dataloaders(batch_size=32, use_augmentation=False)
    
    for split, loader in dataloaders.items():
        print(f"{split} loader: {len(loader)} batches")
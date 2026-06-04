"""
Load raw ECG records from WFDB format
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ECGRecord:
    """Container for ECG record data"""
    record_id: str
    signal: np.ndarray  # Shape: (samples, leads)
    sampling_rate: int
    annotations: Optional[np.ndarray] = None
    annotation_symbols: Optional[List[str]] = None
    patient_info: Optional[Dict] = None
    metadata: Optional[Dict] = None


class ECGLoader:
    """Load and manage ECG records from various formats"""
    
    # MIT-BIH annotation codes (AHA to text mapping)
    MITBIH_SYMBOLS = {
        1: 'Normal',
        2: 'PVC (Premature Ventricular Contraction)',
        3: 'PAC (Premature Atrial Contraction)',
        4: 'Ventricular Escape',
        5: 'Atrial Escape',
        6: 'Nodal Escape',
        7: 'Paced Beat',
        8: 'Fusion of ventricular and normal',
        9: 'Unknown',
        10: 'Ventricular Flutter',
        11: 'AFib (Atrial Fibrillation)',
        12: 'Atrial Flutter',
        13: 'Bradycardia',
        14: 'Tachycardia',
        15: 'Heart Block',
        16: 'Other'
    }
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize loader
        
        Args:
            data_dir: Root directory containing datasets
        """
        self.data_dir = Path(data_dir)
        
    def load_mit_bih_record(self, record_name: Union[str, int], 
                           leads: Optional[List[int]] = None,
                           sampto: Optional[int] = None) -> ECGRecord:
        """
        Load a single MIT-BIH record
        
        Args:
            record_name: Record number (e.g., 100, '100')
            leads: Which leads to load (0=MLII, 1=V5, None=all)
            sampto: Number of samples to load
            
        Returns:
            ECGRecord object
        """
        # Format record name
        record_name = str(record_name).zfill(3)
        
        # Path to MIT-BIH directory
        mitbih_dir = self.data_dir / "mit-bih"
        
        # Load signal
        record = wfdb.rdrecord(str(mitbih_dir / record_name), sampto=sampto)
        
        # Select specific leads if requested
        if leads is not None:
            signal = record.p_signal[:, leads]
            lead_names = [record.sig_name[i] for i in leads]
        else:
            signal = record.p_signal
            lead_names = record.sig_name
            
        # Load annotations if available
        try:
            annotation = wfdb.rdann(str(mitbih_dir / record_name), 'atr')
            annotation_symbols = [self.MITBIH_SYMBOLS.get(code, 'Unknown') 
                                 for code in annotation.symbol]
        except:
            annotation = None
            annotation_symbols = None
            
        return ECGRecord(
            record_id=f"mitbih_{record_name}",
            signal=signal,
            sampling_rate=record.fs,
            annotations=annotation.sample if annotation else None,
            annotation_symbols=annotation_symbols,
            metadata={
                'lead_names': lead_names,
                'record_name': record_name,
                'dataset': 'mit-bih',
                'length_samples': record.sig_len
            }
        )
    
    def load_mit_bih_records(self, record_names: List[Union[str, int]],
                            leads: Optional[List[int]] = None) -> List[ECGRecord]:
        """
        Load multiple MIT-BIH records
        
        Args:
            record_names: List of record numbers
            leads: Which leads to load
            
        Returns:
            List of ECGRecord objects
        """
        records = []
        
        for record_name in record_names:
            try:
                record = self.load_mit_bih_record(record_name, leads)
                records.append(record)
                logger.info(f"Loaded record {record_name}")
            except Exception as e:
                logger.error(f"Failed to load record {record_name}: {e}")
                
        return records
    
    def load_ptb_xl_record(self, record_name: str, 
                          leads: Optional[List[int]] = None) -> ECGRecord:
        """
        Load a single PTB-XL record
        
        Args:
            record_name: PTB-XL record ID (e.g., '00001_hr')
            leads: Which leads to load (0-11 for 12-lead ECG)
            
        Returns:
            ECGRecord object
        """
        ptbxl_dir = self.data_dir / "ptb-xl"
        
        # Load signal
        record = wfdb.rdrecord(str(ptbxl_dir / record_name))
        
        # Select specific leads
        if leads is not None:
            signal = record.p_signal[:, leads]
            lead_names = [record.sig_name[i] for i in leads]
        else:
            signal = record.p_signal
            lead_names = record.sig_name
            
        # Load annotation (SCP codes) from CSV
        try:
            df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
            record_info = df[df['ecg_id'] == int(record_name.split('_')[0])]
            
            if not record_info.empty:
                scp_codes = eval(record_info['scp_codes'].iloc[0])
                # Convert SCP codes to class labels
                label = self._convert_ptbxl_label(scp_codes)
            else:
                label = None
        except:
            label = None
            
        return ECGRecord(
            record_id=f"ptbxl_{record_name}",
            signal=signal,
            sampling_rate=record.fs,
            annotation_symbols=[label] if label else None,
            metadata={
                'lead_names': lead_names,
                'record_name': record_name,
                'dataset': 'ptb-xl',
                'length_samples': record.sig_len
            }
        )
    
    def _convert_ptbxl_label(self, scp_codes: Dict) -> str:
        """
        Convert PTB-XL SCP codes to classification labels
        
        Args:
            scp_codes: Dictionary of SCP codes and their probabilities
            
        Returns:
            Class label string
        """
        # Priority mapping from SCP codes to our 5 classes
        priority_map = [
            ('AFIB', 'AFib'),
            ('VEB', 'PVC'),
            ('BRADY', 'Bradycardia'),
            ('NORM', 'Normal'),
        ]
        
        for scp_code, label in priority_map:
            if scp_code in scp_codes:
                return label
                
        return 'Other'
    
    def load_record_by_path(self, file_path: Union[str, Path]) -> ECGRecord:
        """
        Load ECG record from file path (WFDB format)
        
        Args:
            file_path: Path to WFDB record file (without extension)
            
        Returns:
            ECGRecord object
        """
        file_path = Path(file_path)
        
        # Load record
        record = wfdb.rdrecord(str(file_path.with_suffix('')))
        
        return ECGRecord(
            record_id=file_path.stem,
            signal=record.p_signal,
            sampling_rate=record.fs,
            metadata={
                'lead_names': record.sig_name,
                'length_samples': record.sig_len,
                'file_path': str(file_path)
            }
        )
    
    def get_record_info(self, record: ECGRecord) -> Dict:
        """
        Get summary information about a record
        
        Args:
            record: ECGRecord object
            
        Returns:
            Dictionary with record information
        """
        return {
            'record_id': record.record_id,
            'duration_seconds': record.signal.shape[0] / record.sampling_rate,
            'num_leads': record.signal.shape[1] if len(record.signal.shape) > 1 else 1,
            'num_samples': record.signal.shape[0],
            'sampling_rate': record.sampling_rate,
            'has_annotations': record.annotations is not None,
            'unique_labels': list(set(record.annotation_symbols)) if record.annotation_symbols else [],
            'metadata': record.metadata
        }


if __name__ == "__main__":
    # Test loader
    loader = ECGLoader()
    
    # Load one MIT-BIH record
    record = loader.load_mit_bih_record(100, leads=[0, 1])
    print(f"Loaded: {record.record_id}")
    print(f"Signal shape: {record.signal.shape}")
    print(f"Sampling rate: {record.sampling_rate} Hz")
    print(f"Info: {loader.get_record_info(record)}")
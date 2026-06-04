"""
Unit tests for data loader module
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import wfdb

from src.data.loader import ECGLoader, ECGRecord


class TestECGLoader:
    """Test suite for ECGLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create ECGLoader instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ECGLoader(data_dir=tmpdir)
            yield loader
            
    def test_initialization(self, loader):
        """Test loader initialization"""
        assert loader.data_dir is not None
        assert loader.data_dir.exists()
        
    def test_load_mit_bih_record_mock(self, loader, monkeypatch):
        """Test loading MIT-BIH record (mocked)"""
        # Mock wfdb.rdrecord
        class MockRecord:
            def __init__(self, *args, **kwargs):
                self.p_signal = np.random.randn(3600, 2)
                self.fs = 360
                self.sig_name = ['MLII', 'V5']
                self.sig_len = 3600
                
        monkeypatch.setattr(wfdb, 'rdrecord', MockRecord)
        
        # Mock wfdb.rdann
        class MockAnnotation:
            def __init__(self, *args, **kwargs):
                self.sample = np.array([100, 500, 900])
                self.symbol = [1, 1, 2]
                
        monkeypatch.setattr(wfdb, 'rdann', MockAnnotation)
        
        # Test loading
        record = loader.load_mit_bih_record(100)
        
        assert record is not None
        assert record.record_id == "mitbih_100"
        assert record.sampling_rate == 360
        assert record.signal.shape[0] == 3600
        
    def test_load_mit_bih_record_invalid(self, loader):
        """Test loading invalid MIT-BIH record"""
        with pytest.raises(Exception):
            loader.load_mit_bih_record(99999)
            
    def test_load_multiple_records(self, loader, monkeypatch):
        """Test loading multiple records"""
        # Mock wfdb.rdrecord
        class MockRecord:
            def __init__(self, *args, **kwargs):
                self.p_signal = np.random.randn(3600, 2)
                self.fs = 360
                self.sig_name = ['MLII', 'V5']
                self.sig_len = 3600
                
        monkeypatch.setattr(wfdb, 'rdrecord', MockRecord)
        monkeypatch.setattr(wfdb, 'rdann', lambda *args, **kwargs: None)
        
        records = loader.load_mit_bih_records([100, 101, 102], leads=[0])
        
        assert len(records) == 3
        for record in records:
            assert record.signal.shape[1] == 1
            
    def test_get_record_info(self, loader):
        """Test getting record info"""
        # Create mock record
        record = ECGRecord(
            record_id="test_001",
            signal=np.random.randn(3600, 2),
            sampling_rate=360,
            metadata={'test': 'value'}
        )
        
        info = loader.get_record_info(record)
        
        assert info['record_id'] == "test_001"
        assert info['duration_seconds'] == 10.0  # 3600/360
        assert info['num_leads'] == 2
        assert info['sampling_rate'] == 360
        assert info['has_annotations'] is False
        
    def test_mitbih_symbols_mapping(self, loader):
        """Test MIT-BIH symbol mapping"""
        assert loader.MITBIH_SYMBOLS[1] == 'Normal'
        assert loader.MITBIH_SYMBOLS[2] == 'PVC (Premature Ventricular Contraction)'
        assert loader.MITBIH_SYMBOLS[11] == 'AFib (Atrial Fibrillation)'


class TestECGRecord:
    """Test suite for ECGRecord dataclass"""
    
    def test_ecg_record_creation(self):
        """Test creating ECG record"""
        record = ECGRecord(
            record_id="test_001",
            signal=np.random.randn(1000, 1),
            sampling_rate=360,
            annotations=np.array([100, 200, 300]),
            annotation_symbols=['Normal', 'Normal', 'PVC']
        )
        
        assert record.record_id == "test_001"
        assert record.signal.shape == (1000, 1)
        assert record.sampling_rate == 360
        assert len(record.annotations) == 3
        
    def test_ecg_record_with_metadata(self):
        """Test ECG record with metadata"""
        record = ECGRecord(
            record_id="test_002",
            signal=np.random.randn(1000, 1),
            sampling_rate=360,
            metadata={'patient_id': 'P001', 'age': 65}
        )
        
        assert record.metadata['patient_id'] == 'P001'
        assert record.metadata['age'] == 65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
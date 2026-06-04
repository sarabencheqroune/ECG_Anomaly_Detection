"""
Download ECG datasets from PhysioNet
Supports MIT-BIH Arrhythmia Database and PTB-XL
"""

import os
import wfdb
import shutil
import requests
from pathlib import Path
from typing import Optional, List, Dict
import logging
from tqdm import tqdm
import zipfile
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGDownloader:
    """Download and manage ECG datasets from PhysioNet"""
    
    # Dataset URLs
    DATASETS = {
        'mit-bih': {
            'url': 'https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip',
            'records': [f"{i:03d}" for i in range(100, 235) if i not in [102, 104, 107, 217]],
            'description': 'MIT-BIH Arrhythmia Database (48 recordings)'
        },
        'ptb-xl': {
            'url': 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip',
            'records': '21,000+ recordings',
            'description': 'PTB-XL ECG Dataset (21,837 clinical 12-lead ECGs)'
        }
    }
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize downloader
        
        Args:
            data_dir: Root directory for raw data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_mit_bih(self, force: bool = False) -> Path:
        """
        Download MIT-BIH Arrhythmia Database
        
        Args:
            force: Force re-download if already exists
            
        Returns:
            Path to downloaded dataset
        """
        dataset_dir = self.data_dir / "mit-bih"
        
        if dataset_dir.exists() and not force:
            logger.info(f"MIT-BIH already exists at {dataset_dir}")
            return dataset_dir
            
        logger.info("Downloading MIT-BIH Arrhythmia Database...")
        
        # Download zip file
        zip_path = self.data_dir / "mit-bih.zip"
        self._download_file(self.DATASETS['mit-bih']['url'], zip_path)
        
        # Extract
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        # Clean up zip
        zip_path.unlink()
        
        # Verify download
        records = self.DATASETS['mit-bih']['records']
        missing = []
        
        for record in records[:5]:  # Check first 5 records
            record_path = dataset_dir / f"{record}"
            if not (dataset_dir / f"{record}.dat").exists():
                missing.append(record)
                
        if missing:
            logger.warning(f"Missing records: {missing}")
        else:
            logger.info(f"Successfully downloaded MIT-BIH with {len(records)} recordings")
            
        return dataset_dir
    
    def download_ptb_xl(self, force: bool = False) -> Path:
        """
        Download PTB-XL dataset
        
        Args:
            force: Force re-download if already exists
            
        Returns:
            Path to downloaded dataset
        """
        dataset_dir = self.data_dir / "ptb-xl"
        
        if dataset_dir.exists() and not force:
            logger.info(f"PTB-XL already exists at {dataset_dir}")
            return dataset_dir
            
        logger.info("Downloading PTB-XL dataset (this may take a while)...")
        
        # Download zip file
        zip_path = self.data_dir / "ptb-xl.zip"
        self._download_file(self.DATASETS['ptb-xl']['url'], zip_path, show_progress=True)
        
        # Extract
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        # Clean up
        zip_path.unlink()
        
        # Find the actual data directory (may be nested)
        extracted_contents = list(dataset_dir.iterdir())
        if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
            # Move contents up one level
            temp_dir = extracted_contents[0]
            for item in temp_dir.iterdir():
                shutil.move(str(item), str(dataset_dir / item.name))
            temp_dir.rmdir()
            
        logger.info(f"Successfully downloaded PTB-XL to {dataset_dir}")
        return dataset_dir
    
    def download_all(self, datasets: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download multiple datasets
        
        Args:
            datasets: List of dataset names ('mit-bih', 'ptb-xl'), or None for all
            
        Returns:
            Dictionary mapping dataset names to paths
        """
        if datasets is None:
            datasets = list(self.DATASETS.keys())
            
        results = {}
        
        for dataset in datasets:
            if dataset == 'mit-bih':
                results[dataset] = self.download_mit_bih()
            elif dataset == 'ptb-xl':
                results[dataset] = self.download_ptb_xl()
            else:
                logger.warning(f"Unknown dataset: {dataset}")
                
        return results
    
    def _download_file(self, url: str, dest: Path, show_progress: bool = True):
        """
        Download a file with progress bar
        
        Args:
            url: URL to download
            dest: Destination path
            show_progress: Show progress bar
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            if show_progress:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        logger.info(f"Downloaded {url} to {dest}")
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about available datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        for name, meta in self.DATASETS.items():
            dataset_path = self.data_dir / name
            info[name] = {
                'description': meta['description'],
                'path': str(dataset_path),
                'exists': dataset_path.exists(),
                'size_mb': self._get_directory_size(dataset_path) if dataset_path.exists() else 0
            }
            
        return info
    
    def _get_directory_size(self, path: Path) -> float:
        """Calculate directory size in MB"""
        total = 0
        for file in path.rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total / (1024 * 1024)


if __name__ == "__main__":
    # Test downloader
    downloader = ECGDownloader()
    
    # Download MIT-BIH only
    mitbih_path = downloader.download_mit_bih()
    print(f"MIT-BIH downloaded to: {mitbih_path}")
    
    # Get info
    info = downloader.get_dataset_info()
    print(json.dumps(info, indent=2))
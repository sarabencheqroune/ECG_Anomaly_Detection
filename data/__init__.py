"""
Data management module for ECG datasets
"""

from pathlib import Path

# Define data directories
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
SPLITS_DIR = DATA_DIR / "splits"

# Create directories if they don't exist
for dir_path in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, SPLITS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Subdirectories
BEATS_DIR = PROCESSED_DIR / "beats_npy"
BEATS_DIR.mkdir(exist_ok=True)

__all__ = [
    'DATA_DIR',
    'RAW_DIR',
    'PROCESSED_DIR', 
    'CACHE_DIR',
    'SPLITS_DIR',
    'BEATS_DIR'
]
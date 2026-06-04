#!/usr/bin/env python
"""
Download ECG datasets from PhysioNet
Usage: python download_datasets.py --datasets mit-bih ptb-xl --data-dir ./data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import ECGDownloader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download ECG datasets from PhysioNet')
    
    parser.add_argument('--datasets', nargs='+', 
                       default=['mit-bih', 'ptb-xl'],
                       choices=['mit-bih', 'ptb-xl'],
                       help='Datasets to download')
    
    parser.add_argument('--data-dir', type=str, default='./data/raw',
                       help='Directory to store raw data')
    
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if exists')
    
    parser.add_argument('--list', action='store_true',
                       help='List available datasets and exit')
    
    return parser.parse_args()


def list_datasets():
    """List available datasets information"""
    downloader = ECGDownloader()
    info = downloader.get_dataset_info()
    
    print("\n" + "=" * 60)
    print("Available ECG Datasets")
    print("=" * 60)
    
    for name, meta in info.items():
        status = "✓ Downloaded" if meta['exists'] else "✗ Not downloaded"
        size_mb = f"{meta['size_mb']:.2f} MB" if meta['size_mb'] > 0 else "N/A"
        
        print(f"\n{name.upper()}:")
        print(f"  Description: {meta['description']}")
        print(f"  Status: {status}")
        print(f"  Size: {size_mb}")
        print(f"  Path: {meta['path']}")
        
    print("\n" + "=" * 60)


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.list:
        list_datasets()
        return
    
    logger.info("=" * 60)
    logger.info("ECG Dataset Downloader")
    logger.info("=" * 60)
    
    downloader = ECGDownloader(data_dir=args.data_dir)
    
    for dataset in args.datasets:
        logger.info(f"\nDownloading {dataset}...")
        
        if dataset == 'mit-bih':
            downloader.download_mit_bih(force=args.force)
        elif dataset == 'ptb-xl':
            downloader.download_ptb_xl(force=args.force)
            
    # Show final status
    list_datasets()
    
    logger.info("\n✅ Download completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Generate patient-wise train/val/test splits for ECG datasets
Usage: python split_generator.py --data-dir ./data --seed 42
"""

import argparse
import numpy as np
from pathlib import Path
import json
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SplitGenerator:
    """Generate patient-wise data splits"""
    
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        
        self.splits_dir = self.data_dir / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_mitbih_splits(self, train_ratio: float = 0.7, 
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15) -> dict:
        """Generate splits for MIT-BIH dataset"""
        
        # MIT-BIH patient mapping (simplified - each record is a different patient)
        records = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116,
                   117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 208,
                   209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228,
                   230, 231, 232, 233, 234]
        
        patient_ids = [f"100{str(r).zfill(3)}" for r in records]
        
        # Shuffle patients
        shuffled = patient_ids.copy()
        np.random.shuffle(shuffled)
        
        # Split
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_patients = shuffled[:n_train]
        val_patients = shuffled[n_train:n_train + n_val]
        test_patients = shuffled[n_train + n_val:]
        
        splits = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        # Save splits
        for split_name, patients in splits.items():
            filepath = self.splits_dir / f"mitbih_{split_name}_patients.txt"
            with open(filepath, 'w') as f:
                f.write(f"# MIT-BIH {split_name.upper()} set\n")
                f.write(f"# Total patients: {len(patients)}\n\n")
                for patient in patients:
                    f.write(f"{patient}\n")
                    
        return splits
    
    def generate_ptbxl_splits(self, train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> dict:
        """Generate splits for PTB-XL dataset"""
        
        # In production, load actual patient IDs from PTB-XL database
        # Here we generate synthetic patient IDs
        np.random.seed(self.seed)
        
        # PTB-XL has ~18,885 unique patients
        n_patients = 18885
        patient_ids = [f"2{str(i).zfill(5)}" for i in range(1, n_patients + 1)]
        
        # Shuffle
        shuffled = patient_ids.copy()
        np.random.shuffle(shuffled)
        
        # Split
        n_train = int(n_patients * train_ratio)
        n_val = int(n_patients * val_ratio)
        
        train_patients = shuffled[:n_train]
        val_patients = shuffled[n_train:n_train + n_val]
        test_patients = shuffled[n_train + n_val:]
        
        splits = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        # Save splits
        for split_name, patients in splits.items():
            filepath = self.splits_dir / f"ptbxl_{split_name}_patients.txt"
            with open(filepath, 'w') as f:
                f.write(f"# PTB-XL {split_name.upper()} set\n")
                f.write(f"# Total patients: {len(patients)}\n\n")
                # Write first 1000 for brevity (in production, write all)
                for patient in patients[:1000]:
                    f.write(f"{patient}\n")
                if len(patients) > 1000:
                    f.write(f"\n# ... and {len(patients) - 1000} more patients\n")
                    
        return splits
    
    def generate_combined_splits(self) -> dict:
        """Generate combined train/val/test splits"""
        
        # Load individual splits
        mitbih_train = self._load_patients("mitbih_train_patients.txt")
        mitbih_val = self._load_patients("mitbih_val_patients.txt")
        mitbih_test = self._load_patients("mitbih_test_patients.txt")
        
        ptbxl_train = self._load_patients("ptbxl_train_patients.txt")
        ptbxl_val = self._load_patients("ptbxl_val_patients.txt")
        ptbxl_test = self._load_patients("ptbxl_test_patients.txt")
        
        splits = {
            'train': mitbih_train + ptbxl_train,
            'val': mitbih_val + ptbxl_val,
            'test': mitbih_test + ptbxl_test
        }
        
        # Save combined splits
        for split_name, patients in splits.items():
            filepath = self.splits_dir / f"{split_name}_patients.txt"
            with open(filepath, 'w') as f:
                f.write(f"# Combined {split_name.upper()} set\n")
                f.write(f"# Total patients: {len(patients)}\n")
                f.write(f"# MIT-BIH: {len(mitbih_train if split_name == 'train' else mitbih_val if split_name == 'val' else mitbih_test)}\n")
                f.write(f"# PTB-XL: {len(ptbxl_train if split_name == 'train' else ptbxl_val if split_name == 'val' else ptbxl_test)}\n\n")
                for patient in patients:
                    f.write(f"{patient}\n")
                    
        return splits
    
    def _load_patients(self, filename: str) -> list:
        """Load patient IDs from file"""
        filepath = self.splits_dir / filename
        if not filepath.exists():
            return []
        
        patients = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patients.append(line)
        return patients
    
    def generate_split_report(self) -> dict:
        """Generate report of split statistics"""
        report = {
            'dataset': {},
            'combined': {}
        }
        
        for dataset in ['mitbih', 'ptbxl']:
            report['dataset'][dataset] = {}
            for split in ['train', 'val', 'test']:
                patients = self._load_patients(f"{dataset}_{split}_patients.txt")
                report['dataset'][dataset][split] = len(patients)
                
        # Combined
        for split in ['train', 'val', 'test']:
            patients = self._load_patients(f"{split}_patients.txt")
            report['combined'][split] = len(patients)
            
        return report
    
    def run(self):
        """Generate all splits"""
        print("=" * 60)
        print("Generating Patient-Wise Data Splits")
        print("=" * 60)
        
        # Generate MIT-BIH splits
        print("\n1. Generating MIT-BIH splits...")
        mitbih_splits = self.generate_mitbih_splits()
        print(f"   Train: {len(mitbih_splits['train'])} patients")
        print(f"   Val: {len(mitbih_splits['val'])} patients")
        print(f"   Test: {len(mitbih_splits['test'])} patients")
        
        # Generate PTB-XL splits
        print("\n2. Generating PTB-XL splits...")
        ptbxl_splits = self.generate_ptbxl_splits()
        print(f"   Train: {len(ptbxl_splits['train'])} patients")
        print(f"   Val: {len(ptbxl_splits['val'])} patients")
        print(f"   Test: {len(ptbxl_splits['test'])} patients")
        
        # Generate combined splits
        print("\n3. Generating combined splits...")
        combined_splits = self.generate_combined_splits()
        print(f"   Train: {len(combined_splits['train'])} patients")
        print(f"   Val: {len(combined_splits['val'])} patients")
        print(f"   Test: {len(combined_splits['test'])} patients")
        
        # Generate report
        report = self.generate_split_report()
        
        # Save report
        report_path = self.splits_dir / "split_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n✅ Split report saved to {report_path}")
        
        return report


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data splits')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = SplitGenerator(args.data_dir, args.seed)
    report = generator.run()
    
    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)
    print(f"Dataset splits saved to: {generator.splits_dir}")
    print(f"Total combined patients: {report['combined']['train'] + report['combined']['val'] + report['combined']['test']}")
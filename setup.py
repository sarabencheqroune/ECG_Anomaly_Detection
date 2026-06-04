#!/usr/bin/env python
"""
Setup script for ECG Anomaly Detection package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ecg-anomaly-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time ECG anomaly detection with human-in-the-loop capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ecg-anomaly-detection",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "nvidia-cublas-cu11>=11.10.3.66",
            "nvidia-cudnn-cu11>=8.5.0.96",
        ],
        "ml": [
            "mlflow>=2.3.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "nvidia-cublas-cu11>=11.10.3.66",
            "nvidia-cudnn-cu11>=8.5.0.96",
            "mlflow>=2.3.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecg-train=scripts.train:main",
            "ecg-evaluate=scripts.evaluate:main",
            "ecg-dashboard=src.dashboard.app:main",
            "ecg-download=scripts.download_datasets:main",
            "ecg-preprocess=scripts.preprocess_all:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
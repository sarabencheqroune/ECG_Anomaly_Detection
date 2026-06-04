# ECG Anomaly Detection System

A real-time ECG anomaly detection system with Human-in-the-Loop (HITL) capabilities for cardiac arrhythmia detection.

==> ** Disclaimer**: This is a hobby/side project developed for learning purposes. It is **not** intended for clinical use. The system is a proof-of-concept and requires significant additional development, testing, and validation before any real-world deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Real-time Inference](#real-time-inference)
- [Dashboard](#dashboard)
- [HITL System](#hitl-system)
- [Testing](#testing)
- [Project Status](#project-status)
- [Known Issues & Limitations](#known-issues--limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Overview

This project implements a complete MLOps pipeline for detecting cardiac arrhythmias from ECG signals. It combines traditional signal processing techniques with modern deep learning approaches to identify:

- **Normal Sinus Rhythm** - Healthy heart rhythm
- **Atrial Fibrillation (AFib)** - Irregular rhythm, stroke risk
- **Premature Ventricular Contractions (PVC)** - Early heartbeats
- **Bradycardia** - Abnormally slow heart rate (<60 BPM)
- **Other Anomalies** - Unclassified abnormalities

The system features real-time ECG streaming, a WebSocket-based dashboard, and a sophisticated Human-in-the-Loop feedback system for continuous model improvement.

## Features

###  Data Pipeline
- Multi-dataset support (MIT-BIH, PTB-XL)
- Complete signal preprocessing (baseline removal, filtering, normalization)
- Beat segmentation with configurable windows (100ms pre, 420ms post)
- ECG-specific data augmentation (noise, baseline wander, time warp)

###  Feature Engineering
- **30+ HRV features** (time domain, frequency domain, non-linear)
- **Wavelet decomposition** (Daubechies, Symlets, Coiflets)
- **PQRST morphology** (intervals, amplitudes, areas, slopes)

###  Model Architectures
- **1D-CNN Classifier** - Supervised beat classification
- **Convolutional Autoencoder** - Unsupervised anomaly detection
- **Hybrid Model** - Combines classifier + autoencoder
- **LSTM/Transformer** - Sequence models for rhythm analysis

###  Training Infrastructure
- Mixed precision training (AMP)
- Multi-GPU support (DataParallel)
- Comprehensive callbacks (checkpointing, early stopping, TensorBoard)
- Multiple optimizers and schedulers

###  Real-time Inference
- Online R-peak detection (adaptive threshold)
- Multi-lead beat detection
- Asynchronous stream processing
- Batch inference optimization

###  Human-in-the-Loop
- Priority-based review queue (4 levels)
- Multiple active learning strategies
- Persistent feedback storage
- Redis backend for distributed deployment

###  Dashboard
- Real-time ECG waveform rendering (60 FPS)
- Live predictions with confidence calibration
- Heart rate monitoring
- Dark/light theme support
- WebSocket-based real-time updates


## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Languages** | Python 3.9+, JavaScript, HTML/CSS |
| **Deep Learning** | PyTorch 2.0+, TorchScript, ONNX |
| **Signal Processing** | SciPy, NeuroKit2, WFDB, Biosppy |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **Web Framework** | Flask, Flask-SocketIO, Eventlet |
| **Databases** | PostgreSQL, Redis, SQLite |
| **DevOps** | Docker, Docker Compose |

## Project Structure
ecg-anomaly-detection/
├── data/
│ ├── raw/                # Raw ECG datasets
│ ├── processed/          # Preprocessed beats
│ └── cache/              # Cache files
├── src/
│ ├── data/               # Data pipeline
│ │ ├── loader.py         # ECG data loading
│ │ ├── preprocessor.py   # Signal preprocessing
│ │ ├── segmenter.py      # Beat segmentation
│ │ └── augmenter.py      # Data augmentation
│ ├── features/           # Feature engineering
│ │ ├── rr_intervals.py   # HRV features
│ │ ├── wavelet.py        # Wavelet decomposition
│ │ ├── morphological.py  # PQRST morphology
│ │ └── fusion.py         # Feature fusion
│ ├── models/             # Model architectures
│ │ ├── cnn_classifier.py # 1D-CNN classifier
│ │ ├── autoencoder.py    # Convolutional autoencoder
│ │ ├── hybrid_model.py   # Hybrid model
│ │ └── losses.py         # Loss functions
│ ├── training/           # Training infrastructure
│ │ ├── trainer.py        # Main training loop
│ │ ├── callback.py       # Training callbacks
│ │ └── metrics.py        # Evaluation metrics
│ ├── inference/          # Inference pipeline
│ │ ├── beat_detector.py  # Online R-peak detection
│ │ ├── predictor.py      # Model inference wrapper
│ │ └── stream_processor.py # Real-time stream processing
│ ├── hitl/               # Human-in-the-Loop
│ │ ├── review_queue.py   # Priority-based queue
│ │ ├── active_learning.py # Sample selection
│ │ └── feedback_store.py # Feedback persistence
│ └── dashboard/          # Web dashboard
│ ├── app.py              # Flask application
│ ├── routes.py           # API endpoints
│ ├── socket_handlers.py  # WebSocket events
│ └── templates/
│ └── index.html          # Frontend dashboard
├── notebooks/            # Jupyter notebooks
│ ├── 01_explore_mit_bih.ipynb
│ ├── 02_preprocessing_pipeline.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_training.ipynb
│ ├── 05_error_analysis.ipynb
│ └── 06_hitl_simulation.ipynb
├── scripts/              # Utility scripts
│ ├── download_datasets.py # Dataset downloader
│ ├── preprocess_all.py   # Batch preprocessing
│ ├── train.py            # Training entrypoint
│ ├── evaluate.py         # Model evaluation
│ ├── export_model.py     # Model export
│ ├── simulate_stream.py  # Stream simulation
│ └── generate_reports.py # Report generation
├── tests/                # Unit tests
│ ├── test_data/
│ ├── test_features/
│ ├── test_models/
│ ├── test_training/
│ └── test_inference/
├── config/               # Configuration files
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── reports/              # Generated reports
├── docker-compose.yml    # Docker orchestration
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file



## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for training)
- 8GB RAM minimum, 16GB recommended
- 10GB free disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ecg-anomaly-detection.git
cd ecg-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# Download datasets
python scripts/download_datasets.py --datasets mit-bih ptb-xl

# Preprocess data
python scripts/preprocess_all.py --data-dir ./data --output-dir ./data/processed

# Train a model
python scripts/train.py --model hybrid --epochs 100 --batch-size 64

# Evaluate the model
python scripts/evaluate.py --checkpoint ./checkpoints/best_model.pth --model hybrid

# Run inference on a single beat
python -c "from src.inference.predictor import InferencePredictor; \
           predictor = InferencePredictor('./checkpoints/best_model.pth'); \
           result = predictor.predict(your_beat_array)"

# Start the web dashboard
python -m src.dashboard.app

# Or use Docker
docker-compose up -d


### Datasets

This project uses publicly available datasets from PhysioNet:

Dataset		Records		Sampling Rate	Duration
MIT-BIH		48		360 Hz		~30 min
PTB-XL		21,837		500 Hz		10 sec
# ECG_Anomaly_Detection

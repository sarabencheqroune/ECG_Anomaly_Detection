# Multi-stage build for ECG Anomaly Detection System

# ============================================
# Stage 1: Base image
# ============================================
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    wget \
    libpq-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Builder for dependencies
# ============================================
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 3: Final image
# ============================================
FROM base as final

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Copy requirements for validation
COPY requirements.txt .

# Make scripts executable
RUN chmod +x scripts/*.py

# Create necessary directories
RUN mkdir -p data/raw data/processed data/cache models logs checkpoints

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    DEBUG=False \
    LOG_LEVEL=INFO \
    CORS_ALLOWED_ORIGINS=http://localhost:5001

# Expose ports
EXPOSE 5000 8000 5555

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Entry point
ENTRYPOINT ["python"]

# Default command (can be overridden)
CMD ["-m", "src.dashboard.app"]

# ============================================
# Development image
# ============================================
FROM final as dev

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    pre-commit \
    ipython \
    jupyter

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ============================================
# GPU image (for CUDA support)
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5000

# Run with GPU support
CMD ["python3", "-m", "src.dashboard.app"]

# ============================================
# Production image (optimized)
# ============================================
FROM final as production

# Install production server
RUN pip install --no-cache-dir gunicorn gevent

# Copy optimized settings
COPY config/gunicorn.conf.py /app/

# Run with gunicorn
CMD ["gunicorn", "--config", "config/gunicorn.conf.py", "src.dashboard.app:app"]
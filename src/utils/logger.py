"""
Centralized logging configuration for the ECG project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class LoggerConfig:
    """Configuration for logger setup"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_dir: str = "./logs"
    log_to_file: bool = True
    log_to_console: bool = True
    max_file_size_mb: int = 100
    backup_count: int = 5


# Global logger instances
_loggers = {}
_default_config = LoggerConfig()


def setup_logger(name: str = "ecg_project",
                 config: Optional[LoggerConfig] = None,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup a logger with both console and file handlers
    
    Args:
        name: Logger name
        config: Logger configuration
        log_file: Specific log file name (overrides config)
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = _default_config
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    # Create formatter
    formatter = logging.Formatter(config.log_format, config.date_format)
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # File handler
    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
            
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also add a rotating file handler for large logs
        try:
            from logging.handlers import RotatingFileHandler
            rotating_handler = RotatingFileHandler(
                log_dir / f"{name}.log",
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count
            )
            rotating_handler.setFormatter(formatter)
            logger.addHandler(rotating_handler)
        except Exception as e:
            logger.warning(f"Could not setup rotating file handler: {e}")
            
    # Store logger instance
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "ecg_project") -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding contextual information
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
        
    def process(self, msg, kwargs):
        context = " | ".join([f"{k}={v}" for k, v in self.extra.items()])
        return f"[{context}] {msg}", kwargs


class MetricsLogger:
    """
    Specialized logger for tracking metrics during training
    """
    
    def __init__(self, log_dir: str = "./logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.logger = get_logger("metrics")
        
    def log_metric(self, name: str, value: float, 
                   step: int, epoch: int, **kwargs):
        """
        Log a single metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Epoch number
            **kwargs: Additional metadata
        """
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "value": value,
            "step": step,
            "epoch": epoch,
            **kwargs
        }
        
        # Write to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")
            
        # Also log to console
        self.logger.info(f"Metric: {name}={value:.4f} (epoch={epoch}, step={step})")
        
    def log_metrics(self, metrics: Dict[str, float], 
                    step: int, epoch: int, prefix: str = ""):
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metric name-value pairs
            step: Training step
            epoch: Epoch number
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.log_metric(full_name, value, step, epoch)
            
    def log_dict(self, data: Dict[str, Any], step: int, epoch: int):
        """
        Log a dictionary of arbitrary data
        
        Args:
            data: Dictionary to log
            step: Training step
            epoch: Epoch number
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "dict",
            "step": step,
            "epoch": epoch,
            "data": data
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def read_metrics(self, metric_name: Optional[str] = None) -> list:
        """
        Read logged metrics from file
        
        Args:
            metric_name: Filter by metric name (optional)
            
        Returns:
            List of metric entries
        """
        metrics = []
        
        if not self.metrics_file.exists():
            return metrics
            
        with open(self.metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if metric_name is None or entry.get("name") == metric_name:
                        metrics.append(entry)
                        
        return metrics


# Convenience function for quick logging
def log_important(message: str, level: str = "INFO"):
    """Log important messages with visual emphasis"""
    logger = get_logger()
    border = "=" * 80
    if level.upper() == "INFO":
        logger.info(border)
        logger.info(f"🌟 {message}")
        logger.info(border)
    elif level.upper() == "WARNING":
        logger.warning(border)
        logger.warning(f"⚠️ {message}")
        logger.warning(border)
    elif level.upper() == "ERROR":
        logger.error(border)
        logger.error(f"❌ {message}")
        logger.error(border)


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test", LoggerConfig(log_level="DEBUG"))
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test metrics logger
    metrics_logger = MetricsLogger()
    metrics_logger.log_metric("accuracy", 0.95, step=100, epoch=10)
    metrics_logger.log_metrics({"loss": 0.1, "f1": 0.94}, step=100, epoch=10)
    
    # Test important message
    log_important("Training completed successfully!")
    
    print("Logger test completed")
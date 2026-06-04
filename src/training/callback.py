"""
Training callbacks for monitoring and control
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class"""
    
    def on_train_start(self):
        """Called at the start of training"""
        pass
    
    def on_train_end(self):
        """Called at the end of training"""
        pass
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_start(self, batch: int):
        """Called at the start of each batch"""
        pass
    
    def on_batch_end(self, batch: int, loss: float):
        """Called at the end of each batch"""
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = True,
                 save_frequency: int = 1,
                 filename_format: str = 'checkpoint_epoch_{epoch}.pth'):
        """
        Initialize model checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_weights_only: Save only model weights (not full checkpoint)
            save_frequency: Save every N epochs
            filename_format: Format for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_frequency = save_frequency
        self.filename_format = filename_format
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Save checkpoint at epoch end"""
        
        # Check if we should save this epoch
        if (epoch + 1) % self.save_frequency != 0:
            return
            
        # Get current metric
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            # Save anyway if no metric to monitor
            self._save_checkpoint(epoch, metrics, is_best=False)
            return
            
        # Check if this is the best model
        is_best = False
        
        if self.mode == 'min':
            if current_metric < self.best_metric - 1e-8:
                self.best_metric = current_metric
                self.best_epoch = epoch
                is_best = True
        else:
            if current_metric > self.best_metric + 1e-8:
                self.best_metric = current_metric
                self.best_epoch = epoch
                is_best = True
                
        # Save checkpoint
        if not self.save_best_only or is_best:
            self._save_checkpoint(epoch, metrics, is_best)
            
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        """Save checkpoint to disk"""
        filename = self.filename_format.format(epoch=epoch + 1)
        filepath = self.checkpoint_dir / filename
        
        # In practice, you'd need access to the model and optimizer
        # This is a placeholder - actual implementation requires trainer reference
        logger.info(f"Saving checkpoint to {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            logger.info(f"Saving best model to {best_path}")


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting
    """
    
    def __init__(self,
                 patience: int = 10,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 min_delta: float = 1e-4,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            monitor: Metric to monitor
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore model to best weights
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_metric = None
        self.best_weights = None
        self.stop_training = False
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Check if training should stop"""
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            return
            
        if self.best_metric is None:
            self.best_metric = current_metric
            return
            
        # Check for improvement
        if self.mode == 'min':
            improved = current_metric < self.best_metric - self.min_delta
        else:
            improved = current_metric > self.best_metric + self.min_delta
            
        if improved:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stop_training = True
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                

class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback
    """
    
    def __init__(self, scheduler, monitor: Optional[str] = None):
        """
        Initialize learning rate scheduler
        
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.monitor = monitor
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Step scheduler at epoch end"""
        if self.monitor and self.monitor in metrics:
            self.scheduler.step(metrics[self.monitor])
        else:
            self.scheduler.step()
            
    def on_batch_end(self, batch: int, loss: float):
        """Step scheduler at batch end (for OneCycleLR)"""
        if hasattr(self.scheduler, 'step') and not self.monitor:
            self.scheduler.step()


class TensorBoardLogger(Callback):
    """
    TensorBoard logging callback
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.log_dir = log_dir
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            
    def on_train_start(self):
        """Log at training start"""
        if self.writer:
            self.writer.add_text('Training', 'Started', 0)
            
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Log metrics at epoch end"""
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)
                    
    def on_batch_end(self, batch: int, loss: float):
        """Log batch metrics"""
        if self.writer:
            self.writer.add_scalar('batch_loss', loss, batch)
            
    def on_train_end(self):
        """Close writer at training end"""
        if self.writer:
            self.writer.close()
            
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """Log histogram"""
        if self.writer:
            self.writer.add_histogram(name, values, step)
            
    def log_graph(self, model, input_tensor):
        """Log model graph"""
        if self.writer:
            self.writer.add_graph(model, input_tensor)


class ProgressBar(Callback):
    """
    Progress bar callback for training monitoring
    """
    
    def __init__(self, total_epochs: int, update_frequency: int = 10):
        """
        Initialize progress bar
        
        Args:
            total_epochs: Total number of epochs
            update_frequency: Update every N batches
        """
        self.total_epochs = total_epochs
        self.update_frequency = update_frequency
        self.current_epoch = 0
        self.current_loss = 0.0
        
    def on_epoch_start(self, epoch: int):
        """Initialize progress bar for epoch"""
        self.current_epoch = epoch
        self.current_loss = 0.0
        
    def on_batch_end(self, batch: int, loss: float):
        """Update progress bar"""
        self.current_loss = loss
        
        if batch % self.update_frequency == 0:
            logger.info(f"Epoch {self.current_epoch + 1}/{self.total_epochs}, "
                       f"Batch {batch}, Loss: {loss:.4f}")
            
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Log epoch summary"""
        loss = metrics.get('loss', self.current_loss)
        accuracy = metrics.get('accuracy', None)
        
        msg = f"Epoch {epoch + 1}/{self.total_epochs} completed. Loss: {loss:.4f}"
        if accuracy:
            msg += f", Accuracy: {accuracy:.4f}"
            
        logger.info(msg)


class GradientMonitor(Callback):
    """
    Monitor gradient norms for debugging
    """
    
    def __init__(self, model: torch.nn.Module, log_frequency: int = 100):
        """
        Initialize gradient monitor
        
        Args:
            model: PyTorch model
            log_frequency: Log every N batches
        """
        self.model = model
        self.log_frequency = log_frequency
        self.gradient_norms = []
        
    def on_batch_end(self, batch: int, loss: float):
        """Monitor gradients after backward pass"""
        if batch % self.log_frequency == 0:
            total_norm = 0.0
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
            total_norm = total_norm ** 0.5
            self.gradient_norms.append(total_norm)
            
            logger.info(f"Batch {batch}, Gradient Norm: {total_norm:.6f}")
            
            # Warn about exploding/vanishing gradients
            if total_norm > 10:
                logger.warning(f"Exploding gradients detected: {total_norm:.2f}")
            elif total_norm < 1e-6:
                logger.warning(f"Vanishing gradients detected: {total_norm:.2e}")
                
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Log gradient statistics"""
        if self.gradient_norms:
            logger.info(f"Gradient stats - Mean: {np.mean(self.gradient_norms):.6f}, "
                       f"Std: {np.std(self.gradient_norms):.6f}, "
                       f"Max: {np.max(self.gradient_norms):.6f}")


class MetricsLogger(Callback):
    """
    Log metrics to file
    """
    
    def __init__(self, log_file: str):
        """
        Initialize metrics logger
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Log metrics to file"""
        entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(entry)
        
        # Save to JSON
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def get_history(self) -> List[Dict]:
        """Get all logged metrics"""
        return self.metrics_history


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when metric plateaus
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer,
                 factor: float = 0.5,
                 patience: int = 5,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 min_lr: float = 1e-7):
        """
        Initialize ReduceLROnPlateau callback
        
        Args:
            optimizer: PyTorch optimizer
            factor: Factor to reduce LR by
            patience: Number of epochs with no improvement
            monitor: Metric to monitor
            mode: 'min' or 'max'
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_lr = min_lr
        
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Check if LR should be reduced"""
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            return
            
        # Check for improvement
        if self.mode == 'min':
            improved = current_metric < self.best_metric - 1e-4
        else:
            improved = current_metric > self.best_metric + 1e-4
            
        if improved:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                logger.info(f"Reducing learning rate to {new_lr:.6f}")
                self.counter = 0
                
                
class TimeStopper(Callback):
    """
    Stop training after a certain amount of time
    """
    
    def __init__(self, max_time_seconds: int = 3600):
        """
        Initialize time stopper
        
        Args:
            max_time_seconds: Maximum training time in seconds
        """
        self.max_time = max_time_seconds
        self.start_time = None
        self.stop_training = False
        
    def on_train_start(self):
        """Record start time"""
        import time
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """Check if time limit exceeded"""
        import time
        elapsed = time.time() - self.start_time
        
        if elapsed >= self.max_time:
            self.stop_training = True
            logger.info(f"Time limit ({self.max_time}s) reached. Stopping training.")


class CheckpointRestore(Callback):
    """
    Restore from checkpoint on resume
    """
    
    def __init__(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Initialize checkpoint restore
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.load_optimizer = load_optimizer
        self.checkpoint = None
        
    def on_train_start(self):
        """Load checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint = torch.load(self.checkpoint_path)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            
    def get_checkpoint(self):
        """Return checkpoint data"""
        return self.checkpoint
"""
Main training loop with support for multiple model types and configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List
import logging
from datetime import datetime
import json

from .metrics import MetricsCalculator
from .callback import (
    Callback, ModelCheckpoint, EarlyStopping, 
    LearningRateScheduler, TensorBoardLogger, ProgressBar
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for ECG models
    
    Supports:
    - Classification models (CNN, LSTM, Transformer)
    - Autoencoders (unsupervised anomaly detection)
    - Hybrid models (classifier + autoencoder)
    - Multi-GPU training
    - Mixed precision training
    - Gradient accumulation
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        
        # Multi-GPU
        if torch.cuda.device_count() > 1 and config.get('multi_gpu', False):
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Metrics calculator
        self.metrics = MetricsCalculator(num_classes=config.get('num_classes', 5))
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = 0.0
        self.best_epoch = -1
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', False) else None
        
        # Callbacks
        self.callbacks = self._create_callbacks()
        
        # Logging
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on {self.device}")
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if opt_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                                  momentum=momentum, weight_decay=weight_decay)
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
            
        logger.info(f"Optimizer: {opt_name} (lr={lr}, weight_decay={weight_decay})")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config"""
        scheduler_name = self.config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'none':
            return None
            
        elif scheduler_name == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
            
        elif scheduler_name == 'cosine':
            T_max = self.config.get('epochs', 100)
            eta_min = self.config.get('eta_min', 0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, eta_min)
            
        elif scheduler_name == 'reduce_on_plateau':
            patience = self.config.get('patience', 10)
            factor = self.config.get('factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
            
        elif scheduler_name == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.get('learning_rate', 1e-3),
                epochs=self.config.get('epochs', 100),
                steps_per_epoch=len(self.train_loader)
            )
            
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
            
        logger.info(f"Scheduler: {scheduler_name}")
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on config"""
        from ..models.losses import get_loss
        
        loss_name = self.config.get('loss', 'cross_entropy')
        loss_kwargs = self.config.get('loss_kwargs', {})
        
        criterion = get_loss(loss_name, **loss_kwargs)
        logger.info(f"Loss function: {loss_name}")
        
        return criterion
    
    def _create_callbacks(self) -> List[Callback]:
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        if self.config.get('checkpoint', True):
            checkpoint = ModelCheckpoint(
                checkpoint_dir=str(self.checkpoint_dir),
                monitor=self.config.get('monitor_metric', 'val_loss'),
                mode=self.config.get('monitor_mode', 'min'),
                save_best_only=True,
                save_weights_only=True
            )
            callbacks.append(checkpoint)
            
        # Early stopping
        if self.config.get('early_stopping', False):
            early_stopping = EarlyStopping(
                patience=self.config.get('early_stopping_patience', 10),
                monitor=self.config.get('monitor_metric', 'val_loss'),
                mode=self.config.get('monitor_mode', 'min'),
                min_delta=self.config.get('min_delta', 1e-4)
            )
            callbacks.append(early_stopping)
            
        # Learning rate scheduler callback
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_callback = LearningRateScheduler(self.scheduler, monitor='val_loss')
            callbacks.append(lr_callback)
            
        # TensorBoard logger
        if self.config.get('tensorboard', True):
            tb_logger = TensorBoardLogger(log_dir=str(self.log_dir))
            callbacks.append(tb_logger)
            
        # Progress bar
        if self.config.get('progress_bar', True):
            progress_bar = ProgressBar()
            callbacks.append(progress_bar)
            
        return callbacks
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                labels = None
                
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(inputs)
                    loss = self._compute_loss(outputs, labels, inputs)
                    
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config['grad_clip'])
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                outputs = self._forward_pass(inputs)
                loss = self._compute_loss(outputs, labels, inputs)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config['grad_clip'])
                    
                self.optimizer.step()
                
            # Update metrics
            total_loss += loss.item()
            
            # Collect predictions for metrics
            if labels is not None and 'logits' in outputs:
                preds = torch.argmax(outputs['logits'], dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Step callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self.current_step, loss.item())
                
            self.current_step += 1
            
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        metrics = {}
        if all_predictions:
            metrics = self.metrics.compute_all(np.array(all_labels), np.array(all_predictions))
            
        metrics['loss'] = avg_loss
        
        # Step scheduler (non-plateau)
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
            
        return metrics
    
    def _forward_pass(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass based on model type"""
        outputs = {}
        
        # Check model type based on output signature
        if hasattr(self.model, 'forward_with_features'):
            # Hybrid model
            logits, recon, features = self.model.forward_with_features(inputs)
            outputs['logits'] = logits
            outputs['reconstruction'] = recon
            outputs['features'] = features
            
        elif hasattr(self.model, 'forward_reconstruction'):
            # Autoencoder
            outputs['reconstruction'] = self.model.forward_reconstruction(inputs)
            outputs['features'] = self.model.encode(inputs)
            
        else:
            # Standard classifier
            outputs['logits'] = self.model(inputs)
            
        return outputs
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                      labels: Optional[torch.Tensor],
                      inputs: torch.Tensor) -> torch.Tensor:
        """Compute loss based on model type"""
        
        if hasattr(self.criterion, 'forward'):
            # Custom loss function
            if 'logits' in outputs and labels is not None:
                if 'reconstruction' in outputs:
                    # Hybrid loss
                    return self.criterion(
                        outputs['logits'], outputs['reconstruction'],
                        inputs, labels
                    )
                else:
                    return self.criterion(outputs['logits'], labels)
                    
            elif 'reconstruction' in outputs:
                # Autoencoder loss
                return self.criterion(outputs['reconstruction'], inputs)
                
        else:
            # Standard loss
            if 'logits' in outputs and labels is not None:
                return self.criterion(outputs['logits'], labels)
            elif 'reconstruction' in outputs:
                return self.criterion(outputs['reconstruction'], inputs)
                
        raise ValueError("Cannot compute loss: model output and loss function mismatch")
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_reconstructions = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    labels = None
                    
                # Forward pass
                outputs = self._forward_pass(inputs)
                
                # Compute loss
                loss = self._compute_loss(outputs, labels, inputs)
                total_loss += loss.item()
                
                # Collect predictions
                if labels is not None and 'logits' in outputs:
                    preds = torch.argmax(outputs['logits'], dim=1)
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                # Collect reconstructions for autoencoder
                if 'reconstruction' in outputs:
                    recon_error = torch.mean((outputs['reconstruction'] - inputs) ** 2, 
                                            dim=(1, 2))
                    all_reconstructions.extend(recon_error.cpu().numpy())
                    
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {'loss': avg_loss}
        
        if all_predictions:
            metrics.update(self.metrics.compute_all(np.array(all_labels), 
                                                    np.array(all_predictions)))
            
        if all_reconstructions:
            metrics['reconstruction_error'] = np.mean(all_reconstructions)
            
        return metrics
    
    def train(self) -> Dict:
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Call on_train_start callbacks
        for callback in self.callbacks:
            callback.on_train_start()
            
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Call on_epoch_start callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch)
                
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_metrics.get('loss', 0):.4f}")
            logger.info(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
            
            if 'accuracy' in train_metrics:
                logger.info(f"  Train Acc: {train_metrics['accuracy']:.4f}")
                logger.info(f"  Val Acc: {val_metrics.get('accuracy', 0):.4f}")
                
            logger.info(f"  LR: {current_lr:.6f}")
            
            # Call on_epoch_end callbacks
            stop_training = False
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, val_metrics)
                if hasattr(callback, 'stop_training') and callback.stop_training:
                    stop_training = True
                    
            if stop_training:
                logger.info("Early stopping triggered")
                break
                
        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end()
            
        logger.info(f"Training completed. Best epoch: {self.best_epoch+1}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Loaded checkpoint from {filename} (epoch {self.current_epoch})")


if __name__ == "__main__":
    # Test trainer
    from torch.utils.data import TensorDataset
    
    # Dummy data
    X = torch.randn(1000, 1, 187)
    y = torch.randint(0, 5, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    # Dummy model
    from ..models.cnn_classifier import ECG1DCNN
    model = ECG1DCNN(input_dim=187, num_classes=5)
    
    # Config
    config = {
        'epochs': 2,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'loss': 'cross_entropy',
        'num_classes': 5
    }
    
    # Create trainer
    trainer = Trainer(model, loader, loader, config)
    history = trainer.train()
    print(f"Training completed! History keys: {history.keys()}")
"""
Training script for Hybrid Quantum-Classical Neural Network.

This script handles model training with comprehensive error handling,
including NaN detection, gradient explosion prevention, and checkpointing.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_preprocessing import create_dataloaders
from hybrid_model import create_hybrid_model, count_parameters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.

        Args:
            score: Current validation metric

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class GradientMonitor:
    """Monitor and handle gradient issues during training."""

    def __init__(
        self,
        clip_value: float = 1.0,
        explosion_threshold: float = 10.0
    ):
        """
        Initialize gradient monitor.

        Args:
            clip_value: Value to clip gradients
            explosion_threshold: Threshold for detecting gradient explosion
        """
        self.clip_value = clip_value
        self.explosion_threshold = explosion_threshold
        self.gradient_norms = []

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return gradient norm.

        Args:
            model: PyTorch model

        Returns:
            Gradient norm before clipping
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.clip_value
        )
        self.gradient_norms.append(grad_norm.item())
        return grad_norm.item()

    def check_explosion(self, grad_norm: float) -> bool:
        """
        Check if gradient has exploded.

        Args:
            grad_norm: Current gradient norm

        Returns:
            True if gradient has exploded
        """
        return grad_norm > self.explosion_threshold


class Trainer:
    """Trainer class for hybrid quantum-classical model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # OPTIMIZATION: Enable mixed precision training for faster computation
        self.use_amp = device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Automatic Mixed Precision (AMP) enabled for faster training")
        else:
            self.scaler = None

        # Setup loss function
        self.criterion = self._setup_loss_function()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup gradient monitor
        self.gradient_monitor = GradientMonitor(
            clip_value=config['training']['gradient_clipping']['max_norm'],
            explosion_threshold=config['regularization']['gradient_explosion']['threshold']
        )

        # Setup early stopping
        if config['training']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=config['training']['early_stopping']['patience'],
                min_delta=config['training']['early_stopping']['min_delta']
            )
        else:
            self.early_stopping = None

        # Setup tensorboard
        if config['logging']['tensorboard']:
            log_dir = config['logging']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration."""
        loss_type = self.config['training']['loss_function']

        if loss_type == 'cross_entropy':
            label_smoothing = self.config['regularization']['label_smoothing']
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler based on configuration."""
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']

        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config['factor']
            )
        else:
            return None

    def _check_nan_loss(self, loss: torch.Tensor) -> bool:
        """
        Check if loss is NaN and handle it.

        Args:
            loss: Loss tensor

        Returns:
            True if loss is NaN
        """
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN or Inf loss detected: {loss.item()}")
            return True
        return False

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        nan_count = 0
        processed_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            try:
                # OPTIMIZATION: Use automatic mixed precision for faster training
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        # Forward pass
                        output = self.model(data)
                        # Calculate loss
                        loss = self.criterion(output, target)
                    
                    # Check for NaN loss
                    if self._check_nan_loss(loss):
                        nan_count += 1
                        if nan_count > 5:
                            logger.error("Too many NaN losses, stopping training")
                            raise RuntimeError("Training failed due to NaN losses")
                        continue

                    # Backward pass with scaled gradients
                    self.scaler.scale(loss).backward()

                    # Clip gradients and check for explosion
                    if self.config['training']['gradient_clipping']['enabled']:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = self.gradient_monitor.clip_gradients(self.model)

                        if self.gradient_monitor.check_explosion(grad_norm):
                            logger.warning(f"Gradient explosion detected: {grad_norm:.2f}")
                            self.optimizer.zero_grad()
                            continue

                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training without AMP
                    # Forward pass
                    output = self.model(data)
                    # Calculate loss
                    loss = self.criterion(output, target)

                    # Check for NaN loss
                    if self._check_nan_loss(loss):
                        nan_count += 1
                        if nan_count > 5:
                            logger.error("Too many NaN losses, stopping training")
                            raise RuntimeError("Training failed due to NaN losses")
                        continue

                    # Backward pass
                    loss.backward()

                    # Clip gradients and check for explosion
                    if self.config['training']['gradient_clipping']['enabled']:
                        grad_norm = self.gradient_monitor.clip_gradients(self.model)

                        if self.gradient_monitor.check_explosion(grad_norm):
                            logger.warning(f"Gradient explosion detected: {grad_norm:.2f}")
                            self.optimizer.zero_grad()
                            continue

                    # Optimizer step
                    self.optimizer.step()

                # Update statistics
                processed_batches += 1
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%',
                    'processed': processed_batches
                })

                # Log to tensorboard
                if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                    if self.config['training']['gradient_clipping']['enabled']:
                        self.writer.add_scalar('Train/GradientNorm', grad_norm, global_step)

            except RuntimeError as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                if "out of memory" in str(e):
                    continue

        # Handle case where no batches were successfully processed
        if total == 0:
            logger.error(f"No batches were successfully processed in this epoch!")
            logger.error(f"Total batches attempted: {len(train_loader)}")
            logger.error(f"Successfully processed: {processed_batches}")
            logger.error(f"NaN count: {nan_count}")
            return float('inf'), 0.0

        avg_loss = total_loss / max(processed_batches, 1)
        accuracy = 100.0 * correct / total

        logger.info(f"Processed {processed_batches}/{len(train_loader)} batches successfully")
        return avg_loss, accuracyt / total

        return avg_loss, accuracy

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating", leave=False):
                data, target = data.to(self.device), target.to(self.device)

                try:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    if not self._check_nan_loss(loss):
                        total_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                except RuntimeError as e:
                    logger.error(f"Error in validation: {e}")
                    continue

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.config['training']['checkpoint']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save last checkpoint
        if self.config['training']['checkpoint']['save_last']:
            last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
            torch.save(checkpoint, last_path)
            logger.info(f"Saved last checkpoint to {last_path}")

        # Save best checkpoint
        if is_best and self.config['training']['checkpoint']['save_best']:
            best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    def plot_training_history(self):
        """Plot and save training history."""
        if not self.config['logging']['save_plots']:
            return

        plot_dir = self.config['logging']['plot_dir']
        os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot learning rate
        if self.history['learning_rates']:
            axes[1, 0].plot(self.history['learning_rates'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)

        # Plot gradient norms
        if self.gradient_monitor.gradient_norms:
            axes[1, 1].plot(self.gradient_monitor.gradient_norms)
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norms During Training')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training plots to {plot_path}")
        plt.close()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        start_epoch: int = 0
    ):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Starting epoch number
        """
        num_epochs = self.config['training']['epochs']

        logger.info("Starting training...")
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")

        try:
            for epoch in range(start_epoch, num_epochs):
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, epoch)

                # Validate
                val_loss, val_acc = self.validate(val_loader)

                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)

                # Log to console
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

                # Log to tensorboard
                if self.writer:
                    self.writer.add_scalar('Train/Loss', train_loss, epoch)
                    self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                    self.writer.add_scalar('Val/Loss', val_loss, epoch)
                    self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                    self.writer.add_scalar('LearningRate', current_lr, epoch)

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc

                # Save checkpoint
                self.save_checkpoint(epoch, val_loss, val_acc, is_best)

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_loss):
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Plot training history
            self.plot_training_history()

            # Close tensorboard writer
            if self.writer:
                self.writer.close()

            logger.info("Training completed!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")


def main():
    """Main training function."""
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, seed=seed)
    logger.info(f"Train batches: {len(train_loader)}, "
               f"Val batches: {len(val_loader)}, "
               f"Test batches: {len(test_loader)}")

    # Create model
    logger.info("Creating hybrid model...")
    model = create_hybrid_model(config, use_ensemble=False)

    # OPTIMIZATION: Compile model with PyTorch 2.0+ for faster execution
    try:
        if hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile for optimized performance...")
            # Only compile classical parts - quantum layer needs dynamic execution
            model.classical_model = torch.compile(model.classical_model, mode='reduce-overhead')
            logger.info("Model compilation successful!")
    except Exception as e:
        logger.warning(f"Model compilation not available or failed: {e}")

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, config, device)

    # Check for existing checkpoint
    checkpoint_dir = config['training']['checkpoint']['checkpoint_dir']
    last_checkpoint = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint):
        logger.info("Found existing checkpoint, resuming training...")
        trainer.load_checkpoint(last_checkpoint)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

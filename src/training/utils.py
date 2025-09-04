"""
Training Utilities

This module provides utility classes for training management.
"""

import torch
import os
import logging
from typing import Optional


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.logger = logging.getLogger(__name__)
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        
        return False


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(self, save_dir: str, save_best: bool = True, save_last: bool = True):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric = -float('inf')
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch: int, model, optimizer, scheduler, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }
        
        if self.save_last:
            last_path = os.path.join(self.save_dir, 'last_checkpoint.pth')
            torch.save(checkpoint, last_path)
        
        if is_best and self.save_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")

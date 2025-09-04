"""
Training Module

This module contains all training-related functionality:
- Training and validation loops
- Metric computation
- Model checkpointing
- Learning rate scheduling
"""

from .trainer import WildlifeTrainer
from .metrics import MetricTracker, ConfusionMatrixTracker
from .utils import EarlyStopping, ModelCheckpoint

__all__ = [
    'WildlifeTrainer',
    'MetricTracker',
    'ConfusionMatrixTracker', 
    'EarlyStopping',
    'ModelCheckpoint'
]

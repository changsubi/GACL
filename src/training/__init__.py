"""
Training Module

This module contains all training-related functionality:
- Training and validation loops
- Metric computation
- Calibration and threshold analysis
- Model checkpointing
- Learning rate scheduling
"""

from .trainer import WildlifeTrainer
from .metrics import MetricTracker, ConfusionMatrixTracker
from .calibration import (
    CalibrationAnalyzer,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_curve,
    threshold_analysis,
    temperature_scale,
    generate_synthetic_predictions,
)
from .utils import EarlyStopping, ModelCheckpoint

__all__ = [
    'WildlifeTrainer',
    'MetricTracker',
    'ConfusionMatrixTracker',
    'CalibrationAnalyzer',
    'expected_calibration_error',
    'maximum_calibration_error',
    'brier_score',
    'negative_log_likelihood',
    'reliability_curve',
    'threshold_analysis',
    'temperature_scale',
    'generate_synthetic_predictions',
    'EarlyStopping',
    'ModelCheckpoint'
]

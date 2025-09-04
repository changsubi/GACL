"""
Inference Module

This module contains inference-related functionality:
- Complete wildlife detection and classification pipeline
- Batch processing capabilities
- Visualization and result analysis
"""

from .pipeline import WildlifePipeline
from .predictor import WildlifePredictor

__all__ = [
    'WildlifePipeline',
    'WildlifePredictor'
]

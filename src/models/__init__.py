"""
Model Architecture Module

This module contains all model components for the GACL wildlife classification system:
- Multi-Dilated Convolutional Networks
- Graph Attention Transformer Encoders
- Vision and Text Encoders
- Contrastive Learning Loss Functions
- Main GACL Model
"""

from .backbone import MultiDilatedConvNet, GATEncoder
from .encoders import VisionEncoder, TextEncoder
from .losses import ParallelContrastiveLoss
from .gacl_model import GACLModel
from .detection import WildlifeDetector

__all__ = [
    'MultiDilatedConvNet',
    'GATEncoder', 
    'VisionEncoder',
    'TextEncoder',
    'ParallelContrastiveLoss',
    'GACLModel',
    'WildlifeDetector'
]

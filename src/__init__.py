"""
GACL Wildlife Classification Package

A comprehensive implementation of Graph Attention Contrastive Learning (GACL) 
for Korean wildlife species classification using multi-modal learning.

This package provides:
- Multi-Dilated Convolutional Networks for feature extraction
- Graph Attention Transformer Encoders for structural learning  
- Parallel Contrastive Learning for multi-modal alignment
- Two-stage detection and classification pipeline
"""

__version__ = "1.0.0"
__author__ = "SPHERE AX AILab"
__email__ = "yuncs@sphereax.com"

# Import main components
from .models import GACLModel, MultiDilatedConvNet, GATEncoder
from .data import KoreanWildlifeDataset, DataAugmentation
from .training import WildlifeTrainer
from .inference import WildlifePipeline
from .utils import create_dummy_dataset

__all__ = [
    'GACLModel',
    'MultiDilatedConvNet', 
    'GATEncoder',
    'KoreanWildlifeDataset',
    'DataAugmentation',
    'WildlifeTrainer',
    'WildlifePipeline',
    'create_dummy_dataset'
]

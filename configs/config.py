"""
Configuration Management for GACL Wildlife Classification

This module contains all configuration parameters for the GACL model,
including dataset settings, model hyperparameters, and training configurations.
"""

import torch
import os
from typing import List, Dict, Any


class Config:
    """
    Main configuration class for GACL Wildlife Classification.
    
    This class contains all the hyperparameters and settings used throughout
    the project, following the specifications from the research paper.
    """
    
    def __init__(self):
        # Dataset Configuration
        self.num_classes: int = 4
        self.class_names: List[str] = ['Wildboar', 'Goral', 'Deers', 'Other']
        self.image_size: int = 224
        self.batch_size: int = 16
        
        # Model Architecture Configuration
        # Multi-Dilated Convolution settings
        self.mdconv_input_channels: int = 3
        self.mdconv_output_channels: int = 256
        self.dilation_rates: List[int] = [1, 2, 4, 5]  # As specified in Table 1
        
        # Graph Attention Transformer settings
        self.gat_hidden_dim: int = 256
        self.gat_num_heads: int = 8
        self.gat_num_layers: int = 3
        self.patch_size: int = 32
        self.graph_k_neighbors: int = 5  # For K-means clustering
        
        # Pre-trained Model Configuration
        self.bert_model_name: str = 'bert-base-uncased'
        self.vit_model_name: str = 'google/vit-base-patch16-224'
        self.embedding_dim: int = 768
        
        # Contrastive Learning Configuration
        self.temperature: float = 0.07  # Temperature parameter for contrastive loss
        self.contrastive_weight: float = 0.5  # Weight for contrastive loss vs classification
        
        # Training Configuration
        self.num_epochs: int = 100
        self.learning_rate: float = 1e-4
        self.weight_decay: float = 1e-5
        self.mixup_alpha: float = 1.0  # Mixup augmentation parameter
        self.mixup_probability: float = 0.5  # Probability of applying mixup
        self.gradient_clip_norm: float = 1.0
        
        # Data Augmentation Configuration
        self.rotation_limit: int = 30
        self.shift_scale_rotate_prob: float = 0.5
        self.brightness_contrast_prob: float = 0.7
        self.hsv_prob: float = 0.7
        self.noise_prob: float = 0.2
        self.blur_prob: float = 0.1
        
        # Hardware Configuration
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers: int = 4
        self.pin_memory: bool = True
        
        # Path Configuration
        self.data_root: str = './korean_wildlife_dataset'
        self.model_save_path: str = './models'
        self.results_path: str = './results'
        self.logs_path: str = './logs'
        
        # Detection Configuration (Stage 1)
        self.detector_model_path: str = 'yolov5s.pt'
        self.detection_confidence_threshold: float = 0.5
        self.classification_confidence_threshold: float = 0.8
        
        # Text Prompts Configuration
        self.frame_prompts: Dict[str, str] = {
            'Wildboar': "A large brown wild boar with coarse fur, stocky body, and prominent snout standing in forest habitat",
            'Goral': "A small gray-brown goral with short curved horns, compact body, and agile stance on rocky terrain",
            'Deers': "A graceful deer with slender legs, alert posture, and distinctive antlers or ears in woodland setting",
            'Other': "Various small to medium mammals including raccoon dogs, badgers, and birds in natural habitat"
        }
        
        self.object_prompts: Dict[str, str] = {
            'Wildboar': "stocky body, coarse fur, prominent snout",
            'Goral': "curved horns, compact build, gray fur",
            'Deers': "slender legs, antlers, alert ears",
            'Other': "diverse mammals, varied features"
        }
        
        # Validation Configuration
        self.validation_frequency: int = 1  # Validate every N epochs
        self.save_checkpoint_frequency: int = 10  # Save checkpoint every N epochs
        self.early_stopping_patience: int = 20  # Stop if no improvement for N epochs
        
        # Logging Configuration
        self.log_frequency: int = 100  # Log training stats every N batches
        self.tensorboard_log_dir: str = './logs/tensorboard'
        
    def get_text_prompts_by_index(self) -> Dict[int, Dict[str, str]]:
        """
        Get text prompts indexed by class index for easier access during training.
        
        Returns:
            Dict mapping class indices to frame and object prompts
        """
        prompts = {}
        for idx, class_name in enumerate(self.class_names):
            prompts[idx] = {
                'frame': self.frame_prompts[class_name],
                'object': self.object_prompts[class_name]
            }
        return prompts
    
    def create_directories(self) -> None:
        """Create necessary directories for the project."""
        directories = [
            self.model_save_path,
            self.results_path, 
            self.logs_path,
            self.tensorboard_log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_config(self) -> None:
        """Validate configuration parameters."""
        assert self.num_classes == len(self.class_names), \
            "Number of classes must match length of class names"
        
        assert len(self.dilation_rates) == 4, \
            "Must have exactly 4 dilation rates for Multi-Dilated ConvNet"
        
        assert self.embedding_dim > 0, "Embedding dimension must be positive"
        assert self.temperature > 0, "Temperature must be positive"
        assert 0 <= self.mixup_probability <= 1, "Mixup probability must be between 0 and 1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for saving/logging."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, torch.device):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        config_str = "GACL Wildlife Classification Configuration:\n"
        config_str += f"  Classes: {self.class_names}\n"
        config_str += f"  Image Size: {self.image_size}x{self.image_size}\n"
        config_str += f"  Batch Size: {self.batch_size}\n"
        config_str += f"  Learning Rate: {self.learning_rate}\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Data Root: {self.data_root}\n"
        return config_str


# Global configuration instance
config = Config()

# Validate configuration on import
config.validate_config()

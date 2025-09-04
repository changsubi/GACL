"""
Data Augmentation Module

This module implements various data augmentation techniques for Korean wildlife
images, including geometric transformations, photometric changes, and mixup augmentation.
"""

import torch
import numpy as np
import albumentations as A
from typing import Tuple, Optional
import logging

from ..configs.config import config


class DataAugmentation:
    """
    Comprehensive data augmentation pipeline for wildlife images.
    
    This class implements various augmentation strategies including:
    - Geometric transformations (rotation, scaling, flipping)
    - Photometric transformations (HSV, brightness/contrast)
    - Color space transformations (RGB adjustments)
    - Noise and blur effects
    - Mixup augmentation for training
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize augmentation pipelines
        self._setup_geometric_transforms()
        self._setup_photometric_transforms()
        self._setup_combined_transforms()
        
        self.logger.info("Data augmentation pipelines initialized")
    
    def _setup_geometric_transforms(self) -> None:
        """Setup geometric transformation pipeline."""
        self.geometric_transforms = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=config.rotation_limit,
                p=config.shift_scale_rotate_prob
            ),
        ])
    
    def _setup_photometric_transforms(self) -> None:
        """Setup photometric transformation pipelines."""
        # HSV color space transformations
        self.hsv_transforms = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=config.hsv_prob
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=config.brightness_contrast_prob
            ),
        ])
        
        # LAB-like color space simulation through RGB adjustments
        self.lab_like_transforms = A.Compose([
            A.RGBShift(
                r_shift_limit=25, 
                g_shift_limit=25, 
                b_shift_limit=25, 
                p=0.5
            ),
            A.ChannelShuffle(p=0.1),
        ])
    
    def _setup_combined_transforms(self) -> None:
        """Setup combined transformation pipelines for training and validation."""
        # Training augmentation pipeline
        self.train_transform = A.Compose([
            self.geometric_transforms,
            A.OneOf([
                self.hsv_transforms, 
                self.lab_like_transforms
            ], p=0.8),
            A.GaussNoise(p=config.noise_prob),
            A.Blur(blur_limit=3, p=config.blur_prob),
        ])
        
        # Validation augmentation pipeline (minimal)
        self.val_transform = A.Compose([
            A.Resize(config.image_size, config.image_size),
        ])
        
        # Test augmentation pipeline (same as validation)
        self.test_transform = self.val_transform
    
    def get_train_transform(self) -> A.Compose:
        """Get training augmentation pipeline."""
        return self.train_transform
    
    def get_val_transform(self) -> A.Compose:
        """Get validation augmentation pipeline."""
        return self.val_transform
    
    def get_test_transform(self) -> A.Compose:
        """Get test augmentation pipeline."""
        return self.test_transform
    
    @staticmethod
    def mixup_data(
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup augmentation to input data and labels.
        
        Mixup creates virtual training examples by combining pairs of examples
        and their labels. This is particularly effective for improving model
        generalization and reducing overfitting.
        
        Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization"
        
        Args:
            x (torch.Tensor): Input images tensor [B, C, H, W]
            y (torch.Tensor): Labels tensor [B] or [B, 1]
            alpha (float, optional): Beta distribution parameter. If None, uses config value
            
        Returns:
            Tuple of:
                - mixed_x (torch.Tensor): Mixed input images
                - y_a (torch.Tensor): First set of labels
                - y_b (torch.Tensor): Second set of labels  
                - lam (float): Mixing parameter
        """
        if alpha is None:
            alpha = config.mixup_alpha
            
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Get both label sets for loss calculation
        y_a = y
        y_b = y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(
        pred: torch.Tensor, 
        y_a: torch.Tensor, 
        y_b: torch.Tensor, 
        lam: float,
        criterion
    ) -> torch.Tensor:
        """
        Calculate loss for mixup augmented data.
        
        Args:
            pred (torch.Tensor): Model predictions
            y_a (torch.Tensor): First set of labels
            y_b (torch.Tensor): Second set of labels
            lam (float): Mixing parameter
            criterion: Loss function to use
            
        Returns:
            torch.Tensor: Mixed loss value
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def apply_test_time_augmentation(
        self, 
        image: np.ndarray, 
        num_augmentations: int = 5
    ) -> list:
        """
        Apply test-time augmentation to improve inference robustness.
        
        Args:
            image (np.ndarray): Input image
            num_augmentations (int): Number of augmented versions to create
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Original image (just resized)
        original = self.val_transform(image=image)['image']
        augmented_images.append(original)
        
        # Create augmented versions
        light_augment = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ])
        
        for _ in range(num_augmentations - 1):
            augmented = light_augment(image=image)['image']
            augmented_images.append(augmented)
        
        return augmented_images
    
    def visualize_augmentations(
        self, 
        image: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize different augmentation effects on a sample image.
        
        Args:
            image (np.ndarray): Input image to augment
            save_path (str, optional): Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        augmentations = {
            'Original': self.val_transform,
            'Geometric': self.geometric_transforms,
            'HSV': self.hsv_transforms,
            'RGB Shift': self.lab_like_transforms,
            'Full Training': self.train_transform
        }
        
        fig, axes = plt.subplots(1, len(augmentations), figsize=(20, 4))
        
        for idx, (name, transform) in enumerate(augmentations.items()):
            try:
                augmented = transform(image=image)['image']
                axes[idx].imshow(augmented)
                axes[idx].set_title(name)
                axes[idx].axis('off')
            except Exception as e:
                self.logger.error(f"Failed to apply {name} augmentation: {str(e)}")
                axes[idx].imshow(image)
                axes[idx].set_title(f"{name} (Failed)")
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Augmentation visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class AdvancedAugmentation:
    """
    Advanced augmentation techniques for challenging training scenarios.
    
    This class provides additional augmentation methods that can be used
    for difficult cases or when standard augmentation is insufficient.
    """
    
    @staticmethod
    def cutmix_data(
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        CutMix combines two images by cutting and pasting patches, which can
        help with localization and reduces overfitting.
        
        Args:
            x (torch.Tensor): Input images
            y (torch.Tensor): Labels
            alpha (float): Beta distribution parameter
            
        Returns:
            Tuple of mixed images and labels
        """
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(x.size()[0]).to(x.device)
        
        y_a = y
        y_b = y[rand_index]
        
        # Generate random bounding box
        W = x.size()[3]
        H = x.size()[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return x, y_a, y_b, lam
    
    @staticmethod
    def random_erasing(
        image: torch.Tensor, 
        probability: float = 0.5, 
        sl: float = 0.02, 
        sh: float = 0.4
    ) -> torch.Tensor:
        """
        Apply random erasing augmentation.
        
        Args:
            image (torch.Tensor): Input image tensor
            probability (float): Probability of applying erasing
            sl (float): Minimum erased area ratio
            sh (float): Maximum erased area ratio
            
        Returns:
            torch.Tensor: Image with random erasing applied
        """
        if np.random.random() > probability:
            return image
        
        for _ in range(100):  # Maximum attempts
            area = image.size()[1] * image.size()[2]
            
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(0.3, 1/0.3)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < image.size()[2] and h < image.size()[1]:
                x1 = np.random.randint(0, image.size()[1] - h)
                y1 = np.random.randint(0, image.size()[2] - w)
                
                image[:, x1:x1+h, y1:y1+w] = torch.randn(image.size()[0], h, w)
                return image
        
        return image

"""
Data Preprocessing Module

This module contains preprocessing utilities for images and text data
used in the GACL wildlife classification system.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, List
import logging

from ..configs.config import config


class ImagePreprocessor:
    """
    Image preprocessing utilities for wildlife classification.
    
    This class handles various image preprocessing tasks including
    normalization, resizing, and format conversions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ImageNet statistics for normalization (commonly used with pre-trained models)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
    
    def resize_image(
        self, 
        image: np.ndarray, 
        size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image (np.ndarray): Input image
            size (Tuple[int, int], optional): Target size (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        if size is None:
            size = (config.image_size, config.image_size)
        
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        mean: List[float] = None,
        std: List[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize image using mean and standard deviation.
        
        Args:
            image: Input image (numpy array or torch tensor)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = self.imagenet_mean
        if std is None:
            std = self.imagenet_std
        
        if isinstance(image, torch.Tensor):
            # Tensor normalization
            if image.dim() == 3:  # [C, H, W]
                for i, (m, s) in enumerate(zip(mean, std)):
                    image[i] = (image[i] - m) / s
            elif image.dim() == 4:  # [B, C, H, W]
                for i, (m, s) in enumerate(zip(mean, std)):
                    image[:, i] = (image[:, i] - m) / s
            return image
        else:
            # Numpy normalization
            image = image.astype(np.float32) / 255.0
            for i, (m, s) in enumerate(zip(mean, std)):
                image[:, :, i] = (image[:, :, i] - m) / s
            return image
    
    def denormalize_image(
        self, 
        image: torch.Tensor,
        mean: List[float] = None,
        std: List[float] = None
    ) -> torch.Tensor:
        """
        Denormalize image for visualization.
        
        Args:
            image (torch.Tensor): Normalized image tensor
            mean: Mean values used for normalization
            std: Standard deviation values used for normalization
            
        Returns:
            torch.Tensor: Denormalized image
        """
        if mean is None:
            mean = self.imagenet_mean
        if std is None:
            std = self.imagenet_std
        
        image = image.clone()
        
        if image.dim() == 3:  # [C, H, W]
            for i, (m, s) in enumerate(zip(mean, std)):
                image[i] = image[i] * s + m
        elif image.dim() == 4:  # [B, C, H, W]
            for i, (m, s) in enumerate(zip(mean, std)):
                image[:, i] = image[:, i] * s + m
        
        return torch.clamp(image, 0, 1)
    
    def convert_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor.
        
        Args:
            image (np.ndarray): Input image in HWC format
            
        Returns:
            torch.Tensor: Image tensor in CHW format
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    def convert_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy image.
        
        Args:
            tensor (torch.Tensor): Input tensor in CHW format
            
        Returns:
            np.ndarray: Image array in HWC format
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert CHW to HWC
        image = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Ensure values are in [0, 1] range
        image = np.clip(image, 0, 1)
        
        return image
    
    def crop_object(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Crop object from image using bounding box with optional padding.
        
        Args:
            image (np.ndarray): Input image
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
            padding (float): Padding ratio around the bounding box
            
        Returns:
            np.ndarray: Cropped image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image (np.ndarray): Input image
            alpha (float): Contrast enhancement factor
            
        Returns:
            np.ndarray: Contrast enhanced image
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced


class TextPreprocessor:
    """
    Text preprocessing utilities for handling text prompts and descriptions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text input.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency (optional)
        # text = text.lower()
        
        return text.strip()
    
    def truncate_text(self, text: str, max_length: int = 512) -> str:
        """
        Truncate text to maximum length while preserving word boundaries.
        
        Args:
            text (str): Input text
            max_length (int): Maximum character length
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can preserve most of the text
            return text[:last_space]
        else:
            return text[:max_length]
    
    def generate_negative_prompts(self, positive_prompts: List[str]) -> List[str]:
        """
        Generate negative text prompts for contrastive learning.
        
        Args:
            positive_prompts (List[str]): List of positive prompts
            
        Returns:
            List[str]: List of negative prompts
        """
        negative_templates = [
            "An empty forest scene with no animals",
            "A landscape photo without any wildlife",
            "Trees and vegetation with no animal presence",
            "A nature scene lacking any mammalian subjects"
        ]
        
        # Create negative prompts by mixing templates
        negative_prompts = []
        for i, template in enumerate(negative_templates):
            if i < len(positive_prompts):
                negative_prompts.append(template)
            else:
                # Cycle through templates if we need more negatives
                negative_prompts.append(negative_templates[i % len(negative_templates)])
        
        return negative_prompts[:len(positive_prompts)]
    
    def create_hierarchical_prompts(self, class_name: str) -> dict:
        """
        Create hierarchical text prompts at different levels of detail.
        
        Args:
            class_name (str): Name of the wildlife class
            
        Returns:
            dict: Dictionary with different prompt levels
        """
        base_prompts = config.frame_prompts
        detail_prompts = config.object_prompts
        
        if class_name not in base_prompts:
            return {
                'general': f"A wild animal in its natural habitat",
                'specific': f"A {class_name.lower()} in the wilderness",
                'detailed': f"Detailed view of a {class_name.lower()}"
            }
        
        return {
            'general': f"A wild animal in its natural habitat", 
            'specific': base_prompts[class_name],
            'detailed': f"{base_prompts[class_name]}. Key features: {detail_prompts[class_name]}"
        }

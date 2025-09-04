"""
Korean Wildlife Dataset Implementation

This module implements the dataset class for loading Korean wildlife images
and their corresponding text prompts for multi-modal learning.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Dict, List, Any, Tuple
import logging

from ..configs.config import config


class KoreanWildlifeDataset(Dataset):
    """
    Korean Wildlife Dataset for GACL Model Training.
    
    This dataset loads camera trap images of Korean wildlife along with
    corresponding text prompts for multi-modal contrastive learning.
    
    Expected dataset structure:
    data_root/
    ├── train/
    │   ├── Wildboar/
    │   │   ├── image_001.jpg
    │   │   └── ...
    │   ├── Goral/
    │   ├── Deers/
    │   └── Other/
    ├── val/
    │   └── [same structure as train]
    └── test/
        └── [same structure as train]
    
    Args:
        data_root (str): Path to the dataset root directory
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (callable, optional): Transform to be applied to images
        max_samples_per_class (int, optional): Limit samples per class for testing
    """
    
    def __init__(
        self, 
        data_root: str, 
        split: str = 'train', 
        transform=None,
        max_samples_per_class: int = None
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        
        # Initialize tokenizer for text processing
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        
        # Load dataset information
        self.data = self._load_data()
        
        # Get text prompts from config
        self.frame_prompts = config.frame_prompts
        self.object_prompts = config.object_prompts
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset file paths and labels.
        
        Returns:
            List of dictionaries containing image paths, labels, and class names
        """
        data = []
        split_dir = os.path.join(self.data_root, self.split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        for class_idx, class_name in enumerate(config.class_names):
            class_dir = os.path.join(split_dir, class_name)
            
            if not os.path.exists(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            images = [
                f for f in os.listdir(class_dir) 
                if f.lower().endswith(image_extensions)
            ]
            
            # Limit samples per class if specified
            if self.max_samples_per_class:
                images = images[:self.max_samples_per_class]
            
            # Add to dataset
            for img_name in images:
                data.append({
                    'image_path': os.path.join(class_dir, img_name),
                    'label': class_idx,
                    'class_name': class_name
                })
        
        if not data:
            raise ValueError(f"No valid images found in {split_dir}")
        
        return data
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Loaded image in RGB format
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using BERT tokenizer.
        
        Args:
            text (str): Input text to tokenize
            max_length (int): Maximum sequence length
            
        Returns:
            Dict containing input_ids and attention_mask tensors
        """
        tokens = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict containing image, labels, and tokenized text prompts
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        
        # Load image
        image = self._load_image(item['image_path'])
        
        # Apply transforms if provided
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                self.logger.warning(f"Transform failed for {item['image_path']}: {str(e)}")
                # Resize as fallback
                image = cv2.resize(image, (config.image_size, config.image_size))
        
        # Convert image to tensor and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # Get text prompts for the class
        class_name = item['class_name']
        frame_prompt = self.frame_prompts[class_name]
        object_prompt = self.object_prompts[class_name]
        
        # Tokenize text prompts
        frame_tokens = self._tokenize_text(frame_prompt, max_length=512)
        object_tokens = self._tokenize_text(object_prompt, max_length=100)
        
        return {
            'image': image,
            'label': torch.LongTensor([item['label']]),
            'frame_input_ids': frame_tokens['input_ids'],
            'frame_attention_mask': frame_tokens['attention_mask'],
            'object_input_ids': object_tokens['input_ids'],
            'object_attention_mask': object_tokens['attention_mask'],
            'image_path': item['image_path'],
            'class_name': class_name
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dict mapping class names to their counts
        """
        distribution = {}
        for item in self.data:
            class_name = item['class_name']
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def get_sample_paths_by_class(self, class_name: str) -> List[str]:
        """
        Get all image paths for a specific class.
        
        Args:
            class_name (str): Name of the class
            
        Returns:
            List of image paths for the specified class
        """
        return [
            item['image_path'] for item in self.data 
            if item['class_name'] == class_name
        ]


def create_datasets(
    data_root: str,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    max_samples_per_class: int = None
) -> Tuple[KoreanWildlifeDataset, KoreanWildlifeDataset, KoreanWildlifeDataset]:
    """
    Create train, validation, and test datasets.
    
    Args:
        data_root (str): Path to dataset root directory
        train_transform: Transform for training data
        val_transform: Transform for validation data  
        test_transform: Transform for test data
        max_samples_per_class (int, optional): Limit samples per class
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = KoreanWildlifeDataset(
        data_root=data_root,
        split='train',
        transform=train_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    val_dataset = KoreanWildlifeDataset(
        data_root=data_root,
        split='val',
        transform=val_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    test_dataset = KoreanWildlifeDataset(
        data_root=data_root,
        split='test', 
        transform=test_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    return train_dataset, val_dataset, test_dataset

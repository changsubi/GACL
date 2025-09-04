"""
Dataset Utility Functions

This module provides utilities for dataset creation, analysis, and management.
"""

import os
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

from ..configs.config import config


def create_dummy_dataset(
    data_root: str, 
    samples_per_class: int = 1000,
    splits: Dict[str, float] = None
) -> None:
    """
    Create dummy dataset structure for testing and development.
    
    Note: This creates random images for testing purposes.
    In production, replace with actual camera trap images.
    
    Args:
        data_root (str): Root directory for dataset
        samples_per_class (int): Number of samples per class per split
        splits (Dict[str, float], optional): Split ratios (train/val/test)
    """
    if splits is None:
        splits = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating dummy dataset at {data_root}")
    
    os.makedirs(data_root, exist_ok=True)
    
    for split_name, split_ratio in splits.items():
        split_samples = int(samples_per_class * split_ratio)
        split_dir = os.path.join(data_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in config.class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            logger.info(f"Creating {split_samples} samples for {class_name} in {split_name}")
            
            for i in range(split_samples):
                # Create random image (replace with real data loading)
                dummy_image = np.random.randint(
                    0, 255, (config.image_size, config.image_size, 3), dtype=np.uint8
                )
                
                # Add some class-specific patterns for variety
                if class_name == 'Wildboar':
                    # Add brown/dark patterns
                    dummy_image[:, :, 0] = np.clip(dummy_image[:, :, 0] * 0.8, 0, 255)
                elif class_name == 'Goral':
                    # Add gray patterns  
                    dummy_image = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2GRAY)
                    dummy_image = cv2.cvtColor(dummy_image, cv2.COLOR_GRAY2RGB)
                elif class_name == 'Deers':
                    # Add lighter brown patterns
                    dummy_image[:, :, 1] = np.clip(dummy_image[:, :, 1] * 1.2, 0, 255)
                
                image_path = os.path.join(class_dir, f'{class_name}_{i:04d}.jpg')
                cv2.imwrite(image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    # Create dataset info file
    dataset_info = {
        'name': 'Korean Wildlife Dataset (Dummy)',
        'classes': config.class_names,
        'num_classes': config.num_classes,
        'splits': splits,
        'samples_per_class_per_split': {
            split: int(samples_per_class * ratio) 
            for split, ratio in splits.items()
        },
        'total_samples': sum(
            int(samples_per_class * ratio) * config.num_classes 
            for ratio in splits.values()
        ),
        'image_size': config.image_size,
        'note': 'This is a dummy dataset for testing. Replace with real camera trap images.'
    }
    
    info_path = os.path.join(data_root, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Dummy dataset created successfully at {data_root}")
    logger.info(f"Dataset info saved to {info_path}")


def analyze_dataset_structure(data_root: str) -> Dict:
    """
    Analyze dataset structure and provide statistics.
    
    Args:
        data_root (str): Path to dataset root
        
    Returns:
        Dict: Dataset analysis results
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(data_root):
        logger.error(f"Dataset root not found: {data_root}")
        return {}
    
    analysis = {
        'data_root': data_root,
        'splits': {},
        'total_samples': 0,
        'class_distribution': defaultdict(lambda: defaultdict(int)),
        'issues': []
    }
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        
        if not os.path.exists(split_dir):
            analysis['issues'].append(f"Missing split directory: {split}")
            continue
        
        split_info = {
            'path': split_dir,
            'classes': {},
            'total_samples': 0
        }
        
        # Check each class
        for class_name in config.class_names:
            class_dir = os.path.join(split_dir, class_name)
            
            if not os.path.exists(class_dir):
                analysis['issues'].append(f"Missing class directory: {split}/{class_name}")
                continue
            
            # Count images
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            images = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(image_extensions)
            ]
            
            num_images = len(images)
            split_info['classes'][class_name] = num_images
            split_info['total_samples'] += num_images
            analysis['class_distribution'][class_name][split] = num_images
        
        analysis['splits'][split] = split_info
        analysis['total_samples'] += split_info['total_samples']
    
    # Check for class imbalance
    for class_name in config.class_names:
        class_counts = list(analysis['class_distribution'][class_name].values())
        if class_counts and (max(class_counts) - min(class_counts)) > max(class_counts) * 0.2:
            analysis['issues'].append(f"Class imbalance detected for {class_name}")
    
    # Log analysis
    logger.info("Dataset Analysis:")
    logger.info(f"  Total samples: {analysis['total_samples']}")
    for split, info in analysis['splits'].items():
        logger.info(f"  {split}: {info['total_samples']} samples")
        for class_name, count in info['classes'].items():
            logger.info(f"    {class_name}: {count}")
    
    if analysis['issues']:
        logger.warning(f"Issues found: {analysis['issues']}")
    
    return analysis


def validate_dataset_format(data_root: str) -> Tuple[bool, List[str]]:
    """
    Validate that dataset follows expected format.
    
    Args:
        data_root (str): Path to dataset root
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check root directory
    if not os.path.exists(data_root):
        errors.append(f"Dataset root directory not found: {data_root}")
        return False, errors
    
    # Check splits
    required_splits = ['train', 'val']  # test is optional
    for split in required_splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            errors.append(f"Required split directory missing: {split}")
    
    # Check classes in each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in config.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                errors.append(f"Class directory missing: {split}/{class_name}")
            else:
                # Check if directory has images
                image_files = [
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
                if not image_files:
                    errors.append(f"No images found in: {split}/{class_name}")
    
    return len(errors) == 0, errors


def create_data_splits(
    source_dir: str,
    output_dir: str, 
    split_ratios: Dict[str, float] = None,
    shuffle: bool = True
) -> None:
    """
    Create train/val/test splits from a source directory of images.
    
    Args:
        source_dir (str): Directory containing class subdirectories
        output_dir (str): Output directory for splits
        split_ratios (Dict[str, float], optional): Split ratios
        shuffle (bool): Whether to shuffle images before splitting
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating data splits from {source_dir} to {output_dir}")
    
    # Validate split ratios
    if abs(sum(split_ratios.values()) - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create output structure
    for split in split_ratios.keys():
        for class_name in config.class_names:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
    
    # Process each class
    for class_name in config.class_names:
        class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # Get all images
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        
        if shuffle:
            np.random.shuffle(image_files)
        
        # Split images
        total_images = len(image_files)
        start_idx = 0
        
        for split, ratio in split_ratios.items():
            end_idx = start_idx + int(total_images * ratio)
            split_images = image_files[start_idx:end_idx]
            
            # Copy images to split directory
            for img_file in split_images:
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(output_dir, split, class_name, img_file)
                
                # Copy file (or create symlink for efficiency)
                import shutil
                shutil.copy2(src_path, dst_path)
            
            logger.info(f"{class_name} {split}: {len(split_images)} images")
            start_idx = end_idx
    
    logger.info("Data splitting completed")

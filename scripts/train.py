#!/usr/bin/env python3
"""
Training Script for GACL Wildlife Classification

This script handles the complete training pipeline for the GACL model.
"""

import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from configs.config import config
from data.dataset import create_datasets
from data.augmentation import DataAugmentation
from models.gacl_model import GACLModel
from training.trainer import WildlifeTrainer
from utils.dataset_utils import create_dummy_dataset, analyze_dataset_structure


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GACL Wildlife Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--data_root', type=str, default=config.data_root,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--create_dummy', action='store_true',
        help='Create dummy dataset for testing'
    )
    parser.add_argument(
        '--dummy_samples', type=int, default=100,
        help='Number of dummy samples per class'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=config.batch_size,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=config.num_epochs,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=config.learning_rate,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=config.weight_decay,
        help='Weight decay for optimizer'
    )
    
    # Model arguments
    parser.add_argument(
        '--num_classes', type=int, default=config.num_classes,
        help='Number of classes'
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=config.embedding_dim,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--freeze_pretrained', action='store_true',
        help='Freeze pre-trained model weights'
    )
    
    # Output arguments
    parser.add_argument(
        '--model_save_path', type=str, default=config.model_save_path,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--results_path', type=str, default=config.results_path,
        help='Directory to save results'
    )
    
    # Resuming training
    parser.add_argument(
        '--resume_from', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Device and logging
    parser.add_argument(
        '--device', type=str, default=str(config.device),
        help='Device to use for training (cuda/cpu)'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--num_workers', type=int, default=config.num_workers,
        help='Number of data loader workers'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GACL Wildlife Classification Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Update config with command line arguments
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_classes = args.num_classes
    config.embedding_dim = args.embedding_dim
    config.model_save_path = args.model_save_path
    config.results_path = args.results_path
    config.device = torch.device(args.device)
    config.num_workers = args.num_workers
    
    # Create directories
    config.create_directories()
    
    # Create dummy dataset if requested
    if args.create_dummy:
        logger.info("Creating dummy dataset...")
        create_dummy_dataset(
            data_root=config.data_root,
            samples_per_class=args.dummy_samples
        )
    
    # Analyze dataset
    logger.info("Analyzing dataset structure...")
    dataset_analysis = analyze_dataset_structure(config.data_root)
    
    if dataset_analysis.get('issues'):
        logger.warning(f"Dataset issues found: {dataset_analysis['issues']}")
    
    # Setup data augmentation
    logger.info("Setting up data augmentation...")
    augmentation = DataAugmentation()
    
    # Create datasets
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset = create_datasets(
            data_root=config.data_root,
            train_transform=augmentation.get_train_transform(),
            val_transform=augmentation.get_val_transform(),
            test_transform=augmentation.get_test_transform()
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.error("Please check your dataset structure and paths")
        return 1
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize model
    logger.info("Initializing GACL model...")
    model = GACLModel(
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        freeze_pretrained=args.freeze_pretrained
    )
    
    model_info = model.get_model_size()
    logger.info(f"Model parameters: {model_info}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = WildlifeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        resume_from_checkpoint=args.resume_from
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        training_history = trainer.train()
        logger.info("Training completed successfully!")
        
        # Save training history
        import json
        history_path = os.path.join(config.results_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

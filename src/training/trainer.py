"""
Wildlife Trainer Implementation

This module implements the main training loop for the GACL wildlife model
with comprehensive logging, validation, and checkpointing capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from ..models.gacl_model import GACLModel
from ..data.augmentation import DataAugmentation
from ..configs.config import config
from .metrics import MetricTracker, ConfusionMatrixTracker
from .utils import EarlyStopping, ModelCheckpoint


class WildlifeTrainer:
    """
    Comprehensive trainer for GACL Wildlife Classification Model.
    
    This trainer handles:
    - Training and validation loops
    - Loss computation and backpropagation
    - Metric tracking and logging
    - Model checkpointing and early stopping
    - Learning rate scheduling
    - Mixup augmentation during training
    
    Args:
        model (GACLModel): The GACL model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config: Configuration object
        resume_from_checkpoint (str, optional): Path to checkpoint to resume from
    """
    
    def __init__(
        self,
        model: GACLModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        
        # Initialize tracking components
        self.metric_tracker = MetricTracker(config.class_names)
        self.confusion_tracker = ConfusionMatrixTracker(config.class_names)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=0.001
        )
        self.checkpoint_manager = ModelCheckpoint(
            save_dir=config.model_save_path,
            save_best=True,
            save_last=True
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Data augmentation for mixup
        self.augmentation = DataAugmentation()
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.logger.info(f"Trainer initialized on device: {config.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for training."""
        # Create logs directory
        os.makedirs(self.config.logs_path, exist_ok=True)
        
        # Setup file handler for training logs
        log_file = os.path.join(self.config.logs_path, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def mixup_criterion(
        self, 
        pred: torch.Tensor, 
        y_a: torch.Tensor, 
        y_b: torch.Tensor, 
        lam: float
    ) -> torch.Tensor:
        """
        Compute mixup loss for mixed samples.
        
        Args:
            pred (torch.Tensor): Model predictions
            y_a (torch.Tensor): First set of labels
            y_b (torch.Tensor): Second set of labels
            lam (float): Mixing parameter
            
        Returns:
            torch.Tensor: Mixed loss value
        """
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict containing training metrics for the epoch
        """
        self.model.train()
        self.metric_tracker.reset()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cont_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Create progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1}/{self.config.num_epochs}',
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            image = batch['image'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            frame_ids = batch['frame_input_ids'].to(self.config.device)
            frame_mask = batch['frame_attention_mask'].to(self.config.device)
            object_ids = batch['object_input_ids'].to(self.config.device)
            object_mask = batch['object_attention_mask'].to(self.config.device)
            
            # Apply mixup augmentation with probability
            use_mixup = random.random() < self.config.mixup_probability
            
            if use_mixup:
                # Apply mixup to images and labels
                mixed_image, y_a, y_b, lam = self.augmentation.mixup_data(
                    image, labels.squeeze(), alpha=self.config.mixup_alpha
                )
                
                # Forward pass with mixed data
                outputs = self.model(
                    image=mixed_image,
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                    object_ids=object_ids,
                    object_mask=object_mask,
                    labels=labels,
                    training=True
                )
                
                # Mixup loss for classification component
                cls_loss_mixed = self.mixup_criterion(outputs['logits'], y_a, y_b, lam)
                
                # Total loss with mixup classification loss
                if 'contrastive_loss' in outputs:
                    total_loss_batch = cls_loss_mixed + self.config.contrastive_weight * outputs['contrastive_loss']
                else:
                    total_loss_batch = cls_loss_mixed
                
                # For metrics, use original labels (approximation)
                predicted = torch.argmax(outputs['logits'], dim=1)
                self.metric_tracker.update(predicted, labels.squeeze())
                
            else:
                # Normal forward pass without mixup
                outputs = self.model(
                    image=image,
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                    object_ids=object_ids,
                    object_mask=object_mask,
                    labels=labels,
                    training=True
                )
                
                total_loss_batch = outputs['total_loss']
                
                # Update metrics
                predicted = torch.argmax(outputs['logits'], dim=1)
                self.metric_tracker.update(predicted, labels.squeeze())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            if 'classification_loss' in outputs:
                total_cls_loss += outputs['classification_loss'].item()
            if 'contrastive_loss' in outputs:
                total_cont_loss += outputs['contrastive_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Cls': f'{total_cls_loss/(batch_idx+1):.4f}',
                'Cont': f'{total_cont_loss/(batch_idx+1):.4f}',
                'Acc': f'{self.metric_tracker.get_accuracy():.2f}%'
            })
            
            # Log batch metrics
            if (batch_idx + 1) % self.config.log_frequency == 0:
                self.logger.info(
                    f'Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: '
                    f'Loss={total_loss_batch.item():.4f}, '
                    f'Acc={self.metric_tracker.get_accuracy():.2f}%'
                )
        
        # Calculate epoch metrics
        epoch_metrics = {
            'total_loss': total_loss / num_batches,
            'classification_loss': total_cls_loss / num_batches,
            'contrastive_loss': total_cont_loss / num_batches,
            'accuracy': self.metric_tracker.get_accuracy(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Add per-class metrics
        class_metrics = self.metric_tracker.get_per_class_metrics()
        epoch_metrics.update(class_metrics)
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        self.metric_tracker.reset()
        self.confusion_tracker.reset()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cont_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            
            for batch in pbar:
                # Move batch to device
                image = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                frame_ids = batch['frame_input_ids'].to(self.config.device)
                frame_mask = batch['frame_attention_mask'].to(self.config.device)
                object_ids = batch['object_input_ids'].to(self.config.device)
                object_mask = batch['object_attention_mask'].to(self.config.device)
                
                # Forward pass
                outputs = self.model(
                    image=image,
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                    object_ids=object_ids,
                    object_mask=object_mask,
                    labels=labels,
                    training=True  # To get loss values
                )
                
                # Accumulate losses
                total_loss += outputs['total_loss'].item()
                if 'classification_loss' in outputs:
                    total_cls_loss += outputs['classification_loss'].item()
                if 'contrastive_loss' in outputs:
                    total_cont_loss += outputs['contrastive_loss'].item()
                
                # Update metrics
                predicted = torch.argmax(outputs['logits'], dim=1)
                self.metric_tracker.update(predicted, labels.squeeze())
                self.confusion_tracker.update(predicted, labels.squeeze())
                
                # Update progress bar
                pbar.set_postfix({
                    'Val Loss': f'{total_loss/(len(pbar.n_fmt)+1):.4f}',
                    'Val Acc': f'{self.metric_tracker.get_accuracy():.2f}%'
                })
        
        # Calculate validation metrics
        val_metrics = {
            'val_total_loss': total_loss / num_batches,
            'val_classification_loss': total_cls_loss / num_batches,
            'val_contrastive_loss': total_cont_loss / num_batches,
            'val_accuracy': self.metric_tracker.get_accuracy()
        }
        
        # Add per-class validation metrics
        val_class_metrics = self.metric_tracker.get_per_class_metrics()
        val_metrics.update({f'val_{k}': v for k, v in val_class_metrics.items()})
        
        return val_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dict containing training history
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training on {len(self.train_loader.dataset)} samples")
        self.logger.info(f"Validating on {len(self.val_loader.dataset)} samples")
        
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = self.validate_epoch(epoch)
                
                # Learning rate scheduling
                self.scheduler.step()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update training history
                training_history['train_loss'].append(train_metrics['total_loss'])
                training_history['train_accuracy'].append(train_metrics['accuracy'])
                training_history['val_loss'].append(val_metrics['val_total_loss'])
                training_history['val_accuracy'].append(val_metrics['val_accuracy'])
                training_history['learning_rate'].append(train_metrics['learning_rate'])
                
                # Store for instance variables (for plotting)
                self.train_losses.append(train_metrics['total_loss'])
                self.val_accuracies.append(val_metrics['val_accuracy'])
                self.val_losses.append(val_metrics['val_total_loss'])
                
                # Log epoch results
                self.logger.info(
                    f'Epoch {epoch+1}/{self.config.num_epochs} - '
                    f'Train Loss: {train_metrics["total_loss"]:.4f}, '
                    f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss: {val_metrics["val_total_loss"]:.4f}, '
                    f'Val Acc: {val_metrics["val_accuracy"]:.2f}%'
                )
                
                # Check for best model
                val_acc = val_metrics['val_accuracy']
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.logger.info(f'New best validation accuracy: {val_acc:.2f}%')
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=epoch_metrics,
                    is_best=is_best
                )
                
                # Check early stopping
                if self.early_stopping.should_stop(val_metrics['val_total_loss']):
                    self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                    break
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config.save_checkpoint_frequency == 0:
                    checkpoint_path = os.path.join(
                        self.config.model_save_path, 
                        f'checkpoint_epoch_{epoch+1}.pth'
                    )
                    self.save_checkpoint(checkpoint_path, epoch_metrics)
        
        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user')
        except Exception as e:
            self.logger.error(f'Training failed with error: {e}')
            raise
        
        # Final evaluation and plotting
        self.logger.info(f'Training completed. Best validation accuracy: {self.best_val_acc:.2f}%')
        self.final_evaluation()
        self.plot_training_curves()
        
        return training_history
    
    def final_evaluation(self):
        """Perform final evaluation with detailed metrics and confusion matrix."""
        self.logger.info("Performing final evaluation...")
        
        self.model.eval()
        self.metric_tracker.reset()
        self.confusion_tracker.reset()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Final Evaluation'):
                image = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                frame_ids = batch['frame_input_ids'].to(self.config.device)
                frame_mask = batch['frame_attention_mask'].to(self.config.device)
                object_ids = batch['object_input_ids'].to(self.config.device)
                object_mask = batch['object_attention_mask'].to(self.config.device)
                
                outputs = self.model(
                    image=image,
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                    object_ids=object_ids,
                    object_mask=object_mask,
                    training=False
                )
                
                predicted = torch.argmax(outputs['logits'], dim=1)
                probabilities = F.softmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                self.metric_tracker.update(predicted, labels.squeeze())
                self.confusion_tracker.update(predicted, labels.squeeze())
        
        # Generate detailed classification report
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL EVALUATION RESULTS")
        self.logger.info("="*60)
        
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.config.class_names, 
            digits=4
        )
        self.logger.info(f"\nClassification Report:\n{report}")
        
        # Save confusion matrix
        self._save_confusion_matrix(all_labels, all_predictions)
        
        # Calculate and log per-class metrics
        self._log_per_class_metrics(all_labels, all_predictions)
        
        # Save detailed results
        self._save_evaluation_results(all_predictions, all_labels, all_probabilities)
    
    def _save_confusion_matrix(self, labels: List[int], predictions: List[int]):
        """Save confusion matrix visualization."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config.class_names,
            yticklabels=self.config.class_names
        )
        plt.title('Confusion Matrix - GACL Wildlife Model')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        os.makedirs(self.config.results_path, exist_ok=True)
        plt.savefig(
            os.path.join(self.config.results_path, 'confusion_matrix.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {self.config.results_path}")
    
    def _log_per_class_metrics(self, labels: List[int], predictions: List[int]):
        """Log detailed per-class metrics."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        self.logger.info("\nPER-CLASS METRICS:")
        self.logger.info("-" * 60)
        self.logger.info(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        self.logger.info("-" * 60)
        
        for i, class_name in enumerate(self.config.class_names):
            self.logger.info(
                f"{class_name:<12} {precision[i]:<12.3f} "
                f"{recall[i]:<12.3f} {f1[i]:<12.3f}"
            )
        
        # Overall metrics
        overall_acc = sum(labels[i] == predictions[i] for i in range(len(labels))) / len(labels)
        self.logger.info(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    def _save_evaluation_results(
        self, 
        predictions: List[int], 
        labels: List[int], 
        probabilities: List[List[float]]
    ):
        """Save detailed evaluation results to JSON."""
        results = {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'class_names': self.config.class_names,
            'overall_accuracy': sum(
                labels[i] == predictions[i] for i in range(len(labels))
            ) / len(labels),
            'best_val_accuracy': self.best_val_acc / 100,  # Convert to decimal
            'total_epochs': self.current_epoch + 1
        }
        
        results_path = os.path.join(self.config.results_path, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Detailed results saved to {results_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Combined loss plot
        if len(self.val_losses) == len(self.train_losses):
            axes[1, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
            axes[1, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
            axes[1, 0].set_title('Training vs Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [self.scheduler.get_last_lr()[0] for _ in epochs]
            axes[1, 1].plot(epochs, lr_history, 'g-', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.results_path, 'training_curves.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        self.logger.info(f"Training curves saved to {self.config.results_path}")
    
    def save_checkpoint(self, filepath: str, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'val_losses': self.val_losses
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f'Checkpoint loaded from {filepath}')
        self.logger.info(f'Resuming from epoch {self.current_epoch}')

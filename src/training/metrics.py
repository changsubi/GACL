"""
Metrics and Evaluation Utilities

This module provides comprehensive metrics tracking for model evaluation.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging


class MetricTracker:
    """
    Comprehensive metric tracker for classification tasks.
    
    Tracks accuracy, precision, recall, F1-score, and other metrics
    across training and validation.
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.labels = []
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with new batch results.
        
        Args:
            predictions (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels
        """
        if predictions.device != torch.device('cpu'):
            predictions = predictions.cpu()
        if labels.device != torch.device('cpu'):
            labels = labels.cpu()
        
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        
        self.correct += (predictions == labels).sum()
        self.total += len(labels)
    
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        if self.total == 0:
            return 0.0
        return (self.correct / self.total) * 100
    
    def get_per_class_metrics(self) -> Dict[str, float]:
        """Get detailed per-class metrics."""
        if not self.predictions:
            return {}
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.labels, self.predictions, average=None, zero_division=0
        )
        
        metrics = {}
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
            metrics[f'{class_name}_support'] = support[i]
        
        # Overall metrics
        metrics['macro_precision'] = precision.mean()
        metrics['macro_recall'] = recall.mean()
        metrics['macro_f1'] = f1.mean()
        
        return metrics


class ConfusionMatrixTracker:
    """Track and compute confusion matrix."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Update confusion matrix."""
        if predictions.device != torch.device('cpu'):
            predictions = predictions.cpu()
        if labels.device != torch.device('cpu'):
            labels = labels.cpu()
        
        for pred, label in zip(predictions.numpy(), labels.numpy()):
            self.matrix[label, pred] += 1
    
    def get_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.matrix
    
    def get_normalized_matrix(self) -> np.ndarray:
        """Get normalized confusion matrix."""
        return self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]

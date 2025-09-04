"""
Loss Functions for GACL Model

This module implements the loss functions used in the GACL model:
- Parallel Contrastive Learning Loss for multi-modal alignment
- Classification Loss
- Combined Loss with weighting strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

from ..configs.config import config


class ParallelContrastiveLoss(nn.Module):
    """
    Parallel Contrastive Learning Loss Implementation.
    
    This loss function implements the 4-way contrastive learning approach
    described in the GACL paper (Figures 8-9):
    1. Global-to-Global (G-T): Global image vs Global text
    2. Global-to-Local (G-t): Global image vs Local text  
    3. Local-to-Global (V-T): Local image vs Global text
    4. Local-to-Local (V-t): Local image vs Local text
    
    Args:
        temperature (float): Temperature parameter for scaling similarities
        reduction (str): Reduction method for loss ('mean', 'sum', 'none')
    """
    
    def __init__(self, temperature: float = None, reduction: str = 'mean'):
        super().__init__()
        
        if temperature is None:
            temperature = config.temperature
            
        self.temperature = temperature
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.logger = logging.getLogger(__name__)
    
    def compute_similarity_matrix(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix between two feature sets.
        
        Args:
            features1 (torch.Tensor): First feature set [B, D]
            features2 (torch.Tensor): Second feature set [B, D]
            
        Returns:
            torch.Tensor: Similarity matrix [B, B]
        """
        # Normalize features to unit vectors
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        # Compute similarity matrix and scale by temperature
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        return similarity_matrix
    
    def contrastive_loss_bidirectional(
        self, 
        similarity_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bidirectional contrastive loss from similarity matrix.
        
        Args:
            similarity_matrix (torch.Tensor): Similarity matrix [B, B]
            
        Returns:
            torch.Tensor: Bidirectional contrastive loss
        """
        batch_size = similarity_matrix.shape[0]
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        
        # Image-to-text loss (each image should match its corresponding text)
        loss_i2t = self.criterion(similarity_matrix, labels)
        
        # Text-to-image loss (each text should match its corresponding image)
        loss_t2i = self.criterion(similarity_matrix.T, labels)
        
        # Average of both directions
        return (loss_i2t + loss_t2i) / 2
    
    def forward(
        self, 
        global_img: torch.Tensor, 
        local_img: torch.Tensor,
        global_text: torch.Tensor, 
        local_text: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute parallel contrastive learning loss.
        
        Args:
            global_img (torch.Tensor): Global image features [B, D]
            local_img (torch.Tensor): Local image features [B, D]
            global_text (torch.Tensor): Global text features [B, D]
            local_text (torch.Tensor): Local text features [B, D]
            
        Returns:
            Tuple of (total_loss, loss_dict):
                - total_loss: Average of all four contrastive losses
                - loss_dict: Dictionary with individual loss values
        """
        # 1. Global Image to Global Text (G-T)
        sim_g2g = self.compute_similarity_matrix(global_img, global_text)
        loss_g2g = self.contrastive_loss_bidirectional(sim_g2g)
        
        # 2. Global Image to Local Text (G-t) 
        sim_g2l = self.compute_similarity_matrix(global_img, local_text)
        loss_g2l = self.contrastive_loss_bidirectional(sim_g2l)
        
        # 3. Local Image to Global Text (V-T)
        sim_l2g = self.compute_similarity_matrix(local_img, global_text)
        loss_l2g = self.contrastive_loss_bidirectional(sim_l2g)
        
        # 4. Local Image to Local Text (V-t)
        sim_l2l = self.compute_similarity_matrix(local_img, local_text)
        loss_l2l = self.contrastive_loss_bidirectional(sim_l2l)
        
        # Average all four losses
        total_loss = (loss_g2g + loss_g2l + loss_l2g + loss_l2l) / 4
        
        # Create detailed loss dictionary
        loss_dict = {
            'g2g': loss_g2g.item() if hasattr(loss_g2g, 'item') else loss_g2g,
            'g2l': loss_g2l.item() if hasattr(loss_g2l, 'item') else loss_g2l,
            'l2g': loss_l2g.item() if hasattr(loss_l2g, 'item') else loss_l2g,
            'l2l': loss_l2l.item() if hasattr(loss_l2l, 'item') else loss_l2l,
            'total': total_loss.item() if hasattr(total_loss, 'item') else total_loss
        }
        
        return total_loss, loss_dict
    
    def get_similarities(
        self,
        global_img: torch.Tensor,
        local_img: torch.Tensor, 
        global_text: torch.Tensor,
        local_text: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get similarity matrices for analysis/visualization.
        
        Returns:
            Dict with similarity matrices for each pairing
        """
        with torch.no_grad():
            similarities = {
                'g2g': self.compute_similarity_matrix(global_img, global_text),
                'g2l': self.compute_similarity_matrix(global_img, local_text),
                'l2g': self.compute_similarity_matrix(local_img, global_text),
                'l2l': self.compute_similarity_matrix(local_img, local_text)
            }
        return similarities


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in wildlife classification.
    
    Focal Loss helps the model focus on hard examples and reduces the
    contribution of easy examples, which is particularly useful when
    there's class imbalance in the wildlife dataset.
    
    Args:
        alpha (float or torch.Tensor): Weighting factor for classes
        gamma (float): Focusing parameter
        reduction (str): Reduction method
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions [B, num_classes]
            targets (torch.Tensor): Ground truth labels [B]
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization.
    
    Label smoothing prevents the model from becoming overconfident
    by assigning some probability mass to incorrect classes.
    
    Args:
        num_classes (int): Number of classes
        smoothing (float): Smoothing factor (0.0 = no smoothing, 1.0 = uniform)
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs (torch.Tensor): Model predictions [B, num_classes]
            targets (torch.Tensor): Ground truth labels [B]
            
        Returns:
            torch.Tensor: Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class CombinedLoss(nn.Module):
    """
    Combined loss function for GACL model training.
    
    This loss combines classification loss with contrastive learning loss
    using configurable weights.
    
    Args:
        contrastive_weight (float): Weight for contrastive loss
        classification_weight (float): Weight for classification loss
        use_focal_loss (bool): Whether to use focal loss for classification
        use_label_smoothing (bool): Whether to use label smoothing
    """
    
    def __init__(
        self,
        contrastive_weight: float = None,
        classification_weight: float = 1.0,
        use_focal_loss: bool = False,
        use_label_smoothing: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        if contrastive_weight is None:
            contrastive_weight = config.contrastive_weight
            
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        # Initialize contrastive loss
        self.contrastive_loss = ParallelContrastiveLoss()
        
        # Initialize classification loss
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif use_label_smoothing:
            self.classification_loss = LabelSmoothingLoss(
                num_classes=config.num_classes,
                smoothing=label_smoothing
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_img: torch.Tensor,
        local_img: torch.Tensor,
        global_text: torch.Tensor,
        local_text: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            logits (torch.Tensor): Classification logits [B, num_classes]
            labels (torch.Tensor): Ground truth labels [B]
            global_img (torch.Tensor): Global image features [B, D]
            local_img (torch.Tensor): Local image features [B, D]
            global_text (torch.Tensor): Global text features [B, D]
            local_text (torch.Tensor): Local text features [B, D]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, labels.squeeze())
        
        # Contrastive loss
        cont_loss, cont_loss_dict = self.contrastive_loss(
            global_img, local_img, global_text, local_text
        )
        
        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss + 
            self.contrastive_weight * cont_loss
        )
        
        # Detailed loss dictionary
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': cont_loss.item(),
            'contrastive_details': cont_loss_dict
        }
        
        return total_loss, loss_dict
    
    def update_weights(self, epoch: int, total_epochs: int) -> None:
        """
        Update loss weights during training (optional curriculum learning).
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs
        """
        # Example: Gradually increase contrastive weight
        progress = epoch / total_epochs
        self.contrastive_weight = config.contrastive_weight * (0.5 + 0.5 * progress)
        
        self.logger.info(f"Updated contrastive weight to {self.contrastive_weight:.3f}")


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning (alternative/additional contrastive approach).
    
    This can be used as an alternative or complement to the parallel contrastive loss.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor samples
            positive (torch.Tensor): Positive samples  
            negative (torch.Tensor): Negative samples
            
        Returns:
            torch.Tensor: Triplet loss
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

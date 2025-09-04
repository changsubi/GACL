"""
Main GACL Model Implementation

This module contains the main GACL (Graph Attention Contrastive Learning) model
that combines all components for Korean wildlife classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .backbone import GATEncoder
from .encoders import VisionEncoder, TextEncoder
from .losses import ParallelContrastiveLoss, CombinedLoss
from ..configs.config import config


class GACLModel(nn.Module):
    """
    Main GACL Model for Korean Wildlife Classification.
    
    This model implements the complete GACL architecture described in the paper,
    combining Graph Attention Networks, Vision Transformers, and BERT for
    multi-modal contrastive learning and wildlife classification.
    
    Architecture Components:
    1. GATEncoder: For global structural features using graph attention
    2. VisionEncoder: For local patch features using Vision Transformer
    3. TextEncoder: For text prompt features using BERT
    4. Classification Head: For final species prediction
    5. Contrastive Learning: For multi-modal alignment
    
    Args:
        num_classes (int): Number of wildlife classes
        embedding_dim (int): Feature embedding dimension
        use_combined_loss (bool): Whether to use combined loss function
        freeze_pretrained (bool): Whether to freeze pre-trained weights
    """
    
    def __init__(
        self,
        num_classes: int = None,
        embedding_dim: int = None,
        use_combined_loss: bool = True,
        freeze_pretrained: bool = False
    ):
        super().__init__()
        
        if num_classes is None:
            num_classes = config.num_classes
        if embedding_dim is None:
            embedding_dim = config.embedding_dim
            
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_combined_loss = use_combined_loss
        
        # Core encoders
        self.gat_encoder = GATEncoder(
            input_dim=config.mdconv_output_channels,
            hidden_dim=config.gat_hidden_dim,
            num_heads=config.gat_num_heads,
            num_layers=config.gat_num_layers
        )
        
        self.vision_encoder = VisionEncoder(
            model_name=config.vit_model_name,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_pretrained
        )
        
        self.text_encoder = TextEncoder(
            model_name=config.bert_model_name,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_pretrained
        )
        
        # Classification head using GAT global features
        self.classifier = nn.Sequential(
            nn.Linear(config.gat_hidden_dim, config.gat_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gat_hidden_dim // 2, config.gat_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.gat_hidden_dim // 4, num_classes)
        )
        
        # Loss functions
        if use_combined_loss:
            self.loss_function = CombinedLoss(
                contrastive_weight=config.contrastive_weight,
                classification_weight=1.0
            )
        else:
            self.contrastive_loss = ParallelContrastiveLoss(temperature=config.temperature)
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Feature fusion layer (optional)
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.gat_hidden_dim + embedding_dim, config.gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.logger = logging.getLogger(__name__)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for newly created layers."""
        for module in [self.classifier, self.feature_fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        image: torch.Tensor,
        frame_ids: torch.Tensor,
        frame_mask: torch.Tensor,
        object_ids: torch.Tensor,
        object_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GACL model.
        
        Args:
            image (torch.Tensor): Input images [B, C, H, W]
            frame_ids (torch.Tensor): Frame text token IDs [B, seq_len]
            frame_mask (torch.Tensor): Frame text attention mask [B, seq_len]
            object_ids (torch.Tensor): Object text token IDs [B, seq_len]
            object_mask (torch.Tensor): Object text attention mask [B, seq_len]
            labels (torch.Tensor, optional): Ground truth labels [B] or [B, 1]
            training (bool): Whether model is in training mode
            
        Returns:
            Dict containing model outputs and losses (if training)
        """
        batch_size = image.size(0)
        
        # 1. Extract features from different encoders
        
        # GAT encoder for global structural features
        global_img_gat = self.gat_encoder(image)  # [B, gat_hidden_dim]
        
        # Vision encoder for local patch features
        global_img_vit, local_img_vit = self.vision_encoder(image)  # [B, embedding_dim]
        
        # Text encoders for frame and object prompts
        global_text_frame, local_text_frame = self.text_encoder(frame_ids, frame_mask)
        global_text_object, local_text_object = self.text_encoder(object_ids, object_mask)
        
        # 2. Feature selection for contrastive learning
        # Use frame prompts for global text features (scene-level descriptions)
        global_text = global_text_frame
        # Use object prompts for local text features (object-level descriptions)
        local_text = local_text_object
        
        # 3. Classification using GAT global features
        # Optionally fuse GAT features with ViT global features
        if hasattr(self, 'feature_fusion'):
            fused_features = torch.cat([global_img_gat, global_img_vit], dim=1)
            fused_features = self.feature_fusion(fused_features)
            logits = self.classifier(fused_features)
        else:
            logits = self.classifier(global_img_gat)
        
        # 4. Prepare outputs
        outputs = {
            'logits': logits,
            'global_img_gat': global_img_gat,
            'global_img_vit': global_img_vit,
            'local_img_vit': local_img_vit,
            'global_text': global_text,
            'local_text': local_text,
            'probabilities': F.softmax(logits, dim=1)
        }
        
        # 5. Compute losses if training
        if training and labels is not None:
            if labels.dim() > 1:
                labels = labels.squeeze()
                
            if self.use_combined_loss:
                # Use combined loss function
                total_loss, loss_dict = self.loss_function(
                    logits=logits,
                    labels=labels,
                    global_img=global_img_gat,
                    local_img=local_img_vit,
                    global_text=global_text,
                    local_text=local_text
                )
                
                outputs.update({
                    'total_loss': total_loss,
                    'classification_loss': torch.tensor(loss_dict['classification']),
                    'contrastive_loss': torch.tensor(loss_dict['contrastive']),
                    'loss_details': loss_dict
                })
                
            else:
                # Separate loss computation
                classification_loss = self.classification_loss(logits, labels)
                contrastive_loss, cont_loss_dict = self.contrastive_loss(
                    global_img_gat, local_img_vit, global_text, local_text
                )
                
                total_loss = classification_loss + config.contrastive_weight * contrastive_loss
                
                outputs.update({
                    'total_loss': total_loss,
                    'classification_loss': classification_loss,
                    'contrastive_loss': contrastive_loss,
                    'loss_details': cont_loss_dict
                })
        
        return outputs
    
    def predict(
        self,
        image: torch.Tensor,
        frame_ids: torch.Tensor,
        frame_mask: torch.Tensor,
        object_ids: torch.Tensor,
        object_mask: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Inference prediction with optional feature extraction.
        
        Args:
            image (torch.Tensor): Input images
            frame_ids (torch.Tensor): Frame text token IDs
            frame_mask (torch.Tensor): Frame text attention mask  
            object_ids (torch.Tensor): Object text token IDs
            object_mask (torch.Tensor): Object text attention mask
            return_features (bool): Whether to return feature representations
            
        Returns:
            Dict containing predictions and optionally features
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                image=image,
                frame_ids=frame_ids,
                frame_mask=frame_mask,
                object_ids=object_ids,
                object_mask=object_mask,
                training=False
            )
            
            # Get predictions
            logits = outputs['logits']
            probabilities = outputs['probabilities']
            predicted_classes = torch.argmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            
            result = {
                'predicted_classes': predicted_classes,
                'probabilities': probabilities,
                'confidence_scores': confidence_scores,
                'logits': logits
            }
            
            if return_features:
                result.update({
                    'global_img_gat': outputs['global_img_gat'],
                    'global_img_vit': outputs['global_img_vit'],
                    'local_img_vit': outputs['local_img_vit'],
                    'global_text': outputs['global_text'],
                    'local_text': outputs['local_text']
                })
            
            return result
    
    def get_feature_representations(
        self,
        image: torch.Tensor,
        frame_ids: torch.Tensor,
        frame_mask: torch.Tensor,
        object_ids: torch.Tensor,
        object_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract feature representations for analysis/visualization.
        
        Returns:
            Dict with all feature representations
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                image=image,
                frame_ids=frame_ids,
                frame_mask=frame_mask,
                object_ids=object_ids,
                object_mask=object_mask,
                training=False
            )
            
            return {
                'global_img_gat': outputs['global_img_gat'],
                'global_img_vit': outputs['global_img_vit'],
                'local_img_vit': outputs['local_img_vit'],
                'global_text': outputs['global_text'],
                'local_text': outputs['local_text']
            }
    
    def freeze_encoders(self, freeze_gat: bool = False, freeze_vision: bool = True, freeze_text: bool = True):
        """
        Freeze specific encoder components.
        
        Args:
            freeze_gat (bool): Whether to freeze GAT encoder
            freeze_vision (bool): Whether to freeze Vision encoder
            freeze_text (bool): Whether to freeze Text encoder
        """
        if freeze_gat:
            for param in self.gat_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Frozen GAT encoder")
            
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Frozen Vision encoder")
            
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Frozen Text encoder")
    
    def unfreeze_encoders(self):
        """Unfreeze all encoder components."""
        for param in self.parameters():
            param.requires_grad = True
        self.logger.info("Unfrozen all encoders")
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        component_params = {
            'gat_encoder': sum(p.numel() for p in self.gat_encoder.parameters()),
            'vision_encoder': sum(p.numel() for p in self.vision_encoder.parameters()),
            'text_encoder': sum(p.numel() for p in self.text_encoder.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters())
        }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': component_params
        }
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True):
        """
        Load pre-trained weights from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            strict (bool): Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict, strict=strict)
        self.logger.info(f"Loaded weights from {checkpoint_path}")
    
    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

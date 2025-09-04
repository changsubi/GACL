"""
Vision and Text Encoders

This module implements the vision and text encoders used in the GACL model:
- VisionEncoder: Uses Vision Transformer (ViT) for image feature extraction
- TextEncoder: Uses BERT for text prompt encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, ViTFeatureExtractor
from typing import Tuple
import logging

from ..configs.config import config


class VisionEncoder(nn.Module):
    """
    Vision Transformer Encoder for extracting local image features.
    
    This encoder uses a pre-trained Vision Transformer to extract both global
    and local visual features from input images. The global features come from
    the [CLS] token, while local features are derived from patch tokens.
    
    Args:
        model_name (str): Pre-trained ViT model name
        embedding_dim (int): Output embedding dimension
        freeze_backbone (bool): Whether to freeze the pre-trained weights
    """
    
    def __init__(
        self, 
        model_name: str = None,
        embedding_dim: int = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        if model_name is None:
            model_name = config.vit_model_name
        if embedding_dim is None:
            embedding_dim = config.embedding_dim
            
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load pre-trained Vision Transformer
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Projection layers to map ViT features to target embedding dimension
        vit_hidden_size = self.vit.config.hidden_size
        self.global_projection = nn.Linear(vit_hidden_size, embedding_dim)
        self.local_projection = nn.Linear(vit_hidden_size, embedding_dim)
        
        # Additional layers for feature refinement
        self.global_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.local_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Vision Encoder.
        
        Args:
            image (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            Tuple of (global_features, local_features):
                - global_features: [B, embedding_dim] from [CLS] token
                - local_features: [B, embedding_dim] from patch tokens
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=image)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]
        
        # Extract global features from [CLS] token (first token)
        global_features_raw = hidden_states[:, 0]  # [B, hidden_size]
        
        # Extract local features from patch tokens (mean pooling)
        patch_tokens = hidden_states[:, 1:]  # [B, num_patches, hidden_size]
        local_features_raw = patch_tokens.mean(dim=1)  # [B, hidden_size]
        
        # Project to target embedding dimension
        global_features = self.global_projection(global_features_raw)
        local_features = self.local_projection(local_features_raw)
        
        # Apply refinement MLPs
        global_features = self.global_mlp(global_features)
        local_features = self.local_mlp(local_features)
        
        # L2 normalization for contrastive learning
        global_features = F.normalize(global_features, dim=-1)
        local_features = F.normalize(local_features, dim=-1)
        
        return global_features, local_features
    
    def get_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get individual patch features for detailed analysis.
        
        Args:
            image (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch features [B, num_patches, embedding_dim]
        """
        outputs = self.vit(pixel_values=image)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, num_patches, hidden_size]
        
        # Project each patch token
        patch_features = self.local_projection(patch_tokens)
        patch_features = F.normalize(patch_features, dim=-1)
        
        return patch_features
    
    def get_attention_maps(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from the Vision Transformer.
        
        Args:
            image (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention maps [B, num_heads, seq_len, seq_len]
        """
        outputs = self.vit(pixel_values=image, output_attentions=True)
        # Return attention from the last layer
        return outputs.attentions[-1]


class TextEncoder(nn.Module):
    """
    BERT-based Text Encoder for processing text prompts.
    
    This encoder processes text prompts (both frame-level and object-level)
    to extract semantic representations for contrastive learning with visual features.
    
    Args:
        model_name (str): Pre-trained BERT model name
        embedding_dim (int): Output embedding dimension
        freeze_backbone (bool): Whether to freeze BERT weights
    """
    
    def __init__(
        self,
        model_name: str = None,
        embedding_dim: int = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        if model_name is None:
            model_name = config.bert_model_name
        if embedding_dim is None:
            embedding_dim = config.embedding_dim
            
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layers
        bert_hidden_size = self.bert.config.hidden_size
        self.global_projection = nn.Linear(bert_hidden_size, embedding_dim)
        self.local_projection = nn.Linear(bert_hidden_size, embedding_dim)
        
        # Feature refinement MLPs
        self.global_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.local_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.logger = logging.getLogger(__name__)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Text Encoder.
        
        Args:
            input_ids (torch.Tensor): Token IDs [B, seq_len]
            attention_mask (torch.Tensor): Attention mask [B, seq_len]
            
        Returns:
            Tuple of (global_features, local_features):
                - global_features: [B, embedding_dim] from [CLS] token
                - local_features: [B, embedding_dim] from mean pooling
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]
        
        # Global features from [CLS] token
        global_features_raw = hidden_states[:, 0]  # [B, hidden_size]
        
        # Local features from mean pooling over valid tokens
        # Mask out [PAD] tokens for accurate mean pooling
        masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
        local_features_raw = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        
        # Project to target embedding dimension
        global_features = self.global_projection(global_features_raw)
        local_features = self.local_projection(local_features_raw)
        
        # Apply refinement MLPs
        global_features = self.global_mlp(global_features)
        local_features = self.local_mlp(local_features)
        
        # L2 normalization for contrastive learning
        global_features = F.normalize(global_features, dim=-1)
        local_features = F.normalize(local_features, dim=-1)
        
        return global_features, local_features
    
    def encode_batch_texts(
        self,
        texts: list,
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of text strings.
        
        Args:
            texts (list): List of text strings
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple of (global_features, local_features)
        """
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Tokenize texts
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        
        return self.forward(input_ids, attention_mask)
    
    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get embeddings for individual tokens.
        
        Args:
            input_ids (torch.Tensor): Token IDs [B, seq_len]
            attention_mask (torch.Tensor): Attention mask [B, seq_len]
            
        Returns:
            torch.Tensor: Token embeddings [B, seq_len, embedding_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        
        # Project to embedding dimension
        token_embeddings = self.local_projection(hidden_states)
        token_embeddings = F.normalize(token_embeddings, dim=-1)
        
        return token_embeddings


class CrossModalEncoder(nn.Module):
    """
    Cross-modal encoder for joint vision-text processing.
    
    This encoder can be used for more sophisticated cross-modal interactions
    beyond simple contrastive learning.
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention layers
        self.vision_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.text_to_vision_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for cross-modal encoding.
        
        Args:
            vision_features (torch.Tensor): Vision features [B, vision_dim]
            text_features (torch.Tensor): Text features [B, text_dim]
            
        Returns:
            torch.Tensor: Cross-modal features [B, hidden_dim]
        """
        # Project to common dimension
        v_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, hidden_dim]
        t_proj = self.text_proj(text_features).unsqueeze(1)      # [B, 1, hidden_dim]
        
        # Cross-modal attention
        v_attended, _ = self.vision_to_text_attention(v_proj, t_proj, t_proj)
        t_attended, _ = self.text_to_vision_attention(t_proj, v_proj, v_proj)
        
        # Combine and project
        combined = torch.cat([v_attended.squeeze(1), t_attended.squeeze(1)], dim=-1)
        output = self.output_proj(combined)
        
        return F.normalize(output, dim=-1)

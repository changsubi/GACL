"""
Backbone Model Components

This module implements the core backbone components of the GACL model:
- Multi-Dilated Convolutional Network for multi-scale feature extraction
- Graph Attention Transformer Encoder for structural relationship learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List
import logging

from ..configs.config import config


class MultiDilatedConvNet(nn.Module):
    """
    Multi-Dilated Convolutional Network for multi-scale feature extraction.
    
    This network applies convolutions with different dilation rates in parallel
    to capture features at different scales, as described in Table 1 of the paper.
    
    Architecture:
    - Dilation rates: [1, 2, 4, 5] for fine-grained to global information
    - Each branch outputs channels/4 features
    - Features are concatenated and normalized
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        out_channels (int): Number of output channels (default: 256)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_channels = out_channels // 4
        
        # Multi-dilated convolution branches
        self.conv1 = nn.Conv2d(
            in_channels, self.branch_channels, 
            kernel_size=3, padding=1, dilation=1
        )  # Fine-grained details
        
        self.conv2 = nn.Conv2d(
            in_channels, self.branch_channels,
            kernel_size=3, padding=2, dilation=2  
        )  # Broader local patterns
        
        self.conv4 = nn.Conv2d(
            in_channels, self.branch_channels,
            kernel_size=3, padding=4, dilation=4
        )  # Structural patterns
        
        self.conv5 = nn.Conv2d(
            in_channels, self.branch_channels,
            kernel_size=3, padding=5, dilation=5
        )  # Global contextual information
        
        # Normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-dilated convolution network.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Multi-scale feature tensor [B, out_channels, H, W]
        """
        # Apply different dilation rates in parallel
        conv1_out = self.conv1(x)  # Dilation 1
        conv2_out = self.conv2(x)  # Dilation 2  
        conv4_out = self.conv4(x)  # Dilation 4
        conv5_out = self.conv5(x)  # Dilation 5
        
        # Concatenate features from different receptive fields
        combined = torch.cat([conv1_out, conv2_out, conv4_out, conv5_out], dim=1)
        
        # Apply normalization and activation
        combined = self.bn(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)
        
        return combined
    
    def get_receptive_field_info(self) -> dict:
        """Get information about receptive fields for each branch."""
        return {
            'dilation_1': {'receptive_field': 3, 'description': 'Fine-grained details'},
            'dilation_2': {'receptive_field': 5, 'description': 'Broader local patterns'},
            'dilation_4': {'receptive_field': 9, 'description': 'Structural patterns'},
            'dilation_5': {'receptive_field': 11, 'description': 'Global contextual information'}
        }


class GATEncoder(nn.Module):
    """
    Graph Attention Transformer Encoder for learning structural relationships.
    
    This encoder creates graph representations of image patches and applies
    Graph Attention Networks to learn spatial relationships and dependencies.
    
    Key Components:
    1. Patch extraction and feature computation using Multi-Dilated ConvNet
    2. Graph construction using K-means clustering for edge weights
    3. Multi-layer Graph Attention Network with residual connections
    4. Global pooling for final representation
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for GAT layers
        num_heads (int): Number of attention heads
        num_layers (int): Number of GAT layers
    """
    
    def __init__(
        self, 
        input_dim: int = 256, 
        hidden_dim: int = 256, 
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-dilated convolution for patch feature extraction
        self.patch_conv = MultiDilatedConvNet(3, input_dim)
        
        # Graph Attention layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        for i in range(num_layers):
            # GAT layer
            in_dim = input_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=num_heads, concat=False)
            )
            
            # Layer normalization
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            
            # MLP for residual connection
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ))
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        self.logger = logging.getLogger(__name__)
        
    def create_patches(
        self, 
        image: torch.Tensor, 
        patch_size: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create overlapping image patches for graph construction.
        
        Args:
            image (torch.Tensor): Input image [B, C, H, W]
            patch_size (int, optional): Size of patches
            
        Returns:
            Tuple of (patches, positions):
                - patches: [B, N, C, patch_size, patch_size]
                - positions: [N, 2] center positions of patches
        """
        if patch_size is None:
            patch_size = config.patch_size
            
        B, C, H, W = image.shape
        patches = []
        positions = []
        
        # Create overlapping patches with stride = patch_size // 2
        stride = patch_size // 2
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = image[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                # Store center position
                positions.append([i + patch_size//2, j + patch_size//2])
        
        patches = torch.stack(patches, dim=1)  # [B, N, C, H, W]
        positions = torch.FloatTensor(positions)  # [N, 2]
        
        return patches, positions
    
    def create_graph(
        self, 
        patch_features: torch.Tensor, 
        positions: torch.Tensor = None,
        k: int = None
    ) -> Batch:
        """
        Create graph structure using K-means clustering for edge weights.
        
        Args:
            patch_features (torch.Tensor): Patch features [B, N, D]
            positions (torch.Tensor, optional): Patch positions [N, 2]
            k (int, optional): Number of clusters for K-means
            
        Returns:
            Batch: PyTorch Geometric batch of graph data
        """
        if k is None:
            k = config.graph_k_neighbors
            
        B, N, D = patch_features.shape
        graphs = []
        
        for b in range(B):
            features = patch_features[b].detach().cpu().numpy()
            
            # K-means clustering for semantic grouping
            if N > k and k > 1:
                try:
                    kmeans = KMeans(n_clusters=min(k, N), random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features)
                except Exception as e:
                    self.logger.warning(f"K-means clustering failed: {e}")
                    clusters = np.arange(N) % k
            else:
                clusters = np.arange(N) % max(1, k)
            
            # Create adjacency matrix based on clustering and spatial proximity
            edge_indices = []
            edge_weights = []
            
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_indices.append([i, j])
                        
                        # Semantic similarity weight
                        semantic_weight = 1.0 if clusters[i] == clusters[j] else 0.3
                        
                        # Spatial proximity weight (if positions available)
                        spatial_weight = 1.0
                        if positions is not None:
                            dist = torch.norm(positions[i] - positions[j])
                            max_dist = torch.norm(torch.tensor([config.image_size, config.image_size]).float())
                            spatial_weight = 1.0 - (dist / max_dist).item()
                        
                        # Combined weight
                        weight = 0.7 * semantic_weight + 0.3 * spatial_weight
                        edge_weights.append(weight)
            
            # Create edge tensors
            if len(edge_indices) > 0:
                edge_index = torch.LongTensor(edge_indices).t().contiguous()
                edge_weight = torch.FloatTensor(edge_weights)
            else:
                edge_index = torch.LongTensor([[], []])
                edge_weight = torch.FloatTensor([])
            
            # Create graph data
            graph_data = Data(
                x=patch_features[b],
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            graphs.append(graph_data)
        
        return Batch.from_data_list(graphs)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Graph Attention Transformer.
        
        Args:
            image (torch.Tensor): Input image [B, C, H, W]
            
        Returns:
            torch.Tensor: Global graph features [B, hidden_dim]
        """
        B = image.shape[0]
        
        # Extract patches and their positions
        patches, positions = self.create_patches(image)
        B, N, C, H, W = patches.shape
        
        # Apply multi-dilated convolution to each patch
        patch_features = []
        for i in range(N):
            patch_feat = self.patch_conv(patches[:, i])  # [B, D, H, W]
            # Global average pooling to get patch representation
            patch_feat = F.adaptive_avg_pool2d(patch_feat, 1).squeeze(-1).squeeze(-1)
            patch_features.append(patch_feat)
        
        patch_features = torch.stack(patch_features, dim=1)  # [B, N, D]
        
        # Create graph structure
        batch_graph = self.create_graph(patch_features, positions)
        
        # Apply Graph Attention layers with residual connections
        x = batch_graph.x
        for i, (gat_layer, layer_norm, mlp) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.mlps)
        ):
            residual = x
            
            # Graph Attention layer
            x = gat_layer(x, batch_graph.edge_index, batch_graph.edge_attr)
            
            # Layer normalization
            x = layer_norm(x)
            
            # MLP with residual connection
            mlp_out = mlp(x)
            x = residual + mlp_out  # Residual connection
        
        # Global pooling to get graph-level representation
        global_features = self.global_pool(x, batch_graph.batch)
        
        return global_features
    
    def get_attention_weights(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from GAT layers for visualization.
        
        Args:
            image (torch.Tensor): Input image
            
        Returns:
            List[torch.Tensor]: Attention weights from each layer
        """
        # This would require modifications to GATv2Conv to return attention weights
        # For now, return placeholder
        self.logger.warning("Attention weight extraction not implemented")
        return []

import torch
import torch.nn as nn
from timm import create_model
from timm.models.vision_transformer import VisionTransformer
from avalanche.models import MultiTaskModule, MultiHeadClassifier

class ViT64SingleHead(nn.Module):
    """ViT for class-incremental learning (all 60 classes)"""
    
    def __init__(self, 
                 num_classes=60,
                 patch_size=8,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 dropout=0.1):
        super().__init__()
        
        self.vit = VisionTransformer(
            img_size=64,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            class_token=True,
            global_pool='token',
        )
        
        # Store embedding matrix reference for analysis
        self.head = self.vit.head
        
    def forward(self, x):
        return self.vit(x)


class ViT64MultiHead(MultiTaskModule):
    """ViT for task-incremental learning (10 classes per task)"""
    
    def __init__(self,
                 num_classes_per_task=10,
                 patch_size=8,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 dropout=0.1):
        super().__init__()
        
        # Create ViT without classification head
        self.vit = VisionTransformer(
            img_size=64,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,  # No head, we'll use MultiHeadClassifier
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            class_token=True,
            global_pool='',  # Don't pool, we'll handle it
        )
        
        # Multi-head classifier for different tasks
        self.classifier = MultiHeadClassifier(embed_dim, initial_out_features=num_classes_per_task)
        
    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        # Get CLS token representation
        x = self.vit.forward_features(x)
        cls_token = x[:, 0]  # Extract CLS token
        
        # Pass through task-specific head
        return self.classifier(cls_token, task_label)
    
    def forward(self, x, task_labels):
        """Forward for potentially mixed batches"""
        x = self.vit.forward_features(x)
        cls_token = x[:, 0]
        return self.classifier(cls_token, task_labels)
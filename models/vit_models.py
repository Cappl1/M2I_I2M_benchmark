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


class ViT64MultiHead(nn.Module):
    """ViT for task-incremental learning with our own clean multi-head implementation"""
    
    def __init__(self,
                 num_classes_per_task=10,
                 num_tasks=2,  # Default to 2 tasks for binary pairs
                 patch_size=8,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 dropout=0.1):
        super().__init__()
        
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        
        # Create ViT without classification head
        self.vit = VisionTransformer(
            img_size=64,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,  # No head, we'll add our own
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            class_token=True,
            global_pool='',  # Don't pool, we'll handle it
        )
        
        # Our own clean multi-head implementation
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes_per_task) 
            for _ in range(num_tasks)
        ])
        
        print(f"  → Created ViT with {num_tasks} heads, {num_classes_per_task} classes each")
        
    def forward_single_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward for a specific task - this is what we'll use for training/eval"""
        # Get features from ViT
        features = self.vit.forward_features(x)  # (batch, num_tokens, embed_dim)
        cls_token = features[:, 0]  # Extract CLS token (batch, embed_dim)
        
        # Pass through specific task head
        if task_id >= len(self.heads):
            raise ValueError(f"Task ID {task_id} >= number of heads {len(self.heads)}")
        
        return self.heads[task_id](cls_token)
    
    def forward_head(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Alias for forward_single_task for compatibility"""
        return self.forward_single_task(x, task_id)
    
    def forward(self, x, task_id=None):
        """Default forward - use task 0 if no task specified"""
        if task_id is None:
            task_id = 0
        return self.forward_single_task(x, task_id)
    
    def get_head_weights(self, task_id: int) -> torch.Tensor:
        """Get weights for a specific task head - for analysis"""
        if task_id >= len(self.heads):
            raise ValueError(f"Task ID {task_id} >= number of heads {len(self.heads)}")
        return self.heads[task_id].weight
    
    def add_head(self, num_classes: int = None) -> int:
        """Add a new head for a new task (if needed for dynamic scenarios)"""
        if num_classes is None:
            num_classes = self.num_classes_per_task
        
        new_head = nn.Linear(self.embed_dim, num_classes)
        self.heads.append(new_head)
        
        # Move to same device as existing heads
        if len(self.heads) > 1:
            device = next(self.heads[0].parameters()).device
            new_head.to(device)
        
        new_task_id = len(self.heads) - 1
        print(f"  → Added head for task {new_task_id} with {num_classes} classes")
        return new_task_id

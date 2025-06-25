import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

class ViTClassProjectionAnalyzer:
    """Analyzes class representations in ViT following Vilas et al."""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model
        self.device = device
        self.representations = defaultdict(dict)  # {block_idx: {token_idx: tensor}}
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to extract intermediate representations"""
        self.remove_hooks()
        
        # Handle wrapped models
        if hasattr(self.model, 'base') and hasattr(self.model.base, 'vit'):
            vit_model = self.model.base.vit
        elif hasattr(self.model, 'vit'):
            vit_model = self.model.vit
        else:
            raise AttributeError("Cannot find ViT model in the provided model")
        
        # Register hooks on the actual ViT blocks
        for idx, block in enumerate(vit_model.blocks):
            hook = block.register_forward_hook(
                lambda m, inp, out, idx=idx: self._save_representations(out, idx)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _save_representations(self, output, block_idx):
        """Save representations from a specific block"""
        # output shape: (batch, num_tokens, embed_dim)
        self.representations[block_idx]['output'] = output.detach()
    
    def compute_class_identifiability(self, 
                                    hidden_states: torch.Tensor, 
                                    embedding_matrix: torch.Tensor,
                                    true_labels: torch.Tensor) -> Dict:
        """
        Project hidden states onto class space and compute identifiability.
        
        Args:
            hidden_states: (batch, num_tokens, embed_dim)
            embedding_matrix: (num_classes, embed_dim) - the classifier weights
            true_labels: (batch,) - true class labels
            
        Returns:
            Dict with identifiability scores per token
        """
        batch_size, num_tokens, _ = hidden_states.shape
        
        # Project onto class space: (batch, num_tokens, num_classes)
        logits = torch.matmul(hidden_states, embedding_matrix.T)
        
        # Compute identifiability for each token
        results = {
            'cls_token': [],
            'image_tokens': [],
            'all_tokens': []
        }
        
        for b in range(batch_size):
            true_class = true_labels[b].item()
            
            # For CLS token (index 0)
            cls_logits = logits[b, 0]
            cls_rank = (cls_logits.argsort(descending=True) == true_class).nonzero().item()
            cls_identifiability = 1.0 - (cls_rank / embedding_matrix.shape[0])
            results['cls_token'].append(cls_identifiability)
            
            # For image tokens (indices 1+)
            img_identifiabilities = []
            for t in range(1, num_tokens):
                token_logits = logits[b, t]
                rank = (token_logits.argsort(descending=True) == true_class).nonzero().item()
                identifiability = 1.0 - (rank / embedding_matrix.shape[0])
                img_identifiabilities.append(identifiability)
            
            results['image_tokens'].append(np.mean(img_identifiabilities))
            results['all_tokens'].append(np.mean([cls_identifiability] + img_identifiabilities))
        
        return {k: np.mean(v) for k, v in results.items()}
    
    def analyze_task_representations(self, 
                                   dataloader,
                                   task_id: int,
                                   num_classes_per_task: int = 10):
        """Analyze representations for a specific task during CL"""
        self.register_hooks()
        
        all_results = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                # Forward pass (hooks will save representations)
                _ = self.model(inputs)
                
                # Get embedding matrix (classifier weights)
                if hasattr(self.model, 'classifier'):  # Multi-head
                    # For task-incremental with MultiHeadClassifier
                    task_classifier = self.model.classifier.classifiers[str(task_id)]
                    # IncrementalClassifier has a 'classifier' Linear layer inside
                    if hasattr(task_classifier, 'classifier'):
                        embed_matrix = task_classifier.classifier.weight
                    elif hasattr(task_classifier, 'weight'):
                        embed_matrix = task_classifier.weight
                    else:
                        # Try to find the linear layer
                        for module in task_classifier.modules():
                            if isinstance(module, torch.nn.Linear):
                                embed_matrix = module.weight
                                break
                        else:
                            raise AttributeError(f"Cannot find weight matrix in {type(task_classifier)}")
                else:
                    # For class-incremental with single head
                    embed_matrix = self.model.head.weight
                    # Only use relevant classes for this task
                    start_idx = task_id * num_classes_per_task
                    end_idx = (task_id + 1) * num_classes_per_task
                    embed_matrix = embed_matrix[start_idx:end_idx]
                    labels = labels - start_idx  # Adjust labels to 0-9 range
                
                # Analyze each block
                for block_idx, reps in self.representations.items():
                    hidden_states = reps['output']
                    scores = self.compute_class_identifiability(
                        hidden_states, embed_matrix, labels
                    )
                    
                    for key, value in scores.items():
                        all_results[f'block_{block_idx}_{key}'].append(value)
                
                # Clear representations for next batch
                self.representations.clear()
        
        self.remove_hooks()
        
        # Average across batches
        return {k: np.mean(v) for k, v in all_results.items()}
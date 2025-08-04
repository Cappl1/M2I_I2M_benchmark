import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

class ViTClassProjectionAnalyzer:
    """Analyzes class representations in ViT following Vilas et al."""
    
    def __init__(self, model: nn.Module, device: Union[str, torch.device] = 'cuda'):
        self.model = model
        self.device = str(device) if isinstance(device, torch.device) else device
        self.representations = defaultdict(dict)  # {block_idx: {token_idx: tensor}}
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to extract intermediate representations"""
        self.remove_hooks()
        
        # Handle wrapped models - try different common ViT model structures
        vit_model: Optional[nn.Module] = None
        if hasattr(self.model, 'base') and hasattr(self.model.base, 'vit'):
            vit_model = getattr(self.model.base, 'vit')
        elif hasattr(self.model, 'vit'):
            vit_model = getattr(self.model, 'vit')
        elif hasattr(self.model, 'encoder'):  # Some ViT implementations use 'encoder'
            vit_model = getattr(self.model, 'encoder')
        elif hasattr(self.model, 'transformer'):  # Another common name
            vit_model = getattr(self.model, 'transformer')
        else:
            # Try to find transformer blocks directly in the model
            for name, module in self.model.named_modules():
                if 'block' in name.lower() or 'layer' in name.lower():
                    if hasattr(module, 'attention') or hasattr(module, 'attn'):
                        # Found transformer-like blocks, use the parent container
                        parts = name.split('.')
                        if len(parts) > 1:
                            parent_name = '.'.join(parts[:-1])
                            vit_model = dict(self.model.named_modules())[parent_name]
                            break
        
        if vit_model is None:
            raise AttributeError("Cannot find ViT transformer blocks in the provided model")
        
        # Register hooks on the transformer blocks
        blocks: Optional[Any] = None
        if hasattr(vit_model, 'blocks'):
            blocks = getattr(vit_model, 'blocks')
        elif hasattr(vit_model, 'layers'):
            blocks = getattr(vit_model, 'layers')
        elif hasattr(vit_model, 'encoder_layers'):
            blocks = getattr(vit_model, 'encoder_layers')
        else:
            # Try to find blocks by iterating through children
            blocks = []
            for child in vit_model.children():
                if hasattr(child, 'attention') or hasattr(child, 'attn'):
                    blocks.append(child)
            
            if not blocks:
                raise AttributeError(f"Cannot find transformer blocks in ViT model: {type(vit_model)}")
        
        if blocks is not None and hasattr(blocks, '__len__'):
            print(f"  → Found {len(blocks)} transformer blocks for analysis")
        
        if blocks is not None and hasattr(blocks, '__iter__'):
            for idx, block in enumerate(blocks):
                if hasattr(block, 'register_forward_hook'):
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
    
    def compute_class_identifiability_fast(self, 
                                          hidden_states: torch.Tensor, 
                                          embedding_matrix: torch.Tensor,
                                          true_labels: torch.Tensor,
                                          sample_tokens: bool = True) -> Dict:
        """
        OPTIMIZED: Project hidden states onto class space and compute identifiability.
        
        Args:
            hidden_states: (batch, num_tokens, embed_dim)
            embedding_matrix: (num_classes, embed_dim) - the classifier weights
            true_labels: (batch,) - true class labels
            sample_tokens: If True, only analyze a subset of image tokens for speed
            
        Returns:
            Dict with identifiability scores per token
        """
        batch_size, num_tokens, _ = hidden_states.shape
        num_classes = embedding_matrix.shape[0]
        
        # Project onto class space: (batch, num_tokens, num_classes)
        logits = torch.matmul(hidden_states, embedding_matrix.T)
        
        # OPTIMIZATION 1: Vectorized rank computation
        # Get logits for true classes
        true_labels_expanded = true_labels.unsqueeze(1).unsqueeze(2).expand(-1, num_tokens, 1)
        true_class_logits = torch.gather(logits, 2, true_labels_expanded).squeeze(2)  # (batch, num_tokens)
        
        # Count how many classes have higher logits (vectorized)
        ranks = (logits > true_class_logits.unsqueeze(2)).sum(dim=2).float()  # (batch, num_tokens)
        identifiabilities = 1.0 - (ranks / num_classes)
        
        # OPTIMIZATION 2: Sample image tokens if requested
        if sample_tokens and num_tokens > 10:
            # Sample 10 random image tokens instead of all 196
            token_indices = torch.randperm(num_tokens - 1)[:10] + 1  # Skip CLS token
            sampled_identifiabilities = identifiabilities[:, token_indices]
        else:
            sampled_identifiabilities = identifiabilities[:, 1:]  # All image tokens
        
        # Compute mean scores
        results = {
            'cls_token': identifiabilities[:, 0].mean().item(),  # CLS token
            'image_tokens': sampled_identifiabilities.mean().item(),  # Image tokens
            'all_tokens': identifiabilities.mean().item()  # All tokens
        }
        
        return results
    
    def analyze_task_representations(self, 
                                dataloader,
                                task_id: int,
                                num_classes_per_task: int = 10,
                                max_batches: int = 50,
                                sample_tokens: bool = True):
        """Analyze representations for a specific task during CL"""
        self.register_hooks()
        
        all_results = defaultdict(list)
        batch_count = 0
        
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break
                        
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    
                    # Forward pass - handle our clean multi-head implementation
                    if hasattr(self.model, 'forward_single_task'):
                        # Our clean multi-head implementation
                        _ = self.model.forward_single_task(inputs, task_id)
                    elif hasattr(self.model, 'forward_head'):
                        _ = self.model.forward_head(inputs, task_id)
                    else:
                        # Single-head model
                        _ = self.model(inputs)
                    
                    # Get embedding matrix using our clean implementation
                    embed_matrix: Optional[torch.Tensor] = None
                    
                    # Handle our clean multi-head implementation
                    if hasattr(self.model, 'heads'):
                        try:
                            embed_matrix = self.model.heads[task_id].weight
                            print(f"  → Found head {task_id} with shape {embed_matrix.shape}")
                        except (IndexError, AttributeError) as e:
                            print(f"  → Error accessing head {task_id}: {e}")
                            continue
                    
                    # Handle our clean single-head implementation  
                    elif hasattr(self.model, 'head') and hasattr(self.model.head, 'weight'):
                        embed_matrix = self.model.head.weight
                        # For class-incremental, slice to get relevant classes
                        start_idx = task_id * num_classes_per_task
                        end_idx = (task_id + 1) * num_classes_per_task
                        if end_idx <= embed_matrix.shape[0]:
                            embed_matrix = embed_matrix[start_idx:end_idx]
                            labels = labels - start_idx  # Adjust labels to 0-9 range
                            print(f"  → Using single head classes {start_idx}-{end_idx-1}")
                    
                    # Handle ViT head directly
                    elif hasattr(self.model, 'vit') and hasattr(self.model.vit, 'head') and hasattr(self.model.vit.head, 'weight'):
                        embed_matrix = self.model.vit.head.weight
                        start_idx = task_id * num_classes_per_task
                        end_idx = (task_id + 1) * num_classes_per_task
                        if end_idx <= embed_matrix.shape[0]:
                            embed_matrix = embed_matrix[start_idx:end_idx]
                            labels = labels - start_idx
                            print(f"  → Using ViT head classes {start_idx}-{end_idx-1}")
                    
                    if embed_matrix is None:
                        print("  → Warning: Could not find classifier weights, skipping batch")
                        batch_count += 1
                        continue
                    
                    # Analyze each block
                    for block_idx, reps in self.representations.items():
                        hidden_states = reps['output']
                        scores = self.compute_class_identifiability_fast(
                            hidden_states, embed_matrix, labels, sample_tokens=sample_tokens
                        )
                        
                        for key, value in scores.items():
                            all_results[f'block_{block_idx}_{key}'].append(value)
                    
                    # Clear representations for next batch
                    self.representations.clear()
                    batch_count += 1
        except Exception as e:
            print(f"  → Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.remove_hooks()
        
        # Average across batches
        final_results = {k: np.mean(v) for k, v in all_results.items() if v}
        print(f"  → Analyzed {batch_count} batches, found {len(final_results)} metrics")
        return final_results
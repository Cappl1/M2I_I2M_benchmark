
    
    
    
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


class ViTPatchImportanceAnalyzer:
    """Analyzes per-patch importance for class identifiability in ViT models."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', patch_size: Optional[int] = None):
        self.model = model
        self.device = device
        self.patch_size = patch_size or self._infer_patch_size()  # Auto-detect if not provided
        self.representations = defaultdict(dict)
        self.hooks = []
        print(f"  → Using patch grid size: {self.patch_size}x{self.patch_size} ({self.patch_size**2} patches total)")
    
    def _infer_patch_size(self) -> int:
        """Infer patch size from the model architecture."""
        # Try to find patch_embed in the ViT model
        vit_model = self._find_vit_model()
        
        # Look for patch embedding
        if hasattr(vit_model, 'patch_embed'):
            img_size = getattr(vit_model.patch_embed, 'img_size', None)
            patch_size = getattr(vit_model.patch_embed, 'patch_size', None)
            
            if img_size and patch_size:
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                
                patches_per_dim = img_size // patch_size
                print(f"  → Detected: {img_size}x{img_size} image, {patch_size}x{patch_size} patches → {patches_per_dim}x{patches_per_dim} grid")
                return patches_per_dim
        
        # Fallback: try to detect from a forward pass
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 64, 64).to(self.device)  # Assume 64x64 input
                _ = self.model(dummy_input)
                # If this works, likely 64x64 input with 8x8 patches = 8x8 grid
                return 8
        except:
            pass
        
        # Final fallback
        print("  → Warning: Could not detect patch size, defaulting to 8x8")
        return 8
        
    def register_hooks(self):
        """Register hooks to extract intermediate representations."""
        self.remove_hooks()
        
        # Find ViT blocks (same logic as before)
        vit_model = self._find_vit_model()
        blocks = self._find_transformer_blocks(vit_model)
        
        # Check if blocks is None or empty
        if blocks is None:
            raise ValueError("Could not find transformer blocks")
        
        if hasattr(blocks, '__len__'):
            print(f"  → Found {len(blocks)} transformer blocks for patch analysis")
        
        # Register hooks - ensure blocks is iterable
        try:
            for idx, block in enumerate(blocks):
                hook = block.register_forward_hook(
                    lambda m, inp, out, idx=idx: self._save_representations(out, idx)
                )
                self.hooks.append(hook)
        except TypeError as e:
            raise ValueError(f"Blocks object is not iterable: {e}")
    
    def _find_vit_model(self):
        """Find the ViT model within potentially wrapped architectures."""
        if hasattr(self.model, 'base') and hasattr(self.model.base, 'vit'):
            return self.model.base.vit
        elif hasattr(self.model, 'vit'):
            return self.model.vit
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer
        else:
            # Try to find transformer blocks directly
            for name, module in self.model.named_modules():
                if 'block' in name.lower() or 'layer' in name.lower():
                    if hasattr(module, 'attention') or hasattr(module, 'attn'):
                        parts = name.split('.')
                        if len(parts) > 1:
                            parent_name = '.'.join(parts[:-1])
                            return dict(self.model.named_modules())[parent_name]
        
        raise AttributeError("Cannot find ViT transformer blocks")
    
    def _find_transformer_blocks(self, vit_model):
        """Find transformer blocks within the ViT model."""
        if hasattr(vit_model, 'blocks'):
            return vit_model.blocks
        elif hasattr(vit_model, 'layers'):
            return vit_model.layers
        elif hasattr(vit_model, 'encoder_layers'):
            return vit_model.encoder_layers
        else:
            # Find blocks by iterating
            blocks = []
            for child in vit_model.children():
                if hasattr(child, 'attention') or hasattr(child, 'attn'):
                    blocks.append(child)
            return blocks if blocks else None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _save_representations(self, output, block_idx):
        """Save representations from a specific block."""
        self.representations[block_idx]['output'] = output.detach()
    
    def compute_patch_identifiability(self, 
                                    hidden_states: torch.Tensor,
                                    embedding_matrix: torch.Tensor,
                                    true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute identifiability for each patch individually.
        
        Returns:
            torch.Tensor: (batch, num_patches) identifiability scores
        """
        batch_size, num_tokens, embed_dim = hidden_states.shape
        num_classes = embedding_matrix.shape[0]
        
        # Project onto class space
        logits = torch.matmul(hidden_states, embedding_matrix.T)  # (batch, num_tokens, num_classes)
        
        # Get logits for true classes
        true_labels_expanded = true_labels.unsqueeze(1).unsqueeze(2).expand(-1, num_tokens, 1)
        true_class_logits = torch.gather(logits, 2, true_labels_expanded).squeeze(2)
        
        # Compute ranks for each token
        ranks = (logits > true_class_logits.unsqueeze(2)).sum(dim=2).float()
        identifiabilities = 1.0 - (ranks / num_classes)
        
        return identifiabilities
    
    def analyze_patch_importance(self,
                               dataloader,
                               num_classes: int = 10,
                               max_batches: Optional[int] = None) -> Dict:
        """
        Analyze importance of each patch position across the dataset.
        
        Returns:
            Dict containing patch importance maps for each layer
        """
        self.register_hooks()
        
        # Initialize storage for patch importance
        # patch_importance[block_idx][patch_idx] = list of identifiability scores
        patch_importance = defaultdict(lambda: defaultdict(list))
        cls_importance = defaultdict(list)
        
        batch_count = 0
        
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches and batch_idx >= max_batches:
                        break
                    
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    
                    # Forward pass
                    _ = self.model(inputs)
                    
                    # Get embedding matrix
                    embed_matrix = self._get_embedding_matrix(num_classes)
                    if embed_matrix is None:
                        continue
                    
                    # Analyze each block
                    for block_idx, reps in self.representations.items():
                        hidden_states = reps['output']
                        
                        # Compute identifiability for each patch
                        patch_scores = self.compute_patch_identifiability(
                            hidden_states, embed_matrix, labels
                        )  # (batch, num_tokens)
                        
                        # Store CLS token scores
                        cls_importance[block_idx].extend(patch_scores[:, 0].cpu().numpy())
                        
                        # Store image patch scores (tokens 1 to num_patches)
                        expected_patches = self.patch_size * self.patch_size
                        actual_patches = patch_scores.shape[1] - 1  # Exclude CLS token
                        
                        if actual_patches != expected_patches:
                            print(f"  → Warning: Expected {expected_patches} patches but found {actual_patches}")
                        
                        for patch_idx in range(1, patch_scores.shape[1]):
                            patch_importance[block_idx][patch_idx-1].extend(
                                patch_scores[:, patch_idx].cpu().numpy()
                            )
                    
                    self.representations.clear()
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        print(f"    Analyzed {batch_count} batches...")
                        
        finally:
            self.remove_hooks()
        
        # Compute average importance for each patch position
        results = {
            'patch_importance_maps': {},
            'cls_token_importance': {},
            'statistics': {}
        }
        
        for block_idx in sorted(patch_importance.keys()):
            # Create patch importance map (14x14)
            importance_map = np.zeros((self.patch_size, self.patch_size))
            
            for patch_idx in range(self.patch_size * self.patch_size):
                if patch_idx in patch_importance[block_idx]:
                    scores = patch_importance[block_idx][patch_idx]
                    avg_importance = np.mean(scores)
                    
                    # Convert linear index to 2D position
                    row = patch_idx // self.patch_size
                    col = patch_idx % self.patch_size
                    importance_map[row, col] = avg_importance
            
            results['patch_importance_maps'][f'block_{block_idx}'] = importance_map
            results['cls_token_importance'][f'block_{block_idx}'] = np.mean(cls_importance[block_idx])
            
            # Compute statistics
            flat_importance = importance_map.flatten()
            results['statistics'][f'block_{block_idx}'] = {
                'mean': float(np.mean(flat_importance)),
                'std': float(np.std(flat_importance)),
                'min': float(np.min(flat_importance)),
                'max': float(np.max(flat_importance)),
                'top_5_patches': [int(idx) for idx in np.argsort(flat_importance)[-5:][::-1]],
                'bottom_5_patches': [int(idx) for idx in np.argsort(flat_importance)[:5]]
            }
        
        print(f"  → Completed patch analysis on {batch_count} batches")
        return results
    
    def _get_embedding_matrix(self, num_classes: int) -> Optional[torch.Tensor]:
        """Extract the classifier embedding matrix."""
        if hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'weight'):
                return self.model.classifier.weight[:num_classes]
        elif hasattr(self.model, 'head') and hasattr(self.model.head, 'weight'):
            return self.model.head.weight[:num_classes]
        elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'weight'):
            return self.model.fc.weight[:num_classes]
        
        # Try to find linear layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == num_classes:
                return module.weight
        
        print("  → Warning: Could not find classifier weights")
        return None
    
    def visualize_patch_importance(self, 
                                 results: Dict,
                                 save_dir: Path,
                                 epoch: int,
                                 dataset_name: str):
        """Create visualizations of patch importance maps."""
        patch_maps = results['patch_importance_maps']
        num_blocks = len(patch_maps)
        
        # Create grid plot
        cols = 4
        rows = (num_blocks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        vmin = min(m.min() for m in patch_maps.values())
        vmax = max(m.max() for m in patch_maps.values())
        
        for idx, (block_name, importance_map) in enumerate(sorted(patch_maps.items())):
            if idx < len(axes):
                ax = axes[idx]
                
                # Create heatmap
                im = ax.imshow(importance_map, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f'{block_name.replace("_", " ").title()}')
                ax.set_xlabel('Patch Column')
                ax.set_ylabel('Patch Row')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add grid
                ax.set_xticks(np.arange(-0.5, self.patch_size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, self.patch_size, 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2)
        
        # Hide empty subplots
        for idx in range(len(patch_maps), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Patch Importance Maps - {dataset_name} (Epoch {epoch})', fontsize=16)
        plt.tight_layout()
        
        save_path = save_dir / f'patch_importance_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary heatmap showing evolution
        self._create_summary_heatmap(results, save_dir, epoch, dataset_name)
        
        return save_path
    
    def _create_summary_heatmap(self, results: Dict, save_dir: Path, epoch: int, dataset_name: str):
        """Create a summary showing most/least important patches."""
        stats = results['statistics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Average importance across layers
        blocks = sorted([int(b.split('_')[1]) for b in stats.keys()])
        avg_importance = [stats[f'block_{b}']['mean'] for b in blocks]
        std_importance = [stats[f'block_{b}']['std'] for b in blocks]
        
        ax1.errorbar(blocks, avg_importance, yerr=std_importance, 
                    marker='o', linewidth=2, capsize=5)
        ax1.set_xlabel('Transformer Block')
        ax1.set_ylabel('Average Patch Importance')
        ax1.set_title('Patch Importance Across Layers')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance heatmap
        variance_map = np.zeros((len(blocks), 5))  # Top 5 most variable positions
        
        for i, block in enumerate(blocks):
            block_map = results['patch_importance_maps'][f'block_{block}']
            flat_map = block_map.flatten()
            # Get indices of patches with highest variance across spatial positions
            variances = []
            for patch_idx in range(len(flat_map)):
                row = patch_idx // self.patch_size
                col = patch_idx % self.patch_size
                # Simple variance metric: distance from center
                center = self.patch_size // 2
                center_dist = np.sqrt((row - center)**2 + (col - center)**2)
                variances.append(flat_map[patch_idx] * center_dist)
            
            top_var_indices = np.argsort(variances)[-5:][::-1]
            for j, idx in enumerate(top_var_indices):
                variance_map[i, j] = flat_map[idx]
        
        im = ax2.imshow(variance_map.T, aspect='auto', cmap='plasma')
        ax2.set_xlabel('Transformer Block')
        ax2.set_ylabel('Top Variable Patches')
        ax2.set_title('Most Variable Patch Positions')
        plt.colorbar(im, ax=ax2)
        
        plt.suptitle(f'Patch Importance Summary - {dataset_name} (Epoch {epoch})', fontsize=14)
        plt.tight_layout()
        
        save_path = save_dir / f'patch_summary_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, results: Dict, save_dir: Path, epoch: int):
        """Save detailed patch importance results for later analysis."""
        # Save as numpy for easier loading (this is the main format we use)
        np_path = save_dir / f'patch_importance_epoch_{epoch:03d}.npz'
        np.savez_compressed(
            np_path,
            **{k: v for k, v in results['patch_importance_maps'].items()},
            cls_importance=np.array(list(results['cls_token_importance'].values())),
            block_names=list(results['patch_importance_maps'].keys())
        )
        
        # Also save as JSON (properly converted) for human readability
        json_path = None
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'epoch': epoch,
                'patch_importance_maps': {},
                'cls_token_importance': {},
                'statistics': results['statistics']
            }
            
            # Convert numpy arrays to Python lists
            for block_name, importance_map in results['patch_importance_maps'].items():
                json_results['patch_importance_maps'][block_name] = importance_map.astype(float).tolist()
            
            # Convert cls token importance (numpy scalars to Python floats)
            for block_name, importance in results['cls_token_importance'].items():
                json_results['cls_token_importance'][block_name] = float(importance)
            
            # Save as JSON
            json_path = save_dir / f'patch_importance_epoch_{epoch:03d}.json'
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
                
        except Exception as e:
            print(f"  → Warning: Could not save JSON (not critical): {e}")
            json_path = None
        
        return json_path, np_path
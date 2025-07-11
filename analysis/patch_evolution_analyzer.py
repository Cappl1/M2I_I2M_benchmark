#!/usr/bin/env python3
"""
Analyze patch importance evolution across training epochs.
Compare patch importance patterns across different datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse


class PatchEvolutionAnalyzer:
    """Analyze how patch importance evolves during training."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.patch_dir = self.experiment_dir / 'patch_analysis'
        
        if not self.patch_dir.exists():
            raise ValueError(f"No patch analysis directory found at {self.patch_dir}")
        
        # Load summary
        self.summary = self._load_summary()
        self.dataset_name = self.summary.get('dataset', 'unknown')
        
        # Infer patch size from data
        self.patch_size = self._infer_patch_size()
        print(f"  → Detected patch grid size: {self.patch_size}x{self.patch_size}")
    
    def _infer_patch_size(self) -> int:
        """Infer patch size from the saved data."""
        # Load first available patch data to get dimensions
        for npz_file in sorted(self.patch_dir.glob('patch_importance_epoch_*.npz')):
            try:
                data = np.load(npz_file)
                for key in data.files:
                    if key.startswith('block_') and key != 'block_names':
                        arr = data[key]
                        if arr.ndim == 2:  # Should be square patch grid
                            return arr.shape[0]
            except Exception:
                continue
        
        # Fallback
        print("  → Warning: Could not detect patch size, defaulting to 8")
        return 8
        
    def _load_summary(self) -> Dict:
        """Load patch importance summary."""
        summary_file = self.patch_dir / 'patch_importance_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_patch_data(self) -> Dict[int, Dict]:
        """Load all patch importance data across epochs."""
        patch_data = {}
        
        # Find all patch importance files
        for npz_file in sorted(self.patch_dir.glob('patch_importance_epoch_*.npz')):
            epoch = int(npz_file.stem.split('_')[-1])
            
            # Load numpy data
            data = np.load(npz_file)
            
            # Extract block data (exclude metadata like 'block_names', 'cls_importance')
            block_data = {}
            for key in data.files:
                if key.startswith('block_') and key != 'block_names':
                    # Ensure we're loading numerical data, not strings
                    arr = data[key]
                    if arr.dtype.kind in 'fc':  # float or complex numbers
                        block_data[key] = arr
                    else:
                        print(f"  → Warning: Skipping non-numeric data in key '{key}' (dtype: {arr.dtype})")
            
            if block_data:  # Only add if we found valid data
                patch_data[epoch] = block_data
            else:
                print(f"  → Warning: No valid patch data found for epoch {epoch}")
        
        return patch_data
    
    def analyze_patch_evolution(self, output_dir: Path):
        """Analyze how patch importance evolves during training."""
        patch_data = self.load_patch_data()
        epochs = sorted(patch_data.keys())
        
        if len(epochs) < 2:
            print("Not enough epochs for evolution analysis")
            return
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Patch stability analysis
        self._analyze_patch_stability(patch_data, epochs, output_dir)
        
        # 2. Important patch consistency
        self._analyze_important_patches(patch_data, epochs, output_dir)
        
        # 3. Layer-wise evolution
        self._analyze_layer_evolution(patch_data, epochs, output_dir)
        
        # 4. Spatial patterns
        self._analyze_spatial_patterns(patch_data, epochs, output_dir)
    
    def _analyze_patch_stability(self, patch_data: Dict, epochs: List[int], output_dir: Path):
        """Analyze stability of patch importance across training."""
        # Calculate correlation between consecutive epochs
        correlations = {block: [] for block in patch_data[epochs[0]].keys()}
        
        for i in range(len(epochs) - 1):
            epoch1, epoch2 = epochs[i], epochs[i+1]
            
            for block in patch_data[epoch1].keys():
                if block in patch_data[epoch2]:
                    map1 = patch_data[epoch1][block].flatten()
                    map2 = patch_data[epoch2][block].flatten()
                    
                    # Ensure both maps are numeric and same size
                    if map1.dtype.kind in 'fc' and map2.dtype.kind in 'fc' and len(map1) == len(map2):
                        try:
                            corr = np.corrcoef(map1, map2)[0, 1]
                            if not np.isnan(corr):  # Only add valid correlations
                                correlations[block].append(corr)
                        except Exception as e:
                            print(f"  → Warning: Could not compute correlation for {block} between epochs {epoch1}-{epoch2}: {e}")
                    else:
                        print(f"  → Warning: Invalid data for {block} between epochs {epoch1}-{epoch2}")
        
        # Plot stability
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for block, corrs in sorted(correlations.items()):
            block_num = int(block.split('_')[1])
            ax.plot(epochs[1:], corrs, 'o-', label=f'Block {block_num}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation with Previous Epoch')
        ax.set_title(f'Patch Importance Stability - {self.dataset_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = output_dir / 'patch_stability.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_important_patches(self, patch_data: Dict, epochs: List[int], output_dir: Path):
        """Track which patches remain important throughout training."""
        # Get top-k patches for each block and epoch
        k = 10  # Top 10 patches
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        blocks = sorted(patch_data[epochs[0]].keys())[:12]  # Up to 12 blocks
        
        for idx, block in enumerate(blocks):
            ax = axes[idx]
            
            # Track top patches across epochs
            top_patches_matrix = []
            
            for epoch in epochs:
                if block in patch_data[epoch]:
                    importance_map = patch_data[epoch][block]
                    flat_map = importance_map.flatten()
                    top_indices = np.argsort(flat_map)[-k:][::-1]
                    
                    # Create binary mask
                    mask = np.zeros_like(flat_map)
                    mask[top_indices] = 1
                    top_patches_matrix.append(mask)
            
            if top_patches_matrix:
                # Stack and visualize
                top_patches_matrix = np.array(top_patches_matrix)
                
                # Calculate persistence (how often each patch is in top-k)
                persistence = top_patches_matrix.mean(axis=0)
                persistence_2d = persistence.reshape(self.patch_size, self.patch_size)
                
                im = ax.imshow(persistence_2d, cmap='magma', vmin=0, vmax=1)
                ax.set_title(f'{block.replace("_", " ").title()}')
                ax.axis('off')
                
                # Add colorbar for last plot
                if idx == len(blocks) - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(len(blocks), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Top-{k} Patch Persistence - {self.dataset_name}', fontsize=16)
        plt.tight_layout()
        
        save_path = output_dir / 'important_patches_persistence.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_layer_evolution(self, patch_data: Dict, epochs: List[int], output_dir: Path):
        """Analyze how patch importance evolves across layers."""
        # Calculate average importance per layer over time
        layer_importance = {block: [] for block in patch_data[epochs[0]].keys()}
        
        for epoch in epochs:
            for block in patch_data[epoch].keys():
                block_data = patch_data[epoch][block]
                if block_data.dtype.kind in 'fc':  # Only numeric data
                    try:
                        avg_importance = block_data.mean()
                        if not np.isnan(avg_importance):
                            layer_importance[block].append(avg_importance)
                    except Exception as e:
                        print(f"  → Warning: Could not compute mean for {block} at epoch {epoch}: {e}")
        
        # Create heatmap
        blocks = sorted(layer_importance.keys(), key=lambda x: int(x.split('_')[1]))
        importance_matrix = np.array([layer_importance[block] for block in blocks])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(importance_matrix, 
                    xticklabels=[str(e) for e in epochs],  # Convert to strings
                    yticklabels=[f'Block {b.split("_")[1]}' for b in blocks],
                    cmap='viridis',
                    cbar_kws={'label': 'Average Patch Importance'})
        
        plt.xlabel('Epoch')
        plt.ylabel('Transformer Block')
        plt.title(f'Layer-wise Patch Importance Evolution - {self.dataset_name}')
        plt.tight_layout()
        
        save_path = output_dir / 'layer_evolution_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_spatial_patterns(self, patch_data: Dict, epochs: List[int], output_dir: Path):
        """Analyze spatial patterns in patch importance."""
        # Average importance across all epochs and blocks
        all_maps = []
        
        for epoch in epochs:
            for block_data in patch_data[epoch].values():
                if block_data.dtype.kind in 'fc':  # Only numeric data
                    all_maps.append(block_data)
        
        if not all_maps:
            print("  → Warning: No valid patch data found for spatial analysis")
            return
            
        avg_importance = np.mean(all_maps, axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap
        im1 = ax1.imshow(avg_importance, cmap='viridis')
        ax1.set_title('Average Patch Importance')
        ax1.set_xlabel('Patch Column')
        ax1.set_ylabel('Patch Row')
        plt.colorbar(im1, ax=ax1)
        
        # Radial analysis
        center = self.patch_size // 2  # Center of patch grid
        distances = []
        importances = []
        
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                distances.append(dist)
                importances.append(avg_importance[i, j])
        
        # Bin by distance
        max_dist = np.sqrt(2) * center
        bins = np.linspace(0, max_dist, 8)
        bin_indices = np.digitize(distances, bins)
        
        avg_by_distance = []
        std_by_distance = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                avg_by_distance.append(np.mean(np.array(importances)[mask]))
                std_by_distance.append(np.std(np.array(importances)[mask]))
            else:
                avg_by_distance.append(0)
                std_by_distance.append(0)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.errorbar(bin_centers, avg_by_distance, yerr=std_by_distance, 
                    marker='o', linewidth=2, capsize=5)
        ax2.set_xlabel('Distance from Center')
        ax2.set_ylabel('Average Importance')
        ax2.set_title('Center-Periphery Analysis')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Spatial Patterns in Patch Importance - {self.dataset_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = output_dir / 'spatial_patterns.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_evolution_summary(self, output_dir: Path):
        """Create a comprehensive summary of patch evolution."""
        patch_data = self.load_patch_data()
        epochs = sorted(patch_data.keys())
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary report
        summary = {
            'dataset': self.dataset_name,
            'epochs_analyzed': epochs,
            'findings': {}
        }
        
        # Check if we have enough data
        if len(epochs) < 2:
            summary['findings']['error'] = 'Not enough epochs for analysis (need at least 2)'
        elif not patch_data:
            summary['findings']['error'] = 'No patch data found'
        
        # Calculate key metrics
        if len(epochs) >= 2 and patch_data:
            # 1. Overall stability
            all_correlations = []
            for i in range(len(epochs) - 1):
                for block in patch_data[epochs[i]].keys():
                    if block in patch_data[epochs[i+1]]:
                        map1 = patch_data[epochs[i]][block].flatten()
                        map2 = patch_data[epochs[i+1]][block].flatten()
                        
                        if map1.dtype.kind in 'fc' and map2.dtype.kind in 'fc' and len(map1) == len(map2):
                            try:
                                corr = np.corrcoef(map1, map2)[0, 1]
                                if not np.isnan(corr):
                                    all_correlations.append(corr)
                            except Exception:
                                pass  # Skip problematic correlations
            
            if all_correlations:
                summary['findings']['average_stability'] = float(np.mean(all_correlations))
                summary['findings']['stability_std'] = float(np.std(all_correlations))
            else:
                summary['findings']['average_stability'] = 'No valid correlations found'
                summary['findings']['stability_std'] = 'No valid correlations found'
            
            # 2. Most/least stable blocks
            block_stability = {}
            for block in patch_data[epochs[0]].keys():
                correlations = []
                for i in range(len(epochs) - 1):
                    if block in patch_data[epochs[i]] and block in patch_data[epochs[i+1]]:
                        map1 = patch_data[epochs[i]][block].flatten()
                        map2 = patch_data[epochs[i+1]][block].flatten()
                        
                        if map1.dtype.kind in 'fc' and map2.dtype.kind in 'fc' and len(map1) == len(map2):
                            try:
                                corr = np.corrcoef(map1, map2)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                            except Exception:
                                pass  # Skip problematic correlations
                
                if correlations:
                    block_stability[block] = np.mean(correlations)
            
            if block_stability:
                most_stable = max(block_stability.keys(), key=lambda k: block_stability[k])
                least_stable = min(block_stability.keys(), key=lambda k: block_stability[k])
                
                summary['findings']['most_stable_block'] = most_stable
                summary['findings']['most_stable_correlation'] = float(block_stability[most_stable])
                summary['findings']['least_stable_block'] = least_stable
                summary['findings']['least_stable_correlation'] = float(block_stability[least_stable])
        
        # Save summary
        summary_path = output_dir / 'evolution_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def compare_datasets(experiment_dirs: List[Path], output_dir: Path):
    """Compare patch importance patterns across different datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_patterns = {}
    
    for exp_dir in experiment_dirs:
        try:
            analyzer = PatchEvolutionAnalyzer(exp_dir)
            patch_data = analyzer.load_patch_data()
            
            if patch_data:
                # Get final epoch data
                final_epoch = max(patch_data.keys())
                
                # Average across all blocks
                all_maps = []
                for block_data in patch_data[final_epoch].values():
                    all_maps.append(block_data)
                
                avg_map = np.mean(all_maps, axis=0)
                dataset_patterns[analyzer.dataset_name] = avg_map
        
        except Exception as e:
            print(f"Failed to analyze {exp_dir}: {e}")
    
    # Create comparison visualization
    if len(dataset_patterns) > 1:
        n_datasets = len(dataset_patterns)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
        
        if n_datasets == 1:
            axes = [axes]
        
        vmin = min(m.min() for m in dataset_patterns.values())
        vmax = max(m.max() for m in dataset_patterns.values())
        
        for idx, (dataset, pattern) in enumerate(sorted(dataset_patterns.items())):
            ax = axes[idx]
            im = ax.imshow(pattern, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(dataset.upper())
            ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        
        plt.suptitle('Patch Importance Patterns Across Datasets', fontsize=16)
        plt.tight_layout()
        
        save_path = output_dir / 'dataset_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze patch importance evolution')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output', type=str, default='patch_evolution_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        experiment_dirs = [Path(args.experiment_dir)] + [Path(d) for d in args.compare]
        compare_datasets(experiment_dirs, args.output)
    else:
        # Analyze single experiment
        analyzer = PatchEvolutionAnalyzer(Path(args.experiment_dir))
        output_dir = Path(args.output)
        
        print(f"Analyzing patch evolution for: {analyzer.dataset_name}")
        analyzer.analyze_patch_evolution(output_dir)
        summary = analyzer.create_evolution_summary(output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Key findings:")
        for key, value in summary.get('findings', {}).items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
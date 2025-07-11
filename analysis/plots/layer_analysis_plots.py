"""Plotting utilities for layer analysis results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class LayerAnalysisPlotter:
    """Creates plots from layer analysis results."""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = Path(analysis_dir)
        self.results = self._load_all_results()
        
    def _load_all_results(self) -> List[Dict]:
        """Load all analysis results from the directory."""
        results = []
        for file_path in sorted(self.analysis_dir.glob('epoch_*.json')):
            with open(file_path, 'r') as f:
                results.append(json.load(f))
        return results
    
    def plot_layer_statistics_progression(self, save_path: Optional[str] = None):
        """Plot how layer statistics evolve during training."""
        if not self.results or self.results[0]['analyzer_type'] != 'simple':
            print("No simple layer analysis results found")
            return
            
        # Extract data
        epochs = [r['epoch'] for r in self.results]
        layer_names = list(self.results[0]['layer_stats'].keys())
        
        # Create subplots for different statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Layer Statistics Progression During Training', fontsize=16)
        
        stats = ['mean', 'std', 'norm']
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
        
        for idx, stat in enumerate(stats):
            if idx >= 3:
                break
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            for layer_idx, layer_name in enumerate(layer_names):
                values = []
                for result in self.results:
                    values.append(result['layer_stats'][layer_name][stat])
                
                ax.plot(epochs, values, label=layer_name, color=colors[layer_idx], 
                       marker='o', markersize=4, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{stat.capitalize()}')
            ax.set_title(f'Layer {stat.capitalize()} Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Use the last subplot for layer count/info
        axes[1, 1].axis('off')
        info_text = f"Total Layers: {len(layer_names)}\n"
        info_text += f"Analysis Points: {len(epochs)}\n"
        info_text += f"Epochs: {epochs[0]} - {epochs[-1]}"
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer statistics plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_vit_cls_vs_image_tokens(self, save_path: Optional[str] = None):
        """Plot CLS token vs image token analysis for ViT models."""
        vit_results = [r for r in self.results if r['analyzer_type'] in ['vit_projection']]
        
        if not vit_results:
            print("No ViT projection analysis results found")
            return
            
        epochs = [r['epoch'] for r in vit_results]
        
        # Extract CLS and image token metrics
        cls_metrics = {}
        img_metrics = {}
        
        for result in vit_results:
            epoch = result['epoch']
            scores = result['projection_scores']
            
            # Separate CLS and image token metrics
            for key, value in scores.items():
                if 'cls' in key.lower():
                    if key not in cls_metrics:
                        cls_metrics[key] = []
                    cls_metrics[key].append((epoch, value))
                elif 'token' in key.lower() or 'patch' in key.lower():
                    if key not in img_metrics:
                        img_metrics[key] = []
                    img_metrics[key].append((epoch, value))
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CLS Token vs Image Token Analysis During Training', fontsize=16)
        
        # Plot CLS token metrics
        ax = axes[0, 0]
        for metric_name, values in cls_metrics.items():
            if values:
                epochs_vals, metric_vals = zip(*values)
                ax.plot(epochs_vals, metric_vals, label=metric_name, marker='o', markersize=4)
        ax.set_title('CLS Token Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot image token metrics
        ax = axes[0, 1]
        for metric_name, values in img_metrics.items():
            if values:
                epochs_vals, metric_vals = zip(*values)
                ax.plot(epochs_vals, metric_vals, label=metric_name, marker='s', markersize=4)
        ax.set_title('Image Token Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Combined comparison for key metrics
        ax = axes[1, 0]
        key_cls_metric = list(cls_metrics.keys())[0] if cls_metrics else None
        key_img_metric = list(img_metrics.keys())[0] if img_metrics else None
        
        if key_cls_metric and key_img_metric:
            cls_epochs, cls_vals = zip(*cls_metrics[key_cls_metric])
            img_epochs, img_vals = zip(*img_metrics[key_img_metric])
            
            ax.plot(cls_epochs, cls_vals, label=f'CLS: {key_cls_metric}', 
                   marker='o', markersize=6, linewidth=2, color='blue')
            ax.plot(img_epochs, img_vals, label=f'Image: {key_img_metric}', 
                   marker='s', markersize=6, linewidth=2, color='red')
            
        ax.set_title('CLS vs Image Token Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"Analysis Epochs: {len(epochs)}\n"
        summary_text += f"CLS Metrics: {len(cls_metrics)}\n" 
        summary_text += f"Image Metrics: {len(img_metrics)}\n"
        if epochs:
            summary_text += f"Epoch Range: {min(epochs)} - {max(epochs)}"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CLS vs Image tokens plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_block_progression_heatmap(self, save_path: Optional[str] = None):
        """Plot heatmap showing how different blocks evolve during training."""
        vit_results = [r for r in self.results if r['analyzer_type'] in ['vit_projection']]
        
        if not vit_results:
            print("No ViT projection analysis results found")
            return
            
        epochs = [r['epoch'] for r in vit_results]
        
        # Extract block-wise metrics
        block_data = {}
        
        for result in vit_results:
            epoch = result['epoch']
            scores = result['projection_scores']
            
            for metric_name, value in scores.items():
                if 'block_' in metric_name:
                    if metric_name not in block_data:
                        block_data[metric_name] = {}
                    block_data[metric_name][epoch] = value
        
        if not block_data:
            print("No block-specific metrics found")
            return
            
        # Create DataFrame for heatmap
        df_data = []
        for metric_name, epoch_values in block_data.items():
            for epoch, value in epoch_values.items():
                df_data.append({
                    'Metric': metric_name,
                    'Epoch': epoch,
                    'Value': value
                })
        
        if not df_data:
            return
            
        df = pd.DataFrame(df_data)
        pivot_df = df.pivot(index='Metric', columns='Epoch', values='Value')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Block-wise Metric Progression During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Block Metric')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Block progression heatmap saved to {save_path}")
        else:
            plt.show()
    
    def create_all_plots(self, output_dir: str):
        """Create all available plots and save them."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating plots in {output_path}")
        
        # Layer statistics plot
        self.plot_layer_statistics_progression(
            save_path=output_path / "layer_statistics_progression.png"
        )
        
        # ViT-specific plots
        self.plot_vit_cls_vs_image_tokens(
            save_path=output_path / "cls_vs_image_tokens.png"  
        )
        
        self.plot_block_progression_heatmap(
            save_path=output_path / "block_progression_heatmap.png"
        )
        
        print(f"All plots created in {output_path}")


def main():
    """Example usage of the plotter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create plots from layer analysis results")
    parser.add_argument("--analysis_dir", required=True, 
                       help="Directory containing layer analysis JSON files")
    parser.add_argument("--output_dir", default="plots",
                       help="Directory to save plots")
    parser.add_argument("--plot_type", choices=['all', 'layer_stats', 'cls_vs_img', 'heatmap'],
                       default='all', help="Type of plot to create")
    
    args = parser.parse_args()
    
    plotter = LayerAnalysisPlotter(args.analysis_dir)
    
    if args.plot_type == 'all':
        plotter.create_all_plots(args.output_dir)
    elif args.plot_type == 'layer_stats':
        plotter.plot_layer_statistics_progression()
    elif args.plot_type == 'cls_vs_img':
        plotter.plot_vit_cls_vs_image_tokens()
    elif args.plot_type == 'heatmap':
        plotter.plot_block_progression_heatmap()


if __name__ == "__main__":
    main() 
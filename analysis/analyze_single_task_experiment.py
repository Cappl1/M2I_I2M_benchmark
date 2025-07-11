import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.gridspec as gridspec

class ViTProjectionAnalyzer:
    """Analyze and visualize ViT class projection results."""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = Path(analysis_dir)
        self.data = self.load_analysis_data()
        
    def load_analysis_data(self) -> Dict:
        """Load all analysis JSON files from the directory."""
        data = {}
        
        for json_file in sorted(self.analysis_dir.glob("epoch_*.json")):
            epoch = int(json_file.stem.split('_')[1])
            with open(json_file, 'r') as f:
                data[epoch] = json.load(f)
                
        print(f"Loaded analysis data for {len(data)} epochs")
        return data
    
    def extract_layer_progression(self) -> pd.DataFrame:
        """Extract how identifiability progresses through layers."""
        records = []
        
        for epoch, epoch_data in self.data.items():
            scores = epoch_data['projection_scores']
            
            # Parse scores by block and token type
            for key, value in scores.items():
                if key.startswith('block_'):
                    parts = key.split('_')
                    block_idx = int(parts[1])
                    token_type = '_'.join(parts[2:])
                    
                    # Skip all_tokens as it's dominated by image tokens
                    if token_type == 'all_tokens':
                        continue
                    
                    records.append({
                        'epoch': epoch,
                        'block': block_idx,
                        'token_type': token_type,
                        'identifiability': value
                    })
        
        return pd.DataFrame(records)
    
    def plot_layer_progression(self, epoch: int = None):
        """Plot how class identifiability progresses through transformer layers."""
        df = self.extract_layer_progression()
        
        if epoch is not None:
            df = df[df['epoch'] == epoch]
            title_suffix = f" (Epoch {epoch})"
        else:
            # Use last epoch
            epoch = df['epoch'].max()
            df = df[df['epoch'] == epoch]
            title_suffix = f" (Epoch {epoch})"
        
        plt.figure(figsize=(10, 6))
        
        # Plot different token types (excluding all_tokens)
        for token_type in df['token_type'].unique():
            token_df = df[df['token_type'] == token_type]
            token_df = token_df.sort_values('block')
            
            label = {
                'cls_token': 'CLS Token',
                'image_tokens': 'Image Tokens (avg)'
            }.get(token_type, token_type)
            
            plt.plot(token_df['block'], token_df['identifiability'], 
                    marker='o', linewidth=2, markersize=8, label=label)
        
        plt.xlabel('Transformer Block', fontsize=12)
        plt.ylabel('Class Identifiability', fontsize=12)
        plt.title(f'Class Information Across ViT Layers{title_suffix}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_training_evolution(self):
        """Plot how identifiability evolves during training."""
        df = self.extract_layer_progression()
        
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)
        
        # 1. CLS token evolution
        ax1 = fig.add_subplot(gs[0])
        cls_df = df[df['token_type'] == 'cls_token']
        pivot_cls = cls_df.pivot(index='epoch', columns='block', values='identifiability')
        
        im1 = ax1.imshow(pivot_cls.T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Block')
        ax1.set_title('CLS Token Identifiability')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Image tokens evolution
        ax2 = fig.add_subplot(gs[1])
        img_df = df[df['token_type'] == 'image_tokens']
        pivot_img = img_df.pivot(index='epoch', columns='block', values='identifiability')
        
        im2 = ax2.imshow(pivot_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Block')
        ax2.set_title('Image Tokens Identifiability')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Difference (CLS - Image)
        ax3 = fig.add_subplot(gs[2])
        diff = pivot_cls - pivot_img
        im3 = ax3.imshow(diff.T, aspect='auto', cmap='RdBu_r', origin='lower', 
                        vmin=-0.3, vmax=0.3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Block')
        ax3.set_title('CLS - Image Tokens Difference')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        return fig
    
    def plot_early_vs_late_layers(self):
        """Compare early vs late layer dynamics during training."""
        df = self.extract_layer_progression()
        
        # Define early and late blocks
        all_blocks = sorted(df['block'].unique())
        early_blocks = all_blocks[:3]  # First 3 blocks
        late_blocks = all_blocks[-3:]   # Last 3 blocks
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot early blocks
        for token_type in ['cls_token', 'image_tokens']:
            early_df = df[(df['token_type'] == token_type) & 
                         (df['block'].isin(early_blocks))]
            early_avg = early_df.groupby('epoch')['identifiability'].mean()
            
            label = 'CLS' if token_type == 'cls_token' else 'Image'
            ax1.plot(early_avg.index, early_avg.values, 
                    marker='o', label=label, linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Identifiability')
        ax1.set_title(f'Early Layers (Blocks {early_blocks[0]}-{early_blocks[-1]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot late blocks
        for token_type in ['cls_token', 'image_tokens']:
            late_df = df[(df['token_type'] == token_type) & 
                        (df['block'].isin(late_blocks))]
            late_avg = late_df.groupby('epoch')['identifiability'].mean()
            
            label = 'CLS' if token_type == 'cls_token' else 'Image'
            ax2.plot(late_avg.index, late_avg.values, 
                    marker='o', label=label, linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Avg Identifiability')
        ax2.set_title(f'Late Layers (Blocks {late_blocks[0]}-{late_blocks[-1]})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_insights(self) -> Dict[str, any]:
        """Generate insights from the analysis."""
        df = self.extract_layer_progression()
        insights = {}
        
        # 1. Which layers show strongest class separation?
        last_epoch = df['epoch'].max()
        last_df = df[df['epoch'] == last_epoch]
        
        best_cls_block = last_df[last_df['token_type'] == 'cls_token'].nlargest(1, 'identifiability')
        best_img_block = last_df[last_df['token_type'] == 'image_tokens'].nlargest(1, 'identifiability')
        
        insights['best_cls_block'] = int(best_cls_block['block'].values[0])
        insights['best_cls_identifiability'] = float(best_cls_block['identifiability'].values[0])
        insights['best_img_block'] = int(best_img_block['block'].values[0])
        insights['best_img_identifiability'] = float(best_img_block['identifiability'].values[0])
        
        # 2. How much does identifiability improve during training?
        first_epoch = df['epoch'].min()
        
        first_cls = df[(df['epoch'] == first_epoch) & 
                      (df['token_type'] == 'cls_token')]['identifiability'].mean()
        last_cls = df[(df['epoch'] == last_epoch) & 
                     (df['token_type'] == 'cls_token')]['identifiability'].mean()
        
        insights['cls_improvement'] = (last_cls - first_cls) / first_cls * 100
        
        # 3. CLS vs Image token specialization
        cls_final = last_df[last_df['token_type'] == 'cls_token']['identifiability'].mean()
        img_final = last_df[last_df['token_type'] == 'image_tokens']['identifiability'].mean()
        
        insights['cls_specialization'] = (cls_final - img_final) / img_final * 100
        
        # 4. Layer-wise progression pattern
        cls_progression = last_df[last_df['token_type'] == 'cls_token'].sort_values('block')
        blocks = cls_progression['block'].values
        scores = cls_progression['identifiability'].values
        
        # Check if monotonic increasing
        insights['monotonic_increase'] = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        
        # Find biggest jump
        if len(scores) > 1:
            jumps = np.diff(scores)
            max_jump_idx = np.argmax(jumps)
            insights['biggest_jump_blocks'] = (int(blocks[max_jump_idx]), int(blocks[max_jump_idx + 1]))
            insights['biggest_jump_value'] = float(jumps[max_jump_idx])
        
        return insights
    
    def create_summary_report(self, save_path: str = None):
        """Create a comprehensive summary report without text."""
        # Create figure with subplots (removed text subplot)
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
        
        # 1. Layer progression (final epoch)
        ax1 = fig.add_subplot(gs[0, :])
        
        df = self.extract_layer_progression()
        epoch = df['epoch'].max()
        df_epoch = df[df['epoch'] == epoch]
        
        for token_type in df_epoch['token_type'].unique():
            token_df = df_epoch[df_epoch['token_type'] == token_type].sort_values('block')
            label = {
                'cls_token': 'CLS Token',
                'image_tokens': 'Image Tokens'
            }.get(token_type, token_type)
            ax1.plot(token_df['block'], token_df['identifiability'], 
                    marker='o', linewidth=2, markersize=8, label=label)
        
        ax1.set_xlabel('Transformer Block')
        ax1.set_ylabel('Class Identifiability')
        ax1.set_title(f'Class Information Across ViT Layers (Final Epoch {epoch})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training evolution heatmap
        cls_df = df[df['token_type'] == 'cls_token']
        pivot_cls = cls_df.pivot(index='epoch', columns='block', values='identifiability')
        
        ax2 = fig.add_subplot(gs[1, 0])
        im = ax2.imshow(pivot_cls.T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Block')
        ax2.set_title('CLS Token Evolution')
        plt.colorbar(im, ax=ax2)
        
        # 3. Early vs Late comparison
        ax3 = fig.add_subplot(gs[1, 1])
        all_blocks = sorted(df['block'].unique())
        early_blocks = all_blocks[:3]
        late_blocks = all_blocks[-3:]
        
        for blocks, label, color in [(early_blocks, 'Early Layers', 'blue'), 
                                     (late_blocks, 'Late Layers', 'red')]:
            block_df = cls_df[cls_df['block'].isin(blocks)]
            avg_df = block_df.groupby('epoch')['identifiability'].mean()
            ax3.plot(avg_df.index, avg_df.values, marker='o', 
                    label=label, color=color, linewidth=2)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('CLS Token Identifiability')
        ax3.set_title('Early vs Late Layer Learning Dynamics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('ViT Class Projection Analysis Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to: {save_path}")
        
        return fig


# Example usage
def analyze_vit_results(analysis_dir: str, output_dir: str = None):
    """Run complete analysis on ViT projection results."""
    analyzer = ViTProjectionAnalyzer(analysis_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path(analysis_dir).parent
    
    # Generate all plots
    print("Generating layer progression plot...")
    fig1 = analyzer.plot_layer_progression()
    fig1.savefig(output_dir / 'layer_progression.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating training evolution plot...")
    fig2 = analyzer.plot_training_evolution()
    fig2.savefig(output_dir / 'training_evolution.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating early vs late layers plot...")
    fig3 = analyzer.plot_early_vs_late_layers()
    fig3.savefig(output_dir / 'early_vs_late_layers.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating summary report...")
    fig4 = analyzer.create_summary_report(str(output_dir / 'summary_report.png'))
    plt.close(fig4)
    
    # Save insights as JSON
    insights = analyzer.generate_insights()
    with open(output_dir / 'insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    return analyzer


if __name__ == "__main__":
    # Example: analyze results from a training run
    analysis_dir = "logs/experiment_20250630_095647/layer_analysis"
    analyze_vit_results(analysis_dir)
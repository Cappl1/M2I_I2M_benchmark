import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.gridspec as gridspec

class BinaryPairsViTAnalyzer:
    """Analyze and visualize ViT class projection results for binary pairs experiments."""
    
    def __init__(self, phase1_dir: str, phase2_dir: str):
        self.phase1_dir = Path(phase1_dir)
        self.phase2_dir = Path(phase2_dir)
        self.phase1_data = self.load_phase_data(self.phase1_dir, is_phase1=True)
        self.phase2_data = self.load_phase_data(self.phase2_dir, is_phase1=False)
        
    def load_phase_data(self, analysis_dir: Path, is_phase1: bool) -> Dict:
        """Load analysis data for a specific phase."""
        data = {}
        
        if not analysis_dir.exists():
            print(f"Warning: Analysis directory {analysis_dir} does not exist")
            return data
            
        for json_file in sorted(analysis_dir.glob("epoch_*.json")):
            epoch = int(json_file.stem.split('_')[1])
            with open(json_file, 'r') as f:
                epoch_data = json.load(f)
                
            if is_phase1:
                # Phase 1: Single task structure
                # Find the task data (should be only one task)
                if len(epoch_data) == 1:
                    task_key = list(epoch_data.keys())[0]
                    data[epoch] = epoch_data[task_key]
                else:
                    # Fallback: assume direct structure
                    data[epoch] = epoch_data
            else:
                # Phase 2: Multi-task structure
                data[epoch] = epoch_data
                
        print(f"Loaded {'phase1' if is_phase1 else 'phase2'} data for {len(data)} epochs")
        return data
    
    def extract_task_progression(self, task_id: Optional[int] = None) -> pd.DataFrame:
        """Extract layer progression for specific task(s) across both phases."""
        records = []
        
        # Phase 1 data (single task)
        for epoch, epoch_data in self.phase1_data.items():
            if 'projection_scores' in epoch_data:
                scores = epoch_data['projection_scores']
                self._extract_scores_to_records(scores, epoch, 'phase1', task_id or 0, records)
        
        # Phase 2 data (multi-task)
        for epoch, epoch_data in self.phase2_data.items():
            for task_key, task_data in epoch_data.items():
                if 'projection_scores' in task_data:
                    curr_task_id = task_data.get('task_id', int(task_key.split('_')[-1]))
                    
                    # Filter by task_id if specified
                    if task_id is None or curr_task_id == task_id:
                        scores = task_data['projection_scores']
                        self._extract_scores_to_records(scores, epoch, 'phase2', curr_task_id, records)
        
        return pd.DataFrame(records)
    
    def _extract_scores_to_records(self, scores: Dict, epoch: int, phase: str, task_id: int, records: List):
        """Helper to extract scores into records format."""
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
                    'phase': phase,
                    'task_id': task_id,
                    'block': block_idx,
                    'token_type': token_type,
                    'identifiability': value
                })
    
    def plot_forgetting_dynamics(self, save_path: str = None):
        """Plot how task 0 performance degrades during phase 2 training."""
        # Get data for task 0 across both phases
        task0_df = self.extract_task_progression(task_id=0)
        
        if task0_df.empty:
            print("No data found for task 0")
            return None
            
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
        
        # 1. CLS token forgetting across layers
        ax1 = fig.add_subplot(gs[0, :])
        cls_df = task0_df[task0_df['token_type'] == 'cls_token']
        
        # Separate phases
        phase1_cls = cls_df[cls_df['phase'] == 'phase1']
        phase2_cls = cls_df[cls_df['phase'] == 'phase2']
        
        if not phase1_cls.empty:
            phase1_pivot = phase1_cls.pivot(index='epoch', columns='block', values='identifiability')
            epochs_p1 = phase1_pivot.index
            for block in phase1_pivot.columns:
                ax1.plot(epochs_p1, phase1_pivot[block], 'b-', alpha=0.7, linewidth=1)
        
        if not phase2_cls.empty:
            phase2_pivot = phase2_cls.pivot(index='epoch', columns='block', values='identifiability')
            epochs_p2 = phase2_pivot.index + 50  # Offset for phase 2
            for block in phase2_pivot.columns:
                ax1.plot(epochs_p2, phase2_pivot[block], 'r-', alpha=0.7, linewidth=1)
        
        ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Phase 1 → Phase 2')
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('CLS Token Identifiability')
        ax1.set_title('Task 0 Forgetting Dynamics: CLS Token Across All Layers')
        ax1.legend(['Phase 1 (Learning)', 'Phase 2 (Forgetting)', 'Phase Transition'])
        ax1.grid(True, alpha=0.3)
        
        # 2. Layer-wise forgetting heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        if not phase2_cls.empty:
            phase2_pivot = phase2_cls.pivot(index='epoch', columns='block', values='identifiability')
            im = ax2.imshow(phase2_pivot.T, aspect='auto', cmap='Reds_r', origin='lower')
            ax2.set_xlabel('Epoch (Phase 2)')
            ax2.set_ylabel('Block')
            ax2.set_title('Task 0 Forgetting Heatmap\n(CLS Token)')
            plt.colorbar(im, ax=ax2, label='Identifiability')
        
        # 3. Average forgetting by layer depth
        ax3 = fig.add_subplot(gs[1, 1])
        if not phase2_cls.empty:
            # Compare first and last epoch of phase 2
            first_epoch_p2 = phase2_cls['epoch'].min()
            last_epoch_p2 = phase2_cls['epoch'].max()
            
            first_data = phase2_cls[phase2_cls['epoch'] == first_epoch_p2].sort_values('block')
            last_data = phase2_cls[phase2_cls['epoch'] == last_epoch_p2].sort_values('block')
            
            if not first_data.empty and not last_data.empty:
                ax3.plot(first_data['block'], first_data['identifiability'], 
                        'g-o', label=f'Start Phase 2 (Epoch {first_epoch_p2})', linewidth=2)
                ax3.plot(last_data['block'], last_data['identifiability'], 
                        'r-o', label=f'End Phase 2 (Epoch {last_epoch_p2})', linewidth=2)
                
                ax3.set_xlabel('Transformer Block')
                ax3.set_ylabel('CLS Token Identifiability') 
                ax3.set_title('Layer-wise Forgetting Comparison')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Task comparison in final state
        ax4 = fig.add_subplot(gs[2, :])
        
        # Get both tasks at the end of phase 2
        all_tasks_df = self.extract_task_progression()
        phase2_final = all_tasks_df[
            (all_tasks_df['phase'] == 'phase2') & 
            (all_tasks_df['epoch'] == all_tasks_df[all_tasks_df['phase'] == 'phase2']['epoch'].max()) &
            (all_tasks_df['token_type'] == 'cls_token')
        ]
        
        for task_id in phase2_final['task_id'].unique():
            task_data = phase2_final[phase2_final['task_id'] == task_id].sort_values('block')
            label = f'Task {task_id} {"(Forgotten)" if task_id == 0 else "(Newly Learned)"}'
            color = 'red' if task_id == 0 else 'blue'
            ax4.plot(task_data['block'], task_data['identifiability'], 
                    'o-', label=label, color=color, linewidth=2, markersize=8)
        
        ax4.set_xlabel('Transformer Block')
        ax4.set_ylabel('CLS Token Identifiability')
        ax4.set_title('Final State: Forgotten vs Newly Learned Task Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Binary Pairs Forgetting Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forgetting dynamics plot saved to: {save_path}")
        
        return fig
    
    def plot_learning_vs_forgetting(self, save_path: str = None):
        """Plot direct comparison of learning vs forgetting patterns."""
        all_tasks_df = self.extract_task_progression()
        
        if all_tasks_df.empty:
            print("No data found for learning vs forgetting analysis")
            return None
            
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Filter to CLS tokens only for clarity
        cls_df = all_tasks_df[all_tasks_df['token_type'] == 'cls_token']
        
        # 1. Task 0 across both phases (forgetting trajectory)
        ax1 = fig.add_subplot(gs[0, 0])
        task0_df = cls_df[cls_df['task_id'] == 0]
        
        phase1_data = task0_df[task0_df['phase'] == 'phase1']
        phase2_data = task0_df[task0_df['phase'] == 'phase2']
        
        # Average across blocks for cleaner visualization
        if not phase1_data.empty:
            p1_avg = phase1_data.groupby('epoch')['identifiability'].mean()
            ax1.plot(p1_avg.index, p1_avg.values, 'b-', linewidth=2, label='Phase 1 (Learning)')
        
        if not phase2_data.empty:
            p2_avg = phase2_data.groupby('epoch')['identifiability'].mean()
            # Offset phase 2 epochs
            offset_epochs = p2_avg.index + (phase1_data['epoch'].max() if not phase1_data.empty else 0)
            ax1.plot(offset_epochs, p2_avg.values, 'r-', linewidth=2, label='Phase 2 (Forgetting)')
        
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Avg CLS Identifiability')
        ax1.set_title('Task 0: Learning → Forgetting')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Task 1 in phase 2 (learning trajectory)
        ax2 = fig.add_subplot(gs[0, 1])
        task1_df = cls_df[(cls_df['task_id'] == 1) & (cls_df['phase'] == 'phase2')]
        
        if not task1_df.empty:
            t1_avg = task1_df.groupby('epoch')['identifiability'].mean()
            ax2.plot(t1_avg.index, t1_avg.values, 'g-', linewidth=2, label='Task 1 Learning')
            
            ax2.set_xlabel('Training Epoch (Phase 2)')
            ax2.set_ylabel('Avg CLS Identifiability')
            ax2.set_title('Task 1: New Learning')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Direct comparison at end of phase 2
        ax3 = fig.add_subplot(gs[0, 2])
        phase2_final = cls_df[
            (cls_df['phase'] == 'phase2') & 
            (cls_df['epoch'] == cls_df[cls_df['phase'] == 'phase2']['epoch'].max())
        ]
        
        task0_final = phase2_final[phase2_final['task_id'] == 0].groupby('block')['identifiability'].mean()
        task1_final = phase2_final[phase2_final['task_id'] == 1].groupby('block')['identifiability'].mean()
        
        if not task0_final.empty and not task1_final.empty:
            ax3.plot(task0_final.index, task0_final.values, 'r-o', 
                    linewidth=2, markersize=6, label='Task 0 (Forgotten)')
            ax3.plot(task1_final.index, task1_final.values, 'g-o', 
                    linewidth=2, markersize=6, label='Task 1 (New)')
            
            ax3.set_xlabel('Transformer Block')
            ax3.set_ylabel('CLS Identifiability')
            ax3.set_title('Final Layer-wise Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Forgetting rate by layer (bottom row)
        ax4 = fig.add_subplot(gs[1, :])
        
        # Calculate forgetting rate for each layer
        if not phase2_data.empty:
            first_epoch_p2 = phase2_data['epoch'].min()
            last_epoch_p2 = phase2_data['epoch'].max()
            
            first_state = phase2_data[phase2_data['epoch'] == first_epoch_p2].groupby('block')['identifiability'].mean()
            last_state = phase2_data[phase2_data['epoch'] == last_epoch_p2].groupby('block')['identifiability'].mean()
            
            if not first_state.empty and not last_state.empty:
                forgetting_rate = (first_state - last_state) / first_state * 100
                
                bars = ax4.bar(forgetting_rate.index, forgetting_rate.values, 
                              color='red', alpha=0.7, edgecolor='darkred')
                ax4.set_xlabel('Transformer Block')
                ax4.set_ylabel('Forgetting Rate (%)')
                ax4.set_title('Layer-wise Forgetting Rate (Task 0 degradation during Phase 2)')
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, value in zip(bars, forgetting_rate.values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Learning vs Forgetting Dynamics Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning vs forgetting plot saved to: {save_path}")
        
        return fig
    
    def generate_comparative_insights(self) -> Dict:
        """Generate insights comparing learning and forgetting patterns."""
        insights = {}
        
        # Get all data
        all_df = self.extract_task_progression()
        cls_df = all_df[all_df['token_type'] == 'cls_token']
        
        # Task 0 forgetting analysis
        task0_phase2 = cls_df[(cls_df['task_id'] == 0) & (cls_df['phase'] == 'phase2')]
        if not task0_phase2.empty:
            first_epoch = task0_phase2['epoch'].min()
            last_epoch = task0_phase2['epoch'].max()
            
            first_avg = task0_phase2[task0_phase2['epoch'] == first_epoch]['identifiability'].mean()
            last_avg = task0_phase2[task0_phase2['epoch'] == last_epoch]['identifiability'].mean()
            
            insights['task0_forgetting_percentage'] = (first_avg - last_avg) / first_avg * 100
            insights['task0_initial_phase2'] = first_avg
            insights['task0_final_phase2'] = last_avg
        
        # Task 1 learning analysis
        task1_phase2 = cls_df[(cls_df['task_id'] == 1) & (cls_df['phase'] == 'phase2')]
        if not task1_phase2.empty:
            first_epoch = task1_phase2['epoch'].min()
            last_epoch = task1_phase2['epoch'].max()
            
            first_avg = task1_phase2[task1_phase2['epoch'] == first_epoch]['identifiability'].mean()
            last_avg = task1_phase2[task1_phase2['epoch'] == last_epoch]['identifiability'].mean()
            
            insights['task1_learning_improvement'] = (last_avg - first_avg) / first_avg * 100
            insights['task1_initial_phase2'] = first_avg
            insights['task1_final_phase2'] = last_avg
        
        # Layer-wise forgetting patterns
        if not task0_phase2.empty:
            layer_forgetting = {}
            for block in task0_phase2['block'].unique():
                block_data = task0_phase2[task0_phase2['block'] == block]
                if len(block_data) > 1:
                    first_val = block_data[block_data['epoch'] == block_data['epoch'].min()]['identifiability'].iloc[0]
                    last_val = block_data[block_data['epoch'] == block_data['epoch'].max()]['identifiability'].iloc[0]
                    forgetting_rate = (first_val - last_val) / first_val * 100
                    layer_forgetting[f'block_{block}'] = forgetting_rate
            
            insights['layer_forgetting_rates'] = layer_forgetting
            
            # Find most/least affected layers
            if layer_forgetting:
                most_forgotten_layer = max(layer_forgetting, key=layer_forgetting.get)
                least_forgotten_layer = min(layer_forgetting, key=layer_forgetting.get)
                
                insights['most_forgotten_layer'] = most_forgotten_layer
                insights['most_forgotten_rate'] = layer_forgetting[most_forgotten_layer]
                insights['least_forgotten_layer'] = least_forgotten_layer
                insights['least_forgotten_rate'] = layer_forgetting[least_forgotten_layer]
        
        return insights


def analyze_binary_pairs_results(phase1_dir: str, phase2_dir: str, output_dir: str = None):
    """Run complete analysis on binary pairs ViT projection results."""
    analyzer = BinaryPairsViTAnalyzer(phase1_dir, phase2_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(phase2_dir).parent / 'binary_analysis'
        output_dir.mkdir(exist_ok=True)
    
    print("Generating forgetting dynamics analysis...")
    fig1 = analyzer.plot_forgetting_dynamics(str(output_dir / 'forgetting_dynamics.png'))
    if fig1:
        plt.close(fig1)
    
    print("Generating learning vs forgetting comparison...")
    fig2 = analyzer.plot_learning_vs_forgetting(str(output_dir / 'learning_vs_forgetting.png'))
    if fig2:
        plt.close(fig2)
    
    # Generate insights
    insights = analyzer.generate_comparative_insights()
    with open(output_dir / 'binary_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"\nBinary pairs analysis complete! Results saved to: {output_dir}")
    return analyzer


if __name__ == "__main__":
    # Example usage
    phase1_dir = "logs/BinaryPairsExperiment_20250630_225639/layer_analysis/phase1"
    phase2_dir = "logs/BinaryPairsExperiment_20250630_225639/layer_analysis/phase2"
    analyze_binary_pairs_results(phase1_dir, phase2_dir)
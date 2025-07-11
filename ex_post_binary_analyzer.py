#!/usr/bin/env python3
"""
Ex-post analysis script for Binary Pairs experiments.
Reads experiment configuration from results.json and handles both phases.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.gridspec as gridspec
import argparse
import sys
from datetime import datetime


class ExPostBinaryPairsAnalyzer:
    """Analyze Binary Pairs experiments after completion, reading config from results."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.results_file = self.experiment_dir / 'results.json'
        
        # Load results and config
        self.results = self.load_results()
        self.config = self.results.get('config', {})
        
        # Extract key configuration
        self.scenario_type = self.config.get('scenario_type', 'unknown')
        self.dataset_a = self.config.get('dataset_a', 'unknown')
        self.dataset_b = self.config.get('dataset_b', 'unknown')
        self.experiment_name = f"{self.dataset_a}_{self.dataset_b}_{self.scenario_type}"
        
        # Load phase data if available
        self.phase1_dir = self.experiment_dir / 'layer_analysis' / 'phase1'
        self.phase2_dir = self.experiment_dir / 'layer_analysis' / 'phase2'
        
        self.phase1_data = {}
        self.phase2_data = {}
        
        if self.phase1_dir.exists():
            self.phase1_data = self.load_phase_data(self.phase1_dir, is_phase1=True)
        if self.phase2_dir.exists():
            self.phase2_data = self.load_phase_data(self.phase2_dir, is_phase1=False)
            
    def load_results(self) -> Dict:
        """Load results.json file."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
            
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def load_phase_data(self, analysis_dir: Path, is_phase1: bool) -> Dict:
        """Load analysis data for a specific phase."""
        data = {}
        
        for json_file in sorted(analysis_dir.glob("epoch_*.json")):
            epoch = int(json_file.stem.split('_')[1])
            try:
                with open(json_file, 'r') as f:
                    epoch_data = json.load(f)
                    
                if is_phase1:
                    # Phase 1: Single task structure
                    if isinstance(epoch_data, dict) and len(epoch_data) == 1:
                        task_key = list(epoch_data.keys())[0]
                        data[epoch] = epoch_data[task_key]
                    else:
                        data[epoch] = epoch_data
                else:
                    # Phase 2: Multi-task structure
                    data[epoch] = epoch_data
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                
        return data
    
    def extract_task_progression(self, task_id: Optional[int] = None) -> pd.DataFrame:
        """Extract layer progression for specific task(s) across both phases."""
        records = []
        
        # Phase 1 data
        for epoch, epoch_data in self.phase1_data.items():
            if 'projection_scores' in epoch_data:
                scores = epoch_data['projection_scores']
                self._extract_scores_to_records(scores, epoch, 'phase1', task_id or 0, records)
        
        # Phase 2 data  
        for epoch, epoch_data in self.phase2_data.items():
            if isinstance(epoch_data, dict):
                for task_key, task_data in epoch_data.items():
                    if isinstance(task_data, dict) and 'projection_scores' in task_data:
                        curr_task_id = task_data.get('task_id', int(task_key.split('_')[-1]))
                        
                        if task_id is None or curr_task_id == task_id:
                            scores = task_data['projection_scores']
                            self._extract_scores_to_records(scores, epoch, 'phase2', curr_task_id, records)
        
        return pd.DataFrame(records)
    
    def _extract_scores_to_records(self, scores: Dict, epoch: int, phase: str, task_id: int, records: List):
        """Helper to extract scores into records format."""
        for key, value in scores.items():
            if key.startswith('block_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    block_idx = int(parts[1])
                    token_type = '_'.join(parts[2:])
                    
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
    
    def plot_forgetting_dynamics(self, save_path: Optional[str] = None):
        """Plot forgetting dynamics with experiment configuration in title."""
        task0_df = self.extract_task_progression(task_id=0)
        
        if task0_df.empty:
            print(f"No data found for task 0 in {self.experiment_dir}")
            return None
            
        fig = plt.figure(figsize=(16, 10))
        
        # Create title with configuration info
        title = f'Forgetting Dynamics: {self.dataset_a} → {self.dataset_b}\n'
        title += f'Scenario: {self.scenario_type.replace("_", " ").title()}'
        
        # Main forgetting plot
        ax1 = plt.subplot(2, 2, 1)
        cls_df = task0_df[task0_df['token_type'] == 'cls_token']
        
        # Phase 1 data
        phase1_cls = cls_df[cls_df['phase'] == 'phase1']
        if not phase1_cls.empty:
            phase1_avg = phase1_cls.groupby('epoch')['identifiability'].mean()
            ax1.plot(phase1_avg.index, phase1_avg.values, 'b-', linewidth=2.5, 
                    label=f'Phase 1: Learning {self.dataset_a}')
        
        # Phase 2 data
        phase2_cls = cls_df[cls_df['phase'] == 'phase2']
        if not phase2_cls.empty:
            phase2_avg = phase2_cls.groupby('epoch')['identifiability'].mean()
            # Offset for continuous x-axis
            offset = 50 if not phase1_cls.empty else 0
            ax1.plot(phase2_avg.index + offset, phase2_avg.values, 'r-', linewidth=2.5,
                    label=f'Phase 2: Learning {self.dataset_b}')
        
        ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Phase Transition')
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Average CLS Token Identifiability')
        ax1.set_title(f'Task 0 ({self.dataset_a}) Forgetting During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Layer-wise heatmap
        ax2 = plt.subplot(2, 2, 2)
        if not phase2_cls.empty:
            pivot = phase2_cls.pivot_table(index='block', columns='epoch', values='identifiability')
            im = ax2.imshow(pivot.values, aspect='auto', cmap='RdBu_r', origin='lower')
            ax2.set_xlabel('Epoch (Phase 2)')
            ax2.set_ylabel('Transformer Block')
            ax2.set_title(f'Layer-wise Forgetting Heatmap ({self.dataset_a})')
            plt.colorbar(im, ax=ax2)
        
        # Task comparison
        ax3 = plt.subplot(2, 2, 3)
        all_tasks_df = self.extract_task_progression()
        final_epoch = all_tasks_df[all_tasks_df['phase'] == 'phase2']['epoch'].max() if not all_tasks_df.empty else None
        
        if final_epoch is not None:
            final_data = all_tasks_df[
                (all_tasks_df['phase'] == 'phase2') & 
                (all_tasks_df['epoch'] == final_epoch) &
                (all_tasks_df['token_type'] == 'cls_token')
            ]
            
            for task_id in sorted(final_data['task_id'].unique()):
                task_data = final_data[final_data['task_id'] == task_id].sort_values('block')
                dataset_name = self.dataset_a if task_id == 0 else self.dataset_b
                label = f'Task {task_id} ({dataset_name})'
                color = 'red' if task_id == 0 else 'blue'
                ax3.plot(task_data['block'], task_data['identifiability'], 
                        'o-', label=label, color=color, linewidth=2, markersize=8)
        
        ax3.set_xlabel('Transformer Block')
        ax3.set_ylabel('CLS Token Identifiability')
        ax3.set_title('Final State Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Accuracy evolution from results
        ax4 = plt.subplot(2, 2, 4)
        if 'evolution' in self.results and 'accuracy' in self.results['evolution']:
            acc_data = self.results['evolution']['accuracy']
            epochs = []
            task0_accs = []
            task1_accs = []
            
            for epoch_key, epoch_acc in sorted(acc_data.items()):
                epoch_num = int(epoch_key.split('_')[1])
                epochs.append(epoch_num + 50)  # Phase 2 epochs
                
                if 'task_0' in epoch_acc:
                    task0_accs.append(epoch_acc['task_0'])
                if 'task_1' in epoch_acc:
                    task1_accs.append(epoch_acc['task_1'])
            
            if task0_accs:
                ax4.plot(epochs, task0_accs, 'r-o', linewidth=2, 
                        label=f'{self.dataset_a} (Forgetting)')
            if task1_accs:
                ax4.plot(epochs, task1_accs, 'b-o', linewidth=2,
                        label=f'{self.dataset_b} (Learning)')
        
        ax4.set_xlabel('Training Epoch')
        ax4.set_ylabel('Test Accuracy (%)')
        ax4.set_title('Accuracy Evolution During Phase 2')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_phase_analysis(self, phase: int, save_path: Optional[str] = None):
        """Analyze a single phase independently."""
        phase_name = f'phase{phase}'
        phase_data = self.phase1_data if phase == 1 else self.phase2_data
        
        if not phase_data:
            print(f"No data found for {phase_name}")
            return None
            
        # Extract data for this phase
        df = self.extract_task_progression()
        phase_df = df[df['phase'] == phase_name]
        
        if phase_df.empty:
            print(f"No progression data for {phase_name}")
            return None
        
        if phase == 1:
            # Phase 1: Simple single task analysis
            fig = plt.figure(figsize=(15, 10))
            dataset_name = self.dataset_a
            
            # Create title
            title = f'Phase 1 Analysis: Learning {dataset_name}\n'
            title += f'Experiment: {self.dataset_a} + {self.dataset_b} ({self.scenario_type})'
            
            # Only analyze task 0
            task_df = phase_df[phase_df['task_id'] == 0]
            self._plot_single_task_analysis(fig, task_df, dataset_name, task_id=0)
            
        else:  # phase == 2
            # Phase 2: Analyze BOTH tasks
            fig = plt.figure(figsize=(20, 12))
            
            # Create title
            title = f'Phase 2 Analysis: Learning {self.dataset_b} while Forgetting {self.dataset_a}\n'
            title += f'Experiment: {self.dataset_a} + {self.dataset_b} ({self.scenario_type})'
            
            # Analyze Task 0 (forgetting)
            task0_df = phase_df[phase_df['task_id'] == 0]
            if not task0_df.empty:
                plt.subplot(2, 4, 1)
                self._plot_task_progression(task0_df, f'Task 0 ({self.dataset_a}) Forgetting', 'Reds_r')
                
                plt.subplot(2, 4, 2)
                self._plot_task_heatmap(task0_df, f'Task 0 ({self.dataset_a}) Evolution', 'RdBu_r')
                
                plt.subplot(2, 4, 3)
                self._plot_token_comparison(task0_df, f'Task 0 ({self.dataset_a}) Tokens')
                
                plt.subplot(2, 4, 4)
                self._plot_learning_curve(task0_df, f'Task 0 ({self.dataset_a}) Degradation', 'red')
            
            # Analyze Task 1 (learning)
            task1_df = phase_df[phase_df['task_id'] == 1]
            if not task1_df.empty:
                plt.subplot(2, 4, 5)
                self._plot_task_progression(task1_df, f'Task 1 ({self.dataset_b}) Learning', 'Blues')
                
                plt.subplot(2, 4, 6)
                self._plot_task_heatmap(task1_df, f'Task 1 ({self.dataset_b}) Evolution', 'viridis')
                
                plt.subplot(2, 4, 7)
                self._plot_token_comparison(task1_df, f'Task 1 ({self.dataset_b}) Tokens')
                
                plt.subplot(2, 4, 8)
                self._plot_learning_curve(task1_df, f'Task 1 ({self.dataset_b}) Learning', 'blue')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _plot_single_task_analysis(self, fig, task_df, dataset_name, task_id):
        """Helper for single task analysis layout."""
        # Plot 1: Layer progression
        ax1 = plt.subplot(2, 2, 1)
        self._plot_task_progression(task_df, f'{dataset_name} Learning Progression', 'Blues')
        
        # Plot 2: Evolution heatmap
        ax2 = plt.subplot(2, 2, 2)
        self._plot_task_heatmap(task_df, 'CLS Token Evolution Heatmap', 'viridis')
        
        # Plot 3: Token comparison
        ax3 = plt.subplot(2, 2, 3)
        self._plot_token_comparison(task_df, 'Token Type Comparison')
        
        # Plot 4: Learning curve
        ax4 = plt.subplot(2, 2, 4)
        self._plot_learning_curve(task_df, f'{dataset_name} Learning Curve', 'green')
    
    def _plot_task_progression(self, task_df, title, cmap):
        """Plot layer progression for a task."""
        cls_df = task_df[task_df['token_type'] == 'cls_token']
        
        epochs = sorted(cls_df['epoch'].unique())
        if len(epochs) > 5:
            epochs = epochs[-5:]  # Last 5 epochs
        
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 1.0, len(epochs)))
        
        for i, epoch in enumerate(epochs):
            epoch_data = cls_df[cls_df['epoch'] == epoch].sort_values('block')
            plt.plot(epoch_data['block'], epoch_data['identifiability'],
                    'o-', color=colors[i], alpha=0.8, label=f'Epoch {epoch}')
        
        plt.xlabel('Transformer Block')
        plt.ylabel('CLS Token Identifiability')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_task_heatmap(self, task_df, title, cmap):
        """Plot evolution heatmap for a task."""
        cls_df = task_df[task_df['token_type'] == 'cls_token']
        
        if not cls_df.empty:
            pivot = cls_df.pivot_table(index='block', columns='epoch', values='identifiability')
            im = plt.imshow(pivot.values, aspect='auto', cmap=cmap, origin='lower')
            plt.xlabel('Epoch')
            plt.ylabel('Transformer Block')
            plt.title(title)
            plt.colorbar(im)
    
    def _plot_token_comparison(self, task_df, title):
        """Plot token type comparison."""
        last_epoch = task_df['epoch'].max()
        last_epoch_df = task_df[task_df['epoch'] == last_epoch]
        
        for token_type in ['cls_token', 'image_tokens']:
            token_data = last_epoch_df[last_epoch_df['token_type'] == token_type].sort_values('block')
            label = 'CLS Token' if token_type == 'cls_token' else 'Image Tokens'
            plt.plot(token_data['block'], token_data['identifiability'], 
                    'o-', label=label, linewidth=2)
        
        plt.xlabel('Transformer Block')
        plt.ylabel('Identifiability')
        plt.title(f'{title} (Epoch {last_epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_learning_curve(self, task_df, title, color):
        """Plot learning/forgetting curve."""
        cls_df = task_df[task_df['token_type'] == 'cls_token']
        learning_curve = cls_df.groupby('epoch')['identifiability'].mean()
        plt.plot(learning_curve.index, learning_curve.values, 'o-', 
                color=color, linewidth=2.5)
        plt.xlabel('Epoch')
        plt.ylabel('Average CLS Identifiability')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    def generate_summary_report(self, output_dir: Path):
        """Generate a comprehensive summary with all analyses."""
        summary = {
            'experiment_dir': str(self.experiment_dir),
            'dataset_a': self.dataset_a,
            'dataset_b': self.dataset_b,
            'scenario_type': self.scenario_type,
            'analysis_timestamp': datetime.now().isoformat(),
            'phase1_epochs': len(self.phase1_data),
            'phase2_epochs': len(self.phase2_data),
        }
        
        # Extract key metrics
        all_df = self.extract_task_progression()
        
        if not all_df.empty:
            # Forgetting metrics
            task0_phase2 = all_df[(all_df['task_id'] == 0) & (all_df['phase'] == 'phase2')]
            if not task0_phase2.empty:
                first_epoch = task0_phase2['epoch'].min()
                last_epoch = task0_phase2['epoch'].max()
                
                first_avg = task0_phase2[task0_phase2['epoch'] == first_epoch]['identifiability'].mean()
                last_avg = task0_phase2[task0_phase2['epoch'] == last_epoch]['identifiability'].mean()
                
                summary['forgetting_rate'] = (first_avg - last_avg) / first_avg * 100
                summary['task0_initial_phase2'] = first_avg
                summary['task0_final_phase2'] = last_avg
        
        # Save summary
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def analyze_single_experiment(experiment_dir: Path, output_base: Path):
    """Analyze a single experiment directory."""
    try:
        analyzer = ExPostBinaryPairsAnalyzer(experiment_dir)
        
        # Create output directory
        output_dir = output_base / analyzer.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nAnalyzing: {analyzer.experiment_name}")
        print(f"  - Dataset A: {analyzer.dataset_a}")
        print(f"  - Dataset B: {analyzer.dataset_b}")
        print(f"  - Scenario: {analyzer.scenario_type}")
        
        # Generate all plots
        if analyzer.phase1_data and analyzer.phase2_data:
            # Both phases available - full analysis
            print("  - Generating forgetting dynamics...")
            fig = analyzer.plot_forgetting_dynamics(str(output_dir / 'forgetting_dynamics.png'))
            if fig:
                plt.close(fig)
                
        if analyzer.phase1_data:
            print("  - Analyzing phase 1...")
            fig = analyzer.plot_phase_analysis(1, str(output_dir / 'phase1_analysis.png'))
            if fig:
                plt.close(fig)
                
        if analyzer.phase2_data:
            print("  - Analyzing phase 2...")
            fig = analyzer.plot_phase_analysis(2, str(output_dir / 'phase2_analysis.png'))
            if fig:
                plt.close(fig)
        
        # Generate summary
        print("  - Generating summary...")
        analyzer.generate_summary_report(output_dir)
        
        return True
        
    except Exception as e:
        print(f"  - ERROR: {e}")
        return False


def main():
    """Main function to run ex-post analysis."""
    parser = argparse.ArgumentParser(description='Ex-post analysis for Binary Pairs experiments')
    parser.add_argument('logs_dir', type=str, help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='ex_post_analysis', 
                       help='Output directory for analysis results')
    parser.add_argument('--pattern', type=str, default='*BinaryPairsExperiment*',
                       help='Pattern to match experiment directories')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_base = Path(args.output)
    output_base.mkdir(exist_ok=True)
    
    # Find all experiment directories
    experiment_dirs = list(logs_dir.glob(args.pattern))
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    if not experiment_dirs:
        print("No experiments found!")
        return
    
    # Analyze each experiment
    successful = 0
    failed = 0
    
    for exp_dir in sorted(experiment_dirs):
        if (exp_dir / 'results.json').exists():
            if analyze_single_experiment(exp_dir, output_base):
                successful += 1
            else:
                failed += 1
        else:
            print(f"\nSkipping {exp_dir.name} - no results.json found")
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Results saved to: {output_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
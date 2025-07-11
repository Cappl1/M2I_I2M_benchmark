#!/usr/bin/env python3
"""
Analyze experiments where only Phase 1 completed.
Useful for debugging or when Phase 2 fails.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List
import argparse


class SinglePhaseAnalyzer:
    """Analyze single phase (Phase 1 only) experiments."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.results_file = self.experiment_dir / 'results.json'
        self.phase1_dir = self.experiment_dir / 'layer_analysis' / 'phase1'
        
        # Load configuration
        self.config = self.load_config()
        self.dataset = self.config.get('dataset_a', 'unknown')
        
    def load_config(self) -> Dict:
        """Load experiment configuration."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results = json.load(f)
                return results.get('config', {})
        
        # Fallback: try to load config.yaml
        config_file = self.experiment_dir / 'config.yaml'
        if config_file.exists():
            # Simple YAML parsing (basic implementation)
            config = {}
            with open(config_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        config[key.strip()] = value.strip()
            return config
        
        return {}
    
    def analyze_phase1(self, output_dir: Path):
        """Analyze Phase 1 learning progression."""
        if not self.phase1_dir.exists():
            print(f"No phase 1 data found in {self.experiment_dir}")
            return
        
        # Load all epoch data
        epoch_data = {}
        for json_file in sorted(self.phase1_dir.glob("epoch_*.json")):
            epoch = int(json_file.stem.split('_')[1])
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Handle different data structures
            if isinstance(data, dict) and 'projection_scores' in data:
                epoch_data[epoch] = data
            elif isinstance(data, dict) and len(data) == 1:
                # Wrapped in task key
                task_key = list(data.keys())[0]
                epoch_data[epoch] = data[task_key]
        
        if not epoch_data:
            print("No valid epoch data found")
            return
        
        # Create visualizations
        self.plot_learning_progression(epoch_data, output_dir)
        self.plot_layer_evolution(epoch_data, output_dir)
        self.generate_phase1_summary(epoch_data, output_dir)
    
    def plot_learning_progression(self, epoch_data: Dict, output_dir: Path):
        """Plot how learning progresses through epochs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract CLS token progression
        epochs = sorted(epoch_data.keys())
        
        # Plot 1: Average identifiability over time
        avg_identifiability = []
        for epoch in epochs:
            if 'projection_scores' in epoch_data[epoch]:
                scores = epoch_data[epoch]['projection_scores']
                cls_scores = [v for k, v in scores.items() 
                             if k.startswith('block_') and 'cls_token' in k]
                if cls_scores:
                    avg_identifiability.append(np.mean(cls_scores))
        
        if avg_identifiability:
            ax1.plot(epochs[:len(avg_identifiability)], avg_identifiability, 
                    'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Average CLS Token Identifiability')
            ax1.set_title(f'Learning Curve: {self.dataset}')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Layer-wise progression at different epochs
        sample_epochs = [epochs[0], epochs[len(epochs)//2], epochs[-1]]
        colors = ['lightblue', 'blue', 'darkblue']
        
        for i, epoch in enumerate(sample_epochs):
            if epoch in epoch_data and 'projection_scores' in epoch_data[epoch]:
                scores = epoch_data[epoch]['projection_scores']
                
                # Extract block scores
                block_scores = {}
                for key, value in scores.items():
                    if key.startswith('block_') and 'cls_token' in key:
                        block_num = int(key.split('_')[1])
                        block_scores[block_num] = value
                
                if block_scores:
                    blocks = sorted(block_scores.keys())
                    values = [block_scores[b] for b in blocks]
                    
                    ax2.plot(blocks, values, 'o-', color=colors[i], 
                            label=f'Epoch {epoch}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Transformer Block')
        ax2.set_ylabel('CLS Token Identifiability')
        ax2.set_title('Layer-wise Learning Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Phase 1 Analysis: {self.dataset}\n'
                    f'Experiment: {self.experiment_dir.name}', fontsize=14)
        plt.tight_layout()
        
        save_path = output_dir / f'{self.dataset}_phase1_progression.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    def plot_layer_evolution(self, epoch_data: Dict, output_dir: Path):
        """Create heatmap of layer evolution."""
        # Prepare data for heatmap
        epochs = sorted(epoch_data.keys())
        blocks = set()
        
        # Find all blocks
        for epoch_info in epoch_data.values():
            if 'projection_scores' in epoch_info:
                for key in epoch_info['projection_scores']:
                    if key.startswith('block_') and 'cls_token' in key:
                        block_num = int(key.split('_')[1])
                        blocks.add(block_num)
        
        blocks = sorted(list(blocks))
        
        # Create matrix
        matrix = np.zeros((len(blocks), len(epochs)))
        
        for j, epoch in enumerate(epochs):
            if 'projection_scores' in epoch_data[epoch]:
                scores = epoch_data[epoch]['projection_scores']
                for i, block in enumerate(blocks):
                    key = f'block_{block}_cls_token'
                    if key in scores:
                        matrix[i, j] = scores[key]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(matrix, aspect='auto', cmap='viridis', origin='lower')
        
        plt.colorbar(im, label='CLS Token Identifiability')
        plt.xlabel('Epoch')
        plt.ylabel('Transformer Block')
        plt.title(f'Layer Evolution Heatmap: {self.dataset}')
        
        # Set ticks
        plt.xticks(range(0, len(epochs), max(1, len(epochs)//10)), 
                  epochs[::max(1, len(epochs)//10)])
        plt.yticks(range(len(blocks)), blocks)
        
        save_path = output_dir / f'{self.dataset}_phase1_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    def generate_phase1_summary(self, epoch_data: Dict, output_dir: Path):
        """Generate summary statistics for Phase 1."""
        summary = {
            'experiment': str(self.experiment_dir.name),
            'dataset': self.dataset,
            'scenario_type': self.config.get('scenario_type', 'unknown'),
            'epochs_analyzed': len(epoch_data),
        }
        
        # Calculate improvement
        epochs = sorted(epoch_data.keys())
        if len(epochs) >= 2:
            first_epoch = epochs[0]
            last_epoch = epochs[-1]
            
            # Get average identifiability
            first_scores = []
            last_scores = []
            
            if 'projection_scores' in epoch_data[first_epoch]:
                scores = epoch_data[first_epoch]['projection_scores']
                first_scores = [v for k, v in scores.items() 
                               if k.startswith('block_') and 'cls_token' in k]
            
            if 'projection_scores' in epoch_data[last_epoch]:
                scores = epoch_data[last_epoch]['projection_scores']
                last_scores = [v for k, v in scores.items() 
                              if k.startswith('block_') and 'cls_token' in k]
            
            if first_scores and last_scores:
                first_avg = np.mean(first_scores)
                last_avg = np.mean(last_scores)
                improvement = (last_avg - first_avg) / first_avg * 100
                
                summary['initial_identifiability'] = float(first_avg)
                summary['final_identifiability'] = float(last_avg)
                summary['improvement_percentage'] = float(improvement)
        
        # Find best performing layer
        if epochs and 'projection_scores' in epoch_data[epochs[-1]]:
            final_scores = epoch_data[epochs[-1]]['projection_scores']
            cls_scores = {k: v for k, v in final_scores.items() 
                         if k.startswith('block_') and 'cls_token' in k}
            
            if cls_scores:
                best_layer = max(cls_scores, key=cls_scores.get)
                best_block = int(best_layer.split('_')[1])
                summary['best_performing_block'] = best_block
                summary['best_block_score'] = float(cls_scores[best_layer])
        
        # Save summary
        summary_path = output_dir / f'{self.dataset}_phase1_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved: {summary_path}")
        
        return summary


def analyze_incomplete_experiments(logs_dir: Path, output_dir: Path):
    """Find and analyze all experiments with only Phase 1 data."""
    incomplete_exps = []
    
    # Find experiments with only phase 1
    for exp_dir in logs_dir.glob('*BinaryPairsExperiment*'):
        phase1_exists = (exp_dir / 'layer_analysis' / 'phase1').exists()
        phase2_exists = (exp_dir / 'layer_analysis' / 'phase2').exists()
        
        if phase1_exists and not phase2_exists:
            incomplete_exps.append(exp_dir)
    
    print(f"Found {len(incomplete_exps)} incomplete experiments (Phase 1 only)")
    
    # Analyze each
    for exp_dir in incomplete_exps:
        print(f"\nAnalyzing: {exp_dir.name}")
        try:
            analyzer = SinglePhaseAnalyzer(exp_dir)
            exp_output = output_dir / exp_dir.name
            exp_output.mkdir(parents=True, exist_ok=True)
            analyzer.analyze_phase1(exp_output)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create summary
    create_incomplete_summary(incomplete_exps, output_dir)


def create_incomplete_summary(incomplete_exps: List[Path], output_dir: Path):
    """Create summary of all incomplete experiments."""
    summary_path = output_dir / 'incomplete_experiments_summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# Incomplete Experiments Summary\n\n")
        f.write(f"Found {len(incomplete_exps)} experiments with only Phase 1 data.\n\n")
        
        f.write("## Experiments List\n\n")
        for exp_dir in incomplete_exps:
            f.write(f"- {exp_dir.name}\n")
            
            # Try to get config info
            results_file = exp_dir / 'results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as rf:
                        results = json.load(rf)
                        config = results.get('config', {})
                        f.write(f"  - Dataset A: {config.get('dataset_a', 'unknown')}\n")
                        f.write(f"  - Dataset B: {config.get('dataset_b', 'unknown')}\n")
                        f.write(f"  - Scenario: {config.get('scenario_type', 'unknown')}\n")
                except:
                    pass
        
        f.write("\n## Possible Reasons for Incomplete Runs\n\n")
        f.write("- Training interrupted during Phase 2\n")
        f.write("- Memory/resource constraints\n")
        f.write("- Configuration errors\n")
        f.write("- Dataset loading issues for second task\n")
    
    print(f"\nSaved incomplete experiments summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze incomplete Binary Pairs experiments')
    parser.add_argument('logs_dir', type=str, help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='incomplete_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    analyze_incomplete_experiments(logs_dir, output_dir)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Aggregate binary pairs forgetting analysis with layer-wise identifiability tracking.
Creates separate, detailed charts for comprehensive analysis.
Enhanced with summary table showing all strategies vs dataset pairs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import argparse
try:
    import yaml
except ImportError:
    yaml = None

# Define the canonical dataset complexity order
DATASET_ORDER = ['mnist', 'omniglot', 'fashion_mnist', 'svhn', 'cifar10', 'tinyimagenet']
DATASET_DISPLAY_NAMES = {
    'mnist': 'MNIST',
    'omniglot': 'Omniglot', 
    'fashion_mnist': 'Fashion-MNIST',
    'svhn': 'SVHN',
    'cifar10': 'CIFAR-10',
    'tinyimagenet': 'TinyImageNet'
}

class ComprehensiveBinaryPairsAnalyzer:
    """Comprehensive analysis of binary pairs experiments with layer-wise tracking."""
    
    def __init__(self, results_base_dir: str):
        self.results_base_dir = Path(results_base_dir)
        self.dataset_order = DATASET_ORDER
        self.dataset_complexity = {ds: i for i, ds in enumerate(DATASET_ORDER)}
        
        # Storage for results
        self.experiments = []  # List of experiment dictionaries
        self.layer_evolution_data = defaultdict(list)  # {strategy: [layer_data]}
        
        # Storage for multiple runs
        self.all_runs = defaultdict(list)  # {(dataset_a, dataset_b, strategy): [run_data]}
        
    def load_all_experiments(self):
        """Load all binary pairs experiments with full layer evolution data."""
        print("Loading binary pairs experiments...")
        
        # Look for logs directory or direct strategy_binary_pairs experiments
        logs_dir = self.results_base_dir / 'logs' if (self.results_base_dir / 'logs').exists() else self.results_base_dir
        
        # Find all strategy_binary_pairs experiments
        experiment_dirs = list(logs_dir.glob('strategy_binary_pairs_*'))
        
        if not experiment_dirs:
            # Try looking one level deeper
            for subdir in logs_dir.glob('*/'):
                experiment_dirs.extend(subdir.glob('strategy_binary_pairs_*'))
        
        print(f"Found {len(experiment_dirs)} experiment directories")
        
        for exp_dir in experiment_dirs:
            if not exp_dir.is_dir():
                continue
            
            print(f"  → Processing {exp_dir.name}...")
            
            # Load basic experiment info
            datasets = self.extract_dataset_names_from_dir(exp_dir)
            if not datasets[0] or not datasets[1]:
                print(f"    ✗ Could not extract dataset names, skipping")
                continue
            
            dataset_a, dataset_b = datasets
            print(f"    → Datasets: {dataset_a} → {dataset_b}")
            
            # Load all strategies from this experiment
            strategies_loaded = self.load_strategies_from_experiment(exp_dir, dataset_a, dataset_b)
            print(f"    → Loaded {len(strategies_loaded)} strategies: {strategies_loaded}")
            
            # Load layer evolution data
            self.load_layer_evolution_data(exp_dir, dataset_a, dataset_b, strategies_loaded)
        
        total_experiments = len(self.experiments)
        print(f"\nLoaded {total_experiments} total experiments")
        print(f"Unique dataset pairs: {len(set((exp['dataset_a'], exp['dataset_b']) for exp in self.experiments))}")
        print(f"Unique strategies: {sorted(set(exp['strategy'] for exp in self.experiments))}")
        
        if total_experiments == 0:
            self.debug_directory_structure()
    
    def extract_dataset_names_from_dir(self, experiment_dir: Path) -> Tuple[str, str]:
        """Extract dataset names from experiment directory."""
        # Method 1: Look for config files
        config_files = list(experiment_dir.glob('config_*.yaml')) + list(experiment_dir.glob('*.yml'))
        
        if config_files:
            try:
                if yaml is not None:
                    with open(config_files[0], 'r') as f:
                        config = yaml.safe_load(f)
                    
                    dataset_a = config.get('dataset_a', '').lower()
                    dataset_b = config.get('dataset_b', '').lower()
                    
                    if dataset_a and dataset_b:
                        return dataset_a, dataset_b
            except Exception:
                pass
        
        # Method 2: Look for results.json
        results_file = experiment_dir / 'results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                dataset_a = results.get('dataset_a', '').lower()
                dataset_b = results.get('dataset_b', '').lower()
                
                if dataset_a and dataset_b:
                    return dataset_a, dataset_b
            except Exception:
                pass
        
        # Method 3: Try to infer from log patterns (common fallback)
        # Look for training logs and try to infer
        print(f"    ⚠ Could not auto-detect datasets for {experiment_dir.name}")
        return '', ''
    
    def load_strategies_from_experiment(self, exp_dir: Path, dataset_a: str, dataset_b: str) -> List[str]:
        """Load all strategies from a single experiment directory."""
        strategies_loaded = []
        
        # Find all training log files
        log_files = list(exp_dir.glob('training_log_*.csv'))
        
        for log_file in log_files:
            # Extract strategy name from filename
            strategy_name = log_file.stem.replace('training_log_', '')
            
            try:
                # Load experiment data
                experiment_data = self.load_single_strategy_experiment(log_file, exp_dir, dataset_a, dataset_b, strategy_name)
                
                if experiment_data:
                    self.experiments.append(experiment_data)
                    strategies_loaded.append(strategy_name)
                    
                    # Store for multiple runs analysis
                    key = (dataset_a, dataset_b, strategy_name)
                    self.all_runs[key].append(experiment_data)
                    
            except Exception as e:
                print(f"    ✗ Failed to load {strategy_name}: {e}")
        
        return strategies_loaded
    
    def load_single_strategy_experiment(self, log_file: Path, exp_dir: Path, dataset_a: str, dataset_b: str, strategy: str) -> Optional[Dict]:
        """Load results from a single strategy within an experiment."""
        try:
            # Read the CSV log
            df = pd.read_csv(log_file)
            
            if df.empty:
                return None
            
            # Get final results (last row)
            final_row = df.iloc[-1]
            
            # Extract accuracy values
            task_0_acc = float(final_row['task_0_acc'])
            task_1_acc = float(final_row['task_1_acc'])
            forgetting = float(final_row['forgetting'])
            average_acc = float(final_row['average_acc'])
            
            # Get Phase 1 performance
            phase1_rows = df[df['phase'] == 1]
            if len(phase1_rows) > 0:
                initial_task_0_acc = float(phase1_rows.iloc[-1]['task_0_acc'])
            else:
                initial_task_0_acc = task_0_acc + forgetting
            
            # Categorize direction
            direction = self.categorize_direction(dataset_a, dataset_b)
            complexity_gap = abs(self.dataset_complexity.get(dataset_b, 0) - self.dataset_complexity.get(dataset_a, 0))
            
            return {
                'dataset_a': dataset_a,
                'dataset_b': dataset_b,
                'strategy': strategy,
                'direction': direction,
                'complexity_gap': complexity_gap,
                'initial_task_0_acc': initial_task_0_acc,
                'final_task_0_acc': task_0_acc,
                'final_task_1_acc': task_1_acc,
                'forgetting_absolute': forgetting,
                'forgetting_relative': forgetting / initial_task_0_acc * 100 if initial_task_0_acc > 0 else 0,
                'average_accuracy': average_acc,
                'experiment_dir': exp_dir,
                'log_file': log_file,
                'training_evolution': self.extract_training_evolution(df)
            }
            
        except Exception as e:
            print(f"    ✗ Error loading {strategy}: {e}")
            return None
    
    def extract_training_evolution(self, df: pd.DataFrame) -> Dict:
        """Extract training evolution data from CSV."""
        evolution = {
            'epochs': [],
            'task_0_acc': [],
            'task_1_acc': [],
            'forgetting': []
        }
        
        for _, row in df.iterrows():
            if pd.notna(row.get('phase')) and str(row['phase']).startswith('2.'):  # Phase 2 intermediate results
                epoch = float(str(row['phase']).split('.')[1])
                evolution['epochs'].append(epoch)
                evolution['task_0_acc'].append(float(row['task_0_acc']))
                evolution['task_1_acc'].append(float(row['task_1_acc']))
                evolution['forgetting'].append(float(row['forgetting']))
        
        return evolution
    
    def categorize_direction(self, dataset_a: str, dataset_b: str) -> str:
        """Categorize pair direction based on complexity order."""
        if dataset_a not in self.dataset_complexity or dataset_b not in self.dataset_complexity:
            return 'unknown'
        
        complexity_a = self.dataset_complexity[dataset_a]
        complexity_b = self.dataset_complexity[dataset_b]
        
        return 'forward' if complexity_a < complexity_b else 'backward'
    
    def load_layer_evolution_data(self, exp_dir: Path, dataset_a: str, dataset_b: str, strategies: List[str]):
        """Load layer-wise identifiability evolution data."""
        layer_analysis_dir = exp_dir / 'layer_analysis'
        
        if not layer_analysis_dir.exists():
            print(f"    ⚠ No layer analysis data found in {exp_dir.name}")
            return
        
        # Load data for each phase
        for phase_name in ['phase1', 'phase2']:
            phase_dir = layer_analysis_dir / phase_name
            if not phase_dir.exists():
                continue
            
            # Find all epoch files
            epoch_files = sorted(phase_dir.glob('epoch_*.json'))
            
            for epoch_file in epoch_files:
                epoch_num = int(epoch_file.stem.split('_')[1])
                
                try:
                    with open(epoch_file, 'r') as f:
                        epoch_data = json.load(f)
                    
                    # Process each task in this epoch
                    for task_key, task_data in epoch_data.items():
                        if 'projection_scores' not in task_data:
                            continue
                        
                        task_id = task_data.get('task_id', int(task_key.split('_')[-1]) if '_' in task_key else 0)
                        
                        # Extract layer-wise identifiability
                        layer_scores = self.extract_layer_scores(task_data['projection_scores'])
                        
                        # Store for each strategy (since all strategies share the same analysis)
                        for strategy in strategies:
                            layer_record = {
                                'strategy': strategy,
                                'dataset_a': dataset_a,
                                'dataset_b': dataset_b,
                                'phase': phase_name,
                                'epoch': epoch_num,
                                'task_id': task_id,
                                'dataset_name': dataset_a if task_id == 0 else dataset_b,
                                'layer_scores': layer_scores,
                                'direction': self.categorize_direction(dataset_a, dataset_b)
                            }
                            
                            self.layer_evolution_data[strategy].append(layer_record)
                
                except Exception as e:
                    print(f"    ✗ Error loading {epoch_file}: {e}")
    
    def extract_layer_scores(self, projection_scores: Dict) -> Dict:
        """Extract layer-wise scores from projection scores."""
        layer_scores = {}
        
        for key, value in projection_scores.items():
            if key.startswith('block_') and '_cls_token' in key:
                # Extract block number
                parts = key.split('_')
                if len(parts) >= 2:
                    block_num = int(parts[1])
                    layer_scores[block_num] = value
        
        return layer_scores
    
    def debug_directory_structure(self):
        """Debug helper to show directory structure."""
        print("\nDEBUG: No experiments found. Directory structure:")
        print(f"Base directory: {self.results_base_dir}")
        print("Contents:")
        for item in self.results_base_dir.rglob('*'):
            if item.is_file() and ('training_log' in item.name or 'epoch_' in item.name):
                print(f"  {item}")
    
    def create_all_analyses(self, output_dir: Path):
        """Create comprehensive analysis with separate charts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.experiments:
            print("No experiments to analyze!")
            return
        
        print(f"Creating comprehensive analysis for {len(self.experiments)} experiments...")
        
        # 1. Basic forgetting analysis
        self.create_forgetting_analysis(output_dir)
        
        # 2. Strategy comparison
        self.create_strategy_comparison(output_dir)
        
        # 3. Directional analysis
        self.create_directional_analysis(output_dir)
        
        # 4. Layer evolution analysis
        self.create_layer_evolution_analysis(output_dir)
        
        # 5. Training dynamics analysis
        self.create_training_dynamics_analysis(output_dir)
        
        # 6. Summary insights
        self.create_summary_insights(output_dir)
        
        # 7. NEW: Comprehensive summary table
        self.create_comprehensive_summary_table(output_dir)
        
        print(f"\nAll analyses saved to: {output_dir}")
    
    def create_forgetting_analysis(self, output_dir: Path):
        """Create basic forgetting analysis charts."""
        print("  → Creating forgetting analysis...")
        
        # Prepare data
        df = pd.DataFrame(self.experiments)
        
        # Chart 1: Forgetting by strategy
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='strategy', y='forgetting_relative')
        plt.title('Relative Forgetting by Strategy', fontsize=16)
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Relative Forgetting (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / '1_forgetting_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 2: Forgetting vs complexity gap
        plt.figure(figsize=(12, 8))
        
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            plt.scatter(strategy_data['complexity_gap'], strategy_data['forgetting_relative'], 
                       label=strategy.title(), alpha=0.7, s=80)
        
        plt.xlabel('Complexity Gap', fontsize=12)
        plt.ylabel('Relative Forgetting (%)', fontsize=12)
        plt.title('Forgetting vs Dataset Complexity Gap', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '2_forgetting_vs_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 3: Performance comparison
        plt.figure(figsize=(14, 8))
        
        strategies = df['strategy'].unique()
        x = np.arange(len(strategies))
        width = 0.35
        
        task0_means = [df[df['strategy'] == s]['final_task_0_acc'].mean() for s in strategies]
        task1_means = [df[df['strategy'] == s]['final_task_1_acc'].mean() for s in strategies]
        
        plt.bar(x - width/2, task0_means, width, label='Task 0 (After Forgetting)', alpha=0.8)
        plt.bar(x + width/2, task1_means, width, label='Task 1 (Newly Learned)', alpha=0.8)
        
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Final Accuracy (%)', fontsize=12)
        plt.title('Final Task Performance by Strategy', fontsize=16)
        plt.xticks(x, [s.title() for s in strategies])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / '3_final_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_strategy_comparison(self, output_dir: Path):
        """Create detailed strategy comparison charts."""
        print("  → Creating strategy comparison...")
        
        df = pd.DataFrame(self.experiments)
        strategies = df['strategy'].unique()
        
        # Comprehensive strategy comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Forgetting comparison
        sns.boxplot(data=df, x='strategy', y='forgetting_relative', ax=axes[0,0])
        axes[0,0].set_title('Relative Forgetting by Strategy')
        axes[0,0].set_ylabel('Relative Forgetting (%)')
        
        # Subplot 2: Average accuracy
        sns.boxplot(data=df, x='strategy', y='average_accuracy', ax=axes[0,1])
        axes[0,1].set_title('Average Accuracy by Strategy')
        axes[0,1].set_ylabel('Average Accuracy (%)')
        
        # Subplot 3: Task 0 retention
        df['task_0_retention'] = (df['final_task_0_acc'] / df['initial_task_0_acc']) * 100
        sns.boxplot(data=df, x='strategy', y='task_0_retention', ax=axes[1,0])
        axes[1,0].set_title('Task 0 Retention by Strategy')
        axes[1,0].set_ylabel('Retention (%)')
        
        # Subplot 4: Task 1 learning
        sns.boxplot(data=df, x='strategy', y='final_task_1_acc', ax=axes[1,1])
        axes[1,1].set_title('Task 1 Learning by Strategy')
        axes[1,1].set_ylabel('Task 1 Accuracy (%)')
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Comprehensive Strategy Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / '4_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_directional_analysis(self, output_dir: Path):
        """Create directional analysis (forward vs backward)."""
        print("  → Creating directional analysis...")
        
        df = pd.DataFrame(self.experiments)
        
        # Filter to only known directions
        df_known = df[df['direction'].isin(['forward', 'backward'])]
        
        if df_known.empty:
            print("    ⚠ No directional data available")
            return
        
        # Chart 1: Direction comparison by strategy
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_known, x='strategy', y='forgetting_relative', hue='direction')
        plt.title('Forgetting by Direction and Strategy', fontsize=16)
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Relative Forgetting (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Direction')
        plt.tight_layout()
        plt.savefig(output_dir / '5_directional_forgetting.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Chart 2: Direction summary statistics
        direction_stats = df_known.groupby(['strategy', 'direction']).agg({
            'forgetting_relative': ['mean', 'std', 'count'],
            'average_accuracy': 'mean'
        }).round(2)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Flatten column names and create table
        direction_stats.columns = ['_'.join(col).strip() for col in direction_stats.columns.values]
        table_data = direction_stats.reset_index()
        
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Directional Analysis Summary Statistics', fontsize=16, pad=20)
        plt.savefig(output_dir / '6_directional_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_layer_evolution_analysis(self, output_dir: Path):
        """Create layer-wise identifiability evolution analysis."""
        print("  → Creating layer evolution analysis...")
        
        if not self.layer_evolution_data:
            print("    ⚠ No layer evolution data available")
            return
        
        # Create separate charts for each strategy
        for strategy, strategy_data in self.layer_evolution_data.items():
            if not strategy_data:
                continue
            
            print(f"    → Analyzing {strategy} layer evolution...")
            
            # Convert to DataFrame
            records = []
            for record in strategy_data:
                for layer_num, score in record['layer_scores'].items():
                    records.append({
                        'strategy': record['strategy'],
                        'phase': record['phase'],
                        'epoch': record['epoch'],
                        'task_id': record['task_id'],
                        'dataset_name': record['dataset_name'],
                        'layer': layer_num,
                        'identifiability': score,
                        'direction': record['direction']
                    })
            
            if not records:
                continue
            
            df_layers = pd.DataFrame(records)
            
            # Chart 1: Layer evolution heatmap for this strategy
            self.create_layer_heatmap(df_layers, strategy, output_dir)
            
            # Chart 2: Forgetting dynamics by layer
            self.create_forgetting_dynamics_by_layer(df_layers, strategy, output_dir)
        
        # Chart 3: Cross-strategy layer comparison
        self.create_cross_strategy_layer_comparison(output_dir)
    
    def create_layer_heatmap(self, df_layers: pd.DataFrame, strategy: str, output_dir: Path):
        """Create layer evolution heatmap for a strategy."""
        # Get Task 0 data from both phases
        task0_phase1 = df_layers[(df_layers['task_id'] == 0) & (df_layers['phase'] == 'phase1')]
        task0_phase2 = df_layers[(df_layers['task_id'] == 0) & (df_layers['phase'] == 'phase2')]
        
        if task0_phase2.empty:
            return
        
        # Combine data with baseline from Phase 1
        combined_data = []
        
        # Add baseline from end of Phase 1 (call it epoch 0)
        if not task0_phase1.empty:
            # Get final epoch from Phase 1
            final_phase1_epoch = task0_phase1['epoch'].max()
            baseline_data = task0_phase1[task0_phase1['epoch'] == final_phase1_epoch].copy()
            baseline_data['epoch'] = 0  # Set as epoch 0 for baseline
            combined_data.append(baseline_data)
        
        # Add Phase 2 data
        combined_data.append(task0_phase2)
        
        # Combine all data
        if combined_data:
            all_task0_data = pd.concat(combined_data, ignore_index=True)
        else:
            all_task0_data = task0_phase2
        
        # Create pivot table
        pivot_data = all_task0_data.pivot_table(
            values='identifiability', 
            index='layer', 
            columns='epoch', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Identifiability'})
        plt.title(f'{strategy.title()} - Task 0 Forgetting: Layer-wise Identifiability Evolution', fontsize=14)
        
        # Custom x-axis labels
        epoch_labels = [f'End P1' if col == 0 else f'{int(col)}' for col in pivot_data.columns]
        plt.xlabel('Epoch (Phase 2)', fontsize=12)
        plt.ylabel('Transformer Layer', fontsize=12)
        
        # Set custom tick labels
        ax = plt.gca()
        ax.set_xticklabels(epoch_labels)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'7_layer_evolution_{strategy}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_forgetting_dynamics_by_layer(self, df_layers: pd.DataFrame, strategy: str, output_dir: Path):
        """Create forgetting dynamics by layer."""
        # Task 0 evolution across both phases
        task0_data = df_layers[df_layers['task_id'] == 0]
        
        if task0_data.empty:
            return
        
        plt.figure(figsize=(14, 10))
        
        # Plot evolution for each layer
        for layer in sorted(task0_data['layer'].unique()):
            layer_data = task0_data[task0_data['layer'] == layer].sort_values(['phase', 'epoch'])
            
            # Create continuous epoch numbering (Phase 1: 1-50, Phase 2: 51-100)
            layer_data = layer_data.copy()
            layer_data['continuous_epoch'] = layer_data.apply(
                lambda row: row['epoch'] if row['phase'] == 'phase1' else row['epoch'] + 50, axis=1
            )
            
            plt.plot(layer_data['continuous_epoch'], layer_data['identifiability'], 
                    marker='o', label=f'Layer {layer}', alpha=0.7)
        
        plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Phase 1 → Phase 2')
        plt.xlabel('Training Epoch', fontsize=12)
        plt.ylabel('CLS Token Identifiability', fontsize=12)
        plt.title(f'{strategy.title()} - Task 0 Forgetting Dynamics by Layer', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'8_forgetting_dynamics_{strategy}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cross_strategy_layer_comparison(self, output_dir: Path):
        """Compare layer patterns across strategies with curriculum direction splits."""
        print("    → Creating cross-strategy layer comparison...")
        
        # Aggregate all layer data
        all_records = []
        for strategy, strategy_data in self.layer_evolution_data.items():
            for record in strategy_data:
                for layer_num, score in record['layer_scores'].items():
                    all_records.append({
                        'strategy': record['strategy'],
                        'phase': record['phase'],
                        'epoch': record['epoch'],
                        'task_id': record['task_id'],
                        'layer': layer_num,
                        'identifiability': score,
                        'direction': record['direction'],
                        'dataset_a': record['dataset_a'],
                        'dataset_b': record['dataset_b']
                    })
        
        if not all_records:
            return
        
        df_all = pd.DataFrame(all_records)
        
        # Focus on final state (last epoch of each phase)
        final_phase1 = df_all[(df_all['phase'] == 'phase1') & (df_all['task_id'] == 0)]
        final_phase2 = df_all[(df_all['phase'] == 'phase2') & (df_all['task_id'] == 0)]
        
        if final_phase1.empty or final_phase2.empty:
            return
        
        # Get final epochs
        final_epoch_p1 = final_phase1['epoch'].max()
        final_epoch_p2 = final_phase2['epoch'].max()
        
        final_p1 = final_phase1[final_phase1['epoch'] == final_epoch_p1]
        final_p2 = final_phase2[final_phase2['epoch'] == final_epoch_p2]
        
        # Create three-panel comparison
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Panel 1: Overall comparison
        self._plot_strategy_layer_comparison(final_p1, final_p2, df_all, 'Overall', axes[0])
        
        # Panel 2: Easy-to-Hard (Forward) only
        forward_data = df_all[df_all['direction'] == 'forward']
        if not forward_data.empty:
            forward_p1 = final_p1[final_p1['direction'] == 'forward']
            forward_p2 = final_p2[final_p2['direction'] == 'forward']
            self._plot_strategy_layer_comparison(forward_p1, forward_p2, forward_data, 'Easy→Hard', axes[1])
        else:
            axes[1].text(0.5, 0.5, 'No forward pairs', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Easy→Hard (No Data)')
        
        # Panel 3: Hard-to-Easy (Backward) only
        backward_data = df_all[df_all['direction'] == 'backward']
        if not backward_data.empty:
            backward_p1 = final_p1[final_p1['direction'] == 'backward']
            backward_p2 = final_p2[final_p2['direction'] == 'backward']
            self._plot_strategy_layer_comparison(backward_p1, backward_p2, backward_data, 'Hard→Easy', axes[2])
        else:
            axes[2].text(0.5, 0.5, 'No backward pairs', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Hard→Easy (No Data)')
        
        plt.suptitle('Cross-Strategy Layer Comparison: Task 0 Identifiability by Curriculum Direction', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / '9_cross_strategy_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual detailed charts
        self._create_detailed_curriculum_analysis(df_all, output_dir)
    
    def _plot_strategy_layer_comparison(self, final_p1, final_p2, data_subset, direction_name, ax):
        """Plot strategy layer comparison for a specific direction."""
        strategies = data_subset['strategy'].unique()
        
        for strategy in strategies:
            # Phase 1 data
            p1_data = final_p1[final_p1['strategy'] == strategy].groupby('layer')['identifiability'].mean()
            # Phase 2 data  
            p2_data = final_p2[final_p2['strategy'] == strategy].groupby('layer')['identifiability'].mean()
            
            if not p1_data.empty:
                ax.plot(p1_data.index, p1_data.values, 'o-', 
                       label=f'{strategy.title()} (After Phase 1)', alpha=0.7, linewidth=2, markersize=6)
            if not p2_data.empty:
                ax.plot(p2_data.index, p2_data.values, 's--', 
                       label=f'{strategy.title()} (After Phase 2)', alpha=0.7, linewidth=2, markersize=6)
        
        ax.set_xlabel('Transformer Layer', fontsize=11)
        ax.set_ylabel('CLS Token Identifiability', fontsize=11)
        ax.set_title(f'{direction_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add count information
        if direction_name != 'Overall':
            n_pairs = len(data_subset[['dataset_a', 'dataset_b']].drop_duplicates())
            ax.text(0.02, 0.98, f'n={n_pairs} pairs', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top', fontsize=9)
    
    def _create_detailed_curriculum_analysis(self, df_all: pd.DataFrame, output_dir: Path):
        """Create detailed curriculum direction analysis charts."""
        print("    → Creating detailed curriculum direction analysis...")
        
        # Create separate detailed charts for forward and backward
        for direction in ['forward', 'backward']:
            direction_data = df_all[df_all['direction'] == direction]
            
            if direction_data.empty:
                continue
            
            direction_name = 'Easy→Hard' if direction == 'forward' else 'Hard→Easy'
            
            # Get unique pairs for this direction
            pairs = direction_data[['dataset_a', 'dataset_b']].drop_duplicates()
            
            plt.figure(figsize=(16, 10))
            
            # Focus on Task 0 forgetting across both phases
            task0_data = direction_data[direction_data['task_id'] == 0]
            
            strategies = task0_data['strategy'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
            
            for i, strategy in enumerate(strategies):
                strategy_data = task0_data[task0_data['strategy'] == strategy]
                
                for layer in sorted(strategy_data['layer'].unique()):
                    layer_data = strategy_data[strategy_data['layer'] == layer].sort_values(['phase', 'epoch'])
                    
                    # Create continuous epoch numbering (Phase 1: 1-50, Phase 2: 51-100)
                    layer_data = layer_data.copy()
                    layer_data['continuous_epoch'] = layer_data.apply(
                        lambda row: row['epoch'] if row['phase'] == 'phase1' else row['epoch'] + 50, axis=1
                    )
                    
                    if i == 0:  # Only show layer labels once
                        plt.plot(layer_data['continuous_epoch'], layer_data['identifiability'], 
                               color=colors[i], alpha=0.3, linewidth=1, label=f'Layer {layer}' if strategy == strategies[0] else '')
                    else:
                        plt.plot(layer_data['continuous_epoch'], layer_data['identifiability'], 
                               color=colors[i], alpha=0.3, linewidth=1)
                
                # Add strategy-level average
                strategy_avg = strategy_data.groupby(['phase', 'epoch'])['identifiability'].mean().reset_index()
                strategy_avg['continuous_epoch'] = strategy_avg.apply(
                    lambda row: row['epoch'] if row['phase'] == 'phase1' else row['epoch'] + 50, axis=1
                )
                
                plt.plot(strategy_avg['continuous_epoch'], strategy_avg['identifiability'], 
                        color=colors[i], linewidth=3, label=f'{strategy.title()} (Avg)', marker='o', markersize=4)
            
            plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Phase 1 → Phase 2')
            plt.xlabel('Training Epoch', fontsize=12)
            plt.ylabel('CLS Token Identifiability', fontsize=12)
            plt.title(f'{direction_name} Curriculum: Task 0 Forgetting by Layer and Strategy', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Add pairs information
            pair_list = [f"{DATASET_DISPLAY_NAMES.get(row['dataset_a'], row['dataset_a'])}→{DATASET_DISPLAY_NAMES.get(row['dataset_b'], row['dataset_b'])}" 
                        for _, row in pairs.iterrows()]
            pair_text = f"Pairs ({len(pair_list)}): " + ", ".join(pair_list[:3])
            if len(pair_list) > 3:
                pair_text += f" (+{len(pair_list)-3} more)"
            
            plt.figtext(0.02, 0.02, pair_text, fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'9b_curriculum_{direction}_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_training_dynamics_analysis(self, output_dir: Path):
        """Create training dynamics analysis."""
        print("  → Creating training dynamics analysis...")
        
        # Filter experiments with evolution data
        experiments_with_evolution = [exp for exp in self.experiments if exp['training_evolution']['epochs']]
        
        if not experiments_with_evolution:
            print("    ⚠ No training dynamics data available")
            return
        
        # Create dynamics plot for each strategy
        strategies = set(exp['strategy'] for exp in experiments_with_evolution)
        
        for strategy in strategies:
            strategy_experiments = [exp for exp in experiments_with_evolution if exp['strategy'] == strategy]
            
            if not strategy_experiments:
                continue
            
            plt.figure(figsize=(14, 10))
            
            for i, exp in enumerate(strategy_experiments):
                evolution = exp['training_evolution']
                if not evolution['epochs']:
                    continue
                
                pair_name = f"{exp['dataset_a']} → {exp['dataset_b']}"
                
                # Plot both tasks
                plt.plot(evolution['epochs'], evolution['task_0_acc'], 
                        'o-', label=f'{pair_name} (Task 0)', alpha=0.7)
                plt.plot(evolution['epochs'], evolution['task_1_acc'], 
                        's--', label=f'{pair_name} (Task 1)', alpha=0.7)
            
            plt.xlabel('Epoch (Phase 2)', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'{strategy.title()} - Training Dynamics During Phase 2', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'10_training_dynamics_{strategy}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_insights(self, output_dir: Path):
        """Create summary insights and statistics."""
        print("  → Creating summary insights...")
        
        # Generate comprehensive insights
        insights = self.generate_insights()
        
        # Save insights as JSON
        insights_path = output_dir / 'comprehensive_insights.json'
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Create summary visualization
        self.create_summary_table(insights, output_dir)
        
        print(f"    → Insights saved to {insights_path}")
    
    def create_summary_table(self, insights: Dict, output_dir: Path):
        """Create summary table visualization."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        summary_data.append(['Metric', 'Value'])
        summary_data.append(['Total Experiments', str(insights['summary']['total_experiments'])])
        summary_data.append(['Strategies Analyzed', ', '.join(insights['summary']['strategies'])])
        summary_data.append(['Dataset Pairs', str(insights['summary']['unique_pairs'])])
        
        if 'strategy_performance' in insights:
            summary_data.append(['', ''])  # Separator
            summary_data.append(['Strategy Performance', ''])
            
            for strategy, perf in insights['strategy_performance'].items():
                summary_data.append([f'{strategy.title()} Avg Forgetting', f"{perf['mean_forgetting']:.1f}%"])
                summary_data.append([f'{strategy.title()} Avg Accuracy', f"{perf['mean_accuracy']:.1f}%"])
        
        # Create table
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j)] if i == 0 else table[(i-1, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                elif summary_data[i][0] == '':  # Separator
                    cell.set_facecolor('#f0f0f0')
                elif 'Performance' in summary_data[i][0]:  # Section header
                    cell.set_facecolor('#e6e6e6')
                    cell.set_text_props(weight='bold')
        
        plt.title('Comprehensive Analysis Summary', fontsize=16, pad=20, weight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / '11_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_summary_table(self, output_dir: Path):
        """Create comprehensive summary table showing all strategies vs dataset pairs."""
        print("  → Creating comprehensive summary table...")
        
        # Aggregate data by dataset pair and strategy
        summary_data = defaultdict(lambda: defaultdict(list))
        
        for (dataset_a, dataset_b, strategy), runs in self.all_runs.items():
            key = (dataset_a, dataset_b)
            
            # Extract metrics from all runs
            task0_accs = [run['final_task_0_acc'] for run in runs]
            task1_accs = [run['final_task_1_acc'] for run in runs]
            forgettings = [run['forgetting_absolute'] for run in runs]
            
            summary_data[key][strategy] = {
                'task0_mean': np.mean(task0_accs),
                'task0_std': np.std(task0_accs),
                'task1_mean': np.mean(task1_accs),
                'task1_std': np.std(task1_accs),
                'forgetting_mean': np.mean(forgettings),
                'forgetting_std': np.std(forgettings),
                'n_runs': len(runs)
            }
        
        # Get all unique strategies
        all_strategies = sorted(set(strategy for pair_data in summary_data.values() 
                                  for strategy in pair_data.keys()))
        
        # Create DataFrame for CSV export
        rows = []
        for (dataset_a, dataset_b), strategies_data in sorted(summary_data.items()):
            row = {
                'dataset_pair': f"{DATASET_DISPLAY_NAMES.get(dataset_a, dataset_a)}→{DATASET_DISPLAY_NAMES.get(dataset_b, dataset_b)}",
                'dataset_a': dataset_a,
                'dataset_b': dataset_b
            }
            
            for strategy in all_strategies:
                if strategy in strategies_data:
                    data = strategies_data[strategy]
                    row.update({
                        f'{strategy}_task0_mean': data['task0_mean'],
                        f'{strategy}_task0_std': data['task0_std'],
                        f'{strategy}_task1_mean': data['task1_mean'],
                        f'{strategy}_task1_std': data['task1_std'],
                        f'{strategy}_forgetting_mean': data['forgetting_mean'],
                        f'{strategy}_forgetting_std': data['forgetting_std'],
                        f'{strategy}_n_runs': data['n_runs']
                    })
                else:
                    # No data for this strategy-pair combination
                    row.update({
                        f'{strategy}_task0_mean': np.nan,
                        f'{strategy}_task0_std': np.nan,
                        f'{strategy}_task1_mean': np.nan,
                        f'{strategy}_task1_std': np.nan,
                        f'{strategy}_forgetting_mean': np.nan,
                        f'{strategy}_forgetting_std': np.nan,
                        f'{strategy}_n_runs': 0
                    })
            
            rows.append(row)
        
        # Save to CSV
        df_summary = pd.DataFrame(rows)
        csv_path = output_dir / 'comprehensive_summary_table.csv'
        df_summary.to_csv(csv_path, index=False)
        print(f"    → Summary table saved to {csv_path}")
        
        # Create visual table
        self._create_visual_summary_table(summary_data, all_strategies, output_dir)
        
        # Create detailed tables for each metric
        self._create_metric_specific_tables(summary_data, all_strategies, output_dir)
    
    def _create_visual_summary_table(self, summary_data: Dict, strategies: List[str], output_dir: Path):
        """Create visual summary table with all metrics."""
        # Create three separate tables for better readability
        metrics = ['task0', 'task1', 'forgetting']
        metric_names = {
            'task0': 'Task 0 Accuracy (After Forgetting)',
            'task1': 'Task 1 Accuracy',
            'forgetting': 'Forgetting (Absolute)'
        }
        
        for metric_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.axis('off')
            
            # Prepare table data
            headers = ['Dataset Pair'] + [f'{s.title()}' for s in strategies]
            table_data = []
            
            for (dataset_a, dataset_b), strategies_data in sorted(summary_data.items()):
                pair_name = f"{DATASET_DISPLAY_NAMES.get(dataset_a, dataset_a)}→{DATASET_DISPLAY_NAMES.get(dataset_b, dataset_b)}"
                row = [pair_name]
                
                for strategy in strategies:
                    if strategy in strategies_data:
                        data = strategies_data[strategy]
                        mean_key = f'{metric}_mean'
                        std_key = f'{metric}_std'
                        
                        mean_val = data[mean_key]
                        std_val = data[std_key]
                        n_runs = data['n_runs']
                        
                        if n_runs > 1:
                            cell_text = f"{mean_val:.1f}±{std_val:.1f}\n(n={n_runs})"
                        else:
                            cell_text = f"{mean_val:.1f}\n(n=1)"
                    else:
                        cell_text = "—"
                    
                    row.append(cell_text)
                
                table_data.append(row)
            
            # Create table
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.5)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(len(headers)):
                    if i == 0:  # Header row
                        cell = table[(i, j)]
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell = table[(i, j)]
                        if j == 0:  # Dataset pair column
                            cell.set_facecolor('#f0f0f0')
                        else:
                            # Color code based on metric
                            if table_data[i-1][j] != "—":
                                value_str = table_data[i-1][j].split('±')[0]
                                try:
                                    value = float(value_str)
                                    if metric == 'forgetting':
                                        # Lower forgetting is better (green)
                                        if value < 20:
                                            cell.set_facecolor('#90EE90')
                                        elif value < 40:
                                            cell.set_facecolor('#FFFFE0')
                                        else:
                                            cell.set_facecolor('#FFB6C1')
                                    else:
                                        # Higher accuracy is better
                                        if value > 80:
                                            cell.set_facecolor('#90EE90')
                                        elif value > 60:
                                            cell.set_facecolor('#FFFFE0')
                                        else:
                                            cell.set_facecolor('#FFB6C1')
                                except:
                                    pass
            
            plt.title(f'{metric_names[metric]} by Strategy and Dataset Pair', 
                     fontsize=16, pad=20, weight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'12_summary_table_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_metric_specific_tables(self, summary_data: Dict, strategies: List[str], output_dir: Path):
        """Create separate detailed tables for each metric."""
        # Create a combined heatmap visualization
        n_strategies = len(strategies)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        metrics = ['task0', 'task1', 'forgetting']
        metric_names = ['Task 0 Acc', 'Task 1 Acc', 'Forgetting']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # Create matrix for heatmap
            pairs = sorted(summary_data.keys())
            matrix = np.zeros((len(pairs), n_strategies))
            
            for i, (dataset_a, dataset_b) in enumerate(pairs):
                for j, strategy in enumerate(strategies):
                    if strategy in summary_data[(dataset_a, dataset_b)]:
                        matrix[i, j] = summary_data[(dataset_a, dataset_b)][strategy][f'{metric}_mean']
                    else:
                        matrix[i, j] = np.nan
            
            # Create heatmap
            ax = axes[idx]
            
            # Mask NaN values
            masked_matrix = np.ma.masked_invalid(matrix)
            
            if metric == 'forgetting':
                cmap = 'RdYlGn_r'  # Reversed: green=low (good), red=high (bad)
            else:
                cmap = 'RdYlGn'  # Normal: green=high (good), red=low (bad)
            
            im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto')
            
            # Set ticks
            ax.set_xticks(np.arange(n_strategies))
            ax.set_yticks(np.arange(len(pairs)))
            ax.set_xticklabels([s.title() for s in strategies], rotation=45, ha='right')
            ax.set_yticklabels([f"{DATASET_DISPLAY_NAMES.get(p[0], p[0])}→{DATASET_DISPLAY_NAMES.get(p[1], p[1])}" 
                               for p in pairs])
            
            # Add values to cells
            for i in range(len(pairs)):
                for j in range(n_strategies):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                     ha='center', va='center', color='black', fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_name, rotation=270, labelpad=15)
            
            ax.set_title(metric_name, fontsize=14, fontweight='bold')
        
        plt.suptitle('Performance Summary Heatmaps', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / '13_performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_insights(self) -> Dict:
        """Generate comprehensive insights from all analyses."""
        insights = {
            'summary': {
                'total_experiments': len(self.experiments),
                'strategies': list(set(exp['strategy'] for exp in self.experiments)),
                'unique_pairs': len(set((exp['dataset_a'], exp['dataset_b']) for exp in self.experiments))
            },
            'strategy_performance': {},
            'directional_effects': {},
            'layer_insights': {},
            'recommendations': []
        }
        
        # Strategy performance analysis
        df = pd.DataFrame(self.experiments)
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            insights['strategy_performance'][strategy] = {
                'count': len(strategy_data),
                'mean_forgetting': strategy_data['forgetting_relative'].mean(),
                'std_forgetting': strategy_data['forgetting_relative'].std(),
                'mean_accuracy': strategy_data['average_accuracy'].mean(),
                'best_pair': strategy_data.loc[strategy_data['forgetting_relative'].idxmin()][['dataset_a', 'dataset_b']].tolist(),
                'worst_pair': strategy_data.loc[strategy_data['forgetting_relative'].idxmax()][['dataset_a', 'dataset_b']].tolist()
            }
        
        # Directional effects analysis
        df_known = df[df['direction'].isin(['forward', 'backward'])]
        if not df_known.empty:
            for direction in ['forward', 'backward']:
                direction_data = df_known[df_known['direction'] == direction]
                if not direction_data.empty:
                    insights['directional_effects'][direction] = {
                        'count': len(direction_data),
                        'mean_forgetting': direction_data['forgetting_relative'].mean(),
                        'mean_accuracy': direction_data['average_accuracy'].mean()
                    }
        
        # Generate recommendations
        if insights['strategy_performance']:
            best_strategy = min(insights['strategy_performance'].keys(), 
                              key=lambda s: insights['strategy_performance'][s]['mean_forgetting'])
            insights['recommendations'].append(f"Best performing strategy: {best_strategy.title()}")
        
        if insights['directional_effects']:
            if len(insights['directional_effects']) == 2:
                forward_forgetting = insights['directional_effects']['forward']['mean_forgetting']
                backward_forgetting = insights['directional_effects']['backward']['mean_forgetting']
                
                if forward_forgetting > backward_forgetting:
                    insights['recommendations'].append("Forward transitions (simple→complex) show more forgetting")
                else:
                    insights['recommendations'].append("Backward transitions (complex→simple) show more forgetting")
        
        return insights


def main():
    parser = argparse.ArgumentParser(description='Comprehensive binary pairs analysis with layer evolution')
    parser.add_argument('results_dir', type=str, 
                       help='Base directory containing results')
    parser.add_argument('--output', type=str, default='comprehensive_analysis',
                       help='Output directory for all analyses')
    
    args = parser.parse_args()
    
    # Handle different directory structures
    results_path = Path(args.results_dir)
    
    # If pointing to a binary_pairs_analysis directory, look for logs inside
    if 'binary_pairs_analysis' in results_path.name:
        if (results_path / 'logs').exists():
            analyzer = ComprehensiveBinaryPairsAnalyzer(results_path / 'logs')
        else:
            analyzer = ComprehensiveBinaryPairsAnalyzer(results_path)
    else:
        analyzer = ComprehensiveBinaryPairsAnalyzer(results_path)
    
    # Load all experiments
    analyzer.load_all_experiments()
    
    # Check if we found any results
    if not analyzer.experiments:
        print("\n❌ No experiments found!")
        print("Make sure you're pointing to the correct directory.")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    analyzer.create_all_analyses(output_dir)
    
    print(f"\n✅ Analysis complete! Check {output_dir} for all charts:")
    print("   1_forgetting_by_strategy.png - Basic forgetting comparison")
    print("   2_forgetting_vs_complexity.png - Complexity effects")
    print("   3_final_performance.png - Performance comparison")
    print("   4_strategy_comparison.png - Detailed strategy analysis")
    print("   5_directional_forgetting.png - Forward vs backward effects")
    print("   6_directional_stats.png - Direction statistics")
    print("   7_layer_evolution_*.png - Layer-wise evolution per strategy")
    print("   8_forgetting_dynamics_*.png - Forgetting dynamics per strategy")
    print("   9_cross_strategy_layers.png - Cross-strategy layer comparison")
    print("   10_training_dynamics_*.png - Training evolution per strategy")
    print("   11_summary_table.png - Summary insights")
    print("   12_summary_table_*.png - Comprehensive performance tables")
    print("   13_performance_heatmaps.png - Performance heatmap visualization")
    print("   comprehensive_insights.json - Detailed insights")
    print("   comprehensive_summary_table.csv - Raw data for all experiments")


if __name__ == "__main__":
    main()
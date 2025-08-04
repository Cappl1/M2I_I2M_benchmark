#!/usr/bin/env python3
"""
Aggregate and analyze binary pairs experiment results across multiple runs.
Works with the actual data structure in /home/brothen/M2I_I2M_benchmark/logs.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
import matplotlib.gridspec as gridspec

class BinaryPairsAggregatedAnalyzer:
    """Analyze binary pairs experiments across multiple runs and strategies."""
    
    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)
        # Look for binary_pairs_analysis directories first, then fall back to logs
        self.experiments = self.discover_experiments()
        self.datasets = ["mnist", "fashion_mnist", "cifar10", "svhn"]
        self.strategies = ["Naive", "Replay", "Cumulative"]
        
        print(f"Found {len(self.experiments)} experiment directories")
        
    def discover_experiments(self) -> List[Dict]:
        """Discover all experiment directories and extract metadata."""
        experiments = []
        
        # First try to find binary_pairs_analysis directories
        analysis_dirs = list(self.logs_dir.glob("binary_pairs_analysis_*"))
        if not analysis_dirs:
            # Fall back to looking directly in logs dir
            search_dirs = [self.logs_dir]
        else:
            # Look inside binary_pairs_analysis directories
            search_dirs = []
            for analysis_dir in analysis_dirs:
                search_dirs.extend(analysis_dir.glob("*"))
        
        for search_dir in search_dirs:
            for exp_dir in search_dir.glob("strategy_binary_pairs_*"):
                if exp_dir.is_dir():
                    results_file = exp_dir / "results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            experiment = {
                                'directory': exp_dir,
                                'strategy': results.get('strategy_name'),
                                'dataset_a': results.get('dataset_a'),
                                'dataset_b': results.get('dataset_b'),
                                'pair_name': f"{results.get('dataset_a')}_to_{results.get('dataset_b')}",
                                'results': results,
                                'timestamp': exp_dir.name.split('_')[-2:],  # Extract timestamp
                            }
                            experiments.append(experiment)
                            
                        except Exception as e:
                            print(f"Warning: Could not parse {results_file}: {e}")
        
        return experiments
    
    def group_experiments_by_run(self) -> Dict[str, List[Dict]]:
        """Group experiments by run (based on timestamp)."""
        runs = defaultdict(list)
        
        for exp in self.experiments:
            # Create run identifier from timestamp
            timestamp = "_".join(exp['timestamp'])
            runs[timestamp].append(exp)
        
        return dict(runs)
    
    def load_performance_data(self) -> pd.DataFrame:
        """Load performance data from all experiments."""
        records = []
        
        for exp in self.experiments:
            results = exp['results']
            final_acc = results.get('final_accuracies', {})
            
            record = {
                'strategy': exp['strategy'],
                'dataset_a': exp['dataset_a'],
                'dataset_b': exp['dataset_b'],
                'pair_name': exp['pair_name'],
                'run_id': "_".join(exp['timestamp']),
                'task_0_acc': final_acc.get(f"task_0_{exp['dataset_a']}", 0.0),
                'task_1_acc': final_acc.get(f"task_1_{exp['dataset_b']}", 0.0),
                'forgetting': final_acc.get('forgetting', 0.0),
                'average_acc': final_acc.get('average', 0.0),
                'directory': str(exp['directory'])
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def load_patch_analysis_data(self, exp_dir: Path) -> Dict:
        """Load patch analysis data from an experiment directory."""
        patch_dir = exp_dir / "patch_analysis"
        patch_data = {
            'task1_learning': defaultdict(dict),       # phase1_mnist 
            'task2_learning': defaultdict(dict),       # phase2_fashion_mnist
            'task1_forgetting': defaultdict(dict)      # phase2_forgetting_mnist
        }
        
        if not patch_dir.exists():
            print(f"    ⚠️  No patch_analysis dir in {exp_dir.name}")
            return patch_data
        
        # Load numpy files with proper task separation
        npz_files = list(patch_dir.glob("patch_importance_*.npz"))
        print(f"    🎨 Found {len(npz_files)} patch .npz files")
        
        for npz_file in npz_files:
            filename = npz_file.name
            
            try:
                # Extract epoch from filename
                epoch_match = re.search(r'epoch_(\d+)', filename)
                if not epoch_match:
                    continue
                epoch = int(epoch_match.group(1))
                
                # Load numpy data
                data = np.load(npz_file)
                
                # Convert to dict format
                patch_maps = {}
                for key in data.files:
                    if key.startswith('block_') and key != 'block_names':
                        patch_maps[key] = data[key]
                
                # Classify by filename pattern - match your exact structure
                if 'phase1' in filename:
                    # phase1_mnist_epoch_050.npz → Task 1 Learning
                    patch_data['task1_learning'][epoch] = {'patch_importance_maps': patch_maps}
                    print(f"      📊 Task1 learning epoch {epoch}: {len(patch_maps)} blocks")
                elif 'phase2_forgetting' in filename:
                    # phase2_forgetting_mnist_epoch_050.npz → Task 1 Forgetting
                    patch_data['task1_forgetting'][epoch] = {'patch_importance_maps': patch_maps}
                    print(f"      📉 Task1 forgetting epoch {epoch}: {len(patch_maps)} blocks")
                elif 'phase2' in filename:
                    # phase2_fashion_mnist_epoch_050.npz → Task 2 Learning
                    patch_data['task2_learning'][epoch] = {'patch_importance_maps': patch_maps}
                    print(f"      📈 Task2 learning epoch {epoch}: {len(patch_maps)} blocks")
                else:
                    print(f"      ❓ Unknown patch file pattern: {filename}")
                    
            except Exception as e:
                print(f"    ❌ Could not load {npz_file}: {e}")
        
        # Debug summary
        task1_learning_epochs = len(patch_data['task1_learning'])
        task2_learning_epochs = len(patch_data['task2_learning'])
        task1_forgetting_epochs = len(patch_data['task1_forgetting'])
        
        print(f"    🎨 Patch summary: Task1_learning={task1_learning_epochs}, Task2_learning={task2_learning_epochs}, Task1_forgetting={task1_forgetting_epochs}")
        
        return patch_data
    
    def load_layer_analysis_data(self, exp_dir: Path) -> Dict:
        """Load layer analysis data from an experiment directory."""
        layer_dir = exp_dir / "layer_analysis"
        layer_data = {'phase1': {}, 'phase2': {'task_0': {}, 'task_1': {}}}
        
        if not layer_dir.exists():
            print(f"    ⚠️  No layer_analysis dir in {exp_dir.name}")
            return layer_data
        
        # Load phase1 data
        phase1_dir = layer_dir / "phase1"
        if phase1_dir.exists():
            phase1_files = list(phase1_dir.glob("epoch_*.json"))
            print(f"    📊 Phase1: Found {len(phase1_files)} epoch files")
            
            for json_file in sorted(phase1_files):
                epoch = int(re.search(r'epoch_(\d+)', json_file.name).group(1))
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Phase1 might have different structures - handle both
                    if 'projection_scores' in data:
                        # Direct structure
                        layer_data['phase1'][epoch] = data
                    elif len(data) == 1:
                        # Single task wrapper
                        task_key = list(data.keys())[0]
                        layer_data['phase1'][epoch] = data[task_key]
                    else:
                        # Unknown structure, store as-is
                        layer_data['phase1'][epoch] = data
                        
                except Exception as e:
                    print(f"    ❌ Could not load {json_file}: {e}")
        else:
            print(f"    ⚠️  No phase1 directory in {exp_dir.name}")
        
        # Load phase2 data with task separation
        phase2_dir = layer_dir / "phase2"
        if phase2_dir.exists():
            phase2_files = list(phase2_dir.glob("epoch_*.json"))
            print(f"    📊 Phase2: Found {len(phase2_files)} epoch files")
            
            for json_file in sorted(phase2_files):
                epoch = int(re.search(r'epoch_(\d+)', json_file.name).group(1))
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Debug: print structure of first file
                    if epoch == 10:  # Just for first epoch
                        print(f"    🔍 Phase2 epoch {epoch} structure: {list(data.keys())}")
                        if 'task_0' in data:
                            print(f"        task_0 keys: {list(data['task_0'].keys())}")
                        if 'task_1' in data:
                            print(f"        task_1 keys: {list(data['task_1'].keys())}")
                    
                    # Separate task_0 and task_1 data
                    if 'task_0' in data:
                        layer_data['phase2']['task_0'][epoch] = data['task_0']
                    if 'task_1' in data:
                        layer_data['phase2']['task_1'][epoch] = data['task_1']
                        
                except Exception as e:
                    print(f"    ❌ Could not load {json_file}: {e}")
        else:
            print(f"    ⚠️  No phase2 directory in {exp_dir.name}")
        
        # Debug summary
        phase1_epochs = len(layer_data['phase1'])
        phase2_task0_epochs = len(layer_data['phase2']['task_0'])
        phase2_task1_epochs = len(layer_data['phase2']['task_1'])
        
        print(f"    📋 Loaded: Phase1={phase1_epochs} epochs, Phase2 Task0={phase2_task0_epochs} epochs, Phase2 Task1={phase2_task1_epochs} epochs")
        
        return layer_data
    
    def create_performance_summary(self, output_dir: Path):
        """Create comprehensive performance summary across all runs."""
        df = self.load_performance_data()
        
        if df.empty:
            print("No performance data found")
            return
        
        # Create performance comparison heatmaps
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.3])
        
        metrics = ['task_0_acc', 'task_1_acc', 'forgetting', 'average_acc']
        titles = ['Task A Retention (%)', 'Task B Accuracy (%)', 'Forgetting (%)', 'Average Accuracy (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            if idx < 2:
                ax = fig.add_subplot(gs[idx, :])
            else:
                ax = fig.add_subplot(gs[2, idx-2])
            
            # Create pivot table for heatmap
            pivot_data = df.groupby(['strategy', 'pair_name'])[metric].mean().unstack(fill_value=0)
            
            # Ensure all strategies and pairs are represented
            for strategy in self.strategies:
                if strategy not in pivot_data.index:
                    pivot_data.loc[strategy] = 0
            
            # Reorder to match strategy order
            pivot_data = pivot_data.reindex(self.strategies)
            
            # Create heatmap
            vmin, vmax = 0, 100
            if metric == 'forgetting':
                cmap = 'Reds'
            else:
                cmap = 'RdYlGn'
            
            sns.heatmap(pivot_data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                       annot=True, fmt='.1f', cbar_kws={'label': title})
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Dataset Pair', fontsize=12)
            ax.set_ylabel('Strategy', fontsize=12)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Binary Pairs Performance Summary (Averaged Across Runs)', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(output_dir / 'performance_summary_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed statistics table
        stats_summary = []
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy]
            if not strategy_data.empty:
                stats = {
                    'Strategy': strategy,
                    'Mean Task A Retention': strategy_data['task_0_acc'].mean(),
                    'Mean Task B Accuracy': strategy_data['task_1_acc'].mean(),
                    'Mean Forgetting': strategy_data['forgetting'].mean(),
                    'Mean Average Accuracy': strategy_data['average_acc'].mean(),
                    'Std Task A Retention': strategy_data['task_0_acc'].std(),
                    'Std Forgetting': strategy_data['forgetting'].std(),
                    'Num Experiments': len(strategy_data),
                }
                stats_summary.append(stats)
        
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(output_dir / 'strategy_performance_stats.csv', index=False)
        
        print(f"Performance summary saved to {output_dir}")
        return stats_df
    
    def analyze_forgetting_patterns(self, output_dir: Path):
        """Analyze forgetting patterns across dataset pairs and strategies."""
        df = self.load_performance_data()
        
        if df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Forgetting by strategy (boxplot)
        strategy_order = ['Naive', 'Replay', 'Cumulative']
        df_plot = df[df['strategy'].isin(strategy_order)]
        
        sns.boxplot(data=df_plot, x='strategy', y='forgetting', ax=ax1, order=strategy_order)
        ax1.set_title('Forgetting Distribution by Strategy')
        ax1.set_ylabel('Forgetting (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Task retention vs new task performance
        for strategy in strategy_order:
            strategy_data = df[df['strategy'] == strategy]
            ax2.scatter(strategy_data['task_0_acc'], strategy_data['task_1_acc'], 
                       label=strategy, alpha=0.7, s=60)
        
        ax2.set_xlabel('Task A Retention (%)')
        ax2.set_ylabel('Task B Accuracy (%)')
        ax2.set_title('Retention vs New Learning Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Dataset pair difficulty analysis
        pair_forgetting = df.groupby('pair_name')['forgetting'].agg(['mean', 'std']).reset_index()
        pair_forgetting = pair_forgetting.sort_values('mean')
        
        ax3.barh(range(len(pair_forgetting)), pair_forgetting['mean'], 
                xerr=pair_forgetting['std'], alpha=0.7)
        ax3.set_yticks(range(len(pair_forgetting)))
        ax3.set_yticklabels([p.replace('_to_', ' → ') for p in pair_forgetting['pair_name']])
        ax3.set_xlabel('Average Forgetting (%)')
        ax3.set_title('Forgetting by Dataset Pair')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Strategy effectiveness heatmap
        effectiveness = df.groupby(['strategy', 'pair_name'])['forgetting'].mean().unstack()
        
        # Calculate relative improvement over Naive
        if 'Naive' in effectiveness.index:
            naive_baseline = effectiveness.loc['Naive']
            relative_improvement = effectiveness.subtract(naive_baseline, axis=1) * -1  # Negative because lower forgetting is better
            
            sns.heatmap(relative_improvement, ax=ax4, cmap='RdBu_r', center=0, 
                       annot=True, fmt='.1f', cbar_kws={'label': 'Forgetting Reduction vs Naive (%)'})
            ax4.set_title('Strategy Improvement over Naive Baseline')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'forgetting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_patch_importance_aggregated(self, output_dir: Path):
        """Analyze patch importance patterns aggregated across runs."""
        # Group experiments by strategy and pair
        strategy_pairs = defaultdict(list)
        
        for exp in self.experiments:
            if (exp['dataset_a'] and exp['dataset_b'] and exp['strategy']):
                key = f"{exp['strategy']}_{exp['pair_name']}"
                strategy_pairs[key].append(exp)
        
        # Analyze a representative subset
        sample_keys = list(strategy_pairs.keys())[:6]  # Analyze first 6 combinations
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, key in enumerate(sample_keys):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            experiments = strategy_pairs[key]
            
            # Aggregate patch importance across runs for this strategy-pair combination
            all_patch_maps = []
            
            for exp in experiments:
                patch_data = self.load_patch_analysis_data(exp['directory'])
                
                # Get final epoch data from any available task phase
                final_data = None
                for task_phase in ['task1_learning', 'task2_learning', 'task1_forgetting']:
                    if patch_data[task_phase]:
                        final_epoch = max(patch_data[task_phase].keys())
                        final_data = patch_data[task_phase][final_epoch]
                        break
                
                if final_data and 'patch_importance_maps' in final_data:
                    # Average across blocks
                    block_maps = []
                    for block_name, importance_map in final_data['patch_importance_maps'].items():
                        if isinstance(importance_map, (list, np.ndarray)):
                            importance_map = np.array(importance_map)
                            block_maps.append(importance_map)
                    
                    if block_maps:
                        avg_map = np.mean(block_maps, axis=0)
                        all_patch_maps.append(avg_map)
            
            # Plot aggregated results
            if all_patch_maps:
                final_map = np.mean(all_patch_maps, axis=0)
                im = ax.imshow(final_map, cmap='viridis', aspect='auto')
                
                strategy, pair = key.split('_', 1)
                pair_display = pair.replace('_to_', ' → ')
                ax.set_title(f'{strategy}\n{pair_display}', fontsize=10)
                ax.axis('off')
                
                # Add colorbar for the last plot
                if idx == len(sample_keys) - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'No patch data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(key.replace('_', ' '), fontsize=10)
                ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(sample_keys), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Aggregated Patch Importance Patterns', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_patch_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_experiment_overview(self, output_dir: Path):
        """Create an overview of all experiments and their status."""
        overview_data = []
        
        for exp in self.experiments:
            exp_dir = exp['directory']
            
            # Check what analysis data is available
            has_layer_analysis = (exp_dir / "layer_analysis").exists()
            has_patch_analysis = (exp_dir / "patch_analysis").exists()
            has_training_log = len(list(exp_dir.glob("training_log*.csv"))) > 0
            
            # Count analysis files
            layer_files = 0
            patch_files = 0
            
            if has_layer_analysis:
                layer_files = len(list((exp_dir / "layer_analysis").rglob("*.json")))
            
            if has_patch_analysis:
                patch_files = len(list((exp_dir / "patch_analysis").glob("*.json")))
            
            overview_data.append({
                'Directory': exp_dir.name,
                'Strategy': exp['strategy'],
                'Dataset A': exp['dataset_a'],
                'Dataset B': exp['dataset_b'],
                'Run ID': "_".join(exp['timestamp']),
                'Has Layer Analysis': has_layer_analysis,
                'Layer Files': layer_files,
                'Has Patch Analysis': has_patch_analysis,
                'Patch Files': patch_files,
                'Has Training Log': has_training_log,
                'Task 0 Acc': exp['results']['final_accuracies'].get(f"task_0_{exp['dataset_a']}", 0),
                'Task 1 Acc': exp['results']['final_accuracies'].get(f"task_1_{exp['dataset_b']}", 0),
                'Forgetting': exp['results']['final_accuracies'].get('forgetting', 0),
            })
        
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_csv(output_dir / 'experiment_overview.csv', index=False)
        
        # Create summary statistics
        runs = self.group_experiments_by_run()
        
        summary = {
            'total_experiments': len(self.experiments),
            'unique_runs': len(runs),
            'strategies_tested': list(set(exp['strategy'] for exp in self.experiments if exp['strategy'])),
            'dataset_pairs_tested': list(set(exp['pair_name'] for exp in self.experiments if exp['pair_name'])),
            'experiments_with_layer_analysis': sum(1 for _, row in overview_df.iterrows() if row['Has Layer Analysis']),
            'experiments_with_patch_analysis': sum(1 for _, row in overview_df.iterrows() if row['Has Patch Analysis']),
        }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment overview saved to {output_dir}")
        return overview_df, summary
    
    def analyze_detailed_strategy_pairs(self, output_dir: Path):
        """Create detailed analysis for each strategy-pair combination across all runs."""
        print("🔬 Creating detailed strategy-pair analysis...")
        
        # Group experiments by strategy and pair
        strategy_pairs = defaultdict(list)
        for exp in self.experiments:
            if exp['strategy'] and exp['pair_name']:
                key = f"{exp['strategy']}_{exp['pair_name']}"
                strategy_pairs[key].append(exp)
        
        detailed_dir = output_dir / "detailed_strategy_pairs"
        detailed_dir.mkdir(exist_ok=True)
        
        for strategy_pair, experiments in strategy_pairs.items():
            if len(experiments) == 0:
                continue
                
            print(f"  📊 Analyzing {strategy_pair} ({len(experiments)} runs)")
            
            strategy, pair_name = strategy_pair.split('_', 1)
            pair_dir = detailed_dir / strategy_pair
            pair_dir.mkdir(exist_ok=True)
            
            # Analyze this specific strategy-pair combination
            self._analyze_single_strategy_pair(experiments, pair_dir, strategy, pair_name)
    
    def _analyze_single_strategy_pair(self, experiments: List[Dict], output_dir: Path, strategy: str, pair_name: str):
        """Analyze a single strategy-pair combination across multiple runs."""
        
        print(f"  📊 Analyzing {strategy}_{pair_name} ({len(experiments)} runs)")
        
        # 1. LAYER ANALYSIS - Aggregate across runs with proper task separation
        all_layer_data = {
            'phase1': defaultdict(list), 
            'phase2_task0': defaultdict(list),
            'phase2_task1': defaultdict(list)
        }
        
        # 2. PATCH ANALYSIS - Aggregate across runs with proper task separation
        all_patch_data = {
            'task1_learning': defaultdict(list),       # Task 1 learning (phase1)
            'task2_learning': defaultdict(list),       # Task 2 learning (phase2)
            'task1_forgetting': defaultdict(list)      # Task 1 forgetting (phase2)
        }
        
        for i, exp in enumerate(experiments):
            print(f"    🔄 Processing run {i+1}/{len(experiments)}: {exp['directory'].name}")
            
            layer_data = self.load_layer_analysis_data(exp['directory'])
            patch_data = self.load_patch_analysis_data(exp['directory'])
            
            # Collect phase1 layer data
            phase1_collected = 0
            for epoch, epoch_data in layer_data['phase1'].items():
                if epoch_data and 'projection_scores' in epoch_data:
                    all_layer_data['phase1'][epoch].append(epoch_data['projection_scores'])
                    phase1_collected += 1
            
            # Collect phase2 layer data for both tasks
            task0_collected = 0
            task1_collected = 0
            
            for epoch, epoch_data in layer_data['phase2']['task_0'].items():
                if epoch_data and 'projection_scores' in epoch_data:
                    all_layer_data['phase2_task0'][epoch].append(epoch_data['projection_scores'])
                    task0_collected += 1
            
            for epoch, epoch_data in layer_data['phase2']['task_1'].items():
                if epoch_data and 'projection_scores' in epoch_data:
                    all_layer_data['phase2_task1'][epoch].append(epoch_data['projection_scores'])
                    task1_collected += 1
            
            print(f"      ✅ Layer data: Phase1={phase1_collected}, Task0={task0_collected}, Task1={task1_collected} epochs")
            
            # Collect patch data with proper task separation
            patch_counts = {'task1_learning': 0, 'task2_learning': 0, 'task1_forgetting': 0}
            
            for epoch, epoch_data in patch_data['task1_learning'].items():
                if 'patch_importance_maps' in epoch_data:
                    all_patch_data['task1_learning'][epoch].append(epoch_data['patch_importance_maps'])
                    patch_counts['task1_learning'] += 1
            
            for epoch, epoch_data in patch_data['task2_learning'].items():
                if 'patch_importance_maps' in epoch_data:
                    all_patch_data['task2_learning'][epoch].append(epoch_data['patch_importance_maps'])
                    patch_counts['task2_learning'] += 1
            
            for epoch, epoch_data in patch_data['task1_forgetting'].items():
                if 'patch_importance_maps' in epoch_data:
                    all_patch_data['task1_forgetting'][epoch].append(epoch_data['patch_importance_maps'])
                    patch_counts['task1_forgetting'] += 1
            
            print(f"      🎨 Patch data: T1_learn={patch_counts['task1_learning']}, T2_learn={patch_counts['task2_learning']}, T1_forget={patch_counts['task1_forgetting']} epochs")
        
        # Debug final aggregation counts
        total_phase1 = sum(len(epoch_data) for epoch_data in all_layer_data['phase1'].values())
        total_task0 = sum(len(epoch_data) for epoch_data in all_layer_data['phase2_task0'].values())
        total_task1 = sum(len(epoch_data) for epoch_data in all_layer_data['phase2_task1'].values())
        
        total_patch_t1_learn = sum(len(epoch_data) for epoch_data in all_patch_data['task1_learning'].values())
        total_patch_t2_learn = sum(len(epoch_data) for epoch_data in all_patch_data['task2_learning'].values())
        total_patch_t1_forget = sum(len(epoch_data) for epoch_data in all_patch_data['task1_forgetting'].values())
        
        print(f"    📊 Final layer data: Phase1={total_phase1}, Task0={total_task0}, Task1={total_task1}")
        print(f"    🎨 Final patch data: T1_learn={total_patch_t1_learn}, T2_learn={total_patch_t2_learn}, T1_forget={total_patch_t1_forget}")
        
        # 3. CREATE THE THREE PLOTS
        if any([all_layer_data['phase1'], all_layer_data['phase2_task0'], all_layer_data['phase2_task1']]):
            self._create_three_task_plots(all_layer_data, all_patch_data, output_dir, strategy, pair_name)
        else:
            print(f"    ⚠️  No layer data found for {strategy}_{pair_name}")
    
    def _create_three_task_plots(self, layer_data, patch_data, output_dir, strategy, pair_name):
        """Create the three plots: Task1, Task2-on-Task1, Task2-on-Task2"""
        
        print(f"    🎨 Creating plots for {strategy}_{pair_name}")
        print(f"        Phase1 epochs: {list(layer_data['phase1'].keys())}")
        print(f"        Phase2 Task0 epochs: {list(layer_data['phase2_task0'].keys())}")
        print(f"        Phase2 Task1 epochs: {list(layer_data['phase2_task1'].keys())}")
        print(f"        Patch T1_learn epochs: {sorted(patch_data['task1_learning'].keys())}")
        print(f"        Patch T2_learn epochs: {sorted(patch_data['task2_learning'].keys())}")
        print(f"        Patch T1_forget epochs: {sorted(patch_data['task1_forgetting'].keys())}")
        
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])
        
        # Plot 1: TASK 1 LEARNING (Phase 1)
        ax1_layer = fig.add_subplot(gs[0, 0:2])
        ax1_patch = fig.add_subplot(gs[0, 2:4])
        
        self._plot_task_learning(layer_data['phase1'], patch_data['task1_learning'], ax1_layer, ax1_patch, 
                               "Task 1 Learning (Phase 1)", "task1_learning")
        
        # Plot 2: TASK 1 FORGETTING (Phase 2, Task 0 performance)
        ax2_layer = fig.add_subplot(gs[1, 0:2])
        ax2_patch = fig.add_subplot(gs[1, 2:4])
        
        self._plot_task_learning(layer_data['phase2_task0'], patch_data['task1_forgetting'], ax2_layer, ax2_patch,
                               "Task 1 Forgetting (Phase 2)", "task1_forgetting")
        
        # Plot 3: TASK 2 LEARNING (Phase 2, Task 1 performance)  
        ax3_layer = fig.add_subplot(gs[2, 0:2])
        ax3_patch = fig.add_subplot(gs[2, 2:4])
        
        self._plot_task_learning(layer_data['phase2_task1'], patch_data['task2_learning'], ax3_layer, ax3_patch,
                               "Task 2 Learning (Phase 2)", "task2_learning")
        
        # Count runs for subtitle
        run_count = 0
        for data_dict in [layer_data['phase1'], layer_data['phase2_task0'], layer_data['phase2_task1']]:
            if data_dict:
                max_runs = max(len(epoch_data) for epoch_data in data_dict.values())
                run_count = max(run_count, max_runs)
        
        plt.suptitle(f'{strategy}: {pair_name.replace("_to_", " → ")} (Aggregated across {run_count} runs)', 
                     fontsize=16)
        plt.tight_layout()
        
        save_path = output_dir / f'{strategy}_{pair_name}_detailed_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Saved: {save_path.name}")
    
    def _plot_task_learning(self, layer_data_dict, patch_data_dict, ax_layer, ax_patch, title, phase_task):
        """Plot layer and patch analysis for a specific task/phase."""
        
        # LAYER ANALYSIS PLOT
        if layer_data_dict:
            epochs = sorted(layer_data_dict.keys())
            
            # Average CLS token identifiability across runs and blocks
            cls_scores_by_epoch = []
            cls_errors_by_epoch = []
            
            for epoch in epochs:
                epoch_runs = layer_data_dict[epoch]
                if not epoch_runs:
                    continue
                    
                # Collect CLS token scores across runs
                epoch_cls_scores = []
                for run_data in epoch_runs:
                    # Extract CLS token scores for this run
                    cls_scores = [value for key, value in run_data.items() if 'cls_token' in key]
                    if cls_scores:
                        epoch_cls_scores.append(np.mean(cls_scores))  # Average across blocks
                
                if epoch_cls_scores:
                    cls_scores_by_epoch.append(np.mean(epoch_cls_scores))  # Average across runs
                    cls_errors_by_epoch.append(np.std(epoch_cls_scores))   # Std across runs
                else:
                    cls_scores_by_epoch.append(0)
                    cls_errors_by_epoch.append(0)
            
            if cls_scores_by_epoch:
                ax_layer.errorbar(epochs, cls_scores_by_epoch, yerr=cls_errors_by_epoch,
                                 fmt='o-', linewidth=2, markersize=6, capsize=5)
                ax_layer.set_xlabel('Epoch')
                ax_layer.set_ylabel('CLS Token Identifiability')
                ax_layer.set_title(f'{title} - Layer Analysis')
                ax_layer.grid(True, alpha=0.3)
                ax_layer.set_ylim(0, 1)
        else:
            ax_layer.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax_layer.transAxes)
            ax_layer.set_title(f'{title} - Layer Analysis')
        
        # PATCH ANALYSIS PLOT
        if patch_data_dict:
            # Get the most representative epoch (last available epoch)
            available_epochs = sorted(patch_data_dict.keys())
            
            if available_epochs:
                patch_epoch = available_epochs[-1]  # Use latest epoch
                epoch_data = patch_data_dict[patch_epoch]
                
                print(f"      🎨 Using patch epoch {patch_epoch} for {phase_task}")
                
                if epoch_data:
                    # Average patch importance across runs and blocks
                    all_patch_maps = []
                    
                    for run_data in epoch_data:
                        if isinstance(run_data, dict):
                            # Average across blocks for this run
                            block_maps = []
                            for block_name, patch_map in run_data.items():
                                if isinstance(patch_map, np.ndarray) and patch_map.ndim == 2:
                                    block_maps.append(patch_map)
                            
                            if block_maps:
                                run_avg = np.mean(block_maps, axis=0)
                                all_patch_maps.append(run_avg)
                    
                    if all_patch_maps:
                        # Average across runs
                        final_patch_map = np.mean(all_patch_maps, axis=0)
                        
                        im = ax_patch.imshow(final_patch_map, cmap='viridis', aspect='auto')
                        ax_patch.set_title(f'{title} - Patch Importance\n(Epoch {patch_epoch})')
                        ax_patch.axis('off')
                        plt.colorbar(im, ax=ax_patch, fraction=0.046, pad=0.04)
                        print(f"      ✅ Plotted patch data for {phase_task} using epoch {patch_epoch}")
                    else:
                        ax_patch.text(0.5, 0.5, f'No valid patch maps\n({phase_task}, epoch {patch_epoch})', 
                                     ha='center', va='center', transform=ax_patch.transAxes)
                        ax_patch.set_title(f'{title} - Patch Importance')
                else:
                    ax_patch.text(0.5, 0.5, f'No patch data for epoch {patch_epoch}\n({phase_task})', 
                                 ha='center', va='center', transform=ax_patch.transAxes)
                    ax_patch.set_title(f'{title} - Patch Importance')
            else:
                ax_patch.text(0.5, 0.5, f'No patch epochs available\n({phase_task})', 
                             ha='center', va='center', transform=ax_patch.transAxes)
                ax_patch.set_title(f'{title} - Patch Importance')
        else:
            ax_patch.text(0.5, 0.5, f'No patch data\n({phase_task})', 
                         ha='center', va='center', transform=ax_patch.transAxes, fontsize=10)
            ax_patch.set_title(f'{title} - Patch Importance')

    def run_complete_analysis(self, output_dir: str = None):
        """Run complete aggregated analysis."""
        if output_dir is None:
            output_dir = self.logs_dir / "aggregated_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("Creating experiment overview...")
        overview_df, summary = self.create_experiment_overview(output_dir)
        
        print("Analyzing performance patterns...")
        stats_df = self.create_performance_summary(output_dir)
        
        print("Analyzing forgetting patterns...")
        self.analyze_forgetting_patterns(output_dir)
        
        print("Analyzing patch importance patterns...")
        self.analyze_patch_importance_aggregated(output_dir)
        
        # NEW: DETAILED STRATEGY-PAIR ANALYSIS
        print("Creating detailed strategy-pair analysis...")
        self.analyze_detailed_strategy_pairs(output_dir)
        
        # Create final report
        with open(output_dir / 'analysis_report.md', 'w') as f:
            f.write("# Binary Pairs Experiment Analysis Report\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Experiments:** {summary['total_experiments']}\n")
            f.write(f"- **Unique Runs:** {summary['unique_runs']}\n")
            f.write(f"- **Strategies Tested:** {', '.join(summary['strategies_tested'])}\n")
            f.write(f"- **Dataset Pairs:** {len(summary['dataset_pairs_tested'])}\n")
            f.write(f"- **Experiments with Layer Analysis:** {summary['experiments_with_layer_analysis']}\n")
            f.write(f"- **Experiments with Patch Analysis:** {summary['experiments_with_patch_analysis']}\n\n")
            
            if stats_df is not None and not stats_df.empty:
                f.write("## Strategy Performance Summary\n\n")
                f.write(stats_df.to_string(index=False))
                f.write("\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `performance_summary_heatmap.png` - Performance comparison across strategies\n")
            f.write("- `forgetting_analysis.png` - Detailed forgetting pattern analysis\n")
            f.write("- `aggregated_patch_importance.png` - Patch importance patterns\n")
            f.write("- `strategy_performance_stats.csv` - Detailed performance statistics\n")
            f.write("- `experiment_overview.csv` - Complete experiment inventory\n")
            f.write("- `analysis_summary.json` - Machine-readable summary\n")
            f.write("- `detailed_strategy_pairs/` - **NEW: Detailed analysis for each strategy-pair combination**\n")
        
        print(f"\n✅ Complete analysis finished!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"🔬 Detailed strategy-pair analysis in: {output_dir}/detailed_strategy_pairs/")
        print(f"📋 View the analysis_report.md for a complete summary")
        
        return output_dir








def main():
    """Main analysis function."""
    logs_dir = "/home/brothen/M2I_I2M_benchmark/binary_pairs_analysis_20250716_145601_5runs_class_incremental"
    
    print("🔍 Starting Binary Pairs Aggregated Analysis...")
    print(f"📁 Scanning directory: {logs_dir}")
    
    analyzer = BinaryPairsAggregatedAnalyzer(logs_dir)
    output_dir = analyzer.run_complete_analysis()
    
    return analyzer, output_dir


if __name__ == "__main__":
    analyzer, output_dir = main()
#!/usr/bin/env python3
"""
Aggregate and visualize patch importance across strategies, runs, and dataset pairs.
Groups by strategy and evaluated dataset, showing three scenarios:
1. Phase 1: Initial learning
2. Phase 2 Forgetting: Backward transfer 
3. Phase 2 New: New task learning
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional
import argparse
import json
try:
    import yaml
except ImportError:
    yaml = None


class PatchImportanceAggregator:
    """Aggregates and visualizes patch importance across multiple experiments."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.patch_data = defaultdict(lambda: defaultdict(list))
        self.datasets = ['mnist', 'fashion_mnist', 'cifar10', 'svhn', 'omniglot', 'tinyimagenet']
        self.strategies = ['naive', 'replay', 'cumulative']
        self.dpi = 150  # Default DPI
        self.patch_subplot_size = 4  # Default patch size
        
    def load_all_patch_data(self):
        """Load all patch importance data from the experiment directory."""
        print("Loading patch importance data...")
        
        # Track which experiment/run/dataset combinations we've already processed
        processed_experiments = set()
        
        # Look for all directories that might contain experiments
        experiment_dirs_with_info = []  # Store (dir, strategy, source) tuples
        
        # FIRST: Look in run_* directories for strategy-specific subdirs (preferred)
        for run_dir in self.base_dir.glob('run_*'):
            if run_dir.is_dir():
                # Look for strategy subdirectories
                for strategy_dir in run_dir.glob('*'):
                    if strategy_dir.is_dir() and strategy_dir.name in self.strategies:
                        strategy = strategy_dir.name
                        # Look for dataset pair directories
                        for pair_dir in strategy_dir.glob('*_to_*'):
                            if pair_dir.is_dir() and (pair_dir / 'patch_analysis').exists():
                                experiment_dirs_with_info.append((pair_dir, strategy, 'run_strategy_pair'))
        
        # SECOND: Only look at strategy_binary_pairs if we didn't find run-specific dirs
        if len(experiment_dirs_with_info) == 0:
            print("  No run_*/strategy/pair directories found, checking strategy_binary_pairs...")
            for strategy_pairs_dir in self.base_dir.rglob('strategy_binary_pairs_*'):
                if strategy_pairs_dir.is_dir() and (strategy_pairs_dir / 'patch_analysis').exists():
                    # These directories might contain multiple strategies
                    experiment_dirs_with_info.append((strategy_pairs_dir, None, 'strategy_binary_pairs'))
        
        print(f"Found {len(experiment_dirs_with_info)} potential experiment directories")
        print(f"  - strategy_binary_pairs dirs: {sum(1 for _, _, src in experiment_dirs_with_info if src == 'strategy_binary_pairs')}")
        print(f"  - run/strategy/pair dirs: {sum(1 for _, _, src in experiment_dirs_with_info if src == 'run_strategy_pair')}")
        
        # Process each directory
        for exp_dir, preset_strategy, source in experiment_dirs_with_info:
            if not exp_dir.is_dir():
                continue
                
            # Look for patch_analysis subdirectory
            patch_dir = exp_dir / 'patch_analysis'
            if not patch_dir.exists():
                continue
                
            print(f"\n  → Processing {exp_dir.name} (source: {source})...")
            
            # Extract experiment info
            exp_info = self._extract_experiment_info_comprehensive(exp_dir, preset_strategy)
            if not exp_info:
                continue
                
            strategy, dataset_a, dataset_b, run_num = exp_info
            
            # Create experiment identifier to check for duplicates
            exp_identifier = f"{dataset_a}_{dataset_b}_run{run_num}"
            if exp_identifier in processed_experiments and source == 'strategy_binary_pairs':
                print(f"    ⚠ Skipping duplicate experiment: {exp_identifier}")
                continue
            
            processed_experiments.add(exp_identifier)
            
            # Debug: show what we found
            print(f"    → Strategy: {strategy}, Datasets: {dataset_a}→{dataset_b}, Run: {run_num}")
            
            # Load patch importance files
            npz_files = list(patch_dir.glob('patch_importance_*.npz'))
            print(f"    Found {len(npz_files)} patch files")
            
            for npz_file in npz_files:
                scenario = self._extract_scenario(npz_file.name, dataset_a, dataset_b)
                if not scenario:
                    print(f"    ⚠ Could not extract scenario from {npz_file.name} (a={dataset_a}, b={dataset_b})")
                    continue
                    
                eval_dataset, phase_type = scenario
                
                # Debug: show what we extracted
                eval_dataset, phase_type = scenario
                print(f"      → {npz_file.name}: eval={eval_dataset}, phase={phase_type}")
                
                # Load the data
                try:
                    data = np.load(npz_file)
                    
                    # Create key for grouping
                    key = (strategy, eval_dataset, phase_type)
                    
                    # Extract patch importance maps for each layer
                    layer_data = {}
                    for key_name in data.files:
                        if key_name.startswith('block_') and key_name != 'block_names':
                            # Parse the block number safely
                            try:
                                parts = key_name.split('_')
                                if len(parts) >= 2 and parts[1].isdigit():
                                    layer_idx = int(parts[1])
                                    layer_data[layer_idx] = data[key_name]
                            except (ValueError, IndexError):
                                # Skip any keys that don't match expected format
                                continue
                    
                    if layer_data:
                        # Create a unique identifier for this experiment to avoid duplicates
                        exp_id = f"{strategy}_{dataset_a}_{dataset_b}_run{run_num}_{phase_type}_{npz_file.stem}"
                        
                        # Check if we've already seen this exact experiment
                        existing_files = [exp['file'].name for exp in self.patch_data[key]['experiments']]
                        if npz_file.name not in existing_files:
                            self.patch_data[key]['experiments'].append({
                                'layer_data': layer_data,
                                'dataset_pair': (dataset_a, dataset_b),
                                'run': run_num,
                                'file': npz_file,
                                'exp_id': exp_id
                            })
                            print(f"    ✓ Loaded {len(layer_data)} layers from {npz_file.name}")
                        else:
                            print(f"    ⚠ Skipping duplicate: {npz_file.name}")
                    else:
                        print(f"    ⚠ No layer data found in {npz_file.name}")
                        
                except Exception as e:
                    print(f"    ✗ Error loading {npz_file.name}: {e}")
                    
        # Detailed summary with experiment breakdown
        print(f"\nDetailed loaded data summary:")
        total_experiments = 0
        for key, data in sorted(self.patch_data.items()):
            strategy, eval_dataset, phase_type = key
            n_exps = len(data['experiments'])
            if n_exps > 0:
                print(f"\n  {strategy} - {eval_dataset} - {phase_type}: {n_exps} experiments")
                # Show dataset pairs for this combination
                pairs_count = defaultdict(int)
                runs_count = defaultdict(int)
                for exp in data['experiments']:
                    pair = f"{exp['dataset_pair'][0]}→{exp['dataset_pair'][1]}"
                    pairs_count[pair] += 1
                    runs_count[exp['run']] += 1
                print(f"    Dataset pairs:")
                for pair, count in sorted(pairs_count.items()):
                    print(f"      {pair}: {count} files")
                print(f"    Runs distribution: {dict(runs_count)}")
                total_experiments += n_exps
                
        if total_experiments == 0:
            print("\n⚠ No patch importance data was successfully loaded!")
            print("Please check:")
            print("  1. The directory structure contains patch_analysis/ subdirectories")
            print("  2. The .npz files contain 'block_N' keys where N is a number")
            print("  3. The config files contain dataset_a and dataset_b information")
        else:
            # Sanity check the counts
            print(f"\n📊 Total experiments loaded: {total_experiments}")
            print("\n🔍 Sanity check:")
            
            # Check for suspicious counts
            for key, data in sorted(self.patch_data.items()):
                strategy, eval_dataset, phase_type = key
                n_exps = len(data['experiments'])
                
                # Expected: For each dataset as task 1, we should have 3 pairs × 5 runs = 15
                # For each dataset as task 2, we should also have 3 pairs × 5 runs = 15
                if n_exps > 0 and n_exps % 5 != 0:
                    print(f"  ⚠ {strategy} - {eval_dataset} - {phase_type}: {n_exps} is not divisible by 5 (runs)")
                
                if n_exps > 20:
                    print(f"  ⚠ {strategy} - {eval_dataset} - {phase_type}: {n_exps} seems too high (expected ~15)")
                    
            # Check phase consistency
            phase1_counts = defaultdict(int)
            phase2_forg_counts = defaultdict(int)
            phase2_new_counts = defaultdict(int)
            
            for key, data in self.patch_data.items():
                strategy, eval_dataset, phase_type = key
                n_exps = len(data['experiments'])
                
                if phase_type == 'phase1_initial':
                    phase1_counts[(strategy, eval_dataset)] = n_exps
                elif phase_type == 'phase2_forgetting':
                    phase2_forg_counts[(strategy, eval_dataset)] = n_exps
                elif phase_type == 'phase2_new':
                    phase2_new_counts[(strategy, eval_dataset)] = n_exps
            
            # Phase 1 and Phase 2 forgetting should have same counts
            print("\n📌 Phase consistency check:")
            for key in phase1_counts:
                if key in phase2_forg_counts:
                    if phase1_counts[key] != phase2_forg_counts[key]:
                        strategy, dataset = key
                        print(f"  ⚠ {strategy} - {dataset}: Phase1={phase1_counts[key]}, Phase2_forgetting={phase2_forg_counts[key]} (should be equal)")
            
    def _extract_experiment_info_comprehensive(self, exp_dir: Path, preset_strategy: Optional[str] = None) -> Optional[Tuple[str, str, str, int]]:
        """Extract strategy, datasets, and run number from directory structure (comprehensive method)."""
        strategy = preset_strategy  # Use preset if available
        dataset_a, dataset_b = None, None
        run_num = 1
        
        # Extract run number from parent directories
        for parent in exp_dir.parents:
            if 'run_' in parent.name:
                try:
                    run_num = int(parent.name.split('_')[1])
                    break
                except:
                    pass
        
        # Method 1: Check if parent directory is a strategy name
        if exp_dir.parent.name in self.strategies:
            strategy = exp_dir.parent.name
            # Try to extract datasets from directory name
            dir_name = exp_dir.name
            for ds_a in self.datasets:
                for ds_b in self.datasets:
                    if ds_a != ds_b and f"{ds_a}_to_{ds_b}" == dir_name:
                        dataset_a, dataset_b = ds_a, ds_b
                        break
                if dataset_a:
                    break
        
        # Method 2: Look for config files
        if not dataset_a or not dataset_b:
            config_files = list(exp_dir.glob('config_*.yaml')) + list(exp_dir.glob('*.yml'))
            
            if config_files and yaml is not None:
                try:
                    with open(config_files[0], 'r') as f:
                        config = yaml.safe_load(f)
                    
                    dataset_a = config.get('dataset_a', '').lower()
                    dataset_b = config.get('dataset_b', '').lower()
                    
                    # Try to get strategy from config
                    if not strategy:
                        strategy_name = config.get('strategy_name', '').lower()
                        if strategy_name in self.strategies:
                            strategy = strategy_name
                        
                except Exception as e:
                    print(f"Error reading config: {e}")
        
        # Method 3: Look for results.json
        if not dataset_a or not dataset_b:
            results_file = exp_dir / 'results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    dataset_a = results.get('dataset_a', '').lower()
                    dataset_b = results.get('dataset_b', '').lower()
                    
                except Exception:
                    pass
        
        # Method 4: Try to infer strategy from training logs (but be careful not to override preset)
        if not strategy:
            # Only look for strategy in logs if we don't have a preset strategy
            # This prevents strategy_binary_pairs dirs from being assigned to all strategies
            for log_file in exp_dir.glob('training_log_*.csv'):
                log_name = log_file.name.lower()
                # Check if this log file is strategy-specific
                for strat in self.strategies:
                    if f'training_log_{strat}' in log_name:
                        strategy = strat
                        break
                if strategy:
                    break
            
            # If we still don't have a strategy and this is a strategy_binary_pairs dir,
            # we might need to skip it or handle it specially
            if not strategy and 'strategy_binary_pairs' in str(exp_dir):
                print(f"  ⚠ Found strategy_binary_pairs dir without clear strategy assignment")
                # Look for ANY training log to determine which strategies are present
                all_strategies_found = set()
                for log_file in exp_dir.glob('training_log_*.csv'):
                    for strat in self.strategies:
                        if strat in log_file.name.lower():
                            all_strategies_found.add(strat)
                
                if len(all_strategies_found) > 1:
                    print(f"  ⚠ Multiple strategies found in same directory: {all_strategies_found}")
                    print(f"  ⚠ Skipping this directory to avoid duplication")
                    return None
                elif len(all_strategies_found) == 1:
                    strategy = all_strategies_found.pop()
                    print(f"  → Inferred single strategy: {strategy}")
        
        # Method 5: Try to find strategy from parent paths
        if not strategy:
            for parent in exp_dir.parents:
                if parent.name in self.strategies:
                    strategy = parent.name
                    break
        
        # Method 6: Last resort - try to parse from grandparent directory name
        if not dataset_a or not dataset_b:
            # Look for patterns like "mnist_to_fashion_mnist" in any parent
            for parent in exp_dir.parents:
                parent_name = parent.name.lower()
                for ds_a in self.datasets:
                    for ds_b in self.datasets:
                        if ds_a != ds_b:
                            pattern = f"{ds_a}_to_{ds_b}"
                            if pattern in parent_name or pattern.replace('_', '-') in parent_name:
                                dataset_a, dataset_b = ds_a, ds_b
                                print(f"  → Found datasets from parent dir: {parent_name}")
                                break
                    if dataset_a:
                        break
                if dataset_a:
                    break
        
        # Validation
        if not strategy or not dataset_a or not dataset_b:
            print(f"  ⚠ Could not extract complete info for {exp_dir}: strategy={strategy}, a={dataset_a}, b={dataset_b}")
            # Print directory structure to help debug
            print(f"    Directory: {exp_dir}")
            print(f"    Parent: {exp_dir.parent}")
            print(f"    Grandparent: {exp_dir.parent.parent if exp_dir.parent else 'N/A'}")
            return None
            
        return strategy, dataset_a, dataset_b, run_num
        
    def _extract_scenario(self, filename: str, dataset_a: str, dataset_b: str) -> Optional[Tuple[str, str]]:
        """Determine evaluation dataset and phase type from filename."""
        # Parse filename patterns:
        # - patch_importance_phase1_mnist_epoch_050.npz (task 1 eval during phase 1)
        # - patch_importance_phase2_forgetting_mnist_epoch_050.npz (task 1 eval during phase 2)
        # - patch_importance_phase2_fashion_mnist_epoch_050.npz (task 2 eval during phase 2)
        
        filename_lower = filename.lower()
        
        # Debug
        # print(f"        Parsing: {filename} (a={dataset_a}, b={dataset_b})")
        
        if 'phase1' in filename_lower:
            # Phase 1: Should ONLY evaluate on task 1 (dataset_a)
            if dataset_a in filename_lower:
                # Verify dataset_b is NOT in the filename (to avoid confusion)
                if dataset_b not in filename_lower or filename_lower.count(dataset_a) >= filename_lower.count(dataset_b):
                    return dataset_a, 'phase1_initial'
        
        elif 'phase2' in filename_lower:
            if 'forgetting' in filename_lower:
                # Phase 2 forgetting: Should evaluate task 1 (dataset_a) after training task 2
                if dataset_a in filename_lower:
                    return dataset_a, 'phase2_forgetting'
            else:
                # Phase 2 new task: Should evaluate task 2 (dataset_b)
                # Be careful to ensure it's really dataset_b and not mislabeled
                if dataset_b in filename_lower:
                    # Extra check: dataset_a should NOT be more prominent
                    if dataset_a not in filename_lower or filename_lower.count(dataset_b) > filename_lower.count(dataset_a):
                        return dataset_b, 'phase2_new'
                    
        # Debug failed extractions
        # print(f"        ⚠ Failed to extract scenario")
        return None
        
    def aggregate_patch_importance(self, experiments: List[Dict]) -> Dict:
        """Aggregate patch importance across multiple experiments."""
        # Collect all layer data
        all_layer_data = defaultdict(list)
        
        for exp in experiments:
            for layer_idx, patch_map in exp['layer_data'].items():
                all_layer_data[layer_idx].append(patch_map)
                
        # Compute mean and std for each layer
        aggregated = {}
        for layer_idx, maps in all_layer_data.items():
            stacked = np.stack(maps, axis=0)
            aggregated[layer_idx] = {
                'mean': np.mean(stacked, axis=0),
                'std': np.std(stacked, axis=0),
                'n_samples': len(maps)
            }
            
        return aggregated
        
    def visualize_strategy_dataset_scenario(self, strategy: str, eval_dataset: str):
        """Create comprehensive visualization for a strategy-dataset combination."""
        print(f"\nVisualizing {strategy} - {eval_dataset}...")
        
        # Collect data for all three scenarios
        scenarios = {
            'phase1_initial': 'Phase 1: Initial Learning',
            'phase2_forgetting': 'Phase 2: Forgetting (Backward Transfer)',
            'phase2_new': 'Phase 2: New Task Learning'
        }
        
        scenario_data = {}
        for phase_type, phase_name in scenarios.items():
            key = (strategy, eval_dataset, phase_type)
            if key in self.patch_data and self.patch_data[key]['experiments']:
                experiments = self.patch_data[key]['experiments']
                scenario_data[phase_type] = {
                    'aggregated': self.aggregate_patch_importance(experiments),
                    'n_experiments': len(experiments),
                    'name': phase_name
                }
                
        if not scenario_data:
            print(f"  No data found for {strategy} - {eval_dataset}")
            return
            
        # Create visualizations
        self._create_layer_wise_visualization(strategy, eval_dataset, scenario_data)
        self._create_collapsed_visualization(strategy, eval_dataset, scenario_data)
        
    def _create_layer_wise_visualization(self, strategy: str, eval_dataset: str, scenario_data: Dict):
        """Create layer-wise patch importance heatmaps - one plot per phase."""
        # Find all unique layers and sort them properly
        all_layers = set()
        for phase_data in scenario_data.values():
            all_layers.update(phase_data['aggregated'].keys())
        sorted_layers = sorted(all_layers)
        n_layers = len(sorted_layers)
        
        # Find global min/max for consistent colormap across all plots
        global_vmin, global_vmax = float('inf'), float('-inf')
        for phase_data in scenario_data.values():
            for layer_data in phase_data['aggregated'].values():
                global_vmin = min(global_vmin, layer_data['mean'].min())
                global_vmax = max(global_vmax, layer_data['mean'].max())
        
        # Create output directory
        output_dir = Path('patch_importance_analysis')
        output_dir.mkdir(exist_ok=True)
        
        # Create a separate plot for each phase
        for phase_idx, (phase_type, phase_data) in enumerate(scenario_data.items()):
            # Determine grid layout based on number of layers
            if n_layers <= 4:
                n_cols = n_layers
                n_rows = 1
            elif n_layers <= 8:
                n_cols = 4
                n_rows = 2
            elif n_layers <= 12:
                n_cols = 4
                n_rows = 3
            else:
                n_cols = 5
                n_rows = (n_layers + n_cols - 1) // n_cols
            
            # Create figure for this phase
            fig_width = self.patch_subplot_size * n_cols
            fig_height = self.patch_subplot_size * n_rows
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            
            # Handle single row/column cases
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]
            
            aggregated = phase_data['aggregated']
            
            # Plot each layer
            plot_idx = 0
            for layer_idx in sorted_layers:
                row_idx = plot_idx // n_cols
                col_idx = plot_idx % n_cols
                ax = axes[row_idx][col_idx]
                
                if layer_idx in aggregated:
                    mean_map = aggregated[layer_idx]['mean']
                    im = ax.imshow(mean_map, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
                    
                    # Add grid
                    patch_size = mean_map.shape[0]
                    ax.set_xticks(np.arange(-0.5, patch_size, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, patch_size, 1), minor=True)
                    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
                    
                    # Add layer title
                    ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
                
                plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, n_rows * n_cols):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                axes[row_idx][col_idx].set_visible(False)
            
            # Add colorbar to the right side
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Patch Importance', fontsize=12)
            
            # Title and labels
            dataset_display = eval_dataset.replace('_', ' ').title()
            phase_name = phase_data['name']
            fig.suptitle(f'{strategy.title()} Strategy - {dataset_display}\n{phase_name} (n={phase_data["n_experiments"]} experiments)',
                         fontsize=16, fontweight='bold')
            
            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])
            
            # Save with phase-specific filename
            phase_suffix = {
                'phase1_initial': 'phase1_initial',
                'phase2_forgetting': 'phase2_forgetting',
                'phase2_new': 'phase2_new'
            }.get(phase_type, phase_type)
            
            save_path = output_dir / f'{strategy}_{eval_dataset}_layerwise_{phase_suffix}.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"  Saved layer-wise visualization to {save_path}")
        
    def _create_collapsed_visualization(self, strategy: str, eval_dataset: str, scenario_data: Dict):
        """Create collapsed (averaged across layers) patch importance visualization."""
        n_scenarios = len(scenario_data)
        
        fig, axes = plt.subplots(n_scenarios, 2, figsize=(12, 5 * n_scenarios))
        if n_scenarios == 1:
            axes = axes.reshape(1, -1)
            
        for row_idx, (phase_type, phase_data) in enumerate(scenario_data.items()):
            aggregated = phase_data['aggregated']
            
            # Collapse across layers
            all_means = []
            all_stds = []
            
            for layer_idx in sorted(aggregated.keys()):
                all_means.append(aggregated[layer_idx]['mean'])
                all_stds.append(aggregated[layer_idx]['std'])
                
            if all_means:
                # Stack and average across layers
                collapsed_mean = np.mean(np.stack(all_means, axis=0), axis=0)
                collapsed_std = np.mean(np.stack(all_stds, axis=0), axis=0)
                
                # Plot mean
                ax_mean = axes[row_idx, 0]
                im1 = ax_mean.imshow(collapsed_mean, cmap='viridis')
                ax_mean.set_title(f"{phase_data['name']}\nMean Importance (n={phase_data['n_experiments']})")
                plt.colorbar(im1, ax=ax_mean, fraction=0.046)
                
                # Add grid
                patch_size = collapsed_mean.shape[0]
                ax_mean.set_xticks(np.arange(-0.5, patch_size, 1), minor=True)
                ax_mean.set_yticks(np.arange(-0.5, patch_size, 1), minor=True)
                ax_mean.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
                
                # Plot std
                ax_std = axes[row_idx, 1]
                im2 = ax_std.imshow(collapsed_std, cmap='plasma')
                ax_std.set_title(f"{phase_data['name']}\nStd Dev Across Experiments")
                plt.colorbar(im2, ax=ax_std, fraction=0.046)
                
                # Add grid
                ax_std.set_xticks(np.arange(-0.5, patch_size, 1), minor=True)
                ax_std.set_yticks(np.arange(-0.5, patch_size, 1), minor=True)
                ax_std.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
                
                # Labels
                for ax in [ax_mean, ax_std]:
                    ax.set_xlabel('Patch Column')
                    ax.set_ylabel('Patch Row')
                    
        # Title
        dataset_display = eval_dataset.replace('_', ' ').title()
        fig.suptitle(f'{strategy.title()} Strategy - {dataset_display} Evaluation\nCollapsed Patch Importance (Averaged Across Layers)',
                     fontsize=16)
        
        plt.tight_layout()
        
        # Save
        output_dir = Path('patch_importance_analysis')
        save_path = output_dir / f'{strategy}_{eval_dataset}_collapsed.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved collapsed visualization to {save_path}")
        
    def create_all_visualizations(self):
        """Create visualizations for all strategy-dataset combinations."""
        print(f"\nCreating all visualizations... (DPI: {self.dpi}, Patch size: {self.patch_subplot_size}\")")
        
        # Get all unique strategy-dataset combinations that have data
        combinations = set()
        for key in self.patch_data.keys():
            strategy, eval_dataset, _ = key
            combinations.add((strategy, eval_dataset))
            
        # Sort for consistent ordering
        sorted_combinations = sorted(combinations)
        
        print(f"Found {len(sorted_combinations)} strategy-dataset combinations")
        
        for strategy, eval_dataset in sorted_combinations:
            self.visualize_strategy_dataset_scenario(strategy, eval_dataset)
            
        print("\nVisualization complete!")
        
    def create_summary_report(self):
        """Create a summary report of all available data."""
        output_dir = Path('patch_importance_analysis')
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / 'patch_importance_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("Patch Importance Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by strategy and dataset
            strategy_dataset_data = defaultdict(lambda: defaultdict(list))
            for key, data in self.patch_data.items():
                strategy, eval_dataset, phase_type = key
                n_exps = len(data['experiments'])
                strategy_dataset_data[strategy][eval_dataset].append((phase_type, n_exps))
                
            # Write summary
            for strategy in sorted(strategy_dataset_data.keys()):
                f.write(f"\n{strategy.upper()} Strategy\n")
                f.write("-" * 30 + "\n")
                
                for dataset in sorted(strategy_dataset_data[strategy].keys()):
                    f.write(f"\n  {dataset}:\n")
                    for phase_type, n_exps in sorted(strategy_dataset_data[strategy][dataset]):
                        phase_name = {
                            'phase1_initial': 'Phase 1 Initial',
                            'phase2_forgetting': 'Phase 2 Forgetting',
                            'phase2_new': 'Phase 2 New Task'
                        }.get(phase_type, phase_type)
                        f.write(f"    - {phase_name}: {n_exps} experiments\n")
        
        # Add note about output files
        f.write("\n\nGenerated Files:\n")
        f.write("-" * 30 + "\n")
        f.write("Layer-wise visualizations (one per phase):\n")
        f.write("  - {strategy}_{dataset}_layerwise_phase1_initial.png\n")
        f.write("  - {strategy}_{dataset}_layerwise_phase2_forgetting.png\n")
        f.write("  - {strategy}_{dataset}_layerwise_phase2_new.png\n")
        f.write("\nCollapsed visualizations:\n")
        f.write("  - {strategy}_{dataset}_collapsed.png\n")
                        
        print(f"\nSummary report saved to {report_path}")


def inspect_npz_file(npz_path: str):
    """Utility function to inspect the structure of a patch importance npz file."""
    print(f"\nInspecting {npz_path}...")
    try:
        data = np.load(npz_path)
        print(f"Keys in file: {list(data.files)}")
        
        for key in data.files:
            arr = data[key]
            print(f"  {key}: shape={arr.shape if hasattr(arr, 'shape') else 'scalar'}, dtype={arr.dtype if hasattr(arr, 'dtype') else type(arr)}")
            
        # Check for block data
        block_keys = [k for k in data.files if k.startswith('block_') and k != 'block_names']
        print(f"\nFound {len(block_keys)} block keys: {block_keys}")
        
    except Exception as e:
        print(f"Error loading file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate and visualize patch importance data')
    parser.add_argument('base_dir', type=str, help='Base directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='patch_importance_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--inspect', type=str, help='Path to a single npz file to inspect')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figures (default: 150)')
    parser.add_argument('--patch-size', type=int, default=4, 
                       help='Size of each patch subplot in inches (default: 4)')
    
    args = parser.parse_args()
    
    # If inspect mode, just inspect the file and exit
    if args.inspect:
        inspect_npz_file(args.inspect)
        return
    
    # Create aggregator
    aggregator = PatchImportanceAggregator(args.base_dir)
    
    # Store settings
    aggregator.dpi = args.dpi
    aggregator.patch_subplot_size = args.patch_size
    
    # Load all data
    aggregator.load_all_patch_data()
    
    # Create visualizations
    aggregator.create_all_visualizations()
    
    # Create summary report
    aggregator.create_summary_report()


if __name__ == "__main__":
    main()
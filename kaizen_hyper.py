#!/usr/bin/env python3
"""
Hyperparameter tuning for Kaizen strategy - Clean Version
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import itertools
import time
from datetime import datetime

# Add project root to path
sys.path.append('/home/brothen/M2I_I2M_benchmark')

# Import exactly the same components as the main experiment
from core.strategies import KaizenStrategy, NaiveStrategy

class KaizenHyperparamTuner:
    """Simple hyperparameter tuning for Kaizen strategy."""
    
    def __init__(self):
        # Base config - same as your debug setup
        self.base_config = {
            'experiment_type': 'StrategyBinaryPairsExperiment',
            'model_name': 'ViT64',
            'num_classes': 10,
            'cuda': 0,
            'dataset_a': 'tiny_imagenet',
            'dataset_b': 'tiny_imagenet',
            'scenario_type': 'class_incremental',
            'balanced': 'balanced',
            'number_of_samples_per_class': 500,
            'strategy_name': 'Kaizen',
            'output_dir': './hyperparam_results'
        }
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(self.base_config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Store results
        self.results = []
        self.baseline_acc = None
    
    def build_model(self):
        """Build model exactly like BaseExperiment."""
        from models.vit_models import ViT64SingleHead
        
        model = ViT64SingleHead(num_classes=self.base_config['num_classes'])
        model.to(self.device)
        return model
    
    def _load_dataset(self, dataset_name):
        """Load a dataset by name."""
        if dataset_name == 'mnist':
            from scenarios.datasets.mnist import load_mnist_with_resize
            return load_mnist_with_resize(
                balanced=True,
                number_of_samples_per_class=self.base_config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'cifar10':
            from scenarios.datasets.cifar import load_resized_cifar10
            return load_resized_cifar10(
                balanced=self.base_config.get('balanced', False),
                number_of_samples_per_class=self.base_config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'fashion_mnist':
            from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
            return load_fashion_mnist_with_resize(
                balanced=True,
                number_of_samples_per_class=self.base_config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'svhn':
            from scenarios.datasets.svhn import load_svhn_resized
            return load_svhn_resized(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.base_config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'tiny_imagenet':
            from scenarios.datasets.load_imagenet import load_imagenet
            from scenarios.utils import filter_classes
            # Load full ImageNet/TinyImageNet and filter to first 10 classes
            train_full, test_full = load_imagenet(
                balanced=self.base_config.get('balanced', True),
                number_of_samples_per_class=self.base_config.get('number_of_samples_per_class')
            )
            # Filter to first 10 classes like in the scenarios
            return filter_classes(train_full, test_full, classes=list(range(10)))
    
    def test_baseline_naive(self):
        """Test simple Naive strategy as baseline."""
        print(f"\n🎯 Testing Naive baseline...")
        
        # Create fresh model
        model = self.build_model()
        
        # Simple config for naive strategy
        naive_config = self.base_config.copy()
        naive_config.update({
            'strategy_name': 'Naive',
            'lr': 0.0003,
            'optimizer': 'adam',
            'epochs': 50,
            'minibatch_size': 128
        })
        
        # Create strategy
        strategy = NaiveStrategy(model, naive_config, str(self.device))
        
        # Load data
        train_ds, test_ds = self._load_dataset('tiny_imagenet')
        
        # Train
        start_time = time.time()
        try:
            result = strategy.train_task(train_ds, task_id=0)
            train_time = time.time() - start_time
            
            # Evaluate
            test_acc = strategy._evaluate_dataset(test_ds)
            
            print(f"    🎯 Naive Baseline: {test_acc:.1f}% | Time: {train_time:.0f}s")
            self.baseline_acc = test_acc
            return test_acc
            
        except Exception as e:
            print(f"    ❌ Baseline failed: {str(e)}")
            self.baseline_acc = 0.0
            return 0.0
    
    def test_single_config(self, hyperparam_config):
        """Test a single hyperparameter configuration."""
        print(f"\n🧪 Testing config: {hyperparam_config}")
        
        # Create fresh model for each test
        model = self.build_model()
        
        # Merge base config with hyperparameters
        full_config = {**self.base_config, **hyperparam_config}
        
        # Create strategy
        strategy = KaizenStrategy(model, full_config, str(self.device))
        
        # Load data
        train_ds, test_ds = self._load_dataset('tiny_imagenet')
        
        # Train
        start_time = time.time()
        try:
            result = strategy.train_task(train_ds, task_id=0)
            train_time = time.time() - start_time
            
            # Evaluate
            test_acc = strategy._evaluate_dataset(test_ds)
            
            # Store result
            result_entry = {
                'config': hyperparam_config.copy(),
                'accuracy': test_acc,
                'train_time': train_time,
                'final_loss': result.get('loss_components', {}).get('total', 0),
                'ssl_loss': result.get('loss_components', {}).get('ssl_ct', 0),
                'ce_loss': result.get('loss_components', {}).get('ce_ct', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"    ✅ Accuracy: {test_acc:.1f}% | Time: {train_time:.0f}s")
            return result_entry
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return {
                'config': hyperparam_config.copy(),
                'accuracy': 0.0,
                'train_time': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_hyperparam_search(self, param_grid):
        """Run hyperparameter search over the given grid."""
        print(f"🚀 Starting hyperparameter search with {len(param_grid)} configurations...")
        
        # First, run baseline to see what we're trying to beat
        baseline_acc = self.test_baseline_naive()
        
        print(f"\n🔍 Target to beat: {baseline_acc:.1f}% (Naive strategy)")
        
        for i, config in enumerate(param_grid):
            print(f"\n[{i+1}/{len(param_grid)}]", end=" ")
            result = self.test_single_config(config)
            self.results.append(result)
            
            # Show progress vs baseline
            if 'accuracy' in result:
                acc = result['accuracy']
                vs_baseline = acc - baseline_acc
                status = "🟢" if vs_baseline > 0 else "🔴" if vs_baseline < -5 else "🟡"
                print(f"    {status} vs baseline: {vs_baseline:+.1f}%")
            
            # Save intermediate results
            self.save_results()
        
        print(f"\n🎯 Search complete! Results saved to {self.output_dir}")
        self.print_summary()
    
    def save_results(self):
        """Save results to file."""
        import json
        
        results_file = self.output_dir / 'hyperparam_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print summary of results."""
        print(f"\n📊 HYPERPARAMETER SEARCH RESULTS")
        print("=" * 60)
        
        # Filter successful runs
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            print("❌ No successful runs!")
            return
        
        # Sort by accuracy
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(self.results) - len(successful_results)}")
        
        if self.baseline_acc:
            print(f"Baseline (Naive): {self.baseline_acc:.1f}%")
            beats_baseline = [r for r in successful_results if r['accuracy'] > self.baseline_acc]
            print(f"Configs beating baseline: {len(beats_baseline)}/{len(successful_results)}")
        
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(successful_results[:5]):
            config = result['config']
            acc = result['accuracy']
            time_taken = result['train_time']
            
            # Show vs baseline
            vs_baseline = ""
            if self.baseline_acc:
                diff = acc - self.baseline_acc
                vs_baseline = f" ({diff:+.1f}% vs baseline)"
            
            print(f"\n{i+1}. Accuracy: {acc:.1f}%{vs_baseline} | Time: {time_taken:.0f}s")
            for key, value in config.items():
                print(f"   {key}: {value}")
        
        # Best configuration
        best = successful_results[0]
        print(f"\n🥇 BEST CONFIGURATION:")
        print(f"   Accuracy: {best['accuracy']:.1f}%")
        if self.baseline_acc:
            diff = best['accuracy'] - self.baseline_acc
            print(f"   vs Baseline: {diff:+.1f}%")
        print(f"   Config: {best['config']}")


def create_param_grid():
    """Create hyperparameter grid for testing."""
    
    # Define parameter ranges to test
    param_ranges = {
        'lr': [0.0001, 0.0003, 0.001],
        'optimizer': ['adam', 'lars'],
        'ssl_method': ['simclr', 'moco'],
        'epochs': [50, 100],
        'kd_classifier_weight': [1.0, 2.0, 4.0],
        'minibatch_size': [64, 128]
    }
    
    # For SimCLR, we can test different temperatures
    simclr_configs = []
    moco_configs = []
    
    # Generate all combinations
    keys = ['lr', 'optimizer', 'epochs', 'kd_classifier_weight', 'minibatch_size']
    base_combinations = list(itertools.product(*[param_ranges[k] for k in keys]))
    
    for combo in base_combinations:
        base_config = dict(zip(keys, combo))
        
        # Add SimCLR configs with temperature variations
        for temp in [0.05, 0.1, 0.2]:
            config = base_config.copy()
            config['ssl_method'] = 'simclr'
            config['ssl_temperature'] = temp
            simclr_configs.append(config)
        
        # Add MOCO config (no temperature)
        config = base_config.copy()
        config['ssl_method'] = 'moco'
        moco_configs.append(config)
    
    all_configs = simclr_configs + moco_configs
    
    print(f"Generated {len(all_configs)} configurations to test")
    print(f"  - SimCLR configs: {len(simclr_configs)}")
    print(f"  - MOCO configs: {len(moco_configs)}")
    
    return all_configs


def create_quick_param_grid():
    """Create a smaller grid for quick testing."""
    
    return [
        # Test different learning rates with current best setup
        {'lr': 0.0001, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.001, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        
        # Test different optimizers
        {'lr': 0.0003, 'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.0003, 'optimizer': 'sgd', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        
        # Test MOCO
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'moco', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128},
        
        # Test different temperatures for SimCLR
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.05},
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.2},
        
        # Test more epochs
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 100, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        
        # Test different KD weights
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 1.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.0003, 'optimizer': 'lars', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 4.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
    ]


def create_mini_param_grid():
    """Create an even smaller grid for very quick testing."""
    
    return [
        # Just test the most promising configs based on your 25% result
        {'lr': 0.0003, 'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.0001, 'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'lr': 0.00003, 'optimizer': 'adam', 'ssl_method': 'moco', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128},
        {'lr': 0.000003, 'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 50, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
    ]


def create_lower_adam_lr_grid():
    """Create grid specifically for testing lower Adam learning rates."""
    
    # Based on the trend showing Adam improves with lower LRs
    lower_lrs = [0.00005, 0.00003, 0.00001, 0.000005]  # 5e-5, 3e-5, 1e-5, 5e-6
    
    configs = []
    
    # Test with your best performing Adam setups
    base_configs = [
        {'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 100, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.1},
        {'optimizer': 'adam', 'ssl_method': 'simclr', 'epochs': 100, 'kd_classifier_weight': 2.0, 'minibatch_size': 128, 'ssl_temperature': 0.05},
        {'optimizer': 'adam', 'ssl_method': 'moco', 'epochs': 100, 'kd_classifier_weight': 2.0, 'minibatch_size': 128},
    ]
    
    for base_config in base_configs:
        for lr in lower_lrs:
            config = base_config.copy()
            config['lr'] = lr
            configs.append(config)
    
    print(f"Generated {len(configs)} lower Adam LR configurations to test")
    return configs

if __name__ == "__main__":
    tuner = KaizenHyperparamTuner()
    
    # Replace the existing choice options with:
    print("Select grid size:")
    print("1. Mini grid (4 configs, ~5 minutes)")
    print("2. Quick grid (12 configs, ~15 minutes)")  
    print("3. Lower Adam LRs (12 configs, ~15 minutes)")  # NEW
    print("4. Full grid (100+ configs, ~2+ hours)")      # Changed from 3 to 4
    
    choice = input("Choice (1, 2, 3, or 4): ").strip()
    
    if choice == "1":
        param_grid = create_mini_param_grid()
    elif choice == "2":
        param_grid = create_quick_param_grid()
    elif choice == "3":
        param_grid = create_lower_adam_lr_grid()  # NEW
    else:
        param_grid = create_param_grid()
    
    tuner.run_hyperparam_search(param_grid)
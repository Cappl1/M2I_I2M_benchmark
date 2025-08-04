#!/usr/bin/env python3
"""
Simple debug script to test Kaizen vs Naive strategies
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/brothen/M2I_I2M_benchmark')

from core.strategies import NaiveStrategy, KaizenStrategy

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION TO EXPERIMENT
# ============================================================================

CONFIG = {
    # Basic setup
    'output_dir': './debug_kaizen_simple',
    'cuda': 1,
    'epochs': 50,                    # Number of epochs per task
    'minibatch_size': 128,
    
    # Datasets (first task -> second task)
    'dataset_a': 'cifar10',            # First dataset: mnist, cifar10, fashion_mnist
    'dataset_b': 'mnist',          # Second dataset: mnist, cifar10, fashion_mnist
    'num_samples_per_class': 500,    # Limit samples for faster debugging
    
    # Model
    'model_name': 'ViT64',
    'num_classes': 20,               # 10 from each dataset
    
    # Kaizen hyperparameters - PLAY WITH THESE!
    'kaizen_config': {
        'lr': 0.00001,                 # Learning rate
        'optimizer': 'adam',         # adam, lars, sgd
        'ssl_method': 'simclr',        # byol, simclr
        'ssl_temperature': 0.05,      # SSL temperature
        'memory_size_percent': 2,    # % of data to store in memory buffer
        'kd_classifier_weight': 1.0, # Knowledge distillation weight
        'momentum': 0.999,           # For BYOL momentum updates
        'weight_decay': 1e-4,        # Regularization
    },
    
    # Analysis (set to False to skip and run faster)
    'enable_analysis': False,         # Enable patch + projection analysis
}

# ============================================================================


class SimpleKaizenDebug:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🚀 Simple Kaizen Debug")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.output_dir}")
        print(f"   Task 1: {config['dataset_a']} (classes 0-9)")
        print(f"   Task 2: {config['dataset_b']} (classes 10-19)")
        print(f"   Analysis: {'ON' if config['enable_analysis'] else 'OFF'}")
    
    def _load_dataset(self, dataset_name):
        """Load a dataset by name."""
        if dataset_name == 'mnist':
            from scenarios.datasets.mnist import load_mnist_with_resize
            return load_mnist_with_resize(
                balanced=True,
                number_of_samples_per_class=self.config['num_samples_per_class']
            )
        elif dataset_name == 'cifar10':
            from scenarios.datasets.cifar import load_resized_cifar10
            return load_resized_cifar10(
                balanced=True,
                number_of_samples_per_class=self.config['num_samples_per_class']
            )
        elif dataset_name == 'fashion_mnist':
            from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
            return load_fashion_mnist_with_resize(
                balanced=True,
                number_of_samples_per_class=self.config['num_samples_per_class']
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _relabel_dataset(self, dataset, offset):
        """Relabel dataset classes by adding an offset."""
        class RelabeledDataset:
            def __init__(self, base_dataset, offset):
                self.base_dataset = base_dataset
                self.offset = offset
                if hasattr(base_dataset, 'targets'):
                    self.targets = [t + offset for t in base_dataset.targets]
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                item = self.base_dataset[idx]
                if len(item) == 2:
                    x, y = item
                    return x, y + self.offset
                elif len(item) == 3:
                    x, y, task_id = item
                    return x, y + self.offset, task_id
                else:
                    # Handle any other format - just offset the second element (label)
                    return (item[0], item[1] + self.offset) + item[2:]
        
        return RelabeledDataset(dataset, offset)
    
    def _build_model(self):
        """Build the model."""
        from models.vit_models import ViT64SingleHead
        model = ViT64SingleHead(num_classes=self.config['num_classes'])
        model.to(self.device)
        return model
    
    def _setup_analysis(self, model):
        """Setup analysis components if enabled."""
        if not self.config['enable_analysis']:
            return None, None
        
        patch_analyzer = None
        projection_analyzer = None
        
        try:
            from analysis.patch_importance_analyzer import ViTPatchImportanceAnalyzer
            patch_analyzer = ViTPatchImportanceAnalyzer(model, str(self.device))
            (self.output_dir / 'patch_analysis').mkdir(exist_ok=True)
            print("   ✅ Patch analysis enabled")
        except Exception as e:
            print(f"   ❌ Patch analysis failed: {e}")
        
        try:
            from analysis.vit_class_projection import ViTClassProjectionAnalyzer
            projection_analyzer = ViTClassProjectionAnalyzer(model, str(self.device))
            (self.output_dir / 'layer_analysis').mkdir(exist_ok=True)
            print("   ✅ Projection analysis enabled")
        except Exception as e:
            print(f"   ❌ Projection analysis failed: {e}")
        
        return patch_analyzer, projection_analyzer
    
    def _run_analysis(self, patch_analyzer, projection_analyzer, test_dataset, 
                     stage_name, dataset_name, task_id):
        """Run analysis if enabled."""
        if not self.config['enable_analysis']:
            return
        
        # Patch analysis
        if patch_analyzer:
            try:
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                patch_results = patch_analyzer.analyze_patch_importance(
                    test_loader, num_classes=10, max_batches=10, task_id=task_id
                )
                patch_analyzer.visualize_patch_importance(
                    patch_results, self.output_dir / 'patch_analysis', 
                    self.config['epochs'], dataset_name, stage_name
                )
                print(f"   📊 Patch analysis saved: {stage_name}_{dataset_name}")
            except Exception as e:
                print(f"   ❌ Patch analysis failed: {e}")
        
        # Projection analysis
        if projection_analyzer:
            try:
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                projection_scores = projection_analyzer.analyze_task_representations(
                    test_loader, task_id=task_id, num_classes_per_task=10, max_batches=10
                )
                
                analysis_dir = self.output_dir / 'layer_analysis'
                analysis_file = analysis_dir / f'{stage_name}_task_{task_id}.json'
                
                import json
                with open(analysis_file, 'w') as f:
                    json.dump({
                        'task_id': task_id,
                        'stage': stage_name,
                        'projection_scores': projection_scores
                    }, f, indent=2)
                
                print(f"   📊 Projection analysis saved: {analysis_file.name}")
            except Exception as e:
                print(f"   ❌ Projection analysis failed: {e}")
    
    def test_strategy(self, strategy_name):
        """Test a single strategy (Naive or Kaizen)."""
        print(f"\n--- Testing {strategy_name} Strategy ---")
        
        # Load datasets
        train_a, test_a = self._load_dataset(self.config['dataset_a'])
        train_b, test_b = self._load_dataset(self.config['dataset_b'])
        
        # Relabel second dataset to classes 10-19
        train_b = self._relabel_dataset(train_b, 10)
        test_b = self._relabel_dataset(test_b, 10)
        
        # Build model and strategy
        model = self._build_model()
        patch_analyzer, projection_analyzer = self._setup_analysis(model)
        
        # Create strategy config
        strategy_config = {
            'epochs': self.config['epochs'],
            'minibatch_size': self.config['minibatch_size'],
            'num_classes': self.config['num_classes'],
            'strategy_name': strategy_name,
        }
        
        if strategy_name == 'Kaizen':
            strategy_config.update(self.config['kaizen_config'])
            strategy = KaizenStrategy(model, strategy_config, str(self.device))
        else:
            strategy_config.update({
                'lr': 0.0003,
                'optimizer': 'adam'
            })
            strategy = NaiveStrategy(model, strategy_config, str(self.device))
        
        # === TASK 1: Train on Dataset A ===
        print(f"🔄 Training on {self.config['dataset_a']} for {self.config['epochs']} epochs...")
        strategy.train_task(train_a, task_id=0)
        
        acc_a_after_task1 = strategy._evaluate_dataset(test_a)
        print(f"   {self.config['dataset_a']} accuracy: {acc_a_after_task1:.1f}%")
        
        # Analysis after task 1
        self._run_analysis(patch_analyzer, projection_analyzer, test_a, 
                          'task1', self.config['dataset_a'], 0)
        
        # === TASK 2: Train on Dataset B ===
        print(f"🔄 Training on {self.config['dataset_b']} for {self.config['epochs']} epochs...")
        result = strategy.train_task(train_b, task_id=1)
        
        # Show Kaizen loss components
        if strategy_name == 'Kaizen' and 'loss_components' in result:
            loss = result['loss_components']
            print(f"   Loss components: SSL={loss['ssl_ct']:.3f}, CE={loss['ce_ct']:.3f}, "
                  f"KD_SSL={loss['ssl_kd']:.3f}, KD_CE={loss['ce_kd']:.3f}")
        
        # Final evaluation
        acc_a_after_task2 = strategy._evaluate_dataset(test_a)  # Check forgetting
        acc_b_after_task2 = strategy._evaluate_dataset(test_b)  # Check new learning
        
        forgetting = acc_a_after_task1 - acc_a_after_task2
        average_acc = (acc_a_after_task2 + acc_b_after_task2) / 2
        
        print(f"\n📊 Results:")
        print(f"   After Task 1 - {self.config['dataset_a']}: {acc_a_after_task1:.1f}%")
        print(f"   After Task 2 - {self.config['dataset_a']}: {acc_a_after_task2:.1f}% (forgetting: {forgetting:.1f}%)")
        print(f"   After Task 2 - {self.config['dataset_b']}: {acc_b_after_task2:.1f}%")
        print(f"   Final Average: {average_acc:.1f}%")
        
        # Analysis after task 2
        self._run_analysis(patch_analyzer, projection_analyzer, test_a,
                          'task2_forgetting', self.config['dataset_a'], 0)
        self._run_analysis(patch_analyzer, projection_analyzer, test_b,
                          'task2_learning', self.config['dataset_b'], 1)
        
        return {
            'strategy': strategy_name,
            'task1_acc': acc_a_after_task1,
            'final_acc_a': acc_a_after_task2,
            'final_acc_b': acc_b_after_task2,
            'forgetting': forgetting,
            'average': average_acc
        }
    
    def compare_strategies(self):
        """Compare Naive vs Kaizen strategies."""
        print("\n" + "="*60)
        print("COMPARING NAIVE vs KAIZEN")
        print("="*60)
        
        # Test both strategies
        naive_results = self.test_strategy('Naive')
        kaizen_results = self.test_strategy('Kaizen')
        
        # Show comparison
        print("\n" + "="*80)
        print("CONTINUAL LEARNING RESULTS")
        print("="*80)
        
        dataset_a = self.config['dataset_a'].upper()
        dataset_b = self.config['dataset_b'].upper()
        
        print(f"\n📈 LEARNING PROGRESSION:")
        print(f"{'Phase':<25} {'Naive':<12} {'Kaizen':<12} {'Kaizen Advantage':<15}")
        print("-" * 68)
        
        # Task 1 learning
        naive_task1 = naive_results['task1_acc']
        kaizen_task1 = kaizen_results['task1_acc']
        task1_diff = kaizen_task1 - naive_task1
        print(f"After {dataset_a} training{'':<8} {naive_task1:<12.1f} {kaizen_task1:<12.1f} {task1_diff:+13.1f}%")
        
        # Task 1 retention after Task 2
        naive_retain = naive_results['final_acc_a']
        kaizen_retain = kaizen_results['final_acc_a']
        retain_diff = kaizen_retain - naive_retain
        print(f"{dataset_a} after {dataset_b}{'':<10} {naive_retain:<12.1f} {kaizen_retain:<12.1f} {retain_diff:+13.1f}%")
        
        # Task 2 learning
        naive_task2 = naive_results['final_acc_b']
        kaizen_task2 = kaizen_results['final_acc_b']
        task2_diff = kaizen_task2 - naive_task2
        print(f"{dataset_b} learning{'':<12} {naive_task2:<12.1f} {kaizen_task2:<12.1f} {task2_diff:+13.1f}%")
        
        # Overall performance
        naive_avg = naive_results['average']
        kaizen_avg = kaizen_results['average']
        avg_diff = kaizen_avg - naive_avg
        print(f"{'Final Average':<25} {naive_avg:<12.1f} {kaizen_avg:<12.1f} {avg_diff:+13.1f}%")
        
        print("-" * 68)
        
        print(f"\n💡 FORGETTING ANALYSIS:")
        naive_forget = naive_results['forgetting']
        kaizen_forget = kaizen_results['forgetting']
        print(f"   Naive:  Lost {naive_forget:.1f}% on {dataset_a} after learning {dataset_b}")
        print(f"   Kaizen: Lost {kaizen_forget:.1f}% on {dataset_a} after learning {dataset_b}")
        forget_improvement = naive_forget - kaizen_forget
        print(f"   → Kaizen reduces forgetting by {forget_improvement:.1f}% points")
        
        

def main():
    debugger = SimpleKaizenDebug(CONFIG)
    results = debugger.compare_strategies()
    
    print(f"\n✅ Done! Results saved to: {debugger.output_dir}")
    if CONFIG['enable_analysis']:
        print(f"📊 Analysis files in: {debugger.output_dir}/patch_analysis and layer_analysis")

if __name__ == "__main__":
    main()
"""Modified experiment to test different continual learning strategies."""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from collections import defaultdict

from core.experiment import BaseExperiment
from core.strategies import (
    BaseStrategy, NaiveStrategy, ReplayStrategy, 
    CumulativeStrategy, KaizenStrategy
)


class StrategyComparisonExperiment(BaseExperiment):
    """Compare different continual learning strategies with representation analysis."""
    
    # Available strategies
    STRATEGIES = {
        'naive': NaiveStrategy,
        'replay': ReplayStrategy,
        'cumulative': CumulativeStrategy,
        'kaizen': KaizenStrategy
    }
    
    # Predefined task orders (reuse from OrderAnalysisExperiment)
    ORDERS = {
        'MTI': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
        'ITM': ["imagenet", "cifar10", "svhn", "fashion_mnist", "omniglot", "mnist"],
        'EASY_TO_HARD': ["mnist", "fashion_mnist", "omniglot", "cifar10", "svhn", "imagenet"],
        'HARD_TO_EASY': ["imagenet", "svhn", "cifar10", "omniglot", "fashion_mnist", "mnist"],
    }
    
    def run(self) -> Dict[str, Any]:
        """Run experiment comparing different strategies."""
        # Get configurations
        strategies_to_test = self.config.get('strategies', ['naive', 'replay', 'cumulative'])
        orders_to_test = self._get_orders_to_test()
        
        all_results = {}
        
        # Test each strategy with each order
        for strategy_name in strategies_to_test:
            print(f"\n{'='*80}")
            print(f"TESTING STRATEGY: {strategy_name.upper()}")
            print(f"{'='*80}")
            
            strategy_results = {}
            
            for order_name, order in orders_to_test.items():
                print(f"\n{'='*60}")
                print(f"Strategy: {strategy_name}, Order: {order_name}")
                print(f"Task sequence: {' → '.join(order)}")
                print(f"{'='*60}")
                
                # Run single experiment
                result = self._run_single_experiment(strategy_name, order_name, order)
                strategy_results[order_name] = result
                
                # Save intermediate results
                all_results[strategy_name] = strategy_results
                self.results['strategies'] = all_results
                self.save_results()
            
            # Compute strategy-specific summary
            all_results[strategy_name]['summary'] = self._summarize_strategy_results(strategy_results)
        
        # Compare across strategies
        self.results['comparison'] = self._compare_strategies(all_results)
        self.save_results()
        
        return self.results
    
    def _get_orders_to_test(self) -> Dict[str, List[str]]:
        """Get task orders to test based on config."""
        # Check if we want to test multiple orders or just one
        test_multiple_orders = self.config.get('test_multiple_orders', False)
        
        if test_multiple_orders:
            # Test with multiple orders
            orders = {}
            for order_name in self.config.get('orders', ['MTI']):
                if order_name in self.ORDERS:
                    orders[order_name] = self.ORDERS[order_name]
            
            # Add random orders if requested
            num_random = self.config.get('num_random_orders', 0)
            base_order = self.ORDERS['MTI'].copy()
            for i in range(num_random):
                random_order = base_order.copy()
                random.shuffle(random_order)
                orders[f'RANDOM_{i+1}'] = random_order
        else:
            # Just use one order (default MTI)
            default_order = self.config.get('default_order', 'MTI')
            orders = {default_order: self.ORDERS[default_order]}
        
        return orders
    
    def _run_single_experiment(self, strategy_name: str, order_name: str, 
                              order: List[str]) -> Dict[str, Any]:
        """Run CL experiment with a specific strategy and order."""
        # Reset model for fair comparison
        self.setup()
        
        # Create strategy instance
        strategy_class = self.STRATEGIES[strategy_name]
        
        # Get strategy-specific config
        strategy_config = self.config.copy()
        strategy_specific = self.config.get(f'{strategy_name}_config', {})
        # Handle case where YAML returns None for empty config sections
        if strategy_specific is None:
            strategy_specific = {}
        strategy_config.update(strategy_specific)
        
        strategy = strategy_class(self.model, strategy_config, self.device)
        
        # Load all datasets
        all_datasets = self._load_all_datasets()
        
        # Results storage
        accuracy_matrix = []
        trajectory_evolution = {}  # Store all trajectory data
        task_metrics = []
        
        # Train on tasks in specified order
        for i, dataset_name in enumerate(order):
            train_ds, test_ds = all_datasets[dataset_name]
            
            print(f"\n=== Task {i}: {dataset_name} (Strategy: {strategy_name}) ===")
            print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
            
            # Train with strategy
            task_result = strategy.train_task(
                train_ds, i, 
                analyzer=self.analyzer,
                all_datasets=all_datasets,
                order=order
            )
            
            # Store trajectory data if available
            if 'trajectory' in task_result:
                trajectory_evolution[f'task_{i}_{dataset_name}'] = task_result['trajectory']
            
            # Store task-specific metrics
            task_metrics.append({
                'task_id': i,
                'dataset': dataset_name,
                'strategy_info': task_result
            })
            
            # Evaluate on all seen tasks
            acc_row = []
            for j in range(i + 1):
                prev_dataset_name = order[j]
                _, prev_test_ds = all_datasets[prev_dataset_name]
                acc = self._evaluate_dataset(prev_test_ds)
                acc_row.append(acc)
            
            accuracy_matrix.append(acc_row)
            print(f"Accuracies after task {i}: {[f'{a:.1f}' for a in acc_row]}")
        
        # Final comprehensive analysis if requested
        if self.config.get('final_analysis', True):
            print("\n📊 Running final comprehensive analysis...")
            final_analysis = self._run_final_analysis(all_datasets, order, strategy)
            trajectory_evolution['final_analysis'] = final_analysis
        
        # Compute summary metrics
        summary = self._compute_summary_metrics(accuracy_matrix)
        
        return {
            'strategy': strategy_name,
            'order': order,
            'accuracy_matrix': accuracy_matrix,
            'trajectory_evolution': trajectory_evolution,
            'task_metrics': task_metrics,
            'summary': summary
        }
    
    def _run_final_analysis(self, all_datasets: Dict, order: List[str], 
                           strategy: BaseStrategy) -> Dict:
        """Run comprehensive final analysis on all tasks."""
        final_results = {}
        
        for i, dataset_name in enumerate(order):
            _, test_ds = all_datasets[dataset_name]
            
            # Get accuracy
            acc = self._evaluate_dataset(test_ds)
            
            # Analyze representations
            if self.analyzer:
                loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                repr_results = self.analyzer.analyze_task_representations(
                    loader, i,
                    num_classes_per_task=self.config.get('num_classes', 10),
                    max_batches=50,
                    sample_tokens=True
                )
            else:
                repr_results = {}
            
            final_results[f'task_{i}_{dataset_name}'] = {
                'accuracy': acc,
                'representations': repr_results
            }
        
        return final_results
    
    def _compute_summary_metrics(self, accuracy_matrix: List[List[float]]) -> Dict:
        """Compute summary metrics from accuracy matrix."""
        if not accuracy_matrix:
            return {}
        
        final_accuracies = accuracy_matrix[-1]
        avg_accuracy = sum(final_accuracies) / len(final_accuracies)
        
        # Compute forgetting
        forgetting = []
        if len(accuracy_matrix) > 1:
            for task_id in range(len(accuracy_matrix) - 1):
                if task_id < len(accuracy_matrix[task_id]):  # Check bounds
                    initial_acc = accuracy_matrix[task_id][task_id]
                    final_acc = accuracy_matrix[-1][task_id]
                    forgetting.append(max(0, initial_acc - final_acc))
        
        avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0
        
        # Compute forward transfer (how well new tasks are learned)
        forward_transfer = []
        for i in range(1, len(accuracy_matrix)):
            # Accuracy on task i right after training
            if i < len(accuracy_matrix[i]):
                forward_transfer.append(accuracy_matrix[i][i])
        
        avg_forward_transfer = sum(forward_transfer) / len(forward_transfer) if forward_transfer else 0
        
        return {
            'final_average_accuracy': avg_accuracy,
            'average_forgetting': avg_forgetting,
            'average_forward_transfer': avg_forward_transfer,
            'final_accuracies': final_accuracies,
            'forgetting_per_task': forgetting
        }
    
    def _summarize_strategy_results(self, strategy_results: Dict) -> Dict:
        """Summarize results across different orders for a strategy."""
        all_accuracies = []
        all_forgetting = []
        all_forward_transfer = []
        
        for order_name, result in strategy_results.items():
            if 'summary' in result:
                summary = result['summary']
                all_accuracies.append(summary['final_average_accuracy'])
                all_forgetting.append(summary['average_forgetting'])
                all_forward_transfer.append(summary.get('average_forward_transfer', 0))
        
        return {
            'mean_accuracy': sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0,
            'std_accuracy': torch.std(torch.tensor(all_accuracies)).item() if len(all_accuracies) > 1 else 0,
            'mean_forgetting': sum(all_forgetting) / len(all_forgetting) if all_forgetting else 0,
            'std_forgetting': torch.std(torch.tensor(all_forgetting)).item() if len(all_forgetting) > 1 else 0,
            'mean_forward_transfer': sum(all_forward_transfer) / len(all_forward_transfer) if all_forward_transfer else 0
        }
    
    def _compare_strategies(self, all_results: Dict) -> Dict:
        """Compare results across different strategies."""
        comparison = {
            'strategy_ranking': [],
            'best_strategy': None,
            'representation_analysis': {}
        }
        
        # Rank strategies by average accuracy
        strategy_scores = []
        for strategy_name, results in all_results.items():
            if 'summary' in results:
                score = results['summary']['mean_accuracy']
                forgetting = results['summary']['mean_forgetting']
                strategy_scores.append({
                    'strategy': strategy_name,
                    'accuracy': score,
                    'forgetting': forgetting
                })
        
        # Sort by accuracy (descending)
        strategy_scores.sort(key=lambda x: x['accuracy'], reverse=True)
        comparison['strategy_ranking'] = strategy_scores
        
        if strategy_scores:
            comparison['best_strategy'] = strategy_scores[0]['strategy']
            comparison['accuracy_improvement'] = (
                strategy_scores[0]['accuracy'] - strategy_scores[-1]['accuracy']
                if len(strategy_scores) > 1 else 0
            )
        
        # Analyze representation differences between strategies
        if self.config.get('compare_representations', True):
            comparison['representation_analysis'] = self._compare_representation_evolution(all_results)
        
        return comparison
    
    def _compare_representation_evolution(self, all_results: Dict) -> Dict:
        """Compare how representations evolve differently across strategies."""
        rep_comparison = {}
        
        # Extract final representation scores for each strategy
        for strategy_name, results in all_results.items():
            # Get the first order's results (assuming consistent ordering)
            order_results = list(results.values())[0] if results else {}
            
            if 'trajectory_evolution' in order_results:
                trajectory = order_results['trajectory_evolution']
                
                # Extract final analysis if available
                if 'final_analysis' in trajectory:
                    final_scores = {}
                    for task_name, task_data in trajectory['final_analysis'].items():
                        if 'representations' in task_data and 'projection_scores' in task_data['representations']:
                            scores = task_data['representations']['projection_scores']
                            # Average across all blocks and token types
                            avg_score = sum(scores.values()) / len(scores) if scores else 0
                            final_scores[task_name] = avg_score
                    
                    rep_comparison[strategy_name] = {
                        'final_representation_scores': final_scores,
                        'mean_score': sum(final_scores.values()) / len(final_scores) if final_scores else 0
                    }
        
        return rep_comparison
    
    def setup(self):
        """Setup experiment components."""
        print("Setting up experiment components...")
        self.model = self.build_model()
        self.analyzer = self.build_analyzer()
        print(f"Model: {type(self.model).__name__}")
        print(f"Analyzer: {type(self.analyzer).__name__ if self.analyzer else 'None'}")
    
    def _evaluate_dataset(self, test_ds) -> float:
        """Evaluate on a dataset."""
        self.model.eval()
        correct = total = 0
        
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        with torch.inference_mode():
            for batch in loader:
                # Robust batch handling
                x, y = batch[0], batch[1]
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total
    
    def _load_all_datasets(self) -> Dict[str, tuple]:
        """Load all datasets manually using direct dataset loaders."""
        print("Loading all datasets...")
        datasets = {}
        
        # Get data loading parameters
        balanced = self.config.get('balanced', 'balanced') == 'balanced'
        num_samples = self.config.get('number_of_samples_per_class', 500)
        
        # Import dataset loaders
        from scenarios.datasets.mnist import load_mnist_with_resize
        from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
        from scenarios.datasets.cifar import load_resized_cifar10
        from scenarios.datasets.svhn import load_svhn_resized
        from scenarios.datasets.load_imagenet import load_imagenet
        from scenarios.datasets.omniglot import _load_omniglot
        from scenarios.utils import filter_classes, transform_from_gray_to_rgb
        from torchvision.transforms import Resize, Compose, ToTensor, Normalize
        
        # Load MNIST
        print(f"  → Loading mnist...")
        train_mnist, test_mnist = load_mnist_with_resize(balanced, num_samples)
        datasets['mnist'] = (train_mnist, test_mnist)
        print(f"    Train: {len(train_mnist)} samples, Test: {len(test_mnist)} samples")
        
        # Load Omniglot with proper transforms
        print(f"  → Loading omniglot...")
        transform_omniglot = Compose([
            ToTensor(),
            Resize((64, 64)),
            transform_from_gray_to_rgb(),
            Normalize(mean=(0.9221,), std=(0.2681,))
        ])
        train_omniglot_full, test_omniglot_full = _load_omniglot(transform_omniglot, balanced=balanced, number_of_samples_per_class=num_samples)
        train_omniglot, test_omniglot = filter_classes(train_omniglot_full, test_omniglot_full, classes=list(range(10)))
        datasets['omniglot'] = (train_omniglot, test_omniglot)
        print(f"    Train: {len(train_omniglot)} samples, Test: {len(test_omniglot)} samples")
        
        # Load Fashion-MNIST
        print(f"  → Loading fashion_mnist...")
        train_fashion, test_fashion = load_fashion_mnist_with_resize(balanced, num_samples)
        datasets['fashion_mnist'] = (train_fashion, test_fashion)
        print(f"    Train: {len(train_fashion)} samples, Test: {len(test_fashion)} samples")
        
        # Load SVHN
        print(f"  → Loading svhn...")
        train_svhn, test_svhn = load_svhn_resized(balanced, num_samples)
        datasets['svhn'] = (train_svhn, test_svhn)
        print(f"    Train: {len(train_svhn)} samples, Test: {len(test_svhn)} samples")
        
        # Load CIFAR-10
        print(f"  → Loading cifar10...")
        train_cifar, test_cifar = load_resized_cifar10(balanced, num_samples)
        datasets['cifar10'] = (train_cifar, test_cifar)
        print(f"    Train: {len(train_cifar)} samples, Test: {len(test_cifar)} samples")
        
        # Load ImageNet (filtered to 10 classes)
        print(f"  → Loading imagenet...")
        train_imagenet_full, test_imagenet_full = load_imagenet(balanced, num_samples)
        train_imagenet, test_imagenet = filter_classes(train_imagenet_full, test_imagenet_full, classes=list(range(10)))
        datasets['imagenet'] = (train_imagenet, test_imagenet)
        print(f"    Train: {len(train_imagenet)} samples, Test: {len(test_imagenet)} samples")
        
        return datasets


# Example configuration for running the experiment
example_config = {
    # Experiment settings
    'experiment_type': 'StrategyComparisonExperiment',
    'experiment_name': 'strategy_comparison',
    
    # Strategies to test
    'strategies': ['naive', 'replay', 'cumulative', 'kaizen'],
    
    # Order settings
    'test_multiple_orders': False,  # Set True to test multiple orders
    'default_order': 'MTI',  # Default order if test_multiple_orders is False
    'orders': ['MTI', 'ITM'],  # Orders to test if test_multiple_orders is True
    'num_random_orders': 0,  # Additional random orders
    
    # Training settings
    'epochs': 50,
    'minibatch_size': 128,
    'lr': 0.0003,
    'optimizer': 'adam',
    
    # Analysis settings
    'analyze_representations': True,
    'analysis_freq': 10,  # Analyze every N epochs
    'final_analysis': True,
    'compare_representations': True,
    
    # Model settings
    'model_name': 'ViT64',
    'num_classes': 10,
    
    # Strategy-specific settings
    'replay_config': {
        'memory_size': 500,  # Samples per task
        'replay_batch_ratio': 0.5
    },
    
    'kaizen_config': {
        'memory_size': 500,
        'ssl_method': 'simclr',
        'kd_weight': 1.0,
        'ssl_weight': 1.0
    },
    
    # Scenario settings
    'scenario_type': 'task_incremental',
    'balanced': 'balanced',
    'number_of_samples_per_class': 500,
    
    # Device
    'cuda': 3
}
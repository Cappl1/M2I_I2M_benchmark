"""Experiment to analyze different task orders."""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from collections import defaultdict

from core.experiment import BaseExperiment
from core.strategies import NaiveStrategy, ReplayStrategy, CumulativeStrategy


class OrderAnalysisExperiment(BaseExperiment):
    """Compare different task orders in continual learning."""
    
    # Predefined orders
    ORDERS = {
        'MTI': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
        'ITM': ["imagenet", "cifar10", "svhn", "fashion_mnist", "omniglot", "mnist"],
        'EASY_TO_HARD': ["mnist", "fashion_mnist", "omniglot", "cifar10", "svhn", "imagenet"],
        'HARD_TO_EASY': ["imagenet", "svhn", "cifar10", "omniglot", "fashion_mnist", "mnist"],
    }
    
    def run(self) -> Dict[str, Any]:
        """Run experiment with multiple task orders."""
        orders_to_test = self._get_orders_to_test()
        all_results = {}
        
        for order_name, order in orders_to_test.items():
            print(f"\n{'='*60}")
            print(f"Testing order: {order_name}")
            print(f"Order: {' → '.join(order)}")
            print(f"{'='*60}")
            
            # Run single order experiment
            order_results = self._run_single_order(order_name, order)
            all_results[order_name] = order_results
            
            # Save intermediate results
            self.results['orders'] = all_results
            self.save_results()
        
        # Compute comparative metrics
        self.results['comparison'] = self._compare_orders(all_results)
        self.save_results()
        
        return self.results
    
    def _get_orders_to_test(self) -> Dict[str, List[str]]:
        """Get task orders to test based on config."""
        orders = {}
        
        # Add requested predefined orders
        for order_name in self.config.get('orders', ['MTI', 'ITM']):
            if order_name in self.ORDERS:
                orders[order_name] = self.ORDERS[order_name]
        
        # Add random orders
        num_random = self.config.get('num_random_orders', 2)
        base_order = self.ORDERS['MTI'].copy()
        
        for i in range(num_random):
            random_order = base_order.copy()
            random.shuffle(random_order)
            orders[f'RANDOM_{i+1}'] = random_order
        
        return orders
    
    def setup(self):
        """Setup experiment components for order analysis."""
        print("Setting up experiment components...")
        self.model = self.build_model()
        self.analyzer = self.build_analyzer()
        print(f"Model: {type(self.model).__name__}")
        print(f"Analyzer: {type(self.analyzer).__name__ if self.analyzer else 'None'}")
    
    def _run_single_order(self, order_name: str, order: List[str]) -> Dict[str, Any]:
        """Run CL experiment with a specific task order."""
        # Reset model and strategy for fair comparison
        self.setup()
        
        # Load all datasets manually
        all_datasets = self._load_all_datasets()
        
        # Results storage
        accuracy_matrix = []
        repr_evolution = defaultdict(dict)  # Now includes epoch-wise trajectories
        
        # Train on tasks in specified order
        for i, dataset_name in enumerate(order):
            train_ds, test_ds = all_datasets[dataset_name]
            
            print(f"\n=== Task {i}: {dataset_name} ===")
            print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
            
            # Train with or without trajectory analysis
            if self.config.get('track_trajectory', False):
                # Manual training with trajectory analysis
                print(f"  → Training with trajectory analysis every {self.config.get('analysis_freq', 10)} epochs")
                trajectory_results = self._train_with_trajectory_analysis(
                    train_ds, i, dataset_name, order, all_datasets
                )
                repr_evolution[f'task_{i}_trajectory'] = trajectory_results
            else:
                # Manual training without trajectory analysis
                self._train_task_manually(train_ds, test_ds, dataset_name)
                
                # Final analysis only (after task completion)
                if self.config.get('analyze_representations', True):
                    task_representations = {}
                    
                    # Analyze ALL tasks seen so far to track forgetting
                    for j in range(i + 1):
                        prev_dataset_name = order[j]
                        _, prev_test_ds = all_datasets[prev_dataset_name]
                        
                        print(f"  → Final analysis for {prev_dataset_name} (task {j})")
                        repr_results = self._analyze_task_representations(prev_test_ds, j, prev_dataset_name)
                        task_representations[f'task_{j}_{prev_dataset_name}'] = repr_results
                    
                    repr_evolution[f'after_task_{i}'] = task_representations
            
            # Evaluate on all seen tasks
            acc_row = []
            for j in range(i + 1):
                prev_dataset_name = order[j]
                _, prev_test_ds = all_datasets[prev_dataset_name]
                acc = self._evaluate_dataset(prev_test_ds)
                acc_row.append(acc)
            
            accuracy_matrix.append(acc_row)
            print(f"Accuracies: {[f'{a:.1f}' for a in acc_row]}")
        
        # Compute summary metrics
        final_accuracies = accuracy_matrix[-1] if accuracy_matrix else []
        avg_accuracy = sum(final_accuracies) / len(final_accuracies) if final_accuracies else 0
        
        # Compute forgetting
        forgetting = []
        if len(accuracy_matrix) > 1:
            for task_id in range(len(accuracy_matrix) - 1):
                initial_acc = accuracy_matrix[task_id][task_id]
                final_acc = accuracy_matrix[-1][task_id]
                forgetting.append(initial_acc - final_acc)
        
        avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0
        
        return {
            'order': order,
            'accuracy_matrix': accuracy_matrix,
            'trajectory_representations': dict(repr_evolution),  # Renamed for clarity
            'summary': {
                'final_average_accuracy': avg_accuracy,
                'average_forgetting': avg_forgetting,
                'final_accuracies': final_accuracies,
                'forgetting_per_task': forgetting
            }
        }
    
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
    
    def _train_with_trajectory_analysis(self, train_ds, task_num: int, dataset_name: str, 
                                       order: List[str], all_datasets: Dict) -> Dict:
        """Train a task manually while tracking representation trajectories every N epochs."""
        analysis_freq = self.config.get('analysis_freq', 10)
        epochs = self.config.get('epochs', 50)
        
        trajectory_data = {}
        
        # Manual training setup
        train_loader = DataLoader(train_ds, batch_size=self.config.get('minibatch_size', 128), shuffle=True)
        optimizer = self._build_optimizer(self.model)
        criterion = nn.CrossEntropyLoss()
        
        print(f"  → Manual training with trajectory analysis every {analysis_freq} epochs")
        
        # Train epoch by epoch with periodic analysis
        for epoch in range(epochs):
            # Train one epoch manually
            self.model.train()
            total_loss = 0.0
            correct = total = 0
            
            for batch in train_loader:
                # Robust batch handling
                x, y = batch[0], batch[1]
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
            
            train_acc = 100.0 * correct / total
            
            # Evaluate on current task's test set for validation accuracy
            if (epoch + 1) % 10 == 0:
                _, current_test_ds = all_datasets[dataset_name]
                val_acc = self._evaluate_dataset(current_test_ds)
                print(f"    Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
            
            # Periodic analysis
            if (epoch + 1) % analysis_freq == 0:
                print(f"    → Epoch {epoch + 1}: Analyzing ALL seen tasks...")
                
                epoch_analysis = {}
                
                # Analyze ALL tasks seen so far (including current one)
                for j in range(task_num + 1):
                    prev_dataset_name = order[j]
                    _, prev_test_ds = all_datasets[prev_dataset_name]
                    
                    # Get accuracy for this task at this checkpoint
                    acc = self._evaluate_dataset(prev_test_ds)
                    
                    # Analyze representations
                    repr_results = self._analyze_task_representations(
                        prev_test_ds, j, f"task{task_num}_epoch{epoch+1}_analyzing_{prev_dataset_name}"
                    )
                    
                    epoch_analysis[f'task_{j}_{prev_dataset_name}'] = {
                        'accuracy': acc,
                        'representations': repr_results
                    }
                    
                    # Print progress
                    if j == task_num:
                        print(f"      → {prev_dataset_name}: CURRENT TASK ({acc:.1f}%)")
                    else:
                        print(f"      → {prev_dataset_name}: FORGETTING ({acc:.1f}%)")
                
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
        
        return trajectory_data
    
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
    
    def _analyze_task_representations(self, test_ds, task_id: int, prefix: str):
        """Analyze ViT representations for a specific task."""
        # Create dataloader for analysis
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        try:
            # Use ViT analyzer if available
            if hasattr(self, 'analyzer') and self.analyzer:
                projection_scores = self.analyzer.analyze_task_representations(
                    loader, 
                    task_id=task_id, 
                    num_classes_per_task=self.config.get('num_classes', 10),
                    max_batches=50,  # Analyze first 50 batches for speed
                    sample_tokens=True  # Sample tokens instead of all 196
                )
                return {
                    'task_id': task_id,
                    'analyzer_type': 'vit_projection',
                    'projection_scores': projection_scores,
                    'prefix': prefix
                }
            else:
                print(f"  → Warning: No analyzer available for task {task_id}")
                return {'error': 'No analyzer available'}
                
        except Exception as e:
            print(f"  → Analysis failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    def _compare_orders(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare results across different orders."""
        comparison = {
            'summary_table': {},
            'best_order': None,
            'worst_order': None
        }
        
        # Create summary table
        for order_name, results in all_results.items():
            summary = results['summary']
            comparison['summary_table'][order_name] = {
                'avg_accuracy': summary['final_average_accuracy'],
                'avg_forgetting': summary['average_forgetting']
            }
        
        # Find best/worst orders
        if comparison['summary_table']:
            best = max(comparison['summary_table'].items(), 
                      key=lambda x: x[1]['avg_accuracy'])
            worst = min(comparison['summary_table'].items(), 
                       key=lambda x: x[1]['avg_accuracy'])
            
            comparison['best_order'] = best[0]
            comparison['worst_order'] = worst[0]
        
        return comparison
    
    def _train_task_manually(self, train_ds, test_ds, dataset_name: str):
        """Simple manual training without analysis."""
        print(f"  → Training on {dataset_name} ({len(train_ds)} samples)")
        
        train_loader = DataLoader(train_ds, batch_size=self.config.get('minibatch_size', 128), shuffle=True)
        optimizer = self._build_optimizer(self.model)
        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get('epochs', 50)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = total = 0
            
            for batch in train_loader:
                x, y = batch[0], batch[1]
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
            
            if (epoch + 1) % 10 == 0:
                train_acc = 100.0 * correct / total
                val_acc = self._evaluate_dataset(test_ds)
                print(f"    Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
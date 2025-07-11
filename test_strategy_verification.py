#!/usr/bin/env python3
"""
Comprehensive test script to verify continual learning strategies are working correctly.
Tests both task incremental and class incremental scenarios with detailed dataflow logging.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, List
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.strategy_comparison import StrategyComparisonExperiment
from core.strategies import NaiveStrategy, ReplayStrategy, CumulativeStrategy, KaizenStrategy


class StrategyVerificationTest:
    """Test suite to verify strategies work correctly with detailed logging."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
    def run_comprehensive_test(self):
        """Run comprehensive verification tests."""
        print("=" * 80)
        print("🧪 CONTINUAL LEARNING STRATEGY VERIFICATION TEST")
        print("=" * 80)
        
        # Test configurations - separate tests for each strategy
        test_configs = [
            # Individual strategy tests for better overview
            {
                'name': 'Naive Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive']
            },
            {
                'name': 'Replay Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental', 
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['replay']
            },
            {
                'name': 'Cumulative Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['cumulative']
            },
            {
                'name': 'Kaizen Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['kaizen']
            },
            # Class incremental tests
            {
                'name': 'Naive Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive']
            },
            {
                'name': 'Replay Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['replay']
            },
            {
                'name': 'Kaizen Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['kaizen']
            }
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"\n{'🎯 ' + config['name']}")
            print("=" * 60)
            
            try:
                test_result = self._run_scenario_test(config)
                results[config['name']] = test_result
                print(f"✅ {config['name']} completed successfully")
                
            except Exception as e:
                print(f"❌ {config['name']} failed: {str(e)}")
                traceback.print_exc()
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        
        return results
    
    def _run_scenario_test(self, test_config: Dict) -> Dict:
        """Run test for a specific scenario configuration."""
        scenario_type = test_config['scenario_type']
        progression = test_config['progression']
        strategies = test_config['strategies']
        
        print(f"📋 Scenario: {scenario_type}")
        print(f"📋 Progression: {' → '.join(progression)}")
        print(f"📋 Strategies: {', '.join(strategies)}")
        
        # Create base configuration
        base_config = self._create_test_config(scenario_type, progression, strategies)
        
        scenario_results = {}
        
        for strategy_name in strategies:
            print(f"\n{'🚀 Testing Strategy: ' + strategy_name.upper()}")
            print("-" * 50)
            
            strategy_result = self._test_single_strategy(
                strategy_name, base_config, scenario_type, progression
            )
            scenario_results[strategy_name] = strategy_result
        
        return scenario_results
    
    def _test_single_strategy(self, strategy_name: str, base_config: Dict, 
                             scenario_type: str, progression: List[str]) -> Dict:
        """Test a single strategy with detailed logging."""
        
        print(f"🔄 Initializing {strategy_name} strategy...")
        
        # Create experiment instance
        config = base_config.copy()
        config['strategies'] = [strategy_name]
        
        try:
            experiment = StrategyComparisonExperiment(config)
            
            # Add custom order to experiment's ORDERS if needed
            if 'ORDERS' in config:
                experiment.ORDERS.update(config['ORDERS'])
                
            experiment.setup()
            
            print(f"✅ Model initialized: {type(experiment.model).__name__}")
            print(f"✅ Analyzer initialized: {type(experiment.analyzer).__name__ if experiment.analyzer else 'None'}")
            
            # Load datasets and verify
            print(f"\n📂 Loading datasets...")
            all_datasets = experiment._load_all_datasets()
            
            for dataset_name in progression:
                train_ds, test_ds = all_datasets[dataset_name]
                print(f"  ✅ {dataset_name}: Train={len(train_ds)}, Test={len(test_ds)}")
            
            # Create strategy instance for detailed testing
            strategy_class = experiment.STRATEGIES[strategy_name]
            strategy_config = config.copy()
            
            # Add strategy-specific config
            strategy_specific = config.get(f'{strategy_name}_config', {})
            if strategy_specific is None:
                strategy_specific = {}
            strategy_config.update(strategy_specific)
            
            strategy = strategy_class(experiment.model, strategy_config, experiment.device)
            
            print(f"\n🔍 Strategy instance created: {type(strategy).__name__}")
            
            # Test training on each task with detailed logging
            strategy_metrics = self._test_strategy_training(
                strategy, all_datasets, progression, experiment, scenario_type
            )
            
            return {
                'status': 'success',
                'metrics': strategy_metrics,
                'model_type': type(experiment.model).__name__,
                'strategy_type': type(strategy).__name__
            }
            
        except Exception as e:
            print(f"❌ Strategy {strategy_name} failed: {str(e)}")
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_strategy_training(self, strategy, all_datasets: Dict, progression: List[str], 
                               experiment, scenario_type: str) -> Dict:
        """Test strategy training with detailed dataflow logging."""
        
        print(f"\n🏋️ Starting continual learning simulation...")
        print(f"📊 Scenario type: {scenario_type}")
        
        task_metrics = []
        accuracy_matrix = []
        memory_evolution = []
        
        for task_id, dataset_name in enumerate(progression):
            print(f"\n{'=' * 60}")
            print(f"📚 TASK {task_id}: {dataset_name.upper()}")
            print(f"{'=' * 60}")
            
            train_ds, test_ds = all_datasets[dataset_name]
            
            # Log pre-training state
            print(f"🔍 Pre-training analysis:")
            print(f"  • Training samples: {len(train_ds)}")
            print(f"  • Test samples: {len(test_ds)}")
            
            # Check memory state for replay strategies
            if hasattr(strategy, 'memory_per_task'):
                print(f"  • Memory tasks stored: {len(strategy.memory_per_task)}")
                total_memory = sum(len(data) for data, _ in strategy.memory_per_task.values())
                print(f"  • Total memory samples: {total_memory}")
            
            # Check cumulative data for cumulative strategy
            if hasattr(strategy, 'all_train_data'):
                total_cumulative = sum(len(ds) for ds in strategy.all_train_data)
                print(f"  • Cumulative training samples: {total_cumulative}")
            
            # Get initial accuracy on current task (should be random)
            print(f"\n🎯 Initial performance on {dataset_name}:")
            initial_acc = experiment._evaluate_dataset(test_ds)
            print(f"  • Accuracy: {initial_acc:.2f}% (should be ~random)")
            
            # Train on current task
            print(f"\n🏋️ Training on task {task_id} ({dataset_name})...")
            
            # Log strategy-specific pre-training information
            self._log_strategy_specific_info(strategy, task_id, "PRE-TRAINING")
            
            task_result = strategy.train_task(
                train_ds, task_id,
                analyzer=experiment.analyzer,
                all_datasets=all_datasets,
                order=progression
            )
            
            # Log strategy-specific post-training information
            self._log_strategy_specific_info(strategy, task_id, "POST-TRAINING")
            
            # Log post-training state
            print(f"\n📊 Post-training analysis:")
            
            # Test on current task
            current_acc = experiment._evaluate_dataset(test_ds)
            print(f"  • Current task ({dataset_name}) accuracy: {current_acc:.2f}%")
            
            # Test on all previous tasks
            task_accuracies = []
            for prev_task_id in range(task_id + 1):
                prev_dataset_name = progression[prev_task_id]
                _, prev_test_ds = all_datasets[prev_dataset_name]
                prev_acc = experiment._evaluate_dataset(prev_test_ds)
                task_accuracies.append(prev_acc)
                
                status = "CURRENT" if prev_task_id == task_id else "PREVIOUS"
                forgetting = initial_acc - prev_acc if prev_task_id < task_id else 0
                print(f"  • Task {prev_task_id} ({prev_dataset_name}) [{status}]: {prev_acc:.2f}% " +
                      (f"(forgot {forgetting:.2f}%)" if forgetting > 0 else ""))
            
            accuracy_matrix.append(task_accuracies.copy())
            
            # Log strategy-specific information
            if hasattr(strategy, 'memory_per_task') and strategy.memory_per_task:
                memory_stats = {
                    'total_tasks_in_memory': len(strategy.memory_per_task),
                    'total_samples': sum(len(data) for data, _ in strategy.memory_per_task.values()),
                    'samples_per_task': {f'task_{tid}': len(data) 
                                       for tid, (data, _) in strategy.memory_per_task.items()}
                }
                memory_evolution.append(memory_stats)
                print(f"  • Memory state: {memory_stats['total_samples']} samples from {memory_stats['total_tasks_in_memory']} tasks")
            
            # Compute and log continual learning metrics
            if task_id > 0:
                # Average accuracy
                avg_acc = sum(task_accuracies) / len(task_accuracies)
                
                # Forgetting (only for tasks we've seen before)
                forgetting_values = []
                for i in range(task_id):  # Exclude current task
                    # Compare to accuracy right after training that task
                    initial_acc_task_i = accuracy_matrix[i][i]
                    current_acc_task_i = task_accuracies[i]
                    forgetting = max(0, initial_acc_task_i - current_acc_task_i)
                    forgetting_values.append(forgetting)
                
                avg_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0
                
                print(f"  • Average accuracy: {avg_acc:.2f}%")
                print(f"  • Average forgetting: {avg_forgetting:.2f}%")
            
            task_metrics.append({
                'task_id': task_id,
                'dataset': dataset_name,
                'initial_accuracy': initial_acc,
                'final_accuracy': current_acc,
                'all_task_accuracies': task_accuracies.copy(),
                'task_result': task_result
            })
        
        # Final comprehensive analysis
        print(f"\n{'=' * 60}")
        print(f"📈 FINAL STRATEGY ANALYSIS")
        print(f"{'=' * 60}")
        
        final_metrics = self._compute_final_metrics(accuracy_matrix, progression)
        
        print(f"🎯 Final Results:")
        print(f"  • Average Accuracy: {final_metrics['avg_accuracy']:.2f}%")
        print(f"  • Average Forgetting: {final_metrics['avg_forgetting']:.2f}%")
        print(f"  • Forward Transfer: {final_metrics['forward_transfer']:.2f}%")
        
        print(f"\n📊 Accuracy Matrix:")
        for i, row in enumerate(accuracy_matrix):
            row_str = " ".join(f"{acc:5.1f}" for acc in row)
            print(f"  After Task {i}: [{row_str}]")
        
        return {
            'task_metrics': task_metrics,
            'accuracy_matrix': accuracy_matrix,
            'memory_evolution': memory_evolution,
            'final_metrics': final_metrics
        }
    
    def _log_strategy_specific_info(self, strategy, task_id: int, phase: str):
        """Log detailed strategy-specific information."""
        strategy_name = type(strategy).__name__
        print(f"\n🔍 {phase} Strategy Details ({strategy_name}):")
        
        # Naive Strategy - should have no special state
        if strategy_name == "NaiveStrategy":
            print("  • Naive strategy: No special state (trains only on current task)")
            
        # Replay Strategy - log memory state
        elif strategy_name == "ReplayStrategy":
            if hasattr(strategy, 'memory_per_task'):
                if strategy.memory_per_task:
                    total_samples = sum(len(data) for data, _ in strategy.memory_per_task.values())
                    print(f"  • Memory buffer: {total_samples} samples from {len(strategy.memory_per_task)} tasks")
                    for tid, (data, labels) in strategy.memory_per_task.items():
                        print(f"    - Task {tid}: {len(data)} samples")
                else:
                    print("  • Memory buffer: Empty (first task)")
                
                if phase == "PRE-TRAINING" and task_id > 0:
                    print(f"  • Will mix current task with replay samples (ratio: {strategy.replay_batch_ratio})")
                elif phase == "POST-TRAINING":
                    print(f"  • Updated memory buffer with samples from task {task_id}")
            
        # Cumulative Strategy - log cumulative dataset size
        elif strategy_name == "CumulativeStrategy":
            if hasattr(strategy, 'all_train_data'):
                if phase == "PRE-TRAINING":
                    if strategy.all_train_data:
                        total_samples = sum(len(ds) for ds in strategy.all_train_data)
                        print(f"  • Cumulative training set: {total_samples} samples from {len(strategy.all_train_data)} tasks")
                    else:
                        print("  • Cumulative training set: Empty (first task)")
                elif phase == "POST-TRAINING":
                    total_samples = sum(len(ds) for ds in strategy.all_train_data)
                    print(f"  • Updated cumulative set: {total_samples} samples from {len(strategy.all_train_data)} tasks")
                    
        # Kaizen Strategy - log previous model and SSL state
        elif strategy_name == "KaizenStrategy":
            has_prev_model = hasattr(strategy, 'previous_model') and strategy.previous_model is not None
            print(f"  • Previous model for distillation: {'Available' if has_prev_model else 'None (first task)'}")
            
            if hasattr(strategy, 'memory_per_task'):
                if strategy.memory_per_task:
                    total_samples = sum(len(data) for data, _ in strategy.memory_per_task.values())
                    print(f"  • SSL memory buffer: {total_samples} samples from {len(strategy.memory_per_task)} tasks")
                else:
                    print("  • SSL memory buffer: Empty")
                    
            if hasattr(strategy, 'ssl_method'):
                print(f"  • SSL method: {strategy.ssl_method}")
                print(f"  • KD weight: {strategy.kd_weight}, SSL weight: {strategy.ssl_weight}")
                
        else:
            print(f"  • Unknown strategy type: {strategy_name}")
    
    def _compute_final_metrics(self, accuracy_matrix: List[List[float]], 
                              progression: List[str]) -> Dict:
        """Compute final continual learning metrics."""
        if not accuracy_matrix:
            return {}
        
        # Average accuracy (final performance on all tasks)
        final_accuracies = accuracy_matrix[-1]
        avg_accuracy = sum(final_accuracies) / len(final_accuracies)
        
        # Forgetting (how much we forgot from initial performance)
        forgetting_values = []
        for task_id in range(len(accuracy_matrix) - 1):  # Exclude current task
            initial_acc = accuracy_matrix[task_id][task_id]  # Accuracy right after training
            final_acc = accuracy_matrix[-1][task_id]  # Final accuracy
            forgetting = max(0, initial_acc - final_acc)
            forgetting_values.append(forgetting)
        
        avg_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0
        
        # Forward transfer (how well we learn new tasks)
        forward_transfer_values = []
        for task_id in range(1, len(accuracy_matrix)):
            # Accuracy on task immediately after training
            new_task_acc = accuracy_matrix[task_id][task_id]
            forward_transfer_values.append(new_task_acc)
        
        forward_transfer = sum(forward_transfer_values) / len(forward_transfer_values) if forward_transfer_values else 0
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_forgetting': avg_forgetting,
            'forward_transfer': forward_transfer,
            'final_accuracies': final_accuracies,
            'forgetting_per_task': forgetting_values
        }
    
    def _create_test_config(self, scenario_type: str, progression: List[str], 
                           strategies: List[str]) -> Dict:
        """Create test configuration."""
        return {
            # Experiment settings
            'experiment_type': 'StrategyComparisonExperiment',
            'experiment_name': f'verification_test_{scenario_type}',
            
            # Strategies to test
            'strategies': strategies,
            
            # Order settings
            'test_multiple_orders': False,
            'default_order': 'CUSTOM',
            'ORDERS': {'CUSTOM': progression},  # Custom progression
            
            # Training settings (reduced for testing)
            'epochs': 10,  # Reduced for quick testing
            'minibatch_size': 64,
            'lr': 0.001,
            'optimizer': 'adam',
            
            # Analysis settings
            'analyze_representations': True,
            'analysis_freq': 5,  # Analyze every 5 epochs
            'final_analysis': True,
            'compare_representations': True,
            
            # Model settings
            'model_name': 'ViT64',
            'num_classes': 10,
            
            # Strategy-specific settings
            'replay_config': {
                'memory_size': 100,  # Reduced for testing
                'replay_batch_ratio': 0.5
            },
            
            'kaizen_config': {
                'memory_size': 100,  # Reduced for testing
                'ssl_method': 'simclr',
                'kd_weight': 1.0,
                'ssl_weight': 1.0
            },
            
            # Scenario settings
            'scenario_type': scenario_type,
            'balanced': 'balanced',
            'number_of_samples_per_class': 200,  # Reduced for testing
            
            # Device
            'cuda': 0 if torch.cuda.is_available() else None
        }
    
    def _print_verification_summary(self, results: Dict):
        """Print final verification summary."""
        print("\n" + "=" * 80)
        print("📋 STRATEGY VERIFICATION SUMMARY")
        print("=" * 80)
        
        for scenario_name, scenario_results in results.items():
            print(f"\n🎯 {scenario_name}:")
            
            if 'error' in scenario_results:
                print(f"  ❌ Failed: {scenario_results['error']}")
                continue
            
            for strategy_name, strategy_result in scenario_results.items():
                if strategy_result['status'] == 'success':
                    metrics = strategy_result['metrics']['final_metrics']
                    print(f"  ✅ {strategy_name.upper()}:")
                    print(f"    • Avg Accuracy: {metrics['avg_accuracy']:.1f}%")
                    print(f"    • Avg Forgetting: {metrics['avg_forgetting']:.1f}%")
                    print(f"    • Forward Transfer: {metrics['forward_transfer']:.1f}%")
                else:
                    print(f"  ❌ {strategy_name.upper()}: {strategy_result['error']}")
        
        print(f"\n✅ Verification test completed!")
        
        # Add strategy behavior verification
        self._verify_strategy_behaviors(results)
    
    def _verify_strategy_behaviors(self, results: Dict):
        """Verify that strategies are showing expected behavioral differences."""
        print("\n" + "=" * 80)
        print("🔍 STRATEGY BEHAVIOR VERIFICATION")
        print("=" * 80)
        
        for scenario_name, scenario_results in results.items():
            if 'error' in scenario_results:
                continue
                
            print(f"\n🎯 {scenario_name}:")
            
            # Extract metrics for comparison
            strategy_metrics = {}
            for strategy_name, strategy_result in scenario_results.items():
                if strategy_result['status'] == 'success':
                    metrics = strategy_result['metrics']['final_metrics']
                    strategy_metrics[strategy_name] = metrics
            
            if len(strategy_metrics) < 2:
                print("  ⚠️  Need at least 2 strategies for comparison")
                continue
                
            # Expected behaviors to verify:
            print("  🔍 Expected behavior verification:")
            
            # 1. Cumulative should have lowest forgetting
            if 'cumulative' in strategy_metrics:
                cumulative_forgetting = strategy_metrics['cumulative']['avg_forgetting']
                other_forgetting = [
                    metrics['avg_forgetting'] for name, metrics in strategy_metrics.items() 
                    if name != 'cumulative'
                ]
                
                if all(cumulative_forgetting <= other for other in other_forgetting):
                    print("  ✅ Cumulative strategy has lowest forgetting (as expected)")
                else:
                    print("  ⚠️  Cumulative strategy should have lowest forgetting")
                    
            # 2. Replay should perform better than naive
            if 'replay' in strategy_metrics and 'naive' in strategy_metrics:
                replay_acc = strategy_metrics['replay']['avg_accuracy']
                naive_acc = strategy_metrics['naive']['avg_accuracy']
                replay_forgetting = strategy_metrics['replay']['avg_forgetting']
                naive_forgetting = strategy_metrics['naive']['avg_forgetting']
                
                if replay_acc >= naive_acc:
                    print("  ✅ Replay strategy has higher or equal accuracy than naive")
                else:
                    print("  ⚠️  Replay should perform better than naive (might need more epochs)")
                    
                if replay_forgetting <= naive_forgetting:
                    print("  ✅ Replay strategy has lower or equal forgetting than naive")
                else:
                    print("  ⚠️  Replay should have less forgetting than naive")
                    
            # 3. Accuracy should generally decrease for naive strategy across tasks
            naive_matrix = None
            replay_matrix = None
            
            for strategy_name, strategy_result in scenario_results.items():
                if strategy_result['status'] == 'success':
                    matrix = strategy_result['metrics']['accuracy_matrix']
                    if strategy_name == 'naive':
                        naive_matrix = matrix
                    elif strategy_name == 'replay':
                        replay_matrix = matrix
            
            if naive_matrix and len(naive_matrix) > 1:
                # Check if first task accuracy decreases (forgetting)
                first_task_initial = naive_matrix[0][0]
                first_task_final = naive_matrix[-1][0]
                
                if first_task_final < first_task_initial:
                    print(f"  ✅ Naive strategy shows catastrophic forgetting ({first_task_initial:.1f}% → {first_task_final:.1f}%)")
                else:
                    print(f"  ⚠️  Expected naive strategy to show more forgetting")
                    
            # 4. Print accuracy matrices for visual inspection
            print(f"\n  📊 Accuracy matrices for visual verification:")
            for strategy_name, strategy_result in scenario_results.items():
                if strategy_result['status'] == 'success':
                    matrix = strategy_result['metrics']['accuracy_matrix']
                    print(f"    {strategy_name.upper()}:")
                    for i, row in enumerate(matrix):
                        row_str = " ".join(f"{acc:5.1f}" for acc in row)
                        print(f"      Task {i}: [{row_str}]")


def main():
    """Run the strategy verification test."""
    print("🔬 Starting Continual Learning Strategy Verification...")
    
    try:
        tester = StrategyVerificationTest()
        results = tester.run_comprehensive_test()
        
        print("\n🎉 All tests completed! Check the detailed logs above for verification.")
        return results
        
    except Exception as e:
        print(f"💥 Test suite failed: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 
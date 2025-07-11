#!/usr/bin/env python3
"""
Simple runner script for strategy verification tests.
Usage: python run_verification_test.py [quick|full|task-only|class-only]
"""

import sys
import argparse
from test_strategy_verification import StrategyVerificationTest


def main():
    parser = argparse.ArgumentParser(description='Run continual learning strategy verification tests')
    parser.add_argument(
        'test_type', 
        choices=['quick', 'full', 'task-only', 'class-only', 'naive', 'replay', 'cumulative', 'kaizen'], 
        default='quick',
        nargs='?',
        help='Type of test to run: quick, full, task-only, class-only, or specific strategy (naive, replay, cumulative, kaizen)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print(f"🔬 Running {args.test_type} verification test...")
    
    # Create custom tester for different test types
    if args.test_type == 'quick':
        tester = QuickVerificationTest()
    elif args.test_type == 'full':
        tester = FullVerificationTest()
    elif args.test_type == 'task-only':
        tester = TaskOnlyVerificationTest()
    elif args.test_type == 'class-only':
        tester = ClassOnlyVerificationTest()
    elif args.test_type in ['naive', 'replay', 'cumulative', 'kaizen']:
        tester = SingleStrategyVerificationTest(args.test_type)
    else:
        tester = StrategyVerificationTest()
    
    try:
        results = tester.run_comprehensive_test()
        
        if results:
            print("\n🎉 Test completed successfully!")
            print("📋 Check the detailed logs above for dataflow verification.")
        else:
            print("\n❌ Test failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        return 1
        
    return 0


class QuickVerificationTest(StrategyVerificationTest):
    """Quick test with reduced datasets and epochs."""
    
    def run_comprehensive_test(self):
        """Run quick verification test."""
        print("=" * 80)
        print("🧪 QUICK STRATEGY VERIFICATION TEST")
        print("=" * 80)
        
        # Quick test configuration - only 3 datasets, 5 epochs, separate strategies
        test_configs = [
            {
                'name': 'Quick Naive Strategy Test',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "fashion_mnist", "cifar10"],
                'strategies': ['naive'],
                'epochs': 5,
                'samples_per_class': 100
            },
            {
                'name': 'Quick Replay Strategy Test',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "fashion_mnist", "cifar10"],
                'strategies': ['replay'],
                'epochs': 5,
                'samples_per_class': 100
            },
            {
                'name': 'Quick Cumulative Strategy Test',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "fashion_mnist", "cifar10"],
                'strategies': ['cumulative'],
                'epochs': 5,
                'samples_per_class': 100
            },
            {
                'name': 'Quick Kaizen Strategy Test',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "fashion_mnist", "cifar10"],
                'strategies': ['kaizen'],
                'epochs': 5,
                'samples_per_class': 100
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
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        self._verify_strategy_behaviors(results)
        
        return results
    
    def _create_test_config(self, scenario_type: str, progression: list, 
                           strategies: list) -> dict:
        """Create quick test configuration."""
        config = super()._create_test_config(scenario_type, progression, strategies)
        
        # Override with quick settings
        config.update({
            'epochs': 5,
            'number_of_samples_per_class': 100,
            'analysis_freq': 2,
            'replay_config': {
                'memory_size': 50,
                'replay_batch_ratio': 0.5
            },
            'kaizen_config': {
                'memory_size': 50,
                'ssl_method': 'simclr',
                'kd_weight': 1.0,
                'ssl_weight': 1.0
            }
        })
        
        return config


class FullVerificationTest(StrategyVerificationTest):
    """Full test with all datasets including full M2I progression."""
    
    def run_comprehensive_test(self):
        """Run full verification test."""
        print("=" * 80)
        print("🧪 FULL STRATEGY VERIFICATION TEST (M2I PROGRESSION)")
        print("=" * 80)
        
        # Full test configuration - separate strategies for clarity
        test_configs = [
            # Task Incremental Tests
            {
                'name': 'Full Naive Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive']
            },
            {
                'name': 'Full Replay Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['replay']
            },
            {
                'name': 'Full Cumulative Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['cumulative']
            },
            {
                'name': 'Full Kaizen Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['kaizen']
            },
            # Class Incremental Tests
            {
                'name': 'Full Naive Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive']
            },
            {
                'name': 'Full Replay Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['replay']
            },
            {
                'name': 'Full Kaizen Strategy - Class Incremental M2I',
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
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        self._verify_strategy_behaviors(results)
        
        return results


class TaskOnlyVerificationTest(StrategyVerificationTest):
    """Test only task incremental scenario."""
    
    def run_comprehensive_test(self):
        """Run task incremental only test."""
        print("=" * 80)
        print("🧪 TASK INCREMENTAL ONLY VERIFICATION TEST")
        print("=" * 80)
        
        test_configs = [
            {
                'name': 'Task Incremental - M2I Progression',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive', 'replay', 'cumulative', 'kaizen']
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
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        self._verify_strategy_behaviors(results)
        
        return results


class ClassOnlyVerificationTest(StrategyVerificationTest):
    """Test only class incremental scenario."""
    
    def run_comprehensive_test(self):
        """Run class incremental only test."""
        print("=" * 80)
        print("🧪 CLASS INCREMENTAL ONLY VERIFICATION TEST")
        print("=" * 80)
        
        test_configs = [
            {
                'name': 'Class Incremental - M2I Progression',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': ['naive', 'replay', 'kaizen']
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
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        self._verify_strategy_behaviors(results)
        
        return results


class SingleStrategyVerificationTest(StrategyVerificationTest):
    """Test a single specific strategy with detailed logging."""
    
    def __init__(self, strategy_name: str):
        super().__init__()
        self.strategy_name = strategy_name
        
    def run_comprehensive_test(self):
        """Run verification test for a single strategy."""
        print("=" * 80)
        print(f"🧪 {self.strategy_name.upper()} STRATEGY VERIFICATION TEST")
        print("=" * 80)
        
        # Test both task and class incremental for the specific strategy
        test_configs = [
            {
                'name': f'{self.strategy_name.title()} Strategy - Task Incremental M2I',
                'scenario_type': 'task_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': [self.strategy_name]
            },
            {
                'name': f'{self.strategy_name.title()} Strategy - Class Incremental M2I',
                'scenario_type': 'class_incremental',
                'progression': ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"],
                'strategies': [self.strategy_name]
            }
        ]
        
        # Skip cumulative for class incremental (doesn't make sense)
        if self.strategy_name == 'cumulative':
            test_configs = test_configs[:1]  # Only task incremental
        
        results = {}
        
        for config in test_configs:
            print(f"\n{'🎯 ' + config['name']}")
            print("=" * 60)
            
            try:
                test_result = self._run_scenario_test(config)
                results[config['name']] = test_result
                print(f"✅ {config['name']} completed successfully")
                
                # Print detailed strategy analysis immediately
                self._print_detailed_strategy_analysis(test_result, self.strategy_name)
                
            except Exception as e:
                print(f"❌ {config['name']} failed: {str(e)}")
                results[config['name']] = {'error': str(e)}
        
        # Print summary
        self._print_verification_summary(results)
        self._verify_strategy_behaviors(results)
        
        return results
    
    def _print_detailed_strategy_analysis(self, test_result: dict, strategy_name: str):
        """Print detailed analysis for a single strategy."""
        print(f"\n🔍 DETAILED {strategy_name.upper()} STRATEGY ANALYSIS")
        print("-" * 50)
        
        for strategy, result in test_result.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                
                print(f"\n📊 Performance Metrics:")
                final_metrics = metrics['final_metrics']
                print(f"  • Final Average Accuracy: {final_metrics['avg_accuracy']:.2f}%")
                print(f"  • Average Forgetting: {final_metrics['avg_forgetting']:.2f}%")
                print(f"  • Forward Transfer: {final_metrics['forward_transfer']:.2f}%")
                
                print(f"\n📈 Accuracy Evolution:")
                matrix = metrics['accuracy_matrix']
                for i, row in enumerate(matrix):
                    task_names = ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"][:len(row)]
                    row_str = " ".join(f"{acc:5.1f}" for acc in row)
                    print(f"  After Task {i} ({task_names[i]}): [{row_str}]")
                
                # Strategy-specific analysis
                if strategy_name == 'naive':
                    print(f"\n🧠 Naive Strategy Behavior:")
                    print(f"  • Expected: Catastrophic forgetting should be visible")
                    print(f"  • Observed: {final_metrics['avg_forgetting']:.1f}% average forgetting")
                    
                elif strategy_name == 'replay':
                    print(f"\n🔄 Replay Strategy Behavior:")
                    print(f"  • Expected: Reduced forgetting compared to naive")
                    print(f"  • Memory usage tracked in logs above")
                    if 'memory_evolution' in metrics:
                        total_memory = sum(mem['total_samples'] for mem in metrics['memory_evolution'])
                        print(f"  • Total memory samples stored: {total_memory}")
                        
                elif strategy_name == 'cumulative':
                    print(f"\n📚 Cumulative Strategy Behavior:")
                    print(f"  • Expected: Minimal forgetting (upper bound)")
                    print(f"  • Should show best performance overall")
                    print(f"  • Trains on all data seen so far")
                    
                elif strategy_name == 'kaizen':
                    print(f"\n🌸 Kaizen Strategy Behavior:")
                    print(f"  • Expected: SSL + Knowledge Distillation benefits")
                    print(f"  • Should balance old and new knowledge")
                    print(f"  • Uses self-supervised learning for representation quality")


if __name__ == "__main__":
    sys.exit(main()) 
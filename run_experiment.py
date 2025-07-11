#!/usr/bin/env python3
"""Main script for running experiments with the new configuration system."""

import argparse
import sys
from pathlib import Path

from config.utils import load_experiment_config, create_experiment_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Run continual learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run order analysis with naive strategy
  python run_experiment.py order_analysis --strategy naive

  # Run single task experiment with custom config
  python run_experiment.py single_task --config my_custom.yml

  # Run binary pairs with EWC strategy
  python run_experiment.py binary_pairs --strategy ewc

Available experiment types:
  - order_analysis: Compare different task orders
  - single_task: Train on single dataset
  - binary_pairs: Compare pairs of datasets
  - task_incremental: Task incremental learning

Available strategies:
  - naive: No continual learning strategy
  - ewc: Elastic Weight Consolidation
  - replay: Experience Replay
  - lwf: Learning without Forgetting
  - gem: Gradient Episodic Memory
  - agem: Averaged Gradient Episodic Memory
        """
    )
    
    parser.add_argument(
        'experiment_type',
        nargs='?',  # Make it optional
        choices=['order_analysis', 'single_task', 'binary_pairs', 'task_incremental', 'strategy_comparison'],
        help='Type of experiment to run (can be read from config file)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        default='naive',
        help='Continual learning strategy to use (default: naive)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to custom configuration file for overrides'
    )
    
    parser.add_argument(
        '--cuda',
        type=int,
        default=0,
        help='CUDA device ID (-1 for CPU, default: 0)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='logs',
        help='Output directory for results (default: logs)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running experiment'
    )
    
    args = parser.parse_args()
    
    # Determine experiment type
    experiment_type = args.experiment_type
    
    # If no experiment type provided, try to read from config
    if not experiment_type and args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            experiment_type = config_data.get('experiment_type')
            if experiment_type:
                # Convert to the format expected by our file naming
                type_mapping = {
                    'OrderAnalysisExperiment': 'order_analysis',
                    'SingleTaskExperiment': 'single_task', 
                    'BinaryPairsExperiment': 'binary_pairs',
                    'TaskIncrementalExperiment': 'task_incremental',
                    'StrategyComparisonExperiment': 'strategy_comparison' 
                }
                experiment_type = type_mapping.get(experiment_type, experiment_type)
        except Exception as e:
            print(f"Warning: Could not read experiment_type from config: {e}")
    
    # If still no experiment type, show error
    if not experiment_type:
        print("Error: experiment_type must be provided either as argument or in config file")
        print("Available types: order_analysis, single_task, binary_pairs, task_incremental")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_experiment_config(
            experiment_type=experiment_type,
            strategy=args.strategy,
            custom_config=args.config
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Available experiment configs: {list(Path('configs/experiments').glob('*_base.yml'))}")
        print(f"Available strategy configs: {list(Path('configs/strategies').glob('*.yml'))}")
        sys.exit(1)
    
    # Override with command line arguments
    config['cuda'] = args.cuda
    config['output_dir'] = args.output_dir
    
    if args.dry_run:
        print("Configuration:")
        print("=" * 50)
        import yaml
        print(yaml.dump(config, default_flow_style=False, indent=2))
        return
    
    # Create and run experiment
    try:
        experiment = create_experiment_from_config(config)
        print(f"Starting {experiment_type} experiment with {args.strategy} strategy...")
        results = experiment.run()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {experiment.output_dir}")
        
        # Print summary results
        if 'comparison' in results:
            print("\nExperiment Summary:")
            print("-" * 30)
            for order_name, metrics in results['comparison']['summary_table'].items():
                print(f"{order_name}: {metrics['avg_accuracy']:.1f}% accuracy, "
                      f"{metrics['avg_forgetting']:.1f}% forgetting")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
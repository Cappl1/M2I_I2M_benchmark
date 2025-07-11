"""Configuration utilities for loading and merging experiment configs."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from paths import ROOT_PATH


def load_config(config_path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(str(config_path), 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_experiment_config(experiment_type: str, strategy: Optional[str] = None, 
                          custom_config: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration with optional strategy and custom overrides."""
    
    # Load base experiment config
    base_config_path = Path(f'configs/experiments/{experiment_type}_base.yml')
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    config = load_config(base_config_path)
    
    # Load strategy-specific config if provided
    if strategy:
        strategy_config_path = Path(f'configs/strategies/{strategy}.yml')
        if strategy_config_path.exists():
            strategy_config = load_config(strategy_config_path)
            config = merge_configs(config, strategy_config)
    
    # Load custom config overrides if provided
    if custom_config:
        custom_config_path = Path(custom_config)
        if custom_config_path.exists():
            custom_overrides = load_config(custom_config_path)
            config = merge_configs(config, custom_overrides)
    
    return config


def create_experiment_from_config(config: Dict[str, Any]):
    """Create an experiment instance from configuration."""
    experiment_type = config.get('experiment_type', 'OrderAnalysisExperiment')
    
    if experiment_type == 'OrderAnalysisExperiment':
        from experiments.order_analysis_exp import OrderAnalysisExperiment
        return OrderAnalysisExperiment(config)
    elif experiment_type == 'SingleTaskExperiment':
        from experiments.single_task_exp import SingleTaskExperiment
        return SingleTaskExperiment(config)
    elif experiment_type == 'BinaryPairsExperiment':
        from experiments.binary_pairs_exp import BinaryPairsExperiment
        return BinaryPairsExperiment(config)
    elif experiment_type == 'StrategyComparisonExperiment':
        from experiments.strategy_comparison import StrategyComparisonExperiment
        return StrategyComparisonExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to a YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

# core/experiment.py
"""Base experiment class that all experiments inherit from."""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics, 
    bwt_metrics, forward_transfer_metrics
)
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin

from models.model_provider import parse_model_name
from scenarios.scenarios_providers import parse_scenario
from strategies.strategies_provider import parse_strategy_name
from analysis.vit_class_projection import ViTClassProjectionAnalyzer


class BaseExperiment(ABC):
    """Base class for all continual learning experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = self._setup_output_dir()
        
        # Core components
        self.model = None
        self.scenario = None
        self.strategy = None
        self.analyzer = None
        
        # Results storage
        self.results = {
            'config': config,
            'metrics': {},
            'representations': {},
            'checkpoints': {}
        }
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.get('cuda', -1) >= 0:
            device = torch.device(f'cuda:{self.config["cuda"]}')
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")
        return device
        
    def _setup_output_dir(self) -> Path:
        """Create output directory for experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get('experiment_name', 'experiment')
        
        output_dir = Path(self.config.get('output_dir', 'logs')) / f"{exp_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        return output_dir
        
    def build_model(self) -> torch.nn.Module:
        """Build model from config."""
        # Convert config dict to args-like object for compatibility
        class Args:
            pass
        args = Args()
        for k, v in self.config.items():
            setattr(args, k, v)
            
        model = parse_model_name(args)
        return model.to(self.device)
        
    def build_scenario(self):
        """Build scenario from config."""
        class Args:
            pass
        args = Args()
        for k, v in self.config.items():
            setattr(args, k, v)
            
        return parse_scenario(args)
        
    def build_strategy(self, model: torch.nn.Module, eval_plugin: EvaluationPlugin):
        """Build strategy from config."""
        class Args:
            pass
        args = Args()
        for k, v in self.config.items():
            setattr(args, k, v)
            
        # Setup optimizer and criterion
        optimizer = self._build_optimizer(model)
        criterion = torch.nn.CrossEntropyLoss()
        
        return parse_strategy_name(
            args=args,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            eval_plugin=eval_plugin
        )
        
    def _build_optimizer(self, model: torch.nn.Module):
        """Build optimizer from config."""
        opt_type = self.config.get('optimizer', 'sgd').lower()
        lr = self.config.get('lr', 0.001)
        
        if opt_type == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
            
    def build_eval_plugin(self) -> EvaluationPlugin:
        """Build evaluation plugin with simple, clean logging."""
        from avalanche.logging import TextLogger
        
        # Create a simple custom logger that only logs what we want
        class CleanLogger:
            def __init__(self, log_file):
                self.log_file = log_file
                with open(log_file, 'w') as f:
                    f.write("epoch,train_acc,train_loss,eval_acc,eval_loss\n")
                self.current_epoch = 0
                self.train_acc = 0
                self.train_loss = 0
                
            def after_training_epoch(self, strategy, metric_values):
                self.current_epoch += 1
                # Extract training metrics
                for name, value in metric_values.items():
                    if 'Top1_Acc_Epoch' in name and 'train' in name:
                        self.train_acc = value
                    elif 'Loss_Epoch' in name and 'train' in name:
                        self.train_loss = value
                        
            def after_eval_exp(self, strategy, metric_values):
                # Extract eval metrics and write to file
                eval_acc = 0
                eval_loss = 0
                for name, value in metric_values.items():
                    if 'Top1_Acc_Exp' in name and 'eval' in name:
                        eval_acc = value
                    elif 'Loss_Exp' in name and 'eval' in name:
                        eval_loss = value
                
                # Write to CSV
                with open(self.log_file, 'a') as f:
                    f.write(f"{self.current_epoch},{self.train_acc:.4f},{self.train_loss:.4f},{eval_acc:.4f},{eval_loss:.4f}\n")
                
                print(f"Epoch {self.current_epoch}: Train Acc: {self.train_acc:.3f}, Eval Acc: {eval_acc:.3f}")
                
            # Dummy methods for other callbacks Avalanche might call
            def before_training(self, *args, **kwargs): pass
            def after_training(self, *args, **kwargs): pass  
            def before_training_exp(self, *args, **kwargs): pass
            def after_training_exp(self, *args, **kwargs): pass
            def before_eval(self, *args, **kwargs): pass
            def after_eval(self, *args, **kwargs): pass
            def before_eval_exp(self, *args, **kwargs): pass
        
        # Create clean logger
        clean_logger = CleanLogger(self.output_dir / 'training_log.csv')
        
        return EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True),
            loss_metrics(epoch=True, experience=True),
            loggers=[clean_logger]
        )
        
    def build_analyzer(self) -> Optional[ViTClassProjectionAnalyzer]:
        """Build representation analyzer if model is ViT."""
        if self.model is None:
            return None
            
        # Check if model is ViT
        if hasattr(self.model, 'vit') or 'vit' in self.config.get('model_name', '').lower():
            return ViTClassProjectionAnalyzer(self.model, self.device)
        return None
        
    def setup(self):
        """Setup experiment components."""
        self.model = self.build_model()
        self.scenario = self.build_scenario()
        eval_plugin = self.build_eval_plugin()
        self.strategy = self.build_strategy(self.model, eval_plugin)
        self.analyzer = self.build_analyzer()
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment. Must be implemented by subclasses."""
        pass
        
    def save_results(self):
        """Save experiment results."""
        # Save main results JSON
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save model checkpoint if requested
        if self.config.get('save_model', True):
            checkpoint_path = self.output_dir / 'model_final.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, checkpoint_path)
            self.results['checkpoints']['final'] = str(checkpoint_path)
            
        print(f"\nResults saved to: {self.output_dir}")
        
    def analyze_representations(self, experience, task_id: int, prefix: str = ""):
        """Analyze representations for a given experience."""
        if self.analyzer is None:
            return {}
            
        from torch.utils.data import DataLoader
        loader = DataLoader(
            experience.dataset,
            batch_size=self.config.get('analysis_batch_size', 64),
            shuffle=False
        )
        
        scores = self.analyzer.analyze_task_representations(
            loader,
            task_id=task_id,
            num_classes_per_task=self.config.get('num_classes_per_task', 10)
        )
        
        # Store with prefix
        key = f"{prefix}_task{task_id}" if prefix else f"task{task_id}"
        return {key: scores}
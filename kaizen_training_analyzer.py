#!/usr/bin/env python3
"""
Fixed debug script to properly test why Kaizen underperforms
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append('/home/brothen/M2I_I2M_benchmark')

# Import exactly the same components as the main experiment
from core.strategies import NaiveStrategy, KaizenStrategy

class FixedKaizenDebug:
    """Debug Kaizen performance issues with proper training setup."""
    
    def __init__(self):
        # Match the EXACT config from your working bash scripts
        self.config = {
            'experiment_type': 'StrategyBinaryPairsExperiment',
            'model_name': 'ViT64',
            'num_classes': 10,  # Single task MNIST
            'lr': 0.001,  # Match the multi-strategy script that works
            'optimizer': 'adam',
            'epochs': 50,  # Full epochs like in real experiments
            'minibatch_size': 128,  # Match the multi-strategy script
            'cuda': 1,
            'dataset_a': 'cifar10',
            'dataset_b': 'cifar10',
            'scenario_type': 'class_incremental',
            'balanced': 'balanced',
            'number_of_samples_per_class': 500,
            'strategy_name': 'Naive',  # Start with Naive
            'output_dir': './debug_kaizen_fixed'
        }
        
        # Initialize exactly like BaseExperiment
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def build_model(self):
        """Build model exactly like BaseExperiment."""
        from models.vit_models import ViT64SingleHead
        
        model = ViT64SingleHead(num_classes=self.config['num_classes'])
        model.to(self.device)
        return model
    
    def _load_dataset(self, dataset_name):
        if dataset_name == 'cifar10':
            from scenarios.datasets.cifar import load_resized_cifar10  # or similar
            return load_resized_cifar10(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.config.get('number_of_samples_per_class')
            )
    
    
    def test_naive_strategy_proper(self):
        """Test Naive strategy with proper setup (no epoch splitting)."""
        print("\n🔬 Testing Naive Strategy (Proper Setup)...")
        
        # Create fresh model
        model = self.build_model()
        
        # Load data
        train_ds, test_ds = self._load_dataset('cifar10')
        
        # Create strategy with proper config
        strategy = NaiveStrategy(model, self.config, str(self.device))
        
        # Train ONCE for full epochs (like in real experiments)
        print(f"  Training for {self.config['epochs']} epochs...")
        result = strategy.train_task(train_ds, task_id=0)
        
        # Evaluate
        test_acc = strategy._evaluate_dataset(test_ds)
        print(f"  Final test accuracy: {test_acc:.1f}%")
        
        return test_acc, result
    
    def test_kaizen_strategy_proper(self):
        """Test Kaizen strategy with proper setup."""
        print("\n🔬 Testing Kaizen Strategy (Proper Setup)...")
        
        # Create fresh model
        model = self.build_model()
        
        # Update config for Kaizen (matching your bash script)
        kaizen_config = self.config.copy()
        kaizen_config.update({
            'strategy_name': 'Kaizen',
            'ssl_method': 'simclr',  # Match your bash script
            'memory_size_percent': 1,
            'kd_classifier_weight': 2.0,
            'lr': 0.001,  # Kaizen uses higher LR in your script
            'minibatch_size': 128,  # Kaizen uses smaller batch in your script
            'epochs': 50  # Kaizen uses more epochs in your script
        })
        
        # Create strategy
        strategy = KaizenStrategy(model, kaizen_config, str(self.device))
        
        # Load data
        train_ds, test_ds = self._load_dataset('cifar10')
        
        # Train
        print(f"  Training for {kaizen_config['epochs']} epochs...")
        print(f"  SSL method: {kaizen_config['ssl_method']}")
        print(f"  LR: {kaizen_config['lr']}, Batch size: {kaizen_config['minibatch_size']}")
        
        result = strategy.train_task(train_ds, task_id=0)
        
        # Evaluate
        test_acc = strategy._evaluate_dataset(test_ds)
        print(f"  Final test accuracy: {test_acc:.1f}%")
        
        # Show loss components
        if 'loss_components' in result:
            loss_comp = result['loss_components']
            print(f"\n  Final loss components:")
            print(f"    Total: {loss_comp['total']:.3f}")
            print(f"    SSL_CT: {loss_comp['ssl_ct']:.3f}")
            print(f"    CE_CT: {loss_comp['ce_ct']:.3f}")
            print(f"    SSL_KD: {loss_comp['ssl_kd']:.3f}")
            print(f"    CE_KD: {loss_comp['ce_kd']:.3f}")
        
        return test_acc, result
    
    def debug_kaizen_training_loop(self):
        """Debug Kaizen's training loop in detail."""
        print("\n🔬 Debugging Kaizen Training Loop...")
        
        # Create model and strategy
        model = self.build_model()
        
        kaizen_config = self.config.copy()
        kaizen_config.update({
            'strategy_name': 'Kaizen',
            'ssl_method': 'moco',
            'memory_size_percent': 1,
            'kd_classifier_weight': 2.0,
            'lr': 0.001,
            'minibatch_size': 128,
            'epochs': 5  # Just a few epochs for debugging
        })
        
        strategy = KaizenStrategy(model, kaizen_config, str(self.device))
        
        # Load data
        train_ds, test_ds = self._load_dataset('cifar10')
        
        # Manually create dataloader to inspect
        train_loader = strategy._create_replay_dataloader(train_ds, task_id=0)
        
        print("  Checking dataloader...")
        batch = next(iter(train_loader))
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Current X shape: {batch['current_x'].shape}")
        print(f"  Has replay mask: {batch['has_replay'].sum().item()}/{len(batch['has_replay'])}")
        
        # Check SSL augmentation
        print("\n  Testing SSL augmentation...")
        x_sample = batch['current_x'][:4].to(self.device)
        x_aug1 = strategy._apply_ssl_augmentation(x_sample)
        x_aug2 = strategy._apply_ssl_augmentation(x_sample)
        
        print(f"  Original range: [{x_sample.min():.2f}, {x_sample.max():.2f}]")
        print(f"  Aug1 range: [{x_aug1.min():.2f}, {x_aug1.max():.2f}]")
        print(f"  Aug2 range: [{x_aug2.min():.2f}, {x_aug2.max():.2f}]")
        print(f"  Aug difference: {(x_aug1 - x_aug2).abs().mean():.4f}")
        
        # Test feature extraction
        print("\n  Testing feature extraction...")
        with torch.no_grad():
            features = strategy._extract_features(x_sample)
            print(f"  Feature shape: {features.shape}")
            print(f"  Feature norm: {features.norm(dim=1).mean():.2f}")
        
        # Test SSL projection
        print("\n  Testing SSL projection...")
        with torch.no_grad():
            proj = strategy.projector_ssl(features)
            print(f"  Projection shape: {proj.shape}")
            print(f"  Projection norm: {proj.norm(dim=1).mean():.2f}")
        
        # Test one training step
        print("\n  Testing one training step...")
        optimizer = strategy._build_optimizer()
        loss_dict = strategy._train_epoch_kaizen(train_loader, optimizer, task_id=0)
        
        print("  Loss components:")
        for k, v in loss_dict.items():
            print(f"    {k}: {v:.4f}")
        
        return loss_dict
    
    
    def run_debug(self):
        """Run complete debugging suite."""
        
        
        # 2. Test Naive strategy with proper setup
        naive_acc, _ = self.test_naive_strategy_proper()
        

        # Debug the training loop
        self.debug_kaizen_training_loop()
        # 3. Test Kaizen strategy
        kaizen_acc, _ = self.test_kaizen_strategy_proper()
        
       
            
        
        
        

if __name__ == "__main__":
    debugger = FixedKaizenDebug()
    debugger.run_debug()
"""Single task experiment with optional patch-level importance analysis."""

import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader
import csv
import os
import json
from pathlib import Path

from core.experiment import BaseExperiment
from analysis.vit_class_projection import ViTClassProjectionAnalyzer
from analysis.patch_importance_analyzer import ViTPatchImportanceAnalyzer


class SingleTaskExperiment(BaseExperiment):
    """Train on a single dataset with optional patch-level analysis."""
    
    def setup(self):
        """Setup experiment components - manual training, no strategies."""
        self.model = self.build_model()
        # Skip scenario and strategy creation for single task
        self.scenario = None
        self.strategy = None
        
        # Setup analyzers
        self.analyzer = ViTClassProjectionAnalyzer(self.model, str(self.device))
        
        # Setup patch analyzer if enabled
        self.enable_patch_analysis = self.config.get('patch_analysis', False)
        if self.enable_patch_analysis:
            print("  → Patch-level analysis ENABLED")
            self.patch_analyzer = ViTPatchImportanceAnalyzer(self.model, str(self.device))
            self.patch_analysis_freq = self.config.get('patch_analysis_freq', 20)  # Less frequent than regular analysis
        else:
            self.patch_analyzer = None
    
    def run(self) -> Dict[str, Any]:
        """Run single task training with manual loop."""
        self.setup()
        
        dataset_name = self.config.get('dataset', 'mnist')
        
        print(f"\n=== Training on {dataset_name.upper()} ===")
        if self.enable_patch_analysis:
            print(f"  → Patch analysis will run every {self.patch_analysis_freq} epochs")
        
        # Load dataset directly
        train_dataset, test_dataset = self._load_single_dataset(dataset_name)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('minibatch_size', 128), 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.get('minibatch_size', 128), 
            shuffle=False
        )
        
        # Setup training
        optimizer = self._build_optimizer(self.model)
        criterion = nn.CrossEntropyLoss()
        
        # Create simple CSV logger
        log_file = self.output_dir / 'training_log.csv'
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_acc', 'train_loss', 'eval_acc', 'eval_loss'])
        
        # Create analysis directories
        analysis_dir = self.output_dir / 'layer_analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        if self.enable_patch_analysis:
            patch_analysis_dir = self.output_dir / 'patch_analysis'
            patch_analysis_dir.mkdir(exist_ok=True)
            
            # Create patch summary file
            patch_summary_file = patch_analysis_dir / 'patch_importance_summary.json'
            patch_summary = {
                'dataset': dataset_name,
                'config': {
                    'patch_analysis_freq': self.patch_analysis_freq,
                    'num_classes': self.config.get('num_classes', 10),
                    'epochs': self.config.get('epochs', 10)
                },
                'epochs_analyzed': []
            }
        
        print("Starting training...")
        best_acc = 0.0
        analysis_freq = self.config.get('analysis_freq', 10)
        
        # Training loop
        for epoch in range(self.config.get('epochs', 10)):
            # Train
            train_acc, train_loss = self._train_epoch(train_loader, optimizer, criterion)
            
            # Evaluate
            eval_acc, eval_loss = self._evaluate_epoch(test_loader, criterion)
            
            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f'{train_acc:.4f}', f'{train_loss:.4f}', 
                               f'{eval_acc:.4f}', f'{eval_loss:.4f}'])
            
            # Print clean progress
            print(f"Epoch {epoch + 1:2d}: Train Acc: {train_acc:.3f}, Train Loss: {train_loss:.3f}, "
                  f"Eval Acc: {eval_acc:.3f}, Eval Loss: {eval_loss:.3f}")
            
            # Standard ViT projection analysis
            if (epoch + 1) % analysis_freq == 0:
                print(f"  → Running ViT class projection analysis...")
                try:
                    analysis_results = self._run_vit_analysis(test_loader, epoch + 1)
                    
                    # Save analysis results
                    analysis_file = analysis_dir / f'epoch_{epoch + 1:03d}.json'
                    with open(analysis_file, 'w') as f:
                        json.dump(analysis_results, f, indent=2)
                        
                    print(f"  → Analysis saved to {analysis_file}")
                except Exception as e:
                    print(f"  → Analysis failed: {e}")
            
            # Patch-level analysis (less frequent)
            if self.enable_patch_analysis and (epoch + 1) % self.patch_analysis_freq == 0:
                print(f"  → Running patch importance analysis...")
                try:
                    patch_results = self._run_patch_analysis(test_loader, epoch + 1, dataset_name)
                    
                    if patch_results:
                        patch_summary['epochs_analyzed'].append({
                            'epoch': epoch + 1,
                            'eval_accuracy': eval_acc,
                            'summary_stats': patch_results.get('statistics', {})
                        })
                        
                        # Update summary file
                        with open(patch_summary_file, 'w') as f:
                            json.dump(patch_summary, f, indent=2)
                        
                        print(f"  → Patch analysis completed and saved")
                except Exception as e:
                    print(f"  → Patch analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Track best accuracy
            if eval_acc > best_acc:
                best_acc = eval_acc
        
        print(f"\nTraining completed! Best accuracy: {best_acc:.3f}")
        
        # Store results
        self.results.update({
            'dataset': dataset_name,
            'final_accuracy': eval_acc,
            'best_accuracy': best_acc,
            'log_file': str(log_file),
            'analysis_dir': str(analysis_dir),
            'patch_analysis_enabled': self.enable_patch_analysis
        })
        
        if self.enable_patch_analysis:
            self.results['patch_analysis_dir'] = str(patch_analysis_dir)
            self.results['patch_analysis_summary'] = str(patch_summary_file)
        
        self.save_results()
        return self.results
    
    def _run_vit_analysis(self, test_loader, epoch):
        """Run OPTIMIZED ViT class projection analysis on the test set."""
        projection_scores = self.analyzer.analyze_task_representations(
            test_loader, 
            task_id=0, 
            num_classes_per_task=self.config.get('num_classes', 10),
            max_batches=50,  # Only analyze first 50 batches
            sample_tokens=True  # Sample 10 tokens instead of all 196
        )
        return {
            'epoch': epoch,
            'analyzer_type': 'vit_projection',
            'projection_scores': projection_scores
        }
    
    def _run_patch_analysis(self, test_loader, epoch, dataset_name):
        """Run detailed patch-level importance analysis."""
        if not self.patch_analyzer:
            return None
        
        # Run patch importance analysis
        max_batches = self.config.get('patch_analysis_max_batches', 100)  # Analyze more batches for stability
        patch_results = self.patch_analyzer.analyze_patch_importance(
            test_loader,
            num_classes=self.config.get('num_classes', 10),
            max_batches=max_batches
        )
        
        # Create visualizations
        patch_analysis_dir = self.output_dir / 'patch_analysis'
        self.patch_analyzer.visualize_patch_importance(
            patch_results,
            patch_analysis_dir,
            epoch,
            dataset_name
        )
        
        # Save detailed results
        self.patch_analyzer.save_detailed_results(
            patch_results,
            patch_analysis_dir,
            epoch
        )
        
        return patch_results
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle variable batch sizes (data, target) or (data, target, task_id)
            if len(batch) == 2:
                data, target = batch
            else:
                data, target = batch[0], batch[1]  # Ignore task_id if present
                
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return accuracy, avg_loss
    
    def _evaluate_epoch(self, test_loader, criterion):
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.inference_mode():
            for batch in test_loader:
                # Handle variable batch sizes (data, target) or (data, target, task_id)
                if len(batch) == 2:
                    data, target = batch
                else:
                    data, target = batch[0], batch[1]  # Ignore task_id if present
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        return accuracy, avg_loss
    
    def _load_single_dataset(self, dataset_name: str):
        """Load a single dataset by name - FIXED for omniglot and imagenet."""
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'mnist':
            from scenarios.datasets.mnist import load_mnist_with_resize
            return load_mnist_with_resize(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'fashion_mnist':
            from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
            return load_fashion_mnist_with_resize(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'cifar10':
            from scenarios.datasets.cifar import load_resized_cifar10
            return load_resized_cifar10(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'svhn':
            from scenarios.datasets.svhn import load_svhn_resized
            return load_svhn_resized(
                balanced=self.config.get('balanced', False),
                number_of_samples_per_class=self.config.get('number_of_samples_per_class')
            )
        elif dataset_name == 'omniglot':
            # Load omniglot the same way as in scenarios_providers.py
            from scenarios.datasets.omniglot import _load_omniglot
            from torchvision.transforms import Resize, Compose, ToTensor, Normalize
            from scenarios.utils import transform_from_gray_to_rgb, filter_classes
            
            transform_with_resize = Compose([
                ToTensor(),
                Resize((64, 64)),
                transform_from_gray_to_rgb(),
                Normalize(mean=(0.9221,), std=(0.2681,))
            ])
            
            # Load full omniglot dataset
            balanced = self.config.get('balanced', False)
            samples_per_class = self.config.get('number_of_samples_per_class')
            
            train_omniglot_full, test_omniglot_full = _load_omniglot(
                transform_with_resize, 
                balanced=balanced, 
                number_of_samples_per_class=samples_per_class
            )
            
            # Filter to only 10 classes (0-9)
            train_omniglot, test_omniglot = filter_classes(
                train_omniglot_full, test_omniglot_full, 
                classes=list(range(10))
            )
            
            return train_omniglot, test_omniglot
            
        elif dataset_name == 'imagenet':
            # Load imagenet the same way as in scenarios_providers.py
            from scenarios.datasets.load_imagenet import load_imagenet
            from scenarios.utils import filter_classes
            
            balanced = self.config.get('balanced', False)
            samples_per_class = self.config.get('number_of_samples_per_class')
            
            # Load full imagenet
            train_imagenet_full, test_imagenet_full = load_imagenet(
                balanced=balanced, 
                number_of_samples_per_class=samples_per_class
            )
            
            # Filter to only 10 classes (0-9)
            train_imagenet, test_imagenet = filter_classes(
                train_imagenet_full, test_imagenet_full,
                classes=list(range(10))
            )
            
            return train_imagenet, test_imagenet
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


class SimpleStream:
    """Simple stream wrapper for Avalanche compatibility."""
    def __init__(self, name: str = "single_task_stream"):
        self.name = name


class SimpleExperience:
    """Simple experience wrapper for single task training - Avalanche compatible."""
    
    def __init__(self, dataset, task_label: int = 0, exp_id: int = 0):
        self.dataset = dataset
        self.task_label = task_label
        self.experience_id = exp_id
        
        # Avalanche compatibility attributes
        self.current_experience = exp_id  # Required by Avalanche logging
        self.task_id = task_label
        self.origin_stream = SimpleStream("single_task")  # Required for logging
        
        # Try to extract classes from dataset
        if hasattr(dataset, 'targets'):
            unique_classes = sorted(set(dataset.targets))
        elif hasattr(dataset, '_targets'):
            unique_classes = sorted(set(dataset._targets))
        else:
            # Fallback: scan through dataset
            targets = []
            for _, target in dataset:
                targets.append(target)
                if len(targets) > 1000:  # Don't scan too much
                    break
            unique_classes = sorted(set(targets))
        
        self.classes_in_this_experience = unique_classes
        self.task_labels = [task_label] * len(dataset)
        
        # Additional Avalanche compatibility
        self.classes_seen_so_far = set(unique_classes)
        self.previous_classes = set()  # For single task, no previous classes
        
    def __len__(self):
        return len(self.dataset)
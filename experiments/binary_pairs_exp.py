"""Binary task pairs experiment with evolution tracking AND patch analysis - FULL VERSION."""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from torch.utils.data import ConcatDataset, DataLoader
import csv
import json
from pathlib import Path

from core.experiment import BaseExperiment


class BinaryPairsExperiment(BaseExperiment):
    """Train on pairs of tasks and track representation evolution with optional patch analysis."""
    
    def setup(self):
        """Setup experiment components."""
        # Set num_classes dynamically based on scenario type
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        if scenario_type == 'class_incremental':
            # Single head with 20 classes (10 from each dataset)
            self.config['num_classes'] = 20
        else:  # task_incremental
            # Multiple heads, 10 classes per head
            self.config['num_classes'] = 10
            self.config['num_tasks'] = 2  # Ensure we have 2 heads for 2 tasks
        
        self.model = self.build_model()
        
        # Build scenario for getting task data using a custom approach for binary pairs
        self.scenario = self._build_binary_scenario()
        
        # Don't use Avalanche strategy - manual training for more control
        self.strategy = None
        
        # Use the SAME analyzer as single task experiment
        from analysis.vit_class_projection import ViTClassProjectionAnalyzer
        self.analyzer = ViTClassProjectionAnalyzer(self.model, str(self.device))
        
        # ADD PATCH ANALYSIS SUPPORT
        self.enable_patch_analysis = self.config.get('patch_analysis', False)
        if self.enable_patch_analysis:
            print("  → Patch-level analysis ENABLED for binary pairs")
            from analysis.patch_importance_analyzer import ViTPatchImportanceAnalyzer
            self.patch_analyzer = ViTPatchImportanceAnalyzer(self.model, str(self.device))
            self.patch_analysis_freq = self.config.get('patch_analysis_freq', 20)
        else:
            self.patch_analyzer = None
            print("  → Patch-level analysis DISABLED")
    
    def _build_binary_scenario(self):
        """Build a scenario specifically for binary pairs experiments with two different datasets."""
        from scenarios.scenarios_providers import create_scenario
        
        # For binary pairs, we need two different datasets
        dataset_a = self.config.get('dataset_a', 'cifar10')
        dataset_b = self.config.get('dataset_b', 'mnist')
        
        train_ds1, test_ds1 = self._load_dataset(dataset_a)  # Task 0
        train_ds2, test_ds2 = self._load_dataset(dataset_b)  # Task 1
        
        class_incremental = self.config.get('scenario_type', 'class_incremental') == 'class_incremental'
        
        if class_incremental:
            # For class incremental: relabel second dataset to classes 10-19
            train_ds2 = self._relabel_dataset(train_ds2, offset=10)
            test_ds2 = self._relabel_dataset(test_ds2, offset=10)
        
        # Create Avalanche scenario with both datasets as separate tasks
        return create_scenario(
            train_dataset=[train_ds1, train_ds2],
            test_dataset=[test_ds1, test_ds2],
            class_incremental=class_incremental
        )
    
    def _relabel_dataset(self, dataset, offset):
        """Relabel dataset classes by adding an offset (for class incremental learning)."""
        class RelabeledDataset:
            def __init__(self, base_dataset, offset):
                self.base_dataset = base_dataset
                self.offset = offset
                # Update targets if they exist
                if hasattr(base_dataset, 'targets'):
                    self.targets = [t + offset for t in base_dataset.targets]
                else:
                    self.targets = None
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                # Handle variable return values (x, y) or (x, y, task_id)
                item = self.base_dataset[idx]
                if len(item) == 2:
                    x, y = item
                    return x, y + self.offset
                elif len(item) == 3:
                    x, y, task_id = item
                    return x, y + self.offset, task_id
                else:
                    raise ValueError(f"Unexpected item format: {len(item)} elements")
        
        return RelabeledDataset(dataset, offset)
    
    def _load_dataset(self, dataset_name):
        """Load a dataset by name - using the same approach as SingleTaskExperiment."""
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
    
    def run(self) -> Dict[str, Any]:
        """Run binary pairs training with periodic analysis."""
        self.setup()
        
        task_x, task_y = self.config.get('task_pair', [0, 1])
        analysis_freq = self.config.get('analysis_freq', 10)
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        
        # Get dataset names for better logging
        dataset_a = self.config.get('dataset_a', 'cifar10')
        dataset_b = self.config.get('dataset_b', 'mnist')
        
        print(f"\n=== Training on {dataset_a.upper()} → {dataset_b.upper()} ({scenario_type}) ===")
        if self.enable_patch_analysis:
            print(f"  → Patch analysis will run every {self.patch_analysis_freq} epochs")
        
        # Get task experiences
        train_x = self.scenario.train_stream[task_x]
        train_y = self.scenario.train_stream[task_y]
        test_x = self.scenario.test_stream[task_x]
        test_y = self.scenario.test_stream[task_y]
        
        print(f"Task {task_x} ({dataset_a}) classes: {train_x.classes_in_this_experience}")
        print(f"Task {task_y} ({dataset_b}) classes: {train_y.classes_in_this_experience}")
        
        # CREATE PATCH ANALYSIS SUMMARY if enabled
        if self.enable_patch_analysis:
            patch_summary = self._create_patch_summary(dataset_a, dataset_b, scenario_type)
        else:
            patch_summary = None
        
        # Sequential training: Task X first, then Task Y
        self._train_sequential(train_x, train_y, test_x, test_y, task_x, task_y, analysis_freq, scenario_type, dataset_a, dataset_b, patch_summary)
        
        # Store results
        self.results['task_pair'] = [task_x, task_y]
        self.results['datasets'] = [dataset_a, dataset_b]
        self.results['scenario_type'] = scenario_type
        self.results['patch_analysis_enabled'] = self.enable_patch_analysis
        
        # SAVE FINAL PATCH SUMMARY
        if self.enable_patch_analysis and patch_summary:
            patch_summary_file = self.output_dir / 'patch_analysis' / 'patch_importance_summary.json'
            with open(patch_summary_file, 'w') as f:
                json.dump(patch_summary, f, indent=2)
            print(f"  → Patch analysis summary saved to {patch_summary_file}")
        
        self.save_results()
        return self.results
    
    def _create_patch_summary(self, dataset_a, dataset_b, scenario_type):
        """Create patch analysis summary structure."""
        return {
            'datasets': [dataset_a, dataset_b],
            'scenario_type': scenario_type,
            'config': {
                'patch_analysis_freq': self.patch_analysis_freq,
                'num_classes': self.config.get('num_classes', 10),
                'epochs': self.config.get('epochs', 50),
                'patch_analysis_max_batches': self.config.get('patch_analysis_max_batches', 100)
            },
            'phases': {
                'phase1': {
                    'dataset': dataset_a, 
                    'task_id': 0,
                    'epochs_analyzed': []
                },
                'phase2': {
                    'dataset': dataset_b, 
                    'task_id': 1,
                    'epochs_analyzed': []
                }
            }
        }
    
    def _train_sequential(self, train_x, train_y, test_x, test_y, task_x, task_y, analysis_freq, scenario_type, dataset_a, dataset_b, patch_summary):
        """Train sequentially: Task X first, then Task Y, with continual evaluation."""
        print(f"\n--- Sequential Training with {'Patch Analysis' if self.enable_patch_analysis else 'Representation Analysis'} ({scenario_type}) ---")
        
        # CREATE PATCH ANALYSIS DIRECTORY if enabled
        if self.enable_patch_analysis:
            patch_dir = self.output_dir / 'patch_analysis'
            patch_dir.mkdir(exist_ok=True)
        
        # Phase 1: Train on Task X only
        print(f"\nPhase 1: Training on {dataset_a.upper()} only")
        self._train_single_task_phase(
            train_x, test_x, task_x, "phase1", analysis_freq, 
            phase=1, dataset_name=dataset_a, patch_summary=patch_summary
        )
        
        # Phase 2: Train on Task Y, evaluate on both tasks (THIS IS THE KEY FIX)
        print(f"\nPhase 2: Training on {dataset_b.upper()}, evaluating forgetting on {dataset_a.upper()}")
        evolution_results = self._train_single_task_phase(
            train_y, test_y, task_y, "phase2", analysis_freq, phase=2, 
            also_test_on=(test_x, task_x), dataset_name=dataset_b, patch_summary=patch_summary
        )
        
        self.results['evolution'] = evolution_results
    
    def _train_single_task_phase(self, train_exp, test_exp, task_id, phase_name, analysis_freq, phase=1, also_test_on=None, dataset_name=None, patch_summary=None):
        """Train on a single task phase with optional patch analysis."""
        optimizer = self._build_optimizer(self.model)
        criterion = nn.CrossEntropyLoss()
        
        # Create analysis directory
        analysis_dir = self.output_dir / 'layer_analysis' / phase_name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV logger for this phase
        log_file = self.output_dir / f'training_log_{phase_name}.csv'
        headers = ['epoch', 'train_acc', 'train_loss', f'task_{task_id}_acc']
        if also_test_on:
            headers.append(f'task_{also_test_on[1]}_acc')
        
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        evolution_results = {'accuracy': {}, 'representations': {}}
        
        # Training loop
        train_loader = DataLoader(train_exp.dataset, batch_size=self.config.get('minibatch_size', 128), shuffle=True)
        
        for epoch in range(self.config.get('epochs', 50)):
            # Train one epoch
            train_acc, train_loss = self._train_epoch_on_loader(train_loader, optimizer, criterion)
            
            # Evaluate on current task
            current_acc = self._evaluate_task_unified(test_exp)
            
            # Also evaluate on previous task if in phase 2
            row_data = [epoch + 1, f'{train_acc:.4f}', f'{train_loss:.4f}', f'{current_acc:.4f}']
            log_msg = f"Epoch {epoch + 1:2d}: Train: {train_acc:.3f}, Task {task_id}: {current_acc:.3f}"
            
            if also_test_on:
                prev_acc = self._evaluate_task_unified(also_test_on[0])
                row_data.append(f'{prev_acc:.4f}')
                log_msg += f", Task {also_test_on[1]}: {prev_acc:.3f}"
            
            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            print(log_msg)
            
            # Periodic representation analysis
            if (epoch + 1) % analysis_freq == 0:
                print(f"  → Running layer analysis for {phase_name}...")
                
                acc_dict = {f'task_{task_id}': current_acc}
                if also_test_on:
                    acc_dict[f'task_{also_test_on[1]}'] = prev_acc
                
                evolution_results['accuracy'][f'epoch_{epoch + 1}'] = acc_dict
                
                # Analyze representations on BOTH tasks in phase 2
                repr_epoch = {}
                
                # Always analyze current task
                repr_epoch[f'task_{task_id}'] = self._run_vit_analysis(
                    DataLoader(test_exp.dataset, batch_size=128, shuffle=False), 
                    task_id, epoch + 1
                )
                
                # In phase 2, also analyze previous task to see forgetting patterns
                if also_test_on:
                    repr_epoch[f'task_{also_test_on[1]}'] = self._run_vit_analysis(
                        DataLoader(also_test_on[0].dataset, batch_size=128, shuffle=False),
                        also_test_on[1], epoch + 1
                    )
                
                evolution_results['representations'][f'epoch_{epoch + 1}'] = repr_epoch
                
                # Save analysis results
                analysis_file = analysis_dir / f'epoch_{epoch + 1:03d}.json'
                with open(analysis_file, 'w') as f:
                    json.dump(repr_epoch, f, indent=2)
                
                print(f"  → Analysis saved to {analysis_file}")
            
            # PATCH ANALYSIS (potentially different frequency)
            # PATCH ANALYSIS (potentially different frequency)
            if self.enable_patch_analysis and (epoch + 1) % self.patch_analysis_freq == 0:
                print(f"  → Running patch analysis for {phase_name}...")
                try:
                    # Always analyze current task
                    test_loader = DataLoader(test_exp.dataset, batch_size=128, shuffle=False)
                    patch_results = self._run_patch_analysis(test_loader, epoch + 1, f"{dataset_name}_{phase_name}")
                    
                    # In phase 2, ALSO analyze previous task to see patch forgetting!
                    if also_test_on:
                        print(f"  → Running patch analysis for previous task (forgetting analysis)...")
                        prev_test_loader = DataLoader(also_test_on[0].dataset, batch_size=128, shuffle=False)
                        prev_patch_results = self._run_patch_analysis(
                            prev_test_loader, 
                            epoch + 1, 
                            f"prev_task_{also_test_on[1]}_{phase_name}"
                        )
                    
                    # UPDATE PATCH SUMMARY
                    if patch_results and patch_summary:
                        patch_summary['phases'][phase_name]['epochs_analyzed'].append({
                            'epoch': epoch + 1,
                            'eval_accuracy': current_acc,
                            'prev_task_accuracy': prev_acc if also_test_on else None,
                            'summary_stats': patch_results.get('statistics', {}),
                            'prev_task_patch_stats': prev_patch_results.get('statistics', {}) if also_test_on else None
                        })
                        
                        print(f"  → Patch analysis completed for {dataset_name}")
                
                except Exception as e:
                    print(f"  → Patch analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        return evolution_results
    
    def _run_vit_analysis(self, test_loader, task_id, epoch):
        """Run ViT class projection analysis - SAME as single task experiment."""
        try:
            projection_scores = self.analyzer.analyze_task_representations(
                test_loader, 
                task_id=task_id, 
                num_classes_per_task=self.config.get('num_classes', 10),
                max_batches=50,  # Analyze first 50 batches
                sample_tokens=True  # Sample tokens instead of all 196
            )
            return {
                'epoch': epoch,
                'task_id': task_id,
                'analyzer_type': 'vit_projection',
                'projection_scores': projection_scores
            }
        except Exception as e:
            print(f"  → Analysis failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    def _run_patch_analysis(self, test_loader, epoch, dataset_phase_name):
        """Run detailed patch-level importance analysis."""
        if not self.patch_analyzer:
            return None
        
        # Run patch importance analysis
        max_batches = self.config.get('patch_analysis_max_batches', 100)
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
            dataset_phase_name
        )
        
        # Save detailed results
        self.patch_analyzer.save_detailed_results(
            patch_results,
            patch_analysis_dir,
            epoch
        )
        
        return patch_results
    
    def _train_epoch_on_loader(self, loader, optimizer, criterion):
        """Train one epoch on a given data loader."""
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        
        for batch in loader:
            # Handle variable batch sizes - take first 2 elements regardless
            x, y = batch[0], batch[1]  # Always works regardless of batch length
            
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
        
        return 100.0 * correct / total, total_loss / len(loader)

    def _evaluate_task_unified(self, test_exp) -> float:
        """Evaluate model on a single task."""
        self.model.eval()
        correct = total = 0
        
        loader = DataLoader(test_exp.dataset, batch_size=128, shuffle=False)
        
        with torch.inference_mode():
            for batch in loader:
                # Handle variable batch sizes - take first 2 elements regardless
                x, y = batch[0], batch[1]  # Always works regardless of batch length
                    
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total
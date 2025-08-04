"""Simple binary pairs experiment A→B with patch analysis."""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from torch.utils.data import DataLoader
import csv
import json
from pathlib import Path

from core.experiment import BaseExperiment


class StrategyBinaryPairsExperiment(BaseExperiment):
    """Simple binary pairs experiment: Train on A, then train on B, analyze forgetting."""
    
    def setup(self):
        """Setup experiment components."""
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        
        # NEW: Set the correct model name based on scenario type
        if scenario_type == 'task_incremental':
            # Use multi-head model for task incremental
            if 'ViT' in self.config.get('model_name', 'ViT64'):
                self.config['model_name'] = 'ViT64_multihead'
            elif 'EfficientNet' in self.config.get('model_name', ''):
                if 'NotPretrained' in self.config['model_name']:
                    self.config['model_name'] = 'EfficientNet_multihead_NotPretrained'
                else:
                    self.config['model_name'] = 'EfficientNet_multihead'
        else:  # class_incremental
            # Use single head model
            if self.config.get('model_name') == 'ViT64_multihead':
                self.config['model_name'] = 'ViT64'
            # Keep other models as-is for class incremental
        
        if scenario_type == 'class_incremental':
            # Single head with 20 classes (10 from each dataset)
            self.config['num_classes'] = 20
        else:  # task_incremental
            # Multiple heads, 10 classes per head
            self.config['num_classes'] = 10
            self.config['num_tasks'] = 2  # Ensure we have 2 heads for 2 tasks


        self.model = self.build_model()
        # Build strategy
        self.strategy = self._build_strategy()


        # Add this debug code to your setup() method after building the model
        print("\n=== TESTING CLEAN MODEL ===")
        try:
            with torch.no_grad():
                dummy_input = torch.randn(2, 3, 64, 64).to(self.device)
                
                if hasattr(self.model, 'heads'):
                    print(f"Multi-head model with {len(self.model.heads)} heads")
                    
                    # Test each head
                    for task_id in range(len(self.model.heads)):
                        output = self.model.forward_single_task(dummy_input, task_id)
                        print(f"  Task {task_id}: output shape {output.shape}, sample prediction {output[0].argmax().item()}")
                        
                        # Test head weights access
                        weights = self.model.get_head_weights(task_id)
                        print(f"  Task {task_id}: head weight shape {weights.shape}")
                else:
                    print("Single-head model")
                    output = self.model(dummy_input)
                    print(f"  Output shape: {output.shape}, sample prediction: {output[0].argmax().item()}")
                    
            print("=== CLEAN MODEL TEST PASSED ===\n")
        except Exception as e:
            print(f"=== CLEAN MODEL TEST FAILED: {e} ===\n")
            import traceback
            traceback.print_exc()
                
        # Setup patch analysis
        self.enable_patch_analysis = self.config.get('patch_analysis', True)
        if self.enable_patch_analysis:
            print("  → Patch analysis ENABLED")
            from analysis.patch_importance_analyzer import ViTPatchImportanceAnalyzer
            self.patch_analyzer = ViTPatchImportanceAnalyzer(self.model, str(self.device))
            self.patch_analysis_freq = self.config.get('patch_analysis_freq', 10)
            self.patch_during_training = self.config.get('patch_analysis_during_training', True)
        else:
            self.patch_analyzer = None
        
        # Setup class projection analysis 
        self.enable_projection_analysis = self.config.get('analyze_representations', True)
        if self.enable_projection_analysis:
            print("  → Class projection analysis ENABLED")
            from analysis.vit_class_projection import ViTClassProjectionAnalyzer
            self.projection_analyzer = ViTClassProjectionAnalyzer(self.model, str(self.device))
        else:
            self.projection_analyzer = None
    
    def _build_strategy(self):
        """Build strategy based on config."""
        from core.strategies import NaiveStrategy, ReplayStrategy, CumulativeStrategy, KaizenStrategy, RDBPStrategy
        
        strategy_name = self.config.get('strategy_name', 'Naive')
        
        if strategy_name == 'Naive':
            return NaiveStrategy(self.model, self.config, str(self.device))
        elif strategy_name == 'Replay':
            return ReplayStrategy(self.model, self.config, str(self.device))
        elif strategy_name == 'Cumulative':
            return CumulativeStrategy(self.model, self.config, str(self.device))
        elif strategy_name == 'Kaizen':
            return KaizenStrategy(self.model, self.config, str(self.device))
        elif strategy_name == 'RDBP':
            return RDBPStrategy(self.model, self.config, str(self.device))
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _load_dataset(self, dataset_name):
        """Load a dataset by name."""
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
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _relabel_dataset(self, dataset, offset):
        """Relabel dataset classes by adding an offset."""
        class RelabeledDataset:
            def __init__(self, base_dataset, offset):
                self.base_dataset = base_dataset
                self.offset = offset
                if hasattr(base_dataset, 'targets'):
                    self.targets = [t + offset for t in base_dataset.targets]
                else:
                    self.targets = None
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
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
    
    def _train_task_with_analysis(self, train_dataset, test_datasets, task_id, task_name, patch_dir, log_file=None):
        """Train a task with periodic patch analysis and evaluation by running multiple shorter training sessions."""
        total_epochs = self.config.get('epochs', 50)
        analysis_freq = self.patch_analysis_freq
        
        # Save original epoch count
        original_epochs = self.config['epochs']
        
        # Train in chunks of analysis_freq epochs
        epochs_completed = 0
        
        while epochs_completed < total_epochs:
            # Calculate how many epochs to train in this chunk
            epochs_this_chunk = min(analysis_freq, total_epochs - epochs_completed)
            
            print(f"  → Training epochs {epochs_completed + 1}-{epochs_completed + epochs_this_chunk}...")
            
            # Temporarily set epochs for this chunk
            self.config['epochs'] = epochs_this_chunk
            
            # Train for this chunk
            self.strategy.train_task(train_dataset, task_id)
            
            epochs_completed += epochs_this_chunk
            
            # Evaluate on all test datasets after this chunk (if not the final chunk)
            if epochs_completed < total_epochs:
                print(f"  → Evaluating at epoch {epochs_completed}...")
                
                # Evaluate on all test datasets
                eval_results = {}
                scenario_type = self.config.get('scenario_type', 'class_incremental')
                for i, (dataset_name, test_dataset) in enumerate(test_datasets.items()):
                    if scenario_type == 'task_incremental':
                        # For task incremental: first dataset=task 0, second dataset=task 1
                        eval_task_id = i
                        acc = self.strategy._evaluate_dataset(test_dataset, task_id=eval_task_id)
                    else:
                        acc = self.strategy._evaluate_dataset(test_dataset)
                    eval_results[dataset_name] = acc
                    print(f"    {dataset_name}: {acc:.1f}%")
                
                # Log to CSV if in Phase 2 (when we have both tasks)
                if log_file and len(test_datasets) == 2:
                    dataset_names = list(test_datasets.keys())
                    acc_a = eval_results.get(dataset_names[0], 0.0)
                    acc_b = eval_results.get(dataset_names[1], 0.0)
                    
                    # Calculate forgetting and average
                    if hasattr(self, 'phase1_final_acc'):
                        forgetting = self.phase1_final_acc - acc_a
                    else:
                        forgetting = 0.0
                    average_acc = (acc_a + acc_b) / 2
                    
                    # Log intermediate results
                    self._log_results(log_file, f"2.{epochs_completed}", acc_a, acc_b, forgetting, average_acc)
                
                # Run patch analysis after this chunk (if enabled)
                if self.enable_patch_analysis and self.patch_during_training:
                    print(f"  → Running patch analysis at epoch {epochs_completed}...")
                    
                    # Run analysis on all test datasets we want to track
                    for dataset_name, test_dataset in test_datasets.items():
                        analysis_name = f"{task_name}_epoch{epochs_completed:02d}_{dataset_name}"
                        self._run_patch_analysis(
                                                test_dataset,
                                                patch_dir,
                                                task_name,            # e.g. "phase1" or "phase2"
                                                dataset_name,         # e.g. "mnist"
                                                epochs_completed,
                                                task_id=0 if dataset_name == self.config.get('dataset_a') else 1
                                    )

                                                        
                # Run class projection analysis (if enabled)
                if self.enable_projection_analysis:
                    print(f"  → Running class projection analysis at epoch {epochs_completed}...")
                    
                    # Determine phase name
                    phase_name = "phase1" if task_id == 0 else "phase2"
                    layer_analysis_dir = self.output_dir / 'layer_analysis' / phase_name
                    
                    # Run analysis on all test datasets
                    repr_epoch = {}
                    for dataset_name, test_dataset in test_datasets.items():
                        # Determine task ID for this dataset
                        dataset_task_id = 0 if dataset_name == list(test_datasets.keys())[0] else 1
                        
                        result = self._run_projection_analysis(test_dataset, dataset_task_id, epochs_completed, phase_name)
                        if result:
                            repr_epoch[f'task_{dataset_task_id}'] = result
                    
                    # Save class projection results in format expected by BinaryPairsViTAnalyzer
                    if repr_epoch:
                        analysis_file = layer_analysis_dir / f'epoch_{epochs_completed:03d}.json'
                        with open(analysis_file, 'w') as f:
                            json.dump(repr_epoch, f, indent=2)
                        print(f"  → Class projection analysis saved to {analysis_file}")
        
        # Restore original epoch count
        self.config['epochs'] = original_epochs
    
    def run(self) -> Dict[str, Any]:
        """Run simple binary pairs training A→B."""
        self.setup()
        
        strategy_name = self.config.get('strategy_name', 'Naive')
        dataset_a = self.config.get('dataset_a', 'mnist')
        dataset_b = self.config.get('dataset_b', 'fashion_mnist')
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        
        print(f"\n=== {strategy_name}: {dataset_a.upper()} → {dataset_b.upper()} ===")
        if hasattr(self.strategy, 'memory_size'):
            try:
                memory_size = getattr(self.strategy, 'memory_size', None)
                if memory_size is not None:
                    print(f"  → Memory size: {memory_size} samples per task")
            except:
                pass
        
        # Load datasets
        train_a, test_a = self._load_dataset(dataset_a)
        train_b, test_b = self._load_dataset(dataset_b)
        
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        if scenario_type == 'class_incremental':
            # Relabel dataset B to classes 10-19
            train_b = self._relabel_dataset(train_b, 10)
            test_b = self._relabel_dataset(test_b, 10)
            print(f"Dataset A ({dataset_a}): classes 0-9")
            print(f"Dataset B ({dataset_b}): classes 10-19")
        else:  # task_incremental
            print(f"Dataset A ({dataset_a}): classes 0-9, task 0")
            print(f"Dataset B ({dataset_b}): classes 0-9, task 1")

        # Create results tracking
        self.results = {
            'strategy_name': strategy_name,
            'dataset_a': dataset_a,
            'dataset_b': dataset_b,
            'task_results': {}
        }
        
        # Create CSV logger
        log_file = self.output_dir / f'training_log_{strategy_name.lower()}.csv'
        self._create_csv_logger(log_file)
        
        # Create patch analysis directory
        if self.enable_patch_analysis:
            patch_dir = self.output_dir / 'patch_analysis'
            patch_dir.mkdir(exist_ok=True)
        
        # Create layer analysis directory for class projection analysis
        if self.enable_projection_analysis:
            layer_analysis_dir = self.output_dir / 'layer_analysis'
            layer_analysis_dir.mkdir(exist_ok=True)
            # Create phase subdirectories
            (layer_analysis_dir / 'phase1').mkdir(exist_ok=True)
            (layer_analysis_dir / 'phase2').mkdir(exist_ok=True)
        
        # PHASE 1: Train on Dataset A
        print(f"\n--- Phase 1: Training on {dataset_a.upper()} ---")
        
        # Train with periodic analysis
        test_datasets_phase1 = {dataset_a: test_a}
        self._train_task_with_analysis(train_a, test_datasets_phase1, 0, f"phase1_{dataset_a}", patch_dir)
        
        # Evaluate after Phase 1
        if scenario_type == 'task_incremental':
            acc_a_phase1 = self.strategy._evaluate_dataset(test_a, task_id=0)
        else:
            acc_a_phase1 = self.strategy._evaluate_dataset(test_a)
        acc_b_phase1 = 0.0  # Haven't seen B yet
        
        # Store for forgetting calculation
        self.phase1_final_acc = acc_a_phase1
        
        print(f"After Phase 1: {dataset_a}={acc_a_phase1:.1f}%, {dataset_b}={acc_b_phase1:.1f}%")
        
        # Log Phase 1
        self._log_results(log_file, 1, acc_a_phase1, acc_b_phase1)
        
        # Final patch analysis after Phase 1
        if self.enable_patch_analysis:
            print("  → Running final patch analysis after Phase 1...")
            self._run_patch_analysis(test_a, patch_dir,
                          "phase1", dataset_a, epoch=50, task_id=0)
        
        # Final class projection analysis after Phase 1
        if self.enable_projection_analysis:
            print("  → Running final class projection analysis after Phase 1...")
            layer_analysis_dir = self.output_dir / 'layer_analysis' / 'phase1'
            
            result = self._run_projection_analysis(test_a, 0, 50, 'phase1')
            if result:
                repr_epoch = {f'task_0': result}
                analysis_file = layer_analysis_dir / f'epoch_050.json'
                with open(analysis_file, 'w') as f:
                    json.dump(repr_epoch, f, indent=2)
                print(f"  → Final class projection analysis saved to {analysis_file}")
        
        # PHASE 2: Train on Dataset B
        print(f"\n--- Phase 2: Training on {dataset_b.upper()} ---")
        
        # Train with periodic analysis (track both datasets now for forgetting)
        test_datasets_phase2 = {dataset_a: test_a, dataset_b: test_b}
        self._train_task_with_analysis(train_b, test_datasets_phase2, 1, f"phase2_{dataset_b}", patch_dir, log_file)
        
        # Evaluate after Phase 2
        if scenario_type == 'task_incremental':
            acc_a_phase2 = self.strategy._evaluate_dataset(test_a, task_id=0)  # Check forgetting
            acc_b_phase2 = self.strategy._evaluate_dataset(test_b, task_id=1)
        else:
            acc_a_phase2 = self.strategy._evaluate_dataset(test_a)  # Check forgetting
            acc_b_phase2 = self.strategy._evaluate_dataset(test_b)
        
        print(f"After Phase 2: {dataset_a}={acc_a_phase2:.1f}%, {dataset_b}={acc_b_phase2:.1f}%")
        
        # Calculate forgetting
        forgetting = acc_a_phase1 - acc_a_phase2
        average_acc = (acc_a_phase2 + acc_b_phase2) / 2
        
        print(f"Forgetting: {forgetting:.1f}%, Average: {average_acc:.1f}%")
        
        # Log Phase 2
        self._log_results(log_file, 2, acc_a_phase2, acc_b_phase2, forgetting, average_acc)
        
        # Final patch analysis after Phase 2
        if self.enable_patch_analysis:
            print("  → Running final patch analysis after Phase 2...")
            self._run_patch_analysis(test_a, patch_dir,
                          "phase2_forgetting", dataset_a, epoch=50, task_id=0)
            self._run_patch_analysis(test_b, patch_dir,
                          "phase2",           dataset_b, epoch=50, task_id=1)
        
        # Final class projection analysis after Phase 2
        if self.enable_projection_analysis:
            print("  → Running final class projection analysis after Phase 2...")
            layer_analysis_dir = self.output_dir / 'layer_analysis' / 'phase2'
            
            # Analyze both tasks
            repr_epoch = {}
            result_a = self._run_projection_analysis(test_a, 0, 50, 'phase2')
            result_b = self._run_projection_analysis(test_b, 1, 50, 'phase2')
            
            if result_a:
                repr_epoch[f'task_0'] = result_a
            if result_b:
                repr_epoch[f'task_1'] = result_b
            
            if repr_epoch:
                analysis_file = layer_analysis_dir / f'epoch_050.json'
                with open(analysis_file, 'w') as f:
                    json.dump(repr_epoch, f, indent=2)
                print(f"  → Final class projection analysis saved to {analysis_file}")
        
        # Store final results
        self.results['final_accuracies'] = {
            f'task_0_{dataset_a}': acc_a_phase2,
            f'task_1_{dataset_b}': acc_b_phase2,
            'forgetting': forgetting,
            'average': average_acc
        }
        
        self.save_results()
        return self.results
    
    def _create_csv_logger(self, log_file: Path):
        """Create CSV logger."""
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['phase', 'strategy', 'task_0_acc', 'task_1_acc', 'forgetting', 'average_acc'])
    
    def _log_results(self, log_file: Path, phase, acc_a: float, acc_b: float, 
                    forgetting: float = 0.0, average: float = None):
        """Log results to CSV."""
        strategy_name = self.config.get('strategy_name', 'Naive')
        if average is None:
            average = (acc_a + acc_b) / 2
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([phase, strategy_name, f"{acc_a:.2f}", f"{acc_b:.2f}", 
                           f"{forgetting:.2f}", f"{average:.2f}"])
    
    def _run_patch_analysis(self, test_dataset, patch_dir: Path, task_name: str, dataset_name: str, epoch: int = 1, task_id: int = 0):
        """Run patch analysis on a dataset."""
        if not self.enable_patch_analysis or not self.patch_analyzer:
            return
        
        # Define analysis_name early to avoid UnboundLocalError
        analysis_name = f"{task_name}_epoch{epoch:02d}_{dataset_name}"
        
        try:
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            patch_results = self.patch_analyzer.analyze_patch_importance(
                test_loader,
                num_classes=self.config.get('num_classes', 10),  # 10 per task for task_incremental
                max_batches=self.config.get('patch_analysis_max_batches', 20),
                task_id=task_id  # Pass the correct task_id
            )
            
            # Create visualizations
            self.patch_analyzer.visualize_patch_importance(
                    patch_results, patch_dir, epoch, dataset_name, task_name
            )
            # Save detailed results
            self.patch_analyzer.save_detailed_results(
                    patch_results, patch_dir, epoch, task_name, dataset_name
            )
            
            print(f"  → Patch analysis completed: {analysis_name}")
            
        except Exception as e:
            print(f"  → Patch analysis failed for {analysis_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_projection_analysis(self, test_dataset, task_id: int, epoch: int, phase_name: str):
        """Run class projection analysis on a dataset."""
        if not self.enable_projection_analysis or not self.projection_analyzer:
            return None
        
        try:
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            projection_scores = self.projection_analyzer.analyze_task_representations(
                test_loader,
                task_id=task_id,
                num_classes_per_task=10,  # Each task has 10 classes
                max_batches=50,
                sample_tokens=True
            )
            
            result = {
                'epoch': epoch,
                'task_id': task_id,
                'analyzer_type': 'vit_projection',
                'projection_scores': projection_scores
            }
            
            print(f"  → Class projection analysis completed for task {task_id}")
            return result
            
        except Exception as e:
            print(f"  → Class projection analysis failed for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
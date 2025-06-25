#!/usr/bin/env python3
"""
Fixed Comprehensive Task-Incremental Learning Order Experiment
=============================================================
Fixes evaluation bugs and adds periodic representation analysis during training.

Key fixes:
1. Proper multihead evaluation with correct task IDs
2. Periodic ViT analysis every N epochs during training
3. Better accuracy extraction from strategy results
4. Proper forgetting matrix computation
"""

import argparse
import json
import os
import random
import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Use vanilla experiment components
from config.utils import load_config
from models.model_provider import parse_model_name
from scenarios.scenarios_providers import parse_scenario, get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet, get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist
from strategies.strategies_provider import parse_strategy_name
from analysis.vit_class_projection import ViTClassProjectionAnalyzer
from paths import ROOT_PATH, LOGS_PATH

# ─────────────────────────────────────────────── Order Configuration ──
MTI_ORDER = ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"]  # MNIST to ImageNet
ITM_ORDER = ["imagenet", "cifar10", "svhn", "fashion_mnist", "omniglot", "mnist"]  # ImageNet to MNIST

def create_custom_order_scenario(order: List[str], args):
    """Create scenario with custom dataset order."""
    if order == MTI_ORDER:
        return get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet(
            class_incremental=(args.scenario_type == 'class_incremental'),
            balanced=(args.balanced == 'balanced'),
            number_of_samples_per_class=getattr(args, 'number_of_samples_per_class', 500)
        )
    elif order == ITM_ORDER:
        return get_short_imagenet_cifar10_svhn_fmnist_omniglot_mnist(
            class_incremental=(args.scenario_type == 'class_incremental'),
            balanced=(args.balanced == 'balanced'),
            number_of_samples_per_class=getattr(args, 'number_of_samples_per_class', 500)
        )
    else:
        # For random orders, use MTI as base and document the limitation
        print(f"Warning: Custom order {order} not fully supported. Using MTI scenario as approximation.")
        return get_short_mnist_omniglot_fmnist_svhn_cifar10_imagenet(
            class_incremental=(args.scenario_type == 'class_incremental'),
            balanced=(args.balanced == 'balanced'),
            number_of_samples_per_class=getattr(args, 'number_of_samples_per_class', 500)
        )


# ─────────────────────────────────────────────── Fixed Training Utils ──
def evaluate_on_experience(model, experience, device, task_id=None):
    """
    Properly evaluate model on an experience, handling multihead models correctly.
    """
    model.eval()
    correct = total = 0
    
    # Create dataloader for the experience
    loader = DataLoader(experience.dataset, batch_size=64, shuffle=False)
    
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:  # x, y, task_labels
                x, y, original_task_labels = batch
                x, y = x.to(device), y.to(device)
                
                # For multihead models, override task labels with the correct task_id
                if task_id is not None:
                    task_labels = torch.full_like(original_task_labels, task_id).to(device)
                    out = model(x, task_labels)
                else:
                    out = model(x, original_task_labels.to(device))
                    
            else:  # x, y (single-head model)
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                
            pred = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total


def train_single_epoch(strategy, train_exp):
    """Train for a single epoch and return any metrics."""
    # This is a simplified version - you might need to adapt based on your strategy implementation
    strategy.train(train_exp, num_workers=0)


# ─────────────────────────────────────────────── Analysis Wrapper ──
class _WrappedModel(nn.Module):
    """Wrap model so analyzer can call it consistently."""

    def __init__(self, base_model, task_id: Optional[int] = None, is_multihead: bool = False):
        super().__init__()
        self.base = base_model
        self.task_id = task_id
        self.is_multihead = is_multihead
        
        # Extract and expose ViT
        if hasattr(base_model, 'vit'):
            self.vit = base_model.vit
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'vit'):
            self.vit = base_model.model.vit
        else:
            self.vit = None
            
        # Expose classifier/head attributes that the analyzer expects
        if hasattr(base_model, 'classifier'):
            self.classifier = base_model.classifier
        elif hasattr(base_model, 'head'):
            self.head = base_model.head
        elif hasattr(base_model, 'vit') and hasattr(base_model.vit, 'head'):
            self.head = base_model.vit.head

    def forward(self, x):
        if self.is_multihead and self.task_id is not None:
            task_labels = torch.full((x.size(0),), self.task_id, dtype=torch.long, device=x.device)
            return self.base(x, task_labels)
        else:
            return self.base(x)


def run_periodic_analysis(model, experience, task_id, epoch, device, is_multihead, analysis_results):
    """Run ViT analysis and store results."""
    if not hasattr(model, 'vit') and not (hasattr(model, 'model') and hasattr(model.model, 'vit')):
        return  # No ViT to analyze
        
    try:
        wrapped_model = _WrappedModel(model, task_id, is_multihead).to(device)
        
        if wrapped_model.vit is not None:
            analyzer = ViTClassProjectionAnalyzer(wrapped_model, device)
            
            # Create loader for analysis (smaller sample for speed)
            dataset_size = len(experience.dataset)
            sample_size = min(500, dataset_size)  # Limit for speed
            indices = torch.randperm(dataset_size)[:sample_size]
            subset = torch.utils.data.Subset(experience.dataset, indices)
            loader = DataLoader(subset, batch_size=64, shuffle=False)
            
            # Analyze representations
            scores = analyzer.analyze_task_representations(
                loader, 
                task_id=task_id, 
                num_classes_per_task=10
            )
            
            # Store results with epoch information
            key = f"task{task_id}_epoch{epoch}"
            analysis_results[key] = {k: float(v) for k, v in scores.items()}
            print(f"  Epoch {epoch}: Analysis complete for task {task_id}")
            
    except Exception as e:
        print(f"  Error in analysis at epoch {epoch}: {e}")


# ─────────────────────────────────────────────── Main Fixed Experiment ──
def run_order_experiment(order: List[str], order_name: str, args):
    """Run experiment for a specific order with proper evaluation and periodic analysis."""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {order_name}")
    print(f"Order: {' → '.join(order)}")
    print(f"{'='*60}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create scenario using vanilla components
    scenario = create_custom_order_scenario(order, args)
    print(f"Created scenario with {len(scenario.train_stream)} tasks")
    
    # Initialize model using vanilla model provider
    model = parse_model_name(args)
    print(f"Using model: {args.model_name}")
    
    # Initialize strategy using vanilla strategy provider
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create minimal evaluation plugin for the strategy
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import TextLogger
    from avalanche.training.plugins import EvaluationPlugin
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True),
        loggers=[TextLogger()]
    )
    
    strategy = parse_strategy_name(
        args=args,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        eval_plugin=eval_plugin
    )
    
    # Track results
    accuracy_matrix: List[List[float]] = []
    periodic_analysis_results = defaultdict(dict)
    
    # Determine if model is multi-head
    is_multihead = hasattr(model, 'classifier') or 'multihead' in args.model_name.lower()
    print(f"Model is multihead: {is_multihead}")
    
    # Store all experiences for evaluation
    all_test_experiences = []
    
    # Train and evaluate each task
    for task_id, (train_exp, test_exp) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
        dataset_name = order[task_id] if task_id < len(order) else f"task_{task_id}"
        print(f"\n=== Task {task_id}: {dataset_name} ===")
        print(f"Classes: {train_exp.classes_in_this_experience}")
        
        # Store test experience for later evaluation
        all_test_experiences.append(test_exp)
        
        # Training with periodic analysis
        print(f"Training on {dataset_name} for {args.epochs} epochs...")
        
        for epoch in range(args.epochs):
            # Train one epoch
            train_single_epoch(strategy, train_exp)
            
            # Periodic analysis every 10 epochs (and final epoch)
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"  Running analysis at epoch {epoch + 1}...")
                run_periodic_analysis(
                    strategy.model, train_exp, task_id, epoch + 1, 
                    device, is_multihead, periodic_analysis_results
                )
        
        # Final evaluation on test set for current task
        test_acc = evaluate_on_experience(strategy.model, test_exp, device, task_id if is_multihead else None)
        print(f"Final test accuracy on {dataset_name}: {test_acc:.2f}%")
        
        # Evaluate on all previous tasks (forgetting matrix)
        accuracy_row = []
        for prev_task_id in range(task_id + 1):
            prev_test_exp = all_test_experiences[prev_task_id]
            prev_dataset_name = order[prev_task_id] if prev_task_id < len(order) else f"task_{prev_task_id}"
            
            # Use proper task_id for multihead models
            eval_task_id = prev_task_id if is_multihead else None
            acc = evaluate_on_experience(strategy.model, prev_test_exp, device, eval_task_id)
            accuracy_row.append(acc)
            
            print(f"  Task {prev_task_id} ({prev_dataset_name}): {acc:.2f}%")
        
        accuracy_matrix.append(accuracy_row)
        print(f"Accuracy row: {[f'{a:.1f}' for a in accuracy_row]}")
    
    # Calculate summary metrics
    final_accuracy = accuracy_matrix[-1][-1] if accuracy_matrix else 0.0
    average_accuracy = sum(row[-1] for row in accuracy_matrix) / len(accuracy_matrix) if accuracy_matrix else 0.0
    
    # Calculate forgetting
    forgetting_scores = []
    if len(accuracy_matrix) > 1:
        for task_id in range(len(accuracy_matrix) - 1):
            initial_acc = accuracy_matrix[task_id][task_id]
            final_acc = accuracy_matrix[-1][task_id]
            forgetting = initial_acc - final_acc
            forgetting_scores.append(forgetting)
    
    average_forgetting = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0.0
    
    results = {
        "order_name": order_name,
        "order": order,
        "config": {
            "model_name": args.model_name,
            "strategy_name": args.strategy_name,
            "scenario": args.scenario,
            "scenario_type": args.scenario_type,
            "epochs": args.epochs,
            "lr": args.lr,
            "minibatch_size": args.minibatch_size,
        },
        "accuracy_matrix": accuracy_matrix,
        "periodic_analysis": dict(periodic_analysis_results),
        "summary": {
            "final_accuracy": final_accuracy,
            "average_accuracy": average_accuracy,
            "average_forgetting": average_forgetting,
            "forgetting_per_task": forgetting_scores
        }
    }
    
    print(f"\n=== {order_name} SUMMARY ===")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print(f"Average accuracy: {average_accuracy:.2f}%")
    print(f"Average forgetting: {average_forgetting:.2f}%")
    print(f"Periodic analysis points: {len(periodic_analysis_results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fixed Comprehensive Order Experiment')
    
    # Use the same arguments as vanilla experiment
    parser.add_argument("--config", type=str, required=True, 
                       help="Configuration file (e.g., config/task_incremental/vit_config_param.yml)")
    parser.add_argument("--logdir", type=str, default="logs/fixed_order_experiments", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_random", type=int, default=2, 
                       help="Number of random orders to test")
    parser.add_argument("--analysis_freq", type=int, default=10,
                       help="Run ViT analysis every N epochs")
    
    # Allow overriding config parameters
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--model_name", type=str, help="Override model from config")
    parser.add_argument("--strategy_name", type=str, help="Override strategy from config")
    
    args = parser.parse_args()
    
    # Load config file like vanilla experiment
    opt = load_config(args.config)
    parser.set_defaults(**opt)
    args = parser.parse_args()  # Reparse with config defaults
    
    print(f"Loaded config: {args.config}")
    print(f"Model: {args.model_name}, Strategy: {args.strategy_name}")
    print(f"Scenario: {args.scenario}, Type: {args.scenario_type}")
    print(f"Analysis frequency: every {args.analysis_freq} epochs")
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    # 1. Standard MTI order (MNIST to ImageNet)
    results = run_order_experiment(MTI_ORDER, "MTI_Standard", args)
    all_results["mti_standard"] = results
    
    # 2. Standard ITM order (ImageNet to MNIST)  
    results = run_order_experiment(ITM_ORDER, "ITM_Backward", args)
    all_results["itm_backward"] = results
    
    # 3. Random orders (note: will use MTI scenario as approximation)
    if args.num_random > 0:
        print(f"\nNote: Random orders will use base scenario structure due to current limitations.")
        for i in range(args.num_random):
            random_order = MTI_ORDER.copy()
            random.shuffle(random_order)
            results = run_order_experiment(random_order, f"Random_{i+1}", args)
            all_results[f"random_{i+1}"] = results
    
    # Save comprehensive results
    output_file = Path(args.logdir) / f"fixed_results_{args.model_name}_{args.strategy_name}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"Tested {len(all_results)} different orders")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Order':<15} {'Final Acc':<10} {'Avg Acc':<10} {'Avg Forget':<12} {'Analysis Pts':<12}")
    print("-" * 65)
    
    for key, result in all_results.items():
        summary = result["summary"]
        analysis_count = len(result.get("periodic_analysis", {}))
        print(f"{result['order_name']:<15} {summary['final_accuracy']:<10.2f} "
              f"{summary['average_accuracy']:<10.2f} {summary['average_forgetting']:<12.2f} "
              f"{analysis_count:<12}")


if __name__ == "__main__":
    main()
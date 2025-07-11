#!/usr/bin/env python3
"""Analyze a completed training run and generate visualizations."""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from plots.layer_analysis_plots import LayerAnalysisPlotter


def analyze_training_logs(log_file: str, output_dir: str):
    """Analyze training logs and create accuracy/loss plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read training log
    df = pd.read_csv(log_file)
    
    # Create training progress plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    axes[0].plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(df['epoch'], df['eval_acc'], label='Eval Accuracy', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Progress - Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(df['epoch'], df['eval_loss'], label='Eval Loss', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Progress - Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    accuracy_plot_path = output_path / "training_progress.png"
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved to {accuracy_plot_path}")
    plt.close()
    
    # Create summary statistics
    summary = {
        'final_train_acc': float(df['train_acc'].iloc[-1]),
        'final_eval_acc': float(df['eval_acc'].iloc[-1]),
        'best_eval_acc': float(df['eval_acc'].max()),
        'final_train_loss': float(df['train_loss'].iloc[-1]),
        'final_eval_loss': float(df['eval_loss'].iloc[-1]),
        'min_eval_loss': float(df['eval_loss'].min()),
        'total_epochs': len(df)
    }
    
    with open(output_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze a completed training run")
    parser.add_argument("--experiment_dir", required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", 
                       help="Output directory for analysis plots (default: experiment_dir/analysis)")
    
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment_dir)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_path / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing experiment: {experiment_path}")
    print(f"Output directory: {output_dir}")
    
    # Analyze training logs
    log_file = experiment_path / "training_log.csv"
    if log_file.exists():
        print("\n1. Analyzing training logs...")
        analyze_training_logs(str(log_file), str(output_dir))
    else:
        print("Warning: No training_log.csv found")
    
    # Analyze layer analysis results
    layer_analysis_dir = experiment_path / "layer_analysis"
    if layer_analysis_dir.exists() and any(layer_analysis_dir.glob("epoch_*.json")):
        print("\n2. Analyzing layer progression...")
        plotter = LayerAnalysisPlotter(str(layer_analysis_dir))
        plotter.create_all_plots(str(output_dir))
    else:
        print("Warning: No layer analysis results found")
    
    # Read experiment results if available
    results_file = experiment_path / "results.json"
    if results_file.exists():
        print("\n3. Reading experiment results...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"Dataset: {results.get('dataset', 'Unknown')}")
        print(f"Final accuracy: {results.get('final_accuracy', 'N/A'):.3f}")
        print(f"Best accuracy: {results.get('best_accuracy', 'N/A'):.3f}")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    # Print summary of what was created
    created_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.json"))
    print(f"Created {len(created_files)} files:")
    for file in sorted(created_files):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main() 
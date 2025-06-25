#!/usr/bin/env python3
"""
Quick Order Test - Simplified version for testing different learning orders
==========================================================================
Quick test of MTI vs ITM orders with reduced epochs for fast experimentation.

Usage:
    python quick_order_test.py --epochs 5 --batch 32 --n_per_class 100
"""

import argparse
from comprehensive_order_experiment import (
    MTI_ORDER, ITM_ORDER, run_order_experiment, DATASETS
)
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Quick Order Test')
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task (reduced for quick test)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--n_per_class", type=int, default=100, help="Samples per class (reduced for quick test)")
    parser.add_argument("--logdir", type=str, default="logs/quick_order_test", help="Output directory")
    parser.add_argument("--test_random", action="store_true", help="Also test one random order")
    
    args = parser.parse_args()
    
    print("Quick Order Test - Reduced epochs/data for fast experimentation")
    print(f"Epochs: {args.epochs}, Samples per class: {args.n_per_class}")
    
    # Create output directory
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test MTI order
    print("\n" + "="*50)
    print("Testing MTI (MNIST → ImageNet) order")
    print("="*50)
    mti_result = run_order_experiment(MTI_ORDER, "MTI_Quick", args)
    results["mti"] = mti_result
    
    # Test ITM order  
    print("\n" + "="*50)
    print("Testing ITM (ImageNet → MNIST) order")
    print("="*50)
    itm_result = run_order_experiment(ITM_ORDER, "ITM_Quick", args)
    results["itm"] = itm_result
    
    # Optional random order
    if args.test_random:
        print("\n" + "="*50)
        print("Testing Random order")
        print("="*50)
        random_order = list(DATASETS.keys())
        random.shuffle(random_order)
        random_result = run_order_experiment(random_order, "Random_Quick", args)
        results["random"] = random_result
    
    # Print comparison
    print(f"\n{'='*60}")
    print("QUICK TEST COMPARISON")
    print(f"{'='*60}")
    print(f"{'Order':<12} {'Final Acc':<10} {'Avg Acc':<10} {'Avg Forget':<12}")
    print("-" * 48)
    
    for name, result in results.items():
        summary = result["summary"]
        print(f"{name.upper():<12} {summary['final_accuracy']:<10.2f} "
              f"{summary['average_accuracy']:<10.2f} {summary['average_forgetting']:<12.2f}")
    
    # Save results
    import json
    output_file = Path(args.logdir) / "quick_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main() 
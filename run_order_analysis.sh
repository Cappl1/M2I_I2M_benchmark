#!/bin/bash

# Run Order Analysis experiments across different task orders
# Usage: ./run_order_analysis.sh

set -e  # Exit on any error

# Configuration
BASE_CONFIG="/home/brothen/M2I_I2M_benchmark/configs/experiments/order_analysis_base.yml"
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"    # Adjust this to your main experiment script
RESULTS_DIR="order_analysis_results_$(date +%Y%m%d_%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Base configuration template for order analysis
cat > "$BASE_CONFIG" << 'EOF'
# Order Analysis Experiment Configuration
experiment_type: OrderAnalysisExperiment
experiment_name: OrderAnalysisExperiment

# Scenario configuration
scenario_type:  task_incremental # or  class_incremental
scenario: short_mnist_omniglot_fmnist_svhn_cifar10_imagenet
resized: resized
balanced: balanced
number_of_samples_per_class: 500

# Orders to test
orders: ["MTI", "ITM", "EASY_TO_HARD", "HARD_TO_EASY"]  # Predefined orders
num_random_orders: 2  # Additional random orders to test

# Training parameters
epochs: 40
minibatch_size: 128
lr: 0.0003
optimizer: adam
momentum: 0.9

# Analysis parameters
analyze_representations: true  # Layer analysis on all previous tasks every 10 epochs
analysis_freq: 10  # Run layer analysis every N epochs
track_trajectory: true  # Enable analysis every N epochs during training (not just final)

# Model parameters
model_name: ViT64
num_classes: 10

# Strategy configuration
strategy_name: Naive  # or EWC, LwF, etc.

# Device configuration
cuda: 2  # Use GPU 3, set to -1 for CPU

# Output configuration
output_dir: logs/order_analysis
EOF

echo "Starting order analysis experiments..."
echo "=================================="

# Function to run order analysis experiment
run_order_experiment() {
    local scenario_type=$1
    echo ""
    echo "🚀 Starting order analysis experiment: $scenario_type"
    echo "Time: $(date)"
    echo "-------------------------------------------"
    
    # Create scenario-specific config
    local config_file="config_order_${scenario_type}.yaml"
    sed "s/scenario_type: class_incremental/scenario_type: $scenario_type/g" "$BASE_CONFIG" > "$config_file"
    
    # Run the experiment
    echo "Running order analysis experiment..."
    if python "$PYTHON_SCRIPT" --config "$config_file"; then
        echo "✅ Order analysis completed successfully for $scenario_type"
    else
        echo "❌ Order analysis failed for $scenario_type"
        rm -f "$config_file"
        return 1
    fi
    
    # Find the latest experiment directory
    local latest_exp_dir=$(find logs/order_analysis -name "experiment_*" -type d | sort | tail -1)
    
    if [ -z "$latest_exp_dir" ]; then
        echo "❌ Could not find experiment directory for $scenario_type"
        rm -f "$config_file"
        return 1
    fi
    
    echo "Found experiment directory: $latest_exp_dir"
    
    # Copy experiment logs to results directory
    local scenario_results_dir="$RESULTS_DIR/${scenario_type}_experiment"
    cp -r "$latest_exp_dir" "$scenario_results_dir"
    echo "📁 Experiment logs copied to: $scenario_results_dir"
    
    # Extract and save key results
    local results_json="${latest_exp_dir}/results.json"
    if [ -f "$results_json" ]; then
        echo "Processing results..."
        python -c "
import json
import pandas as pd
from pathlib import Path

# Load results
with open('$results_json', 'r') as f:
    data = json.load(f)

if 'comparison' in data and 'summary_table' in data['comparison']:
    # Create summary CSV
    summary_data = []
    for order_name, metrics in data['comparison']['summary_table'].items():
        summary_data.append({
            'Order': order_name,
            'Avg_Accuracy': metrics['avg_accuracy'],
            'Avg_Forgetting': metrics['avg_forgetting']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Avg_Accuracy', ascending=False)
    
    # Save to CSV
    csv_path = '$RESULTS_DIR/${scenario_type}_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'Summary saved to: {csv_path}')
    
    # Print results
    print(f'\n=== {scenario_type.upper()} RESULTS ===')
    print(df.to_string(index=False, float_format='%.2f'))
    
    if 'best_order' in data['comparison']:
        print(f'\\nBest order: {data[\"comparison\"][\"best_order\"]}')
        print(f'Worst order: {data[\"comparison\"][\"worst_order\"]}')
else:
    print('No comparison data found in results')
"
    fi
    
    # Clean up config file
    rm -f "$config_file"
    
    echo "✅ Complete pipeline finished for $scenario_type"
}

# Function to create summary report
create_summary() {
    echo ""
    echo "📊 Creating order analysis summary..."
    
    local summary_file="$RESULTS_DIR/summary_report.md"
    
    cat > "$summary_file" << EOF
# Order Analysis Experiment Summary Report

**Generated on:** $(date)
**Scenarios tested:** Class Incremental, Task Incremental

## Experiment Configuration
- Model: ViT64
- Epochs: 50
- Learning Rate: 0.0003
- Batch Size: 128
- Samples per class: 500
- Classes per dataset: 10
- Datasets: MNIST, Omniglot, Fashion-MNIST, SVHN, CIFAR-10, ImageNet

## Analysis Details

The experiment now includes **comprehensive layer analysis**:

### Layer Analysis (`analyze_representations: true`)
- **Every 10 epochs during training**: Analyzes ViT representations for ALL previously seen tasks
- **Tracks forgetting development**: Shows how layer-wise class identifiability degrades over time
- **Cross-task comparison**: Compares representation quality across different tasks at each checkpoint
- **Temporal dynamics**: Reveals when and how catastrophic forgetting occurs in different layers

### Data Structure Generated:
\`\`\`json
{
  "task_0_mnist": {
    "epoch_10": {
      "task_0_mnist": {"accuracy": 85.2, "representations": {...}}
    },
    "epoch_20": {...}
  },
  "task_1_cifar10": {
    "epoch_10": {
      "task_0_mnist": {"accuracy": 78.1, "representations": {...}},
      "task_1_cifar10": {"accuracy": 32.5, "representations": {...}}
    }
  }
}
\`\`\`

This provides unprecedented insight into how different task orders affect layer-wise forgetting patterns in Vision Transformers.

## Task Orders Tested

### Predefined Orders:
- **MTI**: MNIST → Omniglot → Fashion-MNIST → SVHN → CIFAR-10 → ImageNet
- **ITM**: ImageNet → CIFAR-10 → SVHN → Fashion-MNIST → Omniglot → MNIST  
- **EASY_TO_HARD**: MNIST → Fashion-MNIST → Omniglot → CIFAR-10 → SVHN → ImageNet
- **HARD_TO_EASY**: ImageNet → SVHN → CIFAR-10 → Omniglot → Fashion-MNIST → MNIST

### Random Orders:
- 2 additional randomized task orders for comparison

## Results Overview

EOF
    
    # Add scenario-specific results
    for scenario in "class_incremental" "task_incremental"; do
        local csv_file="$RESULTS_DIR/${scenario}_summary.csv"
        if [ -f "$csv_file" ]; then
            echo "### $scenario Results" >> "$summary_file"
            echo "" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            cat "$csv_file" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Files Generated

For each scenario, the following files are available:

### Results (\\\`{scenario}_experiment/\\\`)
- \\\`results.json\\\` - Complete experimental results including:
  - \\\`accuracy_matrix\\\` - Task accuracy after each training phase
  - \\\`trajectory_representations\\\` - Layer analysis data for every 10 epochs
  - \\\`summary\\\` - Average accuracy and forgetting metrics per order
- \\\`training_log.csv\\\` - Basic training metrics (if generated)

### Layer Analysis Data (\\\`trajectory_representations\\\` in results.json)
- **Epoch-wise forgetting trajectories** for all tasks across all task orders  
- **Layer-wise identifiability scores** (CLS tokens, image tokens) per transformer block
- **Accuracy degradation patterns** correlated with representation changes
- **Cross-order comparison data** for identifying optimal task sequences

## Key Research Questions

This experiment setup allows investigation of:

1. **Order sensitivity**: How much does task order affect final performance?
2. **Catastrophic forgetting patterns**: Which orders minimize forgetting?
3. **Transfer learning effects**: Do certain orders facilitate better forward transfer?
4. **Curriculum learning**: Is there an optimal difficulty progression?

## Next Steps

1. Compare results between class-incremental and task-incremental scenarios
2. Identify patterns in successful task orders
3. Analyze representation evolution for different orders
4. Use insights to design curriculum learning strategies

EOF
    
    echo "📄 Summary report created: $summary_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    local failed_scenarios=()
    local successful_scenarios=()
    
    echo "🎯 Starting Order Analysis experiments"
    echo "Scenarios to test: class_incremental, task_incremental"
    
    # Ask for confirmation
    read -p "This will run multiple task orders for each scenario. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    # Run experiments for both scenarios
    for scenario in "class_incremental" "task_incremental"; do
        if run_order_experiment "$scenario"; then
            successful_scenarios+=("$scenario")
        else
            failed_scenarios+=("$scenario")
            echo "⚠️  Continuing with next scenario..."
        fi
    done
    
    # Create summary
    create_summary
    
    # Final report
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo "🎉 ORDER ANALYSIS PIPELINE COMPLETED!"
    echo "======================================"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo "Successful: ${#successful_scenarios[@]} scenarios"
    echo "Failed: ${#failed_scenarios[@]} scenarios"
    
    if [ ${#successful_scenarios[@]} -gt 0 ]; then
        echo "✅ Successful scenarios: ${successful_scenarios[*]}"
    fi
    
    if [ ${#failed_scenarios[@]} -gt 0 ]; then
        echo "❌ Failed scenarios: ${failed_scenarios[*]}"
    fi
    
    echo ""
    echo "📁 All results available in: $RESULTS_DIR"
    echo "📄 Summary report: $RESULTS_DIR/summary_report.md"
    
    # Cleanup
    rm -f "$BASE_CONFIG"
}

# Check if required files exist
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found"
    echo "Please adjust the PYTHON_SCRIPT variable at the top of this script"
    exit 1
fi

# Run main function
main "$@"
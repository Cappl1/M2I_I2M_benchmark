#!/bin/bash

# Run ViT experiments across all datasets and perform analysis
# Usage: ./run_all_datasets.sh

set -e  # Exit on any error

# Configuration
DATASETS=("mnist" "cifar10" "fashion_mnist" "omniglot" "svhn" "imagenet")
BASE_CONFIG="/home/brothen/M2I_I2M_benchmark/configs/experiments/single_task_base.yml"  # Adjust this path if needed
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"    # Adjust this to your main experiment script
ANALYSIS_SCRIPT="/home/brothen/M2I_I2M_benchmark/analysis/analyze_single_task_experiment.py"  # The modified analyzer script
RESULTS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Base configuration template
cat > "$BASE_CONFIG" << 'EOF'
# Single Task Experiment Configuration
experiment: SingleTaskExperiment
experiment_type: SingleTaskExperiment

# Model configuration
model_name: ViT64
num_classes: 10
lr: 0.0003
momentum: 0.9
optimizer: adam
epochs: 50
minibatch_size: 128
hs: 250

# Device configuration
cuda: 3  # -1 for CPU, >= 0 for GPU ID

# Task configuration
dataset: DATASET_PLACEHOLDER
resized: resized
balanced: balanced
number_of_samples_per_class: 500

# Strategy configuration
strategy_name: Naive

# Analysis configuration
analysis_freq: 10  # Run layer analysis every N epochs

# Output configuration
output_dir: logs
save_model: true
use_tensorboard: false
EOF

echo "Starting experiments across all datasets..."
echo "Datasets to process: ${DATASETS[*]}"
echo "=================================="

# Function to run experiment for a single dataset
run_experiment() {
    local dataset=$1
    echo ""
    echo "🚀 Starting experiment for dataset: $dataset"
    echo "Time: $(date)"
    echo "-------------------------------------------"
    
    # Create dataset-specific config
    local config_file="config_${dataset}.yaml"
    sed "s/DATASET_PLACEHOLDER/$dataset/g" "$BASE_CONFIG" > "$config_file"
    
    # Run the experiment
    echo "Running experiment..."
    if python "$PYTHON_SCRIPT" --config "$config_file"; then
        echo "✅ Experiment completed successfully for $dataset"
    else
        echo "❌ Experiment failed for $dataset"
        return 1
    fi
    
    # Find the latest experiment directory
    local latest_exp_dir=$(find logs -name "experiment_*" -type d | sort | tail -1)
    
    if [ -z "$latest_exp_dir" ]; then
        echo "❌ Could not find experiment directory for $dataset"
        return 1
    fi
    
    echo "Found experiment directory: $latest_exp_dir"
    
    # Check if layer analysis directory exists
    local analysis_dir="${latest_exp_dir}/layer_analysis"
    if [ ! -d "$analysis_dir" ]; then
        echo "❌ No layer analysis directory found at: $analysis_dir"
        return 1
    fi
    
    # Run analysis
    echo "Running analysis..."
    if python -c "
import sys
sys.path.append('.')
from analysis.analyze_single_task_experiment import analyze_vit_results
analyze_vit_results('$analysis_dir', '$RESULTS_DIR/${dataset}_analysis')
"; then
        echo "✅ Analysis completed successfully for $dataset"
    else
        echo "❌ Analysis failed for $dataset"
        return 1
    fi
    
    # Copy experiment logs to results directory
    local dataset_results_dir="$RESULTS_DIR/${dataset}_experiment"
    cp -r "$latest_exp_dir" "$dataset_results_dir"
    echo "📁 Experiment logs copied to: $dataset_results_dir"
    
    # Clean up config file
    rm "$config_file"
    
    echo "✅ Complete pipeline finished for $dataset"
}

# Function to create summary report
create_summary() {
    echo ""
    echo "📊 Creating experiment summary..."
    
    local summary_file="$RESULTS_DIR/summary_report.md"
    
    cat > "$summary_file" << EOF
# ViT Experiment Summary Report

**Generated on:** $(date)
**Datasets processed:** ${DATASETS[*]}

## Experiment Configuration
- Model: ViT64
- Epochs: 50
- Learning Rate: 0.0003
- Batch Size: 128
- Samples per class: 500
- Classes per dataset: 10

## Results Overview

EOF
    
    for dataset in "${DATASETS[@]}"; do
        echo "### $dataset" >> "$summary_file"
        echo "" >> "$summary_file"
        
        local insights_file="$RESULTS_DIR/${dataset}_analysis/insights.json"
        if [ -f "$insights_file" ]; then
            echo "- Analysis completed ✅" >> "$summary_file"
            echo "- Results available in: \`${dataset}_analysis/\`" >> "$summary_file"
            echo "- Experiment logs in: \`${dataset}_experiment/\`" >> "$summary_file"
        else
            echo "- Analysis failed ❌" >> "$summary_file"
        fi
        echo "" >> "$summary_file"
    done
    
    cat >> "$summary_file" << EOF

## Files Generated

For each dataset, the following files are available:

### Analysis Results (\`{dataset}_analysis/\`)
- \`layer_progression.png\` - Class identifiability across layers
- \`training_evolution.png\` - Training dynamics heatmaps  
- \`early_vs_late_layers.png\` - Early vs late layer comparison
- \`summary_report.png\` - Combined visual summary
- \`insights.json\` - Quantitative insights

### Experiment Logs (\`{dataset}_experiment/\`)
- Complete experiment logs and checkpoints
- Layer analysis data (JSON files per epoch)
- Model weights (if saved)

## Next Steps

1. Review individual dataset results in their respective directories
2. Compare insights.json files across datasets for patterns
3. Examine summary_report.png files for visual comparisons
4. Use the analysis for further research or model improvements

EOF
    
    echo "📄 Summary report created: $summary_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    local failed_datasets=()
    local successful_datasets=()
    
    echo "🎯 Starting ViT experiments and analysis pipeline"
    echo "Total datasets to process: ${#DATASETS[@]}"
    
    # Run experiments for each dataset
    for dataset in "${DATASETS[@]}"; do
        if run_experiment "$dataset"; then
            successful_datasets+=("$dataset")
        else
            failed_datasets+=("$dataset")
            echo "⚠️  Continuing with next dataset..."
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
    echo "🎉 PIPELINE COMPLETED!"
    echo "======================================"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo "Successful: ${#successful_datasets[@]} datasets"
    echo "Failed: ${#failed_datasets[@]} datasets"
    
    if [ ${#successful_datasets[@]} -gt 0 ]; then
        echo "✅ Successful datasets: ${successful_datasets[*]}"
    fi
    
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo "❌ Failed datasets: ${failed_datasets[*]}"
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

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "❌ Error: Analysis script '$ANALYSIS_SCRIPT' not found"  
    echo "Please ensure the modified analyzer script is available"
    exit 1
fi

# Run main function
main "$@"
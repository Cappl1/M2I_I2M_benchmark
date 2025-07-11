#!/bin/bash

# Run ViT binary pairs experiments across all dataset combinations
# Usage: ./run_binary_pairs_all.sh

set -e  # Exit on any error

# Configuration
DATASETS=("mnist" "cifar10" "fashion_mnist" "omniglot" "svhn" "imagenet")
BASE_CONFIG="/home/brothen/M2I_I2M_benchmark/configs/experiments/binary_pairs_base.yml"
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"   # Adjust this to your main experiment script
ANALYSIS_SCRIPT="/home/brothen/M2I_I2M_benchmark/analysis/binary_pairs_analyzer.py"  # The modified analyzer script
RESULTS_DIR="binary_pairs_results_$(date +%Y%m%d_%H%M%S)"



# Create results directory
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Base configuration template for binary pairs
cat > "$BASE_CONFIG" << 'EOF'
# Binary Pairs Experiment Configuration
experiment_type: BinaryPairsExperiment
experiment_name: BinaryPairsExperiment

# Task pair configuration
task_pair: [0, 1]  # Which two tasks to train on

# Scenario configuration
scenario_type: task_incremental  # or class_incremental
num_tasks: 2  # Binary pairs = 2 tasks
dataset_a: DATASET_A_PLACEHOLDER  # First dataset
dataset_b: DATASET_B_PLACEHOLDER  # Second dataset

resized: resized
balanced: balanced
number_of_samples_per_class: 500

# Training parameters
epochs: 50
minibatch_size: 128
lr: 0.0003
optimizer: adam
momentum: 0.9

# Analysis parameters
analysis_freq: 10  # Run layer analysis every N epochs

# Model parameters
model_name: ViT64
# num_classes will be set dynamically:
# - Class incremental: 20 (10 × 2 datasets, single head)
# - Task incremental: 10 (10 classes per head, multiple heads)

# Device configuration
cuda: 3  # Use GPU 3, set to -1 for CPU

# Output configuration
output_dir: logs
EOF

# Generate all dataset pairs (excluding same dataset pairs)
declare -a DATASET_PAIRS
for i in "${!DATASETS[@]}"; do
    for j in "${!DATASETS[@]}"; do
        if [ $i -ne $j ]; then
            # Use proper delimiter to avoid splitting compound names like fashion_mnist
            DATASET_PAIRS+=("${DATASETS[$i]}|${DATASETS[$j]}")
        fi
    done
done

echo "Starting binary pairs experiments..."
echo "Total dataset pairs to process: ${#DATASET_PAIRS[@]}"
echo "Dataset pairs: ${DATASET_PAIRS[*]}"
echo "=================================="

# Function to run experiment for a single dataset pair
run_binary_experiment() {
    local pair="$1"
    
    # More reliable way to split the pair
    IFS='|' read -r dataset_a dataset_b <<< "$pair"
    
    echo ""
    echo "🚀 Starting binary pairs experiment: $dataset_a + $dataset_b"
    echo "Time: $(date)"
    echo "-------------------------------------------"
    
    # Create pair-specific config (use underscore for filenames)
    local config_file="config_${dataset_a}_${dataset_b}.yaml"
    sed "s/DATASET_A_PLACEHOLDER/$dataset_a/g; s/DATASET_B_PLACEHOLDER/$dataset_b/g" "$BASE_CONFIG" > "$config_file"
    
    # Run the experiment
    echo "Running binary pairs experiment..."
    if python "$PYTHON_SCRIPT" --config "$config_file"; then
        echo "✅ Experiment completed successfully for ${dataset_a}_${dataset_b}"
    else
        echo "❌ Experiment failed for ${dataset_a}_${dataset_b}"
        rm -f "$config_file"
        return 1
    fi
    
    # Find the latest experiment directory (handle both naming conventions)
    local latest_exp_dir=$(find logs -name "*BinaryPairsExperiment*" -type d | sort | tail -1)
    
    if [ -z "$latest_exp_dir" ]; then
        echo "❌ Could not find experiment directory for ${dataset_a}_${dataset_b}"
        rm -f "$config_file"
        return 1
    fi
    
    echo "Found experiment directory: $latest_exp_dir"
    
    # Check if layer analysis directories exist (both phase1 and phase2)
    local analysis_dir_phase1="${latest_exp_dir}/layer_analysis/phase1"
    local analysis_dir_phase2="${latest_exp_dir}/layer_analysis/phase2"
    
    if [ ! -d "$analysis_dir_phase1" ] && [ ! -d "$analysis_dir_phase2" ]; then
        echo "❌ No layer analysis directories found at: $latest_exp_dir/layer_analysis/"
        rm -f "$config_file"
        return 1
    fi
    
    # Run binary pairs analysis (handles both phases together)
    local pair_results_dir="$RESULTS_DIR/${dataset_a}_${dataset_b}_analysis"
    
    if [ -d "$analysis_dir_phase1" ] && [ -d "$analysis_dir_phase2" ]; then
        echo "Running binary pairs analysis..."
        if python -c "
import sys
sys.path.append('.')
from analysis.binary_pairs_analyzer import analyze_binary_pairs_results
analyze_binary_pairs_results('$analysis_dir_phase1', '$analysis_dir_phase2', '$pair_results_dir')
"; then
            echo "✅ Binary pairs analysis completed successfully"
        else
            echo "❌ Binary pairs analysis failed"
        fi
    elif [ -d "$analysis_dir_phase1" ]; then
        echo "⚠️  Only phase1 analysis found, running single task analysis..."
        if python -c "
import sys
sys.path.append('.')
from analysis.vit_analyzer import analyze_vit_results
analyze_vit_results('$analysis_dir_phase1', '$pair_results_dir/phase1_only')
"; then
            echo "✅ Phase1 analysis completed successfully"
        else
            echo "❌ Phase1 analysis failed"
        fi
    else
        echo "❌ No analysis directories found"
        rm -f "$config_file"
        return 1
    fi
    
    # Copy experiment logs to results directory
    local pair_exp_dir="$RESULTS_DIR/${dataset_a}_${dataset_b}_experiment"
    cp -r "$latest_exp_dir" "$pair_exp_dir"
    echo "📁 Experiment logs copied to: $pair_exp_dir"
    
    # Clean up config file
    rm -f "$config_file"
    
    echo "✅ Complete pipeline finished for ${dataset_a}_${dataset_b}"
}

# Function to create summary report
create_summary() {
    echo ""
    echo "📊 Creating binary pairs experiment summary..."
    
    local summary_file="$RESULTS_DIR/summary_report.md"
    
    cat > "$summary_file" << EOF
# Binary Pairs ViT Experiment Summary Report

**Generated on:** $(date)
**Dataset pairs processed:** ${#DATASET_PAIRS[@]} total combinations
**Base datasets:** ${DATASETS[*]}

## Experiment Configuration
- Model: ViT64
- Epochs: 50
- Learning Rate: 0.0003
- Batch Size: 128
- Samples per class: 500
- Classes per dataset: 10
- Scenario: Task Incremental (10 classes total)

## Dataset Pairs Tested

EOF
    
    # Group by first dataset for better organization
    for dataset_a in "${DATASETS[@]}"; do
        echo "### $dataset_a paired with:" >> "$summary_file"
        echo "" >> "$summary_file"
        
        for dataset_b in "${DATASETS[@]}"; do
            if [ "$dataset_a" != "$dataset_b" ]; then
                echo "- **$dataset_a + $dataset_b**" >> "$summary_file"
                
                local binary_insights="$RESULTS_DIR/${dataset_a}_${dataset_b}_analysis/binary_insights.json"
                
                if [ -f "$binary_insights" ]; then
                    echo "  - ✅ Analysis completed" >> "$summary_file"
                    echo "  - Results: \`${dataset_a}_${dataset_b}_analysis/\`" >> "$summary_file"
                    echo "  - Logs: \`${dataset_a}_${dataset_b}_experiment/\`" >> "$summary_file"
                else
                    echo "  - ❌ Analysis failed" >> "$summary_file"
                fi
                echo "" >> "$summary_file"
            fi
        done
        echo "" >> "$summary_file"
    done
    
    cat >> "$summary_file" << EOF

## Results Structure

For each dataset pair, the following files are available:

### Analysis Results (\`{dataset_a}_{dataset_b}_analysis/\`)
- \`forgetting_dynamics.png\` - Complete forgetting analysis across both phases
- \`learning_vs_forgetting.png\` - Direct comparison of learning vs forgetting patterns  
- \`binary_insights.json\` - Quantitative insights about forgetting rates and layer patterns

### Experiment Logs (\`{dataset_a}_{dataset_b}_experiment/\`)
- \`training_log_phase1.csv\` - Training metrics for phase 1
- \`training_log_phase2.csv\` - Training metrics for phase 2 (includes forgetting)
- \`layer_analysis/phase1/\` - Raw analysis data for phase 1
- \`layer_analysis/phase2/\` - Raw analysis data for phase 2

## Key Research Questions

This experiment setup allows investigation of:

1. **Cross-domain forgetting patterns**: How does learning dataset B affect representations of dataset A across different ViT layers?

2. **Dataset-specific layer specialization**: Do different dataset pairs show different patterns of layer-wise class information?

3. **Catastrophic forgetting dynamics**: At which layers does forgetting manifest most strongly?

4. **Dataset similarity effects**: Do more similar datasets (e.g., MNIST/Fashion-MNIST) show different forgetting patterns than dissimilar ones (e.g., MNIST/CIFAR-10)?

## Next Steps

1. Compare phase2 analysis across different dataset pairs to identify forgetting patterns
2. Look for correlations between dataset similarity and forgetting severity
3. Examine which layers are most susceptible to catastrophic forgetting
4. Use insights to design better continual learning strategies

EOF
    
    echo "📄 Summary report created: $summary_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    local failed_pairs=()
    local successful_pairs=()
    
    echo "🎯 Starting Binary Pairs ViT experiments and analysis pipeline"
    echo "Total dataset pairs to process: ${#DATASET_PAIRS[@]}"
    echo "This will take a very long time... (~${#DATASET_PAIRS[@]} * 50 epochs * 2 phases)"
    
    # Ask for confirmation
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    # Run experiments for each dataset pair
    for pair in "${DATASET_PAIRS[@]}"; do
        IFS='|' read -r dataset_a dataset_b <<< "$pair"
        local pair_name="${dataset_a}_${dataset_b}"
        
        if run_binary_experiment "$pair"; then
            successful_pairs+=("$pair_name")
        else
            failed_pairs+=("$pair_name")
            echo "⚠️  Continuing with next pair..."
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
    echo "🎉 BINARY PAIRS PIPELINE COMPLETED!"
    echo "======================================"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo "Successful: ${#successful_pairs[@]} pairs"
    echo "Failed: ${#failed_pairs[@]} pairs"
    
    if [ ${#successful_pairs[@]} -gt 0 ]; then
        echo "✅ Successful pairs: ${successful_pairs[*]}"
    fi
    
    if [ ${#failed_pairs[@]} -gt 0 ]; then
        echo "❌ Failed pairs: ${failed_pairs[*]}"
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
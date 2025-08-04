#!/bin/bash
# Test version - runs just 2 pairs with Naive strategy

set -e

# Configuration
DATASETS=("mnist" "fashion_mnist")  # Just 2 datasets for testing
STRATEGIES=("Naive")  # Just 1 strategy for testing
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
RESULTS_DIR="test_binary_pairs_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "TEST: Binary Pairs Analysis"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo "Total experiments: $((${#DATASETS[@]} * (${#DATASETS[@]} - 1) * ${#STRATEGIES[@]})) (2 pairs × 1 strategy)"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Create directories for each strategy
for strategy in "${STRATEGIES[@]}"; do
    mkdir -p "$RESULTS_DIR/${strategy,,}"
done

echo "================================================"
echo "Running Test Binary Pairs"
echo "================================================"

TOTAL_COUNT=0
for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "🚀 STRATEGY: $strategy"
    echo "========================================"
    
    PAIR_COUNT=0
    for ((i=0; i<${#DATASETS[@]}; i++)); do
        for ((j=0; j<${#DATASETS[@]}; j++)); do
            if [ $i -eq $j ]; then
                continue
            fi
            
            DATASET_A="${DATASETS[i]}"
            DATASET_B="${DATASETS[j]}"
            PAIR_NAME="${DATASET_A}_to_${DATASET_B}"
            PAIR_COUNT=$((PAIR_COUNT + 1))
            TOTAL_COUNT=$((TOTAL_COUNT + 1))
            
            echo ""
            echo "🔥 $strategy: $PAIR_COUNT/2 - $DATASET_A → $DATASET_B"
            echo "----------------------------------------"
            
            CONFIG_FILE="config_test_${strategy,,}_${PAIR_NAME}.yaml"
            
            cat > "$CONFIG_FILE" << EOF
# Test Binary Pairs Configuration
experiment_type: StrategyBinaryPairsExperiment

# Model configuration
model_name: ViT64
num_classes: 20
lr: 0.0003
optimizer: adam
epochs: 10  # Reduced for testing
minibatch_size: 128

# Device configuration  
cuda: 0

# Binary pairs configuration
dataset_a: $DATASET_A
dataset_b: $DATASET_B
scenario_type: class_incremental

# Dataset configuration
balanced: balanced
number_of_samples_per_class: 100  # Reduced for testing

# Strategy configuration
strategy_name: $strategy

# Analysis configuration
analysis_freq: 5  # More frequent for testing
analyze_representations: true

# Minimal patch analysis
patch_analysis: false  # Disabled for testing

# Output configuration
output_dir: logs
save_model: false
use_tensorboard: false
EOF

            echo "Running $strategy: $DATASET_A → $DATASET_B..."
            
            if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE"; then
                echo "✅ $strategy $PAIR_NAME completed"
                
                # Find and copy experiment directory
                EXP_DIR=$(find logs -name "experiment_*" -type d | sort | tail -1)
                if [ -n "$EXP_DIR" ]; then
                    TARGET_DIR="$RESULTS_DIR/${strategy,,}/${PAIR_NAME}"
                    cp -r "$EXP_DIR" "$TARGET_DIR"
                    echo "✅ Results copied to: $TARGET_DIR"
                fi
            else
                echo "❌ $strategy $PAIR_NAME failed"
            fi
            
            rm -f "$CONFIG_FILE"
        done
    done
    
    echo ""
    echo "✅ $strategy strategy completed: $PAIR_COUNT pairs"
done

echo ""
echo "================================================"
echo "✅ TEST Complete!"
echo "================================================"
echo "Results directory: $RESULTS_DIR"
echo "Total experiments: $TOTAL_COUNT"
echo "Runtime: $SECONDS seconds" 
#!/bin/bash
# Run patch analysis on multiple datasets for comparison

set -e

# List of datasets to analyze
DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn")
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
ANALYZER_SCRIPT="/home/brothen/M2I_I2M_benchmark/analysis/patch_evolution_analyzer.py"
RESULTS_DIR="multi_dataset_patch_analysis_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Running Multi-Dataset Patch Analysis"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Array to store experiment directories for comparison
EXPERIMENT_DIRS=()

# Run each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "🔥 RUNNING DATASET: $DATASET"
    echo "----------------------------------------"
    
    # Create config file for this dataset
    CONFIG_FILE="config_patch_${DATASET}.yaml"
    
    cat > "$CONFIG_FILE" << EOF
# Single Task Experiment Configuration with Patch Analysis
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
cuda: 0  # -1 for CPU, >= 0 for GPU ID

# Task configuration
dataset: $DATASET
resized: resized
balanced: balanced
number_of_samples_per_class: 500

# Strategy configuration
strategy_name: Naive

# Standard analysis configuration
analysis_freq: 10  # Run layer analysis every N epochs

# PATCH ANALYSIS CONFIGURATION - ENABLED
patch_analysis: true  # Enable patch-level importance analysis
patch_analysis_freq: 10  # Run patch analysis every N epochs  
patch_analysis_max_batches: 100  # Number of batches to analyze

# Output configuration
output_dir: logs
save_model: false
use_tensorboard: false
EOF

    echo "Running experiment for $DATASET..."
    
    # Run the experiment
    if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE"; then
        echo "✅ $DATASET experiment completed successfully"
        
        # Find the experiment directory
        EXP_DIR=$(find logs -name "experiment_*" -type d | sort | tail -1)
        
        if [ -n "$EXP_DIR" ]; then
            # Copy results
            cp -r "$EXP_DIR" "$RESULTS_DIR/${DATASET}_experiment"
            EXPERIMENT_DIRS+=("$RESULTS_DIR/${DATASET}_experiment")
            echo "✅ Results copied to: $RESULTS_DIR/${DATASET}_experiment"
        else
            echo "❌ Could not find experiment directory for $DATASET"
        fi
    else
        echo "❌ $DATASET experiment failed"
    fi
    
    # Clean up config file
    rm -f "$CONFIG_FILE"
done

echo ""
echo "================================================"
echo "🔍 Running Evolution Analysis for Each Dataset"
echo "================================================"

# Run evolution analysis for each dataset
for DATASET in "${DATASETS[@]}"; do
    DATASET_DIR="$RESULTS_DIR/${DATASET}_experiment"
    
    if [ -d "$DATASET_DIR" ]; then
        echo "Analyzing evolution for $DATASET..."
        
        if python "$ANALYZER_SCRIPT" "$DATASET_DIR" --output "$RESULTS_DIR/${DATASET}_evolution"; then
            echo "✅ $DATASET evolution analysis completed"
        else
            echo "❌ $DATASET evolution analysis failed"
        fi
    fi
done

echo ""
echo "================================================"
echo "🔬 Running Cross-Dataset Comparison"
echo "================================================"

# Run comparison analysis if we have multiple datasets
if [ ${#EXPERIMENT_DIRS[@]} -gt 1 ]; then
    echo "Running cross-dataset comparison..."
    
    # Build comparison command
    COMPARE_CMD="python $ANALYZER_SCRIPT ${EXPERIMENT_DIRS[0]}"
    for ((i=1; i<${#EXPERIMENT_DIRS[@]}; i++)); do
        COMPARE_CMD+=" --compare ${EXPERIMENT_DIRS[i]}"
    done
    COMPARE_CMD+=" --output $RESULTS_DIR/dataset_comparison"
    
    if eval "$COMPARE_CMD"; then
        echo "✅ Cross-dataset comparison completed"
    else
        echo "❌ Cross-dataset comparison failed"
    fi
else
    echo "Not enough datasets for comparison"
fi

echo ""
echo "================================================"
echo "📊 Creating Summary Report"
echo "================================================"

# Create comprehensive summary
cat > "$RESULTS_DIR/README.md" << EOF
# Multi-Dataset Patch Analysis Results

## Experiment Overview
- **Datasets Analyzed**: ${DATASETS[*]}
- **Model**: ViT64 (8x8 patch grid, 64 patches total)
- **Analysis Type**: Patch-level class identifiability
- **Epochs**: 50 per dataset
- **Generated**: $(date)

## Results Structure

### Individual Dataset Results
$(for dataset in "${DATASETS[@]}"; do
    if [ -d "$RESULTS_DIR/${dataset}_experiment" ]; then
        echo "- **${dataset}_experiment/**: Training results and patch analysis"
        echo "  - \`patch_analysis/\`: Patch importance heatmaps per epoch"
        echo "  - \`training_log.csv\`: Training metrics"
        echo "- **${dataset}_evolution/**: Evolution analysis and stability metrics"
    fi
done)

### Cross-Dataset Comparison
- **dataset_comparison/**: Side-by-side patch pattern comparison
- **dataset_comparison.png**: Visual comparison of final patch importance patterns

## Key Questions Addressed

### 1. **Which patches are most important for each dataset?**
- Check individual patch heatmaps in each \`patch_analysis/\` directory
- Look for dataset-specific spatial patterns

### 2. **How stable are patch patterns during training?**
- See \`patch_stability.png\` in each evolution directory
- Higher correlation = more stable patterns

### 3. **Which patches persist as important throughout training?**
- Check \`important_patches_persistence.png\`
- Red/bright areas = consistently important patches

### 4. **Do different datasets show different spatial preferences?**
- Compare \`spatial_patterns.png\` across datasets
- Check center vs periphery preferences

### 5. **How do patch patterns differ between datasets?**
- See \`dataset_comparison.png\` for side-by-side comparison
- Look for dataset-specific "signatures"

## Key Findings Expected

### MNIST
- **Border patches** likely most important (digit shapes/strokes)
- **Center patches** less discriminative
- **High stability** expected (simple, consistent patterns)

### Fashion-MNIST  
- **More distributed importance** (clothing textures)
- **Medium stability** (more variation than MNIST)

### CIFAR-10
- **Complex spatial patterns** (natural images)
- **Lower stability** (more complex features)

### SVHN
- **Center patches** potentially important (digit focus)
- **Variable patterns** (real-world digit variations)

## Applications for Continual Learning

1. **Parameter Protection**: Protect weights corresponding to stable, important patches
2. **Attention Mechanisms**: Focus on dataset-specific important patches  
3. **Replay Strategies**: Prioritize patches that are consistently important
4. **Task Detection**: Use patch importance patterns as task signatures

## Next Steps

1. **Analyze patch protection strategies** in continual learning scenarios
2. **Test attention masking** based on patch importance
3. **Design patch-aware replay buffers**
4. **Compare with other architectures** (ResNet, etc.)
EOF

echo "📝 Summary report created: $RESULTS_DIR/README.md"

echo ""
echo "================================================"
echo "✅ Multi-Dataset Patch Analysis Complete!"
echo "================================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "🔍 Key outputs:"
for dataset in "${DATASETS[@]}"; do
    if [ -d "$RESULTS_DIR/${dataset}_experiment" ]; then
        echo "  - $dataset heatmaps: $RESULTS_DIR/${dataset}_experiment/patch_analysis/"
        echo "  - $dataset evolution: $RESULTS_DIR/${dataset}_evolution/"
    fi
done

if [ -d "$RESULTS_DIR/dataset_comparison" ]; then
    echo "  - Cross-dataset comparison: $RESULTS_DIR/dataset_comparison/"
fi

echo ""
echo "🚀 Ready for continual learning insights!" 
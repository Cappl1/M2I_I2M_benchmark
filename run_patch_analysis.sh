#!/bin/bash
# Run single task experiment with patch analysis enabled

set -e

# Configuration
DATASET="${1:-cifar10}"  # Default to CIFAR-10
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"  # Adjust path as needed
ANALYZER_SCRIPT="/home/brothen/M2I_I2M_benchmark/analysis/patch_evolution_analyzer.py"
RESULTS_DIR="patch_analysis_results_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Running Single Task Experiment with Patch Analysis"
echo "================================================"
echo "Dataset: $DATASET"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Create config file with patch analysis enabled
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

echo "Running experiment with patch analysis..."
echo "-------------------------------------------"

# Run the experiment
if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE"; then
    echo "✅ Experiment completed successfully"
else
    echo "❌ Experiment failed"
    exit 1
fi

# Find the experiment directory
EXP_DIR=$(find logs -name "experiment_*" -type d | sort | tail -1)

if [ -z "$EXP_DIR" ]; then
    echo "❌ Could not find experiment directory"
    exit 1
fi

echo "Found experiment directory: $EXP_DIR"

# Check if patch analysis was created
PATCH_DIR="${EXP_DIR}/patch_analysis"
if [ ! -d "$PATCH_DIR" ]; then
    echo "❌ No patch analysis directory found"
    exit 1
fi

# Copy results
cp -r "$EXP_DIR" "$RESULTS_DIR/${DATASET}_experiment"

# Run evolution analysis
echo ""
echo "Running patch evolution analysis..."
echo "-------------------------------------------"

if python "$ANALYZER_SCRIPT" "$EXP_DIR" --output "$RESULTS_DIR/${DATASET}_evolution"; then
    echo "✅ Evolution analysis completed"
else
    echo "❌ Evolution analysis failed"
fi

# Create summary
echo ""
echo "Creating summary..."
echo "-------------------------------------------"

cat > "$RESULTS_DIR/README.md" << EOF
# Patch Analysis Results for $DATASET

## Experiment Details
- Dataset: $DATASET
- Model: ViT64 (Vision Transformer)
- Epochs: 50
- Patch Analysis Frequency: Every 10 epochs

## Results Structure

### ${DATASET}_experiment/
- \`training_log.csv\` - Training metrics
- \`layer_analysis/\` - Standard layer-wise analysis
- \`patch_analysis/\` - Patch importance analysis
  - \`patch_importance_epoch_*.png\` - Heatmaps for each analyzed epoch
  - \`patch_importance_epoch_*.json\` - Raw importance data
  - \`patch_importance_epoch_*.npz\` - Numpy format for further processing
  - \`patch_importance_summary.json\` - Summary statistics

### ${DATASET}_evolution/
- \`patch_stability.png\` - How stable patch importance is during training
- \`important_patches_persistence.png\` - Which patches remain important
- \`layer_evolution_heatmap.png\` - Layer-wise evolution over time
- \`spatial_patterns.png\` - Spatial analysis of patch importance
- \`evolution_summary.json\` - Quantitative metrics

## Key Questions to Explore

1. **Which patches are most important for classification?**
   - Check the heatmaps to see spatial patterns
   - Look for consistency across layers

2. **How stable is patch importance during training?**
   - See patch_stability.png for correlation analysis
   - Higher correlation = more stable importance patterns

3. **Do important patches cluster spatially?**
   - Check spatial_patterns.png for center vs periphery analysis
   - Dataset-specific patterns may emerge

4. **How can this inform continual learning?**
   - Stable, important patches → parameters to protect
   - Task-specific patterns → potential conflicts to manage

## Next Steps

1. Compare patch patterns across different datasets
2. Use patch importance for parameter protection in CL
3. Design attention mechanisms that focus on important patches
4. Create patch-based replay strategies

Generated on: $(date)
EOF

# Clean up
rm -f "$CONFIG_FILE"

# Final summary
echo ""
echo "================================================"
echo "✅ Patch Analysis Complete!"
echo "================================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key outputs:"
echo "- Experiment data: $RESULTS_DIR/${DATASET}_experiment/"
echo "- Evolution analysis: $RESULTS_DIR/${DATASET}_evolution/"
echo "- Patch heatmaps: $RESULTS_DIR/${DATASET}_experiment/patch_analysis/"
echo ""
echo "To compare multiple datasets, run:"
echo "python $ANALYZER_SCRIPT exp1_dir --compare exp2_dir exp3_dir --output comparison"
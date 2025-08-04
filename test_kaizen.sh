#!/bin/bash
# Simple Kaizen Test - Just get it working first

PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
CUDA_DEVICES="3"

if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

echo "🧪 Testing Kaizen with patch analysis enabled..."

# Create working config
cat > "kaizen_test.yaml" << EOF
# Working Kaizen Test Configuration
experiment_type: StrategyBinaryPairsExperiment

# Model configuration
model_name: ViT64
num_classes: 10
lr: 0.001
optimizer: adam
epochs: 10
minibatch_size: 64

# Device configuration
cuda: 1

# Single task MNIST
dataset_a: mnist
dataset_b: mnist
scenario_type: class_incremental

# Dataset configuration
balanced: balanced
number_of_samples_per_class: 100

# Kaizen configuration
strategy_name: Kaizen
ssl_method: simclr
memory_size_percent: 1
kd_classifier_weight: 1.0

# Enable analysis to avoid the bug
analysis_freq: 10
analyze_representations: true
patch_analysis: true
patch_analysis_freq: 10
patch_analysis_max_batches: 5
analyze_all_tasks: false
track_task_performance: false
analyze_during_training: false

# Output
output_dir: ./kaizen_test_output
save_model: false
use_tensorboard: false
verbose: true
EOF

echo "🚀 Running Kaizen test..."
python "$PYTHON_SCRIPT" --config "kaizen_test.yaml"

echo ""
echo "✅ Test complete! Check kaizen_test_output/ for results"

# Clean up
rm -f "kaizen_test.yaml"
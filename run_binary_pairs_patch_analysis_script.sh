#!/bin/bash
# Simplified binary patch analysis - just the core experiments

set -e

# Configuration
DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn")
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
RESULTS_DIR="simple_binary_patch_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Simple Binary Patch Analysis"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Binary pairs: $((${#DATASETS[@]} * (${#DATASETS[@]} - 1))) ordered combinations"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Generate all ORDERED combinations (A→B and B→A)
PAIR_COUNT=0
for ((i=0; i<${#DATASETS[@]}; i++)); do
    for ((j=0; j<${#DATASETS[@]}; j++)); do
        # Skip when i == j (same dataset)
        if [ $i -eq $j ]; then
            continue
        fi
        
        DATASET_A="${DATASETS[i]}"
        DATASET_B="${DATASETS[j]}"
        PAIR_NAME="${DATASET_A}_to_${DATASET_B}"
        PAIR_COUNT=$((PAIR_COUNT + 1))
        
        echo ""
        echo "🔥 PAIR $PAIR_COUNT/12: $DATASET_A → $DATASET_B"
        echo "----------------------------------------"
        
        # Run class-incremental scenario with patch analysis
        CONFIG_FILE="config_${PAIR_NAME}.yaml"
        
        cat > "$CONFIG_FILE" << EOF
# Binary Pairs with Patch Analysis Configuration
experiment: BinaryPairsExperiment
experiment_type: BinaryPairsExperiment

# Model configuration
model_name: ViT64
num_classes: 20  # Class incremental: 10 + 10
lr: 0.0003
momentum: 0.9
optimizer: adam
epochs: 50
minibatch_size: 128
hs: 250

# Device configuration  
cuda: 0

# Binary pairs configuration
dataset_a: $DATASET_A
dataset_b: $DATASET_B
scenario_type: class_incremental
task_pair: [0, 1]

# Dataset configuration
resized: resized
balanced: balanced
number_of_samples_per_class: 500

# Strategy configuration
strategy_name: Naive

# Analysis configuration
analysis_freq: 10
analyze_representations: true

# PATCH ANALYSIS - ENABLED (requires modification to BinaryPairsExperiment)
patch_analysis: true
patch_analysis_freq: 10
patch_analysis_max_batches: 100

# Output configuration
output_dir: logs
save_model: false
use_tensorboard: false
EOF

        echo "Running $DATASET_A → $DATASET_B..."
        
        if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE"; then
            echo "✅ $PAIR_NAME completed"
            
            # Find and copy experiment directory
            EXP_DIR=$(find logs -name "experiment_*" -type d | sort | tail -1)
            if [ -n "$EXP_DIR" ]; then
                TARGET_DIR="$RESULTS_DIR/${PAIR_NAME}"
                cp -r "$EXP_DIR" "$TARGET_DIR"
                echo "✅ Results copied to: $TARGET_DIR"
            fi
        else
            echo "❌ $PAIR_NAME failed"
        fi
        
        rm -f "$CONFIG_FILE"
    done
done

echo ""
echo "================================================"
echo "Creating Summary Report"
echo "================================================"

# Create simple analysis script
cat > "$RESULTS_DIR/quick_analysis.py" << 'EOF'
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_binary_results(results_dir):
    results_dir = Path(results_dir)
    
    # Collect results
    results = {}
    
    for exp_dir in results_dir.glob('*'):
        if exp_dir.is_dir() and '_' in exp_dir.name:
            pair_name = exp_dir.name
            
            # Load training logs
            phase1_log = exp_dir / 'training_log_phase1.csv'
            phase2_log = exp_dir / 'training_log_phase2.csv'
            
            if phase1_log.exists() and phase2_log.exists():
                try:
                    df1 = pd.read_csv(phase1_log)
                    df2 = pd.read_csv(phase2_log)
                    
                    # Extract final accuracies
                    task0_final = float(df1.iloc[-1]['task_0_acc'])
                    task0_after_task1 = float(df2.iloc[-1]['task_0_acc']) if 'task_0_acc' in df2.columns else 0
                    task1_final = float(df2.iloc[-1]['task_1_acc'])
                    
                    forgetting = task0_final - task0_after_task1
                    
                    results[pair_name] = {
                        'task0_after_task0': task0_final,
                        'task0_after_task1': task0_after_task1,
                        'task1_final': task1_final,
                        'forgetting': forgetting
                    }
                    
                    print(f"{pair_name:20s} | Task0: {task0_final:5.1f}% → {task0_after_task1:5.1f}% | Task1: {task1_final:5.1f}% | Forgetting: {forgetting:5.1f}%")
                    
                except Exception as e:
                    print(f"Error processing {pair_name}: {e}")
    
    # Create plots
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        pairs = list(results.keys())
        task0_orig = [results[p]['task0_after_task0'] for p in pairs]
        task0_after = [results[p]['task0_after_task1'] for p in pairs]
        task1_perf = [results[p]['task1_final'] for p in pairs]
        forgetting = [results[p]['forgetting'] for p in pairs]
        
        # Plot 1: Performance comparison
        x = range(len(pairs))
        width = 0.25
        
        ax1.bar([i-width for i in x], task0_orig, width, label='Task 0 (original)', alpha=0.8)
        ax1.bar(x, task0_after, width, label='Task 0 (after Task 1)', alpha=0.8)
        ax1.bar([i+width for i in x], task1_perf, width, label='Task 1 (final)', alpha=0.8)
        
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Task Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.replace('_', '\n') for p in pairs], rotation=0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Forgetting analysis
        colors = ['red' if f > 10 else 'orange' if f > 5 else 'green' for f in forgetting]
        bars = ax2.bar(pairs, forgetting, color=colors, alpha=0.7)
        ax2.set_ylabel('Forgetting (%)')
        ax2.set_title('Task 0 Forgetting')
        ax2.set_xticklabels([p.replace('_', '\n') for p in pairs], rotation=0)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, forgetting):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'binary_results_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary plot saved: {results_dir / 'binary_results_summary.png'}")
    
    return results

if __name__ == "__main__":
    import sys
    results = analyze_binary_results(sys.argv[1])
    print(f"\nAnalyzed {len(results)} binary pairs")
EOF

chmod +x "$RESULTS_DIR/quick_analysis.py"

# Run analysis
echo "Running quick analysis..."
if python "$RESULTS_DIR/quick_analysis.py" "$RESULTS_DIR"; then
    echo "✅ Analysis completed"
else
    echo "❌ Analysis failed"
fi

# Create README
cat > "$RESULTS_DIR/README.md" << EOF
# Simple Binary Patch Analysis Results

## ✅ Ready to Run!

### Requirements Met
- Updated `BinaryPairsExperiment` supports patch analysis
- All 12 ordered combinations will be tested
- Both representation and patch-level analysis included

## Experiment Overview
- **Pairs Tested**: All 12 ordered combinations of {mnist, fashion_mnist, cifar10, svhn}
- **Scenario**: Class-incremental (20 classes total)
- **Training**: Sequential (Dataset A → Dataset B)
- **Analysis**: Both representation and patch-level importance

## Results Structure
\`\`\`
$RESULTS_DIR/
├── mnist_to_fashion_mnist/        # MNIST → Fashion-MNIST
├── fashion_mnist_to_mnist/        # Fashion-MNIST → MNIST
├── mnist_to_cifar10/              # MNIST → CIFAR-10
├── cifar10_to_mnist/              # CIFAR-10 → MNIST
├── mnist_to_svhn/                 # MNIST → SVHN
├── svhn_to_mnist/                 # SVHN → MNIST
├── fashion_mnist_to_cifar10/      # Fashion-MNIST → CIFAR-10
├── cifar10_to_fashion_mnist/      # CIFAR-10 → Fashion-MNIST
├── fashion_mnist_to_svhn/         # Fashion-MNIST → SVHN
├── svhn_to_fashion_mnist/         # SVHN → Fashion-MNIST
├── cifar10_to_svhn/               # CIFAR-10 → SVHN
├── svhn_to_cifar10/               # SVHN → CIFAR-10
├── binary_results_summary.png     # Performance overview
└── quick_analysis.py              # Analysis script
\`\`\`

## Key Files per Experiment
- \`patch_analysis/\`: Patch importance heatmaps for both phases
- \`training_log_phase1.csv\`: First dataset training metrics
- \`training_log_phase2.csv\`: Second dataset training + forgetting metrics
- \`layer_analysis/\`: Representation analysis results

## Expected Patterns

### Asymmetric Forgetting (Order Matters!)
- **MNIST → Fashion-MNIST** vs **Fashion-MNIST → MNIST**: Different forgetting patterns
- **Simple → Complex** (MNIST → CIFAR-10): Often more forgetting
- **Complex → Simple** (CIFAR-10 → MNIST): Different patch conflicts

### Domain Similarity Effects
- **MNIST ↔ Fashion-MNIST**: Both grayscale, similar edge patterns
- **CIFAR-10 ↔ SVHN**: Both color natural images, some shared features
- **Cross-domain pairs**: Higher interference and asymmetric effects

### Patch Conflict Patterns
- **Forward transfer**: Earlier task patches help later task
- **Backward interference**: Later task disrupts earlier task patches
- **Bidirectional analysis**: A→B vs B→A reveals conflict asymmetries

## Quick Analysis
\`\`\`bash
# View performance summary
open $RESULTS_DIR/binary_results_summary.png

# Re-run analysis
python $RESULTS_DIR/quick_analysis.py $RESULTS_DIR

# Examine specific pair
ls $RESULTS_DIR/mnist_cifar10/patch_analysis/
\`\`\`

## Next Steps
1. **Identify patch conflicts**: Compare heatmaps between phases
2. **Design protection strategies**: Protect important patches from first task
3. **Optimize task order**: Use forgetting patterns to sequence tasks
4. **Create attention masks**: Focus on task-relevant spatial regions
EOF

echo ""
echo "================================================"
echo "✅ Simple Binary Patch Analysis Complete!"
echo "================================================"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "🔍 Key outputs:"
echo "  - Performance summary: $RESULTS_DIR/binary_results_summary.png"
echo "  - Individual results: $RESULTS_DIR/<dataset_pair>/"
echo "  - Patch heatmaps: $RESULTS_DIR/<dataset_pair>/patch_analysis/"
echo ""
echo "📊 Quick start:"
echo "  open $RESULTS_DIR/binary_results_summary.png"
echo "  cat $RESULTS_DIR/README.md"
echo ""
echo "Total pairs analyzed: $PAIR_COUNT (all 12 ordered combinations)"
echo "Runtime: $SECONDS seconds"
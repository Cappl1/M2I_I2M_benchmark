#!/bin/bash
# RDBP Strategy Binary Pairs Analysis
# — outputs are written directly to their final location —
#
# CUDA Configuration Examples:
#   CUDA_DEVICES="0"        # Use only GPU 0 (default)
#   CUDA_DEVICES="1"        # Use only GPU 1
#   CUDA_DEVICES="0,1"      # Use GPUs 0 and 1 (multi-GPU)
#   CUDA_DEVICES=""         # Use CPU only
#
# To change GPU, edit the CUDA_DEVICES variable below

set -e

# ───────────────────────── Configuration ──────────────────────────
DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn")
STRATEGIES=("RDBP")  # Only RDBP strategy
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
RESULTS_DIR="rdbp_binary_pairs_analysis_$(date +%Y%m%d_%H%M%S)"
REPEATS=5

# ───────────────────────── CUDA Configuration ──────────────────────────
# Set which GPU(s) to use (comma-separated for multiple GPUs)
# Examples: 
#   CUDA_DEVICES="0"        # Use only GPU 0
#   CUDA_DEVICES="1"        # Use only GPU 1  
#   CUDA_DEVICES="0,1"      # Use GPUs 0 and 1
#   CUDA_DEVICES=""         # Use CPU only
CUDA_DEVICES="2"

# Set CUDA_VISIBLE_DEVICES environment variable
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "🔧 CUDA_VISIBLE_DEVICES set to: $CUDA_DEVICES"
    # Verify GPU availability
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | while IFS=, read idx name total used; do
            if [[ "$CUDA_DEVICES" == *"$idx"* ]]; then
                echo "  ✅ GPU $idx: $name (${used}MB/${total}MB used)"
            else
                echo "  ❌ GPU $idx: $name (not visible to this script)"
            fi
        done
    fi
else
    export CUDA_VISIBLE_DEVICES=""
    echo "🔧 CUDA disabled - using CPU only"
fi

echo "================================================"
echo "RDBP Strategy Binary Pairs Analysis - $REPEATS Repeats"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Strategy: RDBP (ReLUDown + Decreasing Backpropagation) - matched config"
echo "Repeats: $REPEATS"
echo "CUDA Devices: ${CUDA_DEVICES:-'CPU only'}"
echo "Total experiments per run: $((${#DATASETS[@]} * (${#DATASETS[@]} - 1)))"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

for repeat in $(seq 1 $REPEATS); do
    echo ""
    echo "=============================="
    echo " RUN $repeat / $REPEATS"
    echo "=============================="
    RUN_RESULTS_DIR="${RESULTS_DIR}/run_${repeat}"
    mkdir -p "$RUN_RESULTS_DIR"

    # Create rdbp directory
    mkdir -p "$RUN_RESULTS_DIR/rdbp"

    START_TIME=$SECONDS

    echo "================================================"
    echo "Running RDBP Binary Pairs Analysis (Run $repeat)"
    echo "================================================"

    TOTAL_COUNT=0
    echo ""
    echo "🚀 STRATEGY: RDBP"
    echo "========================================"

    PAIR_COUNT=0
    for ((i=0; i<${#DATASETS[@]}; i++)); do
        for ((j=0; j<${#DATASETS[@]}; j++)); do
            [[ $i -eq $j ]] && continue   # skip A→A

            DATASET_A="${DATASETS[i]}"
            DATASET_B="${DATASETS[j]}"
            PAIR_NAME="${DATASET_A}_to_${DATASET_B}"
            ((++PAIR_COUNT))
            ((++TOTAL_COUNT))

            echo ""
            echo "🔥 RDBP: $PAIR_COUNT/12 - $DATASET_A → $DATASET_B"
            echo "----------------------------------------"

            # ── pair-specific target directory ──────────────────────
            TARGET_DIR="$RUN_RESULTS_DIR/rdbp/${PAIR_NAME}"
            mkdir -p "$TARGET_DIR"
            ABS_TARGET_DIR="$(readlink -f "$TARGET_DIR")"

            CONFIG_FILE="config_rdbp_${PAIR_NAME}.yaml"

            # ── RDBP configuration ────────────────────────────
            RDBP_CONFIG=$'\n'"# RDBP cfg"$'\n'"hinge_point: -3.0"$'\n'"decrease_factor: 0.15"$'\n'"speed_factor: 1.005"

            # ── emit YAML ───────────────────────────────────────────
            cat > "$CONFIG_FILE" << EOF
# Binary Pairs Configuration
experiment_type: StrategyBinaryPairsExperiment

# Model configuration
model_name: ViT64
num_classes: 20
lr: 0.0003
optimizer: adam
epochs: 50
minibatch_size: 128

# Device configuration
cuda: 1

# Binary pairs configuration
dataset_a: $DATASET_A
dataset_b: $DATASET_B
scenario_type: class_incremental

# Dataset configuration
balanced: balanced
number_of_samples_per_class: 500

# Strategy configuration
strategy_name: RDBP$RDBP_CONFIG

# Analysis configuration
analysis_freq: 10
analyze_representations: true

# Patch analysis
patch_analysis: true
patch_analysis_freq: 10
patch_analysis_max_batches: 40
patch_analysis_during_training: true

# Other analysis flags
analyze_all_tasks: false
track_task_performance: true
analyze_during_training: false

# ───── Output goes directly to the pair folder ─────
output_dir: $ABS_TARGET_DIR
save_model: false
use_tensorboard: false
verbose: true
EOF

            echo "Running RDBP: $DATASET_A → $DATASET_B… (logs every 10 epochs, patch analysis every 10)"

            # Check GPU memory before starting (if using GPU)
            if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
                GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
                echo "  📊 GPU memory used before start: ${GPU_MEM}MB"
            fi

            if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" \
                 2>&1 | tee "$TARGET_DIR/run_log.txt"; then
                echo "✅ RDBP $PAIR_NAME completed"
                
                # Log GPU memory after completion
                if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
                    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
                    echo "  📊 GPU memory used after completion: ${GPU_MEM}MB"
                fi
            else
                echo "❌ RDBP $PAIR_NAME failed (see run_log.txt)"
            fi

            rm -f "$CONFIG_FILE"
        done
    done

    echo ""
    echo "✅ RDBP strategy completed: $PAIR_COUNT pairs"

    echo ""
    echo "================================================"
    echo "Verifying Patch Analysis Results (Run $repeat)"
    echo "================================================"

    echo ""
    echo "Checking RDBP results:"
    for pair_dir in "$RUN_RESULTS_DIR/rdbp"/*; do
        [[ -d "$pair_dir" ]] || continue
        pair_name=$(basename "$pair_dir")
        echo "  $pair_name:"

        if [ -d "$pair_dir/patch_analysis" ]; then
            num_files=$(ls "$pair_dir/patch_analysis"/patch_importance_epoch_*.npz 2>/dev/null | wc -l)
            echo "    ✅ Patch analysis: $num_files epoch files"
            epochs=$(ls "$pair_dir/patch_analysis"/patch_importance_epoch_*.npz 2>/dev/null \
                     | sed 's/.*epoch_\([0-9]*\)\.npz/\1/' | sort -n | tr '\n' ' ')
            echo "    Epochs with analysis: $epochs"
        else
            echo "    ❌ No patch analysis found"
        fi

        if ls "$pair_dir"/training_log*.csv >/dev/null 2>&1; then
            echo "    ✅ Training logs found"
        else
            echo "    ❌ No training logs found"
        fi
    done

    echo ""
    echo "================================================"
    echo "Creating Analysis Summary (Run $repeat)"
    echo "================================================"

    cat > "$RUN_RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""Analyze RDBP binary pairs results (including patch importance)."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, json

def analyze_results(results_dir: Path):
    all_results, patch_results = {}, {}

    rdbp_dir = results_dir / 'rdbp'
    if not rdbp_dir.exists():
        print("No rdbp directory found!")
        return
        
    all_results['RDBP'], patch_results['RDBP'] = {}, {}

    for pair_dir in rdbp_dir.glob('*'):
        if not pair_dir.is_dir():
            continue
        pname = pair_dir.name

        # ─ training log ─
        logs = list(pair_dir.glob('training_log*.csv'))
        if logs:
            try:
                df = pd.read_csv(logs[0])
                row = df.iloc[-1]
                all_results['RDBP'][pname] = {
                    'task_0_acc': row.get('task_0_acc', 0.0),
                    'task_1_acc': row.get('task_1_acc', 0.0),
                    'average_acc': row.get('average_acc', row.get('avg_acc', 0.0)),
                    'forgetting': 100 - row.get('task_0_acc', 0.0),
                }
            except Exception as e:
                print(f"Could not read log for RDBP/{pname}: {e}")

        # ─ patch analysis ─
        pdir = pair_dir / 'patch_analysis'
        if pdir.exists():
            epochs = [int(f.stem.split('_')[-1]) for f in pdir.glob('patch_importance_epoch_*.npz')]
            if epochs:
                patch_results['RDBP'][pname] = {
                    'num_epochs': len(epochs),
                    'epochs': sorted(epochs),
                    'has_analysis': True,
                }

    # ─── plots & tables ───
    if all_results['RDBP']:
        rdbp_plot(results_dir, all_results['RDBP'])
        plasticity_stability_plot(results_dir, all_results['RDBP'])
        summary_table(results_dir, all_results)
    if patch_results['RDBP']:
        patch_summary(patch_results)

def rdbp_plot(results_dir: Path, rdbp_res):
    pairs = sorted(rdbp_res.keys())
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    metrics = ['task_0_acc', 'task_1_acc', 'average_acc', 'forgetting']
    titles = ['Task A retention (RDBP)', 'Task B accuracy (RDBP)', 
              'Average accuracy (RDBP)', 'Catastrophic forgetting (RDBP)']
    colors = ['steelblue', 'forestgreen', 'purple', 'crimson']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        values = [rdbp_res[p].get(metric, 0.0) for p in pairs]
        
        bars = ax.bar(range(len(pairs)), values, color=color, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset pair')
        ax.set_ylabel('Accuracy (%)' if metric != 'forgetting' else 'Forgetting (%)')
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([p.replace('_to_', '→') for p in pairs], rotation=45)
        
        if metric == 'forgetting':
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'rdbp_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

def plasticity_stability_plot(results_dir: Path, rdbp_res):
    """Create plasticity vs stability scatter plot."""
    pairs = list(rdbp_res.keys())
    stability = [rdbp_res[p]['task_0_acc'] for p in pairs]  # Task A retention
    plasticity = [rdbp_res[p]['task_1_acc'] for p in pairs]  # Task B learning
    
    plt.figure(figsize=(10, 8))
    plt.scatter(stability, plasticity, s=100, alpha=0.7, c='steelblue', edgecolors='black')
    
    # Add pair labels
    for i, pair in enumerate(pairs):
        plt.annotate(pair.replace('_to_', '→'), 
                    (stability[i], plasticity[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('Stability (Task A Retention %)', fontsize=12)
    plt.ylabel('Plasticity (Task B Learning %)', fontsize=12)
    plt.title('RDBP: Plasticity vs Stability Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal reference line (perfect balance)
    max_val = max(max(stability), max(plasticity))
    min_val = min(min(stability), min(plasticity))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect balance')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'rdbp_plasticity_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

def summary_table(results_dir: Path, res):
    if 'RDBP' not in res or not res['RDBP']:
        print("No RDBP results to summarize")
        return
        
    data = res['RDBP']
    a0 = [d['task_0_acc'] for d in data.values()]
    a1 = [d['task_1_acc'] for d in data.values()]
    avg = [d['average_acc'] for d in data.values()]
    forgetting = [d['forgetting'] for d in data.values()]
    
    if not a0:
        print("No valid results found")
        return
        
    summary = {
        'Strategy': 'RDBP',
        'Mean Task A retention': np.mean(a0),
        'Mean Task B accuracy': np.mean(a1),
        'Mean average': np.mean(avg),
        'Mean forgetting (%)': np.mean(forgetting),
        'Stability (std)': np.std(a0),
        'Plasticity (std)': np.std(a1),
        '#pairs': len(a0),
        'Best pair (avg)': max(data.keys(), key=lambda k: data[k]['average_acc']),
        'Worst pair (avg)': min(data.keys(), key=lambda k: data[k]['average_acc']),
        'Most stable pair': max(data.keys(), key=lambda k: data[k]['task_0_acc']),
        'Most plastic pair': max(data.keys(), key=lambda k: data[k]['task_1_acc']),
    }
    
    df = pd.DataFrame([summary]).round(2)
    df.to_csv(results_dir / 'rdbp_summary.csv', index=False)
    print("\nRDBP Strategy Summary:")
    print("-" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

def patch_summary(pre):
    print("\nPatch analysis coverage")
    print("-" * 32)
    if 'RDBP' in pre:
        pairs = pre['RDBP']
        total = len(pairs)
        present = sum(1 for d in pairs.values() if d['has_analysis'])
        print(f"RDBP: {present}/{total} pairs")
        if present:
            epochs = [e for d in pairs.values() for e in d.get('epochs', [])]
            print(f"  avg epochs/pair: {len(epochs)/present:.1f}  "
                  f"epoch range: {min(epochs)}–{max(epochs)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: analyze_results.py <results_dir>")
        sys.exit(1)
    analyze_results(Path(sys.argv[1]))
EOF
    chmod +x "$RUN_RESULTS_DIR/analyze_results.py"

    echo "Running analysis…"
    python "$RUN_RESULTS_DIR/analyze_results.py" "$RUN_RESULTS_DIR" || echo "⚠ analysis failed"

    # ───────────── README ─────────────
    cat > "$RUN_RESULTS_DIR/README.md" << EOF
# RDBP Strategy Binary Pairs Analysis Results (Run $repeat)

**Date:** $(date)  
**Pairs:** 12 A→B combinations  
**Strategy:** RDBP (ReLUDown + Decreasing Backpropagation)  
**Config:** Standardized settings matching other strategies for fair comparison
**Total experiments:** $TOTAL_COUNT  
**Analysis frequency:** every 10 epochs  
**Patch analysis frequency:** every 10 epochs  
**CUDA Devices:** ${CUDA_DEVICES:-'CPU only'}

## RDBP Configuration
- **ReLUDown activation:** hinge_point = -3.0  
- **Decreasing Backpropagation:**
  - decrease_factor = 0.15
  - speed_factor = 1.005
- **Training:** 50 epochs per task (matching other strategies)
- **Optimizer:** Adam with lr=0.0003  
- **Batch size:** 128
- **GPU(s):** ${CUDA_DEVICES:-'CPU only'}

## Hardware Info
$(if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
    echo "\`\`\`"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while IFS=, read idx name total; do
        if [[ "$CUDA_DEVICES" == *"$idx"* ]]; then
            echo "GPU $idx: $name (${total}MB total memory)"
        fi
    done
    echo "\`\`\`"
else
    echo "Running on CPU"
fi)

## Results
- **rdbp_performance.png:** Bar charts showing task retention, learning, average accuracy, and forgetting
- **rdbp_plasticity_stability.png:** Scatter plot showing plasticity-stability trade-off
- **rdbp_summary.csv:** Numerical summary with statistics

Each pair's detailed outputs (training logs, patch analysis, etc.) are in:

\`\`\`
$RUN_RESULTS_DIR/rdbp/<datasetA>_to_<datasetB>/
\`\`\`

Patch-analysis files are under each pair's *patch_analysis/* sub-folder.

## RDBP Method Overview
The RDBP strategy balances plasticity and stability through:

### 1. **ReLUDown Activation** (Plasticity)
- Modified ReLU: f(x) = max(0, x) - max(0, -x + d) where d = -3.0
- Prevents neuron dormancy and maintains gradient flow
- Preserves network sensitivity to new inputs
- No additional parameters or computational overhead

### 2. **Decreasing Backpropagation (DBP)** (Stability)
- Progressively reduces gradient updates to earlier layers
- Formula: bp_decrease = bp_standard × (1 - (layer × f) + (layer × f) × a^(-task))
- Protects established features while allowing adaptation in later layers
- Mimics biological memory consolidation processes

### Key Advantages
- ✅ Simple baseline requiring no memory buffers
- ✅ Drop-in replacement for standard training
- ✅ Balanced plasticity-stability trade-off
- ✅ Low computational overhead
- ✅ Biologically-inspired gradient scheduling

**Note:** This experiment uses standardized training settings (50 epochs, Adam optimizer, etc.) to enable fair comparison with other continual learning strategies, rather than the original RDBP paper settings.
EOF

    RUNTIME=$((SECONDS - START_TIME))
    echo ""
    echo "================================================"
    echo "✅  RDBP Binary Pairs Analysis Complete (Run $repeat)"
    echo "================================================"
    echo "Results directory: $RUN_RESULTS_DIR"
    echo "Runtime: $RUNTIME s (~$((RUNTIME/60)) min)"

done

echo ""
echo "================================================"
echo "All $REPEATS runs complete!"
echo "Results directories:"
for repeat in $(seq 1 $REPEATS); do
    echo "  $RESULTS_DIR/run_${repeat}"
done
echo ""
echo "🎯 RDBP strategy analysis finished!"
echo "Check rdbp_performance.png, rdbp_plasticity_stability.png and rdbp_summary.csv in each run directory."
echo ""

# Final GPU memory report
if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
    echo "📊 Final GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read idx name used total temp; do
        if [[ "$CUDA_DEVICES" == *"$idx"* ]]; then
            echo "  GPU $idx: ${used}MB/${total}MB used, ${temp}°C"
        fi
    done
fi
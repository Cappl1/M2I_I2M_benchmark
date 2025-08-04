#!/bin/bash
# Kaizen Strategy Binary Pairs Analysis
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
STRATEGIES=("Kaizen")  # Only Kaizen strategy
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
RESULTS_DIR="kaizen_binary_pairs_analysis_$(date +%Y%m%d_%H%M%S)"
REPEATS=5

# ───────────────────────── CUDA Configuration ──────────────────────────
# Set which GPU(s) to use (comma-separated for multiple GPUs)
# Examples: 
#   CUDA_DEVICES="0"        # Use only GPU 0
#   CUDA_DEVICES="1"        # Use only GPU 1  
#   CUDA_DEVICES="0,1"      # Use GPUs 0 and 1
#   CUDA_DEVICES=""         # Use CPU only
CUDA_DEVICES="3"

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
echo "Kaizen Strategy Binary Pairs Analysis - $REPEATS Repeats"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Strategy: Kaizen"
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

    # Create kaizen directory
    mkdir -p "$RUN_RESULTS_DIR/kaizen"

    START_TIME=$SECONDS

    echo "================================================"
    echo "Running Kaizen Binary Pairs Analysis (Run $repeat)"
    echo "================================================"

    TOTAL_COUNT=0
    echo ""
    echo "🚀 STRATEGY: Kaizen"
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
            echo "🔥 Kaizen: $PAIR_COUNT/12 - $DATASET_A → $DATASET_B"
            echo "----------------------------------------"

            # ── pair-specific target directory ──────────────────────
            TARGET_DIR="$RUN_RESULTS_DIR/kaizen/${PAIR_NAME}"
            mkdir -p "$TARGET_DIR"
            ABS_TARGET_DIR="$(readlink -f "$TARGET_DIR")"

            CONFIG_FILE="config_kaizen_${PAIR_NAME}.yaml"

            # ── Kaizen configuration ────────────────────────────
            KAIZEN_CONFIG=$'\n'"# Kaizen cfg"$'\n'"ssl_method: simclr"$'\n'"memory_size_percent: 1"$'\n'"kd_classifier_weight: 2.0"

            # ── emit YAML ───────────────────────────────────────────
            cat > "$CONFIG_FILE" << EOF
# Binary Pairs Configuration
experiment_type: StrategyBinaryPairsExperiment

# Model configuration
model_name: ViT64
num_classes: 20
lr: 0.00001
optimizer: adam
epochs: 50
minibatch_size: 32

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
strategy_name: Kaizen$KAIZEN_CONFIG

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

            echo "Running Kaizen: $DATASET_A → $DATASET_B… (logs every 10 epochs)"

            # Check GPU memory before starting (if using GPU)
            if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
                GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
                echo "  📊 GPU memory used before start: ${GPU_MEM}MB"
            fi

            if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" \
                 2>&1 | tee "$TARGET_DIR/run_log.txt"; then
                echo "✅ Kaizen $PAIR_NAME completed"
                
                # Log GPU memory after completion
                if [ -n "$CUDA_DEVICES" ] && command -v nvidia-smi &> /dev/null; then
                    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
                    echo "  📊 GPU memory used after completion: ${GPU_MEM}MB"
                fi
            else
                echo "❌ Kaizen $PAIR_NAME failed (see run_log.txt)"
            fi

            rm -f "$CONFIG_FILE"
        done
    done

    echo ""
    echo "✅ Kaizen strategy completed: $PAIR_COUNT pairs"

    echo ""
    echo "================================================"
    echo "Verifying Patch Analysis Results (Run $repeat)"
    echo "================================================"

    echo ""
    echo "Checking Kaizen results:"
    for pair_dir in "$RUN_RESULTS_DIR/kaizen"/*; do
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
"""Analyze Kaizen binary pairs results (including patch importance)."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, json

def analyze_results(results_dir: Path):
    all_results, patch_results = {}, {}

    kaizen_dir = results_dir / 'kaizen'
    if not kaizen_dir.exists():
        print("No kaizen directory found!")
        return
        
    all_results['Kaizen'], patch_results['Kaizen'] = {}, {}

    for pair_dir in kaizen_dir.glob('*'):
        if not pair_dir.is_dir():
            continue
        pname = pair_dir.name

        # ─ training log ─
        logs = list(pair_dir.glob('training_log*.csv'))
        if logs:
            try:
                df = pd.read_csv(logs[0])
                row = df.iloc[-1]
                all_results['Kaizen'][pname] = {
                    'task_0_acc': row.get('task_0_acc', 0.0),
                    'task_1_acc': row.get('task_1_acc', 0.0),
                    'average_acc': row.get('average_acc', row.get('avg_acc', 0.0)),
                }
            except Exception as e:
                print(f"Could not read log for Kaizen/{pname}: {e}")

        # ─ patch analysis ─
        pdir = pair_dir / 'patch_analysis'
        if pdir.exists():
            epochs = [int(f.stem.split('_')[-1]) for f in pdir.glob('patch_importance_epoch_*.npz')]
            if epochs:
                patch_results['Kaizen'][pname] = {
                    'num_epochs': len(epochs),
                    'epochs': sorted(epochs),
                    'has_analysis': True,
                }

    # ─── plots & tables ───
    if all_results['Kaizen']:
        kaizen_plot(results_dir, all_results['Kaizen'])
        summary_table(results_dir, all_results)
    if patch_results['Kaizen']:
        patch_summary(patch_results)

def kaizen_plot(results_dir: Path, kaizen_res):
    pairs = sorted(kaizen_res.keys())
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    
    for ax, metric, title in zip(axes,
            ['task_0_acc', 'task_1_acc', 'average_acc'],
            ['Task A retention (Kaizen)', 'Task B accuracy (Kaizen)', 'Average accuracy (Kaizen)']):
        values = [kaizen_res[p].get(metric, 0.0) for p in pairs]
        
        bars = ax.bar(range(len(pairs)), values, color='steelblue', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Dataset pair')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([p.replace('_to_', '→') for p in pairs], rotation=45)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'kaizen_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

def summary_table(results_dir: Path, res):
    if 'Kaizen' not in res or not res['Kaizen']:
        print("No Kaizen results to summarize")
        return
        
    data = res['Kaizen']
    a0 = [d['task_0_acc'] for d in data.values()]
    a1 = [d['task_1_acc'] for d in data.values()]
    avg = [d['average_acc'] for d in data.values()]
    
    if not a0:
        print("No valid results found")
        return
        
    summary = {
        'Strategy': 'Kaizen',
        'Mean Task A retention': np.mean(a0),
        'Mean Task B accuracy': np.mean(a1),
        'Mean average': np.mean(avg),
        'Forgetting (%)': 100 - np.mean(a0),
        '#pairs': len(a0),
        'Best pair (avg)': max(data.keys(), key=lambda k: data[k]['average_acc']),
        'Worst pair (avg)': min(data.keys(), key=lambda k: data[k]['average_acc']),
    }
    
    df = pd.DataFrame([summary]).round(1)
    df.to_csv(results_dir / 'kaizen_summary.csv', index=False)
    print("\nKaizen Strategy Summary:")
    print("-" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

def patch_summary(pre):
    print("\nPatch analysis coverage")
    print("-" * 32)
    if 'Kaizen' in pre:
        pairs = pre['Kaizen']
        total = len(pairs)
        present = sum(1 for d in pairs.values() if d['has_analysis'])
        print(f"Kaizen: {present}/{total} pairs")
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
# Kaizen Strategy Binary Pairs Analysis Results (Run $repeat)

**Date:** $(date)  
**Pairs:** 12 A→B combinations  
**Strategy:** Kaizen (SSL-based continual learning)  
**Total experiments:** $TOTAL_COUNT  
**Analysis frequency:** every 10 epochs  
**CUDA Devices:** ${CUDA_DEVICES:-'CPU only'}

## Configuration
- SSL method: SimCLR
- Memory replay: 1% of dataset
- Classifier KD weight: 2.0
- Epochs: 50 per task
- GPU(s): ${CUDA_DEVICES:-'CPU only'}

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

Open *kaizen_performance.png* for the bar charts, or look at *kaizen_summary.csv* for numbers.  
Each pair's outputs (training logs, patch analysis, etc.) live in:

\`\`\`
$RUN_RESULTS_DIR/kaizen/<datasetA>_to_<datasetB>/
\`\`\`

Patch-analysis files are under each pair's *patch_analysis/* sub-folder.

## Kaizen Method
The Kaizen strategy combines:
1. Self-supervised learning (SSL) for feature extraction
2. Knowledge distillation from previous tasks
3. Memory replay (1% of previous data)
4. Separate optimization for features vs classifier
EOF

    RUNTIME=$((SECONDS - START_TIME))
    echo ""
    echo "================================================"
    echo "✅  Kaizen Binary Pairs Analysis Complete (Run $repeat)"
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
echo "🎯 Kaizen strategy analysis finished!"
echo "Check kaizen_performance.png and kaizen_summary.csv in each run directory."
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
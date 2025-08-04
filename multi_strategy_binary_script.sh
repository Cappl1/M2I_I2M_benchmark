#!/bin/bash
# Fixed Multi-strategy binary pairs analysis
# — outputs are written directly to their final location —

set -e

# ───────────────────────── Configuration ──────────────────────────
DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn")
STRATEGIES=("Naive" "Replay" "Cumulative")
PYTHON_SCRIPT="/home/brothen/M2I_I2M_benchmark/run_experiment.py"
RESULTS_DIR="binary_pairs_analysis_$(date +%Y%m%d_%H%M%S)"
REPEATS=5

echo "================================================"
echo "Binary Pairs Analysis - 3 Strategies, $REPEATS Repeats"
echo "================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo "Repeats: $REPEATS"
echo "Total experiments per run: $((${#DATASETS[@]} * (${#DATASETS[@]} - 1) * ${#STRATEGIES[@]}))"
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

    # strategy-level dirs (naive/, replay/, cumulative/)
    for strategy in "${STRATEGIES[@]}"; do
        mkdir -p "$RUN_RESULTS_DIR/${strategy,,}"
    done

    START_TIME=$SECONDS

    echo "================================================"
    echo "Running Binary Pairs Analysis (Run $repeat)"
    echo "================================================"

    TOTAL_COUNT=0
    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "🚀 STRATEGY: $strategy"
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
                echo "🔥 $strategy: $PAIR_COUNT/12 - $DATASET_A → $DATASET_B"
                echo "----------------------------------------"

                # ── pair-specific target directory ──────────────────────
                TARGET_DIR="$RUN_RESULTS_DIR/${strategy,,}/${PAIR_NAME}"
                mkdir -p "$TARGET_DIR"
                ABS_TARGET_DIR="$(readlink -f "$TARGET_DIR")"

                CONFIG_FILE="config_${strategy,,}_${PAIR_NAME}.yaml"

                # ── strategy-specific tweaks ────────────────────────────
                case $strategy in
                    "Naive")      MEMORY_CONFIG="" ;;
                    "Replay")     MEMORY_CONFIG=$'\n'"# Replay cfg"$'\n'"memory_size: 500"$'\n'"replay_batch_ratio: 0.5" ;;
                    "Cumulative") MEMORY_CONFIG=$'\n'"# Cumulative cfg"$'\n'"cumulative_mode: true" ;;
                esac

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
scenario_type: task_incremental

# Dataset configuration
balanced: balanced
number_of_samples_per_class: 500

# Strategy configuration
strategy_name: $strategy$MEMORY_CONFIG

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

                echo "Running $strategy: $DATASET_A → $DATASET_B… (logs every 10 epochs)"

                if python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" \
                     2>&1 | tee "$TARGET_DIR/run_log.txt"; then
                    echo "✅ $strategy $PAIR_NAME completed"
                else
                    echo "❌ $strategy $PAIR_NAME failed (see run_log.txt)"
                fi

                rm -f "$CONFIG_FILE"
            done
        done

        echo ""
        echo "✅ $strategy strategy completed: $PAIR_COUNT pairs"
    done

    echo ""
    echo "================================================"
    echo "Verifying Patch Analysis Results (Run $repeat)"
    echo "================================================"

    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "Checking $strategy results:"
        strategy_lower="${strategy,,}"
        for pair_dir in "$RUN_RESULTS_DIR/$strategy_lower"/*; do
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
    done

    echo ""
    echo "================================================"
    echo "Creating Analysis Summary (Run $repeat)"
    echo "================================================"

    cat > "$RUN_RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""Analyze binary pairs results (including patch importance)."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, json

def analyze_results(results_dir: Path):
    all_results, patch_results = {}, {}

    for strategy_dir in results_dir.glob('*'):
        if not strategy_dir.is_dir():
            continue
        sname = strategy_dir.name.capitalize()
        if sname not in ['Naive', 'Replay', 'Cumulative']:
            continue
        all_results[sname], patch_results[sname] = {}, {}

        for pair_dir in strategy_dir.glob('*'):
            if not pair_dir.is_dir():
                continue
            pname = pair_dir.name

            # ─ training log ─
            logs = list(pair_dir.glob('training_log*.csv'))
            if logs:
                try:
                    df = pd.read_csv(logs[0])
                    row = df.iloc[-1]
                    all_results[sname][pname] = {
                        'task_0_acc': row.get('task_0_acc', 0.0),
                        'task_1_acc': row.get('task_1_acc', 0.0),
                        'average_acc': row.get('average_acc', row.get('avg_acc', 0.0)),
                    }
                except Exception as e:
                    print(f"Could not read log for {sname}/{pname}: {e}")

            # ─ patch analysis ─
            pdir = pair_dir / 'patch_analysis'
            if pdir.exists():
                epochs = [int(f.stem.split('_')[-1]) for f in pdir.glob('patch_importance_epoch_*.npz')]
                if epochs:
                    patch_results[sname][pname] = {
                        'num_epochs': len(epochs),
                        'epochs': sorted(epochs),
                        'has_analysis': True,
                    }

    # ─── plots & tables ───
    if all_results:
        comp_plot(results_dir, all_results)
        summary_table(results_dir, all_results)
    if patch_results:
        patch_summary(patch_results)

def comp_plot(results_dir: Path, res):
    strategies = list(res.keys())
    pairs = sorted({p for dat in res.values() for p in dat})
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    for ax, metric, title in zip(axes,
            ['task_0_acc', 'task_1_acc', 'average_acc'],
            ['Task A retention', 'Task B accuracy', 'Average accuracy']):
        mat = np.zeros((len(strategies), len(pairs)))
        for i, s in enumerate(strategies):
            for j, p in enumerate(pairs):
                mat[i, j] = res.get(s, {}).get(p, {}).get(metric, 0.0)
        sns.heatmap(mat, ax=ax, cmap='RdYlGn', vmin=0, vmax=100, annot=True, fmt='.1f',
                    xticklabels=[p.replace('_to_', '→') for p in pairs],
                    yticklabels=strategies)
        ax.set_title(title)
        ax.set_xlabel('dataset pair')
        ax.set_ylabel('strategy')
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_comparison.png', dpi=150)
    plt.close()

def summary_table(results_dir: Path, res):
    rows = []
    for strat, data in res.items():
        a0 = [d['task_0_acc'] for d in data.values()]
        a1 = [d['task_1_acc'] for d in data.values()]
        avg = [d['average_acc'] for d in data.values()]
        if not a0: continue
        rows.append({
            'Strategy': strat,
            'Mean Task A retention': np.mean(a0),
            'Mean Task B accuracy': np.mean(a1),
            'Mean average': np.mean(avg),
            'Forgetting (%)': 100 - np.mean(a0),
            '#pairs': len(a0),
        })
    df = pd.DataFrame(rows).round(1)
    df.to_csv(results_dir / 'strategy_summary.csv', index=False)
    print("\nStrategy Summary:\n", df.to_string(index=False))

def patch_summary(pre):
    print("\nPatch analysis coverage")
    print("-" * 32)
    for strat, pairs in pre.items():
        total = len(pairs)
        present = sum(1 for d in pairs.values() if d['has_analysis'])
        print(f"{strat}: {present}/{total} pairs")
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
# Binary Pairs Analysis Results (Run $repeat)

**Date:** $(date)  
**Pairs:** 12 A→B combinations  
**Strategies:** Naive, Replay, Cumulative  
**Total experiments:** $TOTAL_COUNT  
**Analysis frequency:** every 10 epochs  

Open *performance_comparison.png* for the heat-map, or look at *strategy_summary.csv* for numbers.  
Each pair’s outputs (training logs, patch analysis, etc.) live in:

```
$RUN_RESULTS_DIR/{naive|replay|cumulative}/<datasetA>_to_<datasetB>/
```

Patch-analysis files are under each pair’s *patch_analysis/* sub-folder.
EOF

    RUNTIME=$((SECONDS - START_TIME))
    echo ""
    echo "================================================"
    echo "✅  Binary Pairs Analysis Complete (Run $repeat)"
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

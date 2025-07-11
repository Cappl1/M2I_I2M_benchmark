#!/bin/bash
"""
Run single dataset experiments across all available datasets.
This script runs each dataset individually and optionally analyzes results.
"""

set -e  # Exit on any error

# Configuration
PYTHON_CMD="python"
CONFIG_BASE="configs/experiments/single_task_base.yml"
CUDA_DEVICE=3  # Change this to your preferred GPU
RUN_ANALYSIS=true  # Set to false to skip automatic analysis
BASE_OUTPUT_DIR="logs/single_dataset_sweep_$(date +%Y%m%d_%H%M%S)"

# Available datasets
DATASETS=(
    # "mnist"          # Commented out - currently being worked on
    "fashion_mnist"
    "cifar10"
    "svhn"
    "omniglot"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to create a temporary config for each dataset
create_dataset_config() {
    local dataset=$1
    local temp_config="/tmp/single_task_${dataset}.yml"
    
    # Copy base config and modify dataset
    cp "$CONFIG_BASE" "$temp_config"
    
    # Update dataset name in the config
    sed -i "s/dataset: mnist/dataset: $dataset/g" "$temp_config"
    
    # Update CUDA device
    sed -i "s/cuda: [0-9-]*/cuda: $CUDA_DEVICE/g" "$temp_config"
    
    # Update experiment name
    sed -i "s/experiment_name: single_task/experiment_name: single_task_$dataset/g" "$temp_config"
    
    echo "$temp_config"
}

# Function to run experiment for a single dataset
run_dataset_experiment() {
    local dataset=$1
    local temp_config=$(create_dataset_config "$dataset")
    
    print_status "Starting experiment for dataset: $dataset"
    print_status "Using config: $temp_config"
    
    # Run the experiment
    if $PYTHON_CMD run_experiment.py --config "$temp_config"; then
        print_success "Completed experiment for $dataset"
        
        # Find the most recent experiment directory for this dataset
        local exp_dir=$(find logs -name "*single_task_${dataset}*" -type d | sort | tail -1)
        
        if [ -n "$exp_dir" ] && [ "$RUN_ANALYSIS" = true ]; then
            print_status "Running analysis for $dataset experiment in $exp_dir"
            
            if $PYTHON_CMD analysis/analyze_training_run.py --experiment_dir "$exp_dir"; then
                print_success "Analysis completed for $dataset"
            else
                print_warning "Analysis failed for $dataset (experiment data may still be valid)"
            fi
        fi
        
        # Clean up temporary config
        rm -f "$temp_config"
        return 0
    else
        print_error "Experiment failed for $dataset"
        rm -f "$temp_config"
        return 1
    fi
}

# Function to show summary of all experiments
show_summary() {
    print_status "Experiment Summary"
    echo "=================="
    
    local total_experiments=${#DATASETS[@]}
    local successful_experiments=0
    
    for dataset in "${DATASETS[@]}"; do
        local exp_dirs=$(find logs -name "*single_task_${dataset}*" -type d | wc -l)
        if [ "$exp_dirs" -gt 0 ]; then
            local latest_exp=$(find logs -name "*single_task_${dataset}*" -type d | sort | tail -1)
            if [ -f "$latest_exp/results.json" ]; then
                local accuracy=$(python -c "import json; data=json.load(open('$latest_exp/results.json')); print(f'{data.get(\"final_accuracy\", 0):.3f}')" 2>/dev/null || echo "N/A")
                print_success "$dataset: $accuracy accuracy (in $latest_exp)"
                ((successful_experiments++))
            else
                print_warning "$dataset: Experiment directory exists but no results.json found"
            fi
        else
            print_error "$dataset: No experiment directory found"
        fi
    done
    
    echo ""
    print_status "Completed $successful_experiments/$total_experiments experiments successfully"
}

# Main execution
main() {
    print_status "Starting single dataset sweep"
    print_status "CUDA Device: $CUDA_DEVICE"
    print_status "Base Config: $CONFIG_BASE"
    print_status "Auto Analysis: $RUN_ANALYSIS"
    print_status "Datasets to run: ${DATASETS[*]}"
    echo ""
    
    # Check if base config exists
    if [ ! -f "$CONFIG_BASE" ]; then
        print_error "Base config file not found: $CONFIG_BASE"
        exit 1
    fi
    
    # Check if Python and required scripts exist
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python command not found: $PYTHON_CMD"
        exit 1
    fi
    
    if [ ! -f "run_experiment.py" ]; then
        print_error "run_experiment.py not found in current directory"
        exit 1
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Run experiments for each dataset
    local failed_datasets=()
    
    for dataset in "${DATASETS[@]}"; do
        echo ""
        print_status "Processing dataset $dataset..."
        
        if run_dataset_experiment "$dataset"; then
            print_success "✓ $dataset completed successfully"
        else
            print_error "✗ $dataset failed"
            failed_datasets+=("$dataset")
        fi
        
        # Add a small delay between experiments
        sleep 2
    done
    
    echo ""
    echo "=========================================="
    
    # Show final summary
    show_summary
    
    # Report any failures
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo ""
        print_error "Failed datasets: ${failed_datasets[*]}"
        exit 1
    else
        print_success "All experiments completed successfully!"
    fi
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run single dataset experiments across multiple datasets."
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --cuda N   Use CUDA device N (default: $CUDA_DEVICE)"
    echo "  --no-analysis  Skip automatic analysis after each experiment"
    echo "  --config FILE  Use different base config file (default: $CONFIG_BASE)"
    echo ""
    echo "Datasets that will be run: ${DATASETS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with default settings"
    echo "  $0 -c 0               # Use GPU 0"
    echo "  $0 --no-analysis      # Skip analysis"
    echo "  $0 -c -1              # Use CPU only"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--cuda)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --no-analysis)
            RUN_ANALYSIS=false
            shift
            ;;
        --config)
            CONFIG_BASE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main 
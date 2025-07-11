#!/bin/bash

# Strategy Verification Test Runner
# Usage: ./run_strategy_verification.sh [test_type]
# 
# Available test types:
# - quick: Quick test with 3 datasets and 5 epochs (good for testing)
# - full: Full M2I progression with all 6 datasets
# - naive: Test only naive strategy
# - replay: Test only replay strategy  
# - cumulative: Test only cumulative strategy
# - kaizen: Test only kaizen strategy (SSL + Knowledge Distillation)
# - task-only: All strategies with task incremental only
# - class-only: All strategies with class incremental only

set -e  # Exit on any error

# Default test type
TEST_TYPE=${1:-"quick"}

echo "🔬 Running Strategy Verification Test: $TEST_TYPE"
echo "========================================"

# Activate conda environment
echo "🔧 Activating m2i_i2m environment..."
eval "$(conda shell.bash hook)"
conda activate m2i_i2m

# Check if we're in the right directory
if [ ! -f "run_verification_test.py" ]; then
    echo "❌ Error: run_verification_test.py not found. Make sure you're in the M2I_I2M_benchmark directory."
    exit 1
fi

echo "📂 Working directory: $(pwd)"

# Run the verification test
echo "🚀 Starting verification test..."
echo "   Test type: $TEST_TYPE"
echo "   This will test continual learning strategies with detailed dataflow logging."
echo ""

# Add timestamp
echo "⏰ Started at: $(date)"
echo ""

# Run the test with verbose output
python run_verification_test.py "$TEST_TYPE" --verbose

echo ""
echo "⏰ Completed at: $(date)"
echo "✅ Strategy verification test finished!"

# Provide usage examples
echo ""
echo "💡 Usage Examples:"
echo "   ./run_strategy_verification.sh quick      # Quick test (recommended for first run)"
echo "   ./run_strategy_verification.sh kaizen     # Test only Kaizen strategy"
echo "   ./run_strategy_verification.sh replay     # Test only Replay strategy"
echo "   ./run_strategy_verification.sh naive      # Test only Naive strategy"
echo "   ./run_strategy_verification.sh cumulative # Test only Cumulative strategy"
echo "   ./run_strategy_verification.sh full       # Full M2I progression (takes longer)"
echo ""
echo "🔍 Look for the following in the output to verify strategies are working:"
echo "   • 📊 Accuracy matrices showing forgetting patterns"
echo "   • 🔍 Strategy-specific dataflow (memory usage, SSL components, etc.)"
echo "   • ✅ Expected behavior verification (cumulative should have lowest forgetting, etc.)"
echo "   • 📈 Performance metrics (accuracy, forgetting, forward transfer)" 
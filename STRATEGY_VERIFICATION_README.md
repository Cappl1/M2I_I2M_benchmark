# Strategy Verification Tests

This directory contains comprehensive verification tests for continual learning strategies with detailed dataflow logging. These tests help verify that different strategies (Naive, Replay, Cumulative, Kaizen) are working correctly with both Task Incremental and Class Incremental scenarios using the M2I (MNIST → ImageNet) progression.

## 🎯 Purpose

The verification tests are designed to:
- **Verify Strategy Implementation**: Ensure each strategy behaves as expected
- **Detailed Dataflow Logging**: Track memory usage, model states, and learning dynamics
- **Compare Strategy Performance**: See how different approaches handle catastrophic forgetting
- **Test M2I Progression**: Validate on the full MNIST → Omniglot → Fashion-MNIST → SVHN → CIFAR-10 → ImageNet sequence

## 🚀 Quick Start

### Method 1: Using the Shell Script (Recommended)
```bash
# Quick test with 3 datasets (recommended for first run)
./run_strategy_verification.sh quick

# Test specific strategies
./run_strategy_verification.sh kaizen     # Kaizen strategy only
./run_strategy_verification.sh replay     # Replay strategy only  
./run_strategy_verification.sh naive      # Naive strategy only
./run_strategy_verification.sh cumulative # Cumulative strategy only

# Full M2I progression (takes longer)
./run_strategy_verification.sh full
```

### Method 2: Direct Python Execution
```bash
conda activate m2i_i2m

# Quick test
python run_verification_test.py quick

# Test specific strategies
python run_verification_test.py kaizen
python run_verification_test.py replay
python run_verification_test.py naive
python run_verification_test.py cumulative

# Full tests
python run_verification_test.py full
python run_verification_test.py task-only
python run_verification_test.py class-only
```

## 📋 Available Test Types

| Test Type | Description | Duration | Use Case |
|-----------|-------------|----------|----------|
| `quick` | 3 datasets, 5 epochs each | ~15 min | First-time testing, debugging |
| `kaizen` | Kaizen strategy only | ~30 min | Focus on SSL + KD strategy |
| `replay` | Replay strategy only | ~30 min | Focus on experience replay |
| `naive` | Naive strategy only | ~20 min | Baseline catastrophic forgetting |
| `cumulative` | Cumulative strategy only | ~45 min | Upper bound performance |
| `full` | All strategies, full M2I | ~2-3 hours | Comprehensive evaluation |
| `task-only` | Task incremental only | ~1-2 hours | Focus on task boundaries |
| `class-only` | Class incremental only | ~1-2 hours | Focus on class boundaries |

## 🔍 What to Look For in the Output

### 1. Strategy-Specific Dataflow
Each strategy logs detailed information about its internal state:

**Naive Strategy:**
```
🔍 PRE-TRAINING Strategy Details (NaiveStrategy):
  • Naive strategy: No special state (trains only on current task)
```

**Replay Strategy:**
```
🔍 PRE-TRAINING Strategy Details (ReplayStrategy):
  • Memory buffer: 500 samples from 2 tasks
    - Task 0: 100 samples
    - Task 1: 100 samples
  • Will mix current task with replay samples (ratio: 0.5)
```

**Kaizen Strategy:**
```
🔍 PRE-TRAINING Strategy Details (KaizenStrategy):
  • Previous model for distillation: Available
  • SSL memory buffer: 200 samples from 2 tasks
  • SSL method: simclr
  • KD weight: 1.0, SSL weight: 1.0
```

### 2. Accuracy Evolution Matrices
Track how performance changes across tasks:
```
📊 Accuracy Matrix:
  After Task 0: [85.2]
  After Task 1: [72.1  81.5]
  After Task 2: [65.3  78.2  79.8]
```
- Diagonal elements: Performance on task right after training
- Off-diagonal elements: Performance on previous tasks (forgetting)

### 3. Continual Learning Metrics
```
🎯 Final Results:
  • Average Accuracy: 74.5%
  • Average Forgetting: 8.2%
  • Forward Transfer: 78.9%
```

### 4. Expected Behavior Verification
```
🔍 Expected behavior verification:
  ✅ Cumulative strategy has lowest forgetting (as expected)
  ✅ Replay strategy has higher or equal accuracy than naive
  ✅ Naive strategy shows catastrophic forgetting (85.2% → 65.3%)
```

## 🧠 Strategy Details

### Naive Strategy
- **Behavior**: Trains only on current task
- **Expected**: High catastrophic forgetting
- **Verification**: Should show significant accuracy drop on previous tasks

### Replay Strategy  
- **Behavior**: Stores samples from previous tasks in memory buffer
- **Expected**: Reduced forgetting compared to naive
- **Verification**: Memory buffer size should grow, mixed training batches

### Cumulative Strategy
- **Behavior**: Trains on all data seen so far (upper bound)
- **Expected**: Minimal forgetting, best overall performance
- **Verification**: Training set size grows with each task

### Kaizen Strategy (SSL + Knowledge Distillation)
- **Behavior**: 
  - Self-supervised learning for representation quality
  - Knowledge distillation from previous model
  - Memory buffer for SSL augmentations
- **Expected**: Balance between old and new knowledge
- **Verification**: 
  - Previous model available for distillation
  - SSL loss components logged
  - Memory buffer for SSL samples

## 📊 Example Output Structure

```
🧪 KAIZEN STRATEGY VERIFICATION TEST
=====================================

🎯 Kaizen Strategy - Task Incremental M2I
============================================

🔄 Initializing kaizen strategy...
✅ Model initialized: ViT64Model
✅ Analyzer initialized: ViTLayerAnalyzer

📂 Loading datasets...
  ✅ mnist: Train=5000, Test=1000
  ✅ omniglot: Train=5000, Test=1000
  ...

============================================================
📚 TASK 0: MNIST
============================================================

🔍 Pre-training analysis:
  • Training samples: 5000
  • Test samples: 1000

🔍 PRE-TRAINING Strategy Details (KaizenStrategy):
  • Previous model for distillation: None (first task)
  • SSL memory buffer: Empty
  • SSL method: simclr
  • KD weight: 1.0, SSL weight: 1.0

🏋️ Training on task 0 (mnist)...
    Epoch 10/50: SSL=0.234, CE=1.823, KD=0.000, Acc=76.2%
    ...

🔍 POST-TRAINING Strategy Details (KaizenStrategy):
  • Previous model for distillation: None (first task)
  • SSL memory buffer: Empty
  • Updated memory buffer with samples from task 0

📊 Post-training analysis:
  • Current task (mnist) accuracy: 85.2%
  ...
```

## 🔧 Configuration

The tests use optimized configurations for verification:
- **Quick tests**: 5 epochs, 100 samples per class
- **Full tests**: 50 epochs, 500 samples per class
- **Memory sizes**: 100-500 samples per task depending on test type
- **Analysis frequency**: Every 5-10 epochs for representation analysis

## ❗ Troubleshooting

### Common Issues:

1. **Environment Issues**:
   ```bash
   conda activate m2i_i2m
   pip install -r requirements.txt
   ```

2. **CUDA Memory Issues**:
   - Reduce batch size in config
   - Use smaller memory buffer sizes
   - Test with CPU: set `cuda: null` in config

3. **Dataset Loading Issues**:
   - Ensure dataset cache directory exists
   - Check internet connection for ImageNet download
   - Verify dataset paths in config

### Debugging Tips:
- Use `quick` test type for faster debugging
- Check individual strategy tests first
- Look for memory buffer growth in replay strategies
- Verify SSL loss components in Kaizen strategy

## 📈 Interpreting Results

### Good Signs:
- **Naive**: Shows clear catastrophic forgetting (expected)
- **Replay**: Less forgetting than naive, memory buffer grows
- **Cumulative**: Highest accuracy, minimal forgetting  
- **Kaizen**: SSL and KD losses decrease, balanced performance

### Red Flags:
- All strategies perform identically (implementation issue)
- No forgetting in naive strategy (too easy tasks/too few epochs)
- Replay strategy not using memory (implementation bug)
- Kaizen not using SSL/KD components

## 🎯 Next Steps

After running verification tests:
1. **Analyze** the detailed logs and accuracy matrices
2. **Compare** strategy behaviors against expectations
3. **Investigate** any unexpected results
4. **Tune** hyperparameters if needed
5. **Run** full experiments with validated strategies

For more detailed analysis, see the main experiment runners in the `experiments/` directory. 
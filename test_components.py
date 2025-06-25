#!/usr/bin/env python3
"""
Component Test Suite for Task-Incremental Learning Analysis
===========================================================
Tests each component individually with clear documentation of:
- What is being tested
- Expected behavior  
- Actual outcome
- Pass/fail status

Run: python test_components.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from collections import defaultdict

print("🧪 Starting Component Test Suite...")


class TestResult:
    """Stores and displays test results clearly."""
    
    def __init__(self, name: str, description: str, expected: str):
        self.name = name
        self.description = description
        self.expected = expected
        self.outcome = None
        self.passed = False
        self.details = {}
        
    def complete(self, outcome: str, passed: bool, **details):
        self.outcome = outcome
        self.passed = passed
        self.details = details
        
    def show(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"\n{status} {self.name}")
        print(f"   Test: {self.description}")
        print(f"   Expected: {self.expected}")
        print(f"   Result: {self.outcome}")
        if self.details:
            for k, v in self.details.items():
                print(f"   {k}: {v}")
        print("-" * 50)


def test_1_dataset_loaders():
    """Test 1: Dataset Loading Functions"""
    test = TestResult(
        "Dataset Loaders",
        "Check that all 6 datasets (mnist, omniglot, fashion_mnist, svhn, cifar10, imagenet) load correctly",
        "Each dataset returns train/test splits with 10 classes each"
    )
    
    try:
        # Import the datasets
        from task_incremental_baseline import DATASETS
        
        results = {}
        for name, loader in DATASETS.items():
            print(f"   Loading {name}...")
            train_data, test_data = loader(True, 50)  # balanced=True, 50 samples per class
            
            # Sample some data to check classes
            train_classes = set()
            test_classes = set() 
            sample_size = min(100, len(train_data))
            for i in range(sample_size):
                train_classes.add(train_data[i][1])
            sample_size = min(100, len(test_data))
            for i in range(sample_size):
                test_classes.add(test_data[i][1])
                
            results[name] = {
                "train_size": len(train_data),
                "test_size": len(test_data),
                "train_classes": len(train_classes),
                "test_classes": len(test_classes)
            }
        
        all_have_10_classes = all(
            r["train_classes"] == 10 and r["test_classes"] == 10 
            for r in results.values()
        )
        
        test.complete(
            f"Loaded {len(results)}/6 datasets successfully",
            len(results) == 6 and all_have_10_classes,
            details=results
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_2_scenario_creation():
    """Test 2: Scenario Creation"""
    test = TestResult(
        "Scenario Builder",
        "Check that get_scenario() creates proper task-incremental benchmarks",
        "Scenario with 6 tasks, each containing 10 classes, with task labels"
    )
    
    try:
        from task_incremental_baseline import get_scenario, ORDER
        
        scenario = get_scenario(n_per_class=50)
        
        num_tasks = len(scenario.train_stream)
        task_details = []
        
        for i, (train_exp, test_exp) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
            task_details.append({
                "task": i,
                "dataset": ORDER[i],
                "train_samples": len(train_exp.dataset),
                "test_samples": len(test_exp.dataset), 
                "classes": len(train_exp.classes_in_this_experience)
            })
        
        all_have_10_classes = all(t["classes"] == 10 for t in task_details)
        
        test.complete(
            f"Created scenario with {num_tasks} tasks",
            num_tasks == 6 and all_have_10_classes,
            task_details=task_details
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_3_vit_model():
    """Test 3: ViT Model Architecture"""
    test = TestResult(
        "ViT64MultiHead Model",
        "Check ViT model creation and forward pass with 64x64 images",
        "Model processes RGB images correctly and outputs 10 classes per task"
    )
    
    try:
        from task_incremental_baseline import ViT64MultiHead
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ViT64MultiHead().to(device)
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        task_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(x, task_labels)
            cls_features = model._cls(x)
        
        correct_shapes = (
            output.shape == (batch_size, 10) and
            cls_features.shape == (batch_size, 384)
        )
        
        test.complete(
            f"Model works correctly with {total_params:,} parameters",
            correct_shapes,
            output_shape=tuple(output.shape),
            cls_shape=tuple(cls_features.shape),
            device=device
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_4_multihead_tasks():
    """Test 4: Multi-Head Task Functionality"""
    test = TestResult(
        "Multi-Head Tasks",
        "Check that different task heads produce different outputs",
        "Same input should give different outputs for different task IDs"
    )
    
    try:
        from task_incremental_baseline import ViT64MultiHead
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ViT64MultiHead().to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Test different task heads
        outputs = {}
        for task_id in range(3):
            task_labels = torch.full((batch_size,), task_id, dtype=torch.long).to(device)
            output = model(x, task_labels)
            outputs[task_id] = output
            
            # Test single task method too
            single_output = model.forward_single_task(x, task_id)
            if not torch.allclose(output, single_output, atol=1e-6):
                raise ValueError(f"Inconsistent outputs for task {task_id}")
        
        # Different tasks should give different outputs
        different_heads = not torch.allclose(outputs[0], outputs[1], atol=1e-3)
        
        test.complete(
            "Multi-head functionality working correctly",
            different_heads,
            num_heads_tested=len(outputs),
            outputs_differ=different_heads
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_5_accuracy_function():
    """Test 5: Accuracy Calculation"""
    test = TestResult(
        "Accuracy Function",
        "Check batch_accuracy function with controlled data",
        "Function should return 100% accuracy when predictions match labels"
    )
    
    try:
        from task_incremental_baseline import ViT64MultiHead, batch_accuracy
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ViT64MultiHead().to(device)
        
        # Create controlled test: make model always predict class 0
        with torch.no_grad():
            for param in model.classifier.classifiers[0].parameters():
                param.zero_()
            model.classifier.classifiers[0].bias[0] = 10.0  # Heavily favor class 0
        
        # Create test data where all labels are 0
        test_data = []
        for _ in range(20):
            x = torch.randn(3, 64, 64)
            y = 0  # All labels are 0
            task_label = torch.tensor(0)
            test_data.append((x, y, task_label))
        
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        accuracy = batch_accuracy(model, test_loader, device)
        
        # Should get ~100% accuracy
        accuracy_correct = abs(accuracy - 100.0) < 5.0
        
        test.complete(
            f"Accuracy calculation working: {accuracy:.1f}%",
            accuracy_correct,
            measured_accuracy=accuracy,
            expected_range="95-100%"
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_6_training_loop():
    """Test 6: Training Loop"""
    test = TestResult(
        "Training Function",
        "Check that training modifies model parameters",
        "Model parameters should change after training epochs"
    )
    
    try:
        from task_incremental_baseline import ViT64MultiHead, train_task, batch_accuracy
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ViT64MultiHead().to(device)
        
        # Save initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Create dummy training data
        dummy_data = []
        for _ in range(16):
            x = torch.randn(3, 64, 64)
            y = torch.randint(0, 10, (1,)).item()
            task_label = torch.tensor(0)
            dummy_data.append((x, y, task_label))
        
        train_loader = DataLoader(dummy_data, batch_size=4, shuffle=True)
        test_loader = DataLoader(dummy_data, batch_size=4, shuffle=False)
        
        # Train for a few epochs - use correct signature
        train_task(model, train_loader, test_loader, device, epochs=2, lr=1e-3)
        
        # Check that parameters changed
        params_changed = any(
            not torch.equal(initial_p, final_p)
            for initial_p, final_p in zip(initial_params, model.parameters())
        )
        
        final_acc = batch_accuracy(model, test_loader, device)
        
        test.complete(
            f"Training completed, accuracy: {final_acc:.1f}%",
            params_changed,
            parameters_changed=params_changed,
            final_accuracy=final_acc
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_7_analysis_wrapper():
    """Test 7: Analysis Wrapper"""
    test = TestResult(
        "Analysis Wrapper",
        "Check HeadlessWrapper for ViTClassProjectionAnalyzer compatibility",
        "Wrapper should provide same outputs as base model with fixed task_id"
    )
    
    try:
        from task_incremental_baseline import ViT64MultiHead, HeadlessWrapper
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_model = ViT64MultiHead().to(device)
        task_id = 1
        wrapped_model = HeadlessWrapper(base_model, task_id).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Compare outputs
        task_labels = torch.full((batch_size,), task_id, dtype=torch.long).to(device)
        base_output = base_model(x, task_labels)
        wrapped_output = wrapped_model(x)
        
        outputs_match = torch.allclose(base_output, wrapped_output, atol=1e-6)
        
        test.complete(
            "Wrapper produces identical outputs",
            outputs_match,
            outputs_match=outputs_match,
            fixed_task_id=task_id
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def test_8_order_definitions():
    """Test 8: Order Definitions"""
    test = TestResult(
        "Learning Orders",
        "Check MTI and ITM order definitions",
        "MTI should start with mnist→imagenet, ITM should be reverse"
    )
    
    try:
        from task_incremental_baseline import ORDER, DATASETS
        
        # ORDER should contain all datasets
        all_datasets = set(DATASETS.keys())
        order_set = set(ORDER)
        
        # Check completeness
        complete = order_set == all_datasets
        
        # Check specific ordering
        starts_mnist = ORDER[0] == "mnist"
        ends_imagenet = ORDER[-1] == "imagenet"
        
        mti_order = ORDER
        itm_order = ORDER[::-1]
        
        test.complete(
            f"Orders defined correctly: {ORDER[0]}→{ORDER[-1]}",
            complete and starts_mnist and ends_imagenet,
            mti_order=mti_order,
            itm_order=itm_order,
            all_datasets_included=complete
        )
        
    except Exception as e:
        test.complete(f"Failed: {str(e)}", False, error=str(e))
    
    test.show()
    return test


def run_all_tests():
    """Run all component tests and show summary."""
    print("="*60)
    print("TASK-INCREMENTAL LEARNING COMPONENT TESTS")
    print("="*60)
    
    tests = [
        test_1_dataset_loaders(),
        test_2_scenario_creation(), 
        test_3_vit_model(),
        test_4_multihead_tasks(),
        test_5_accuracy_function(),
        test_6_training_loop(),
        test_7_analysis_wrapper(),
        test_8_order_definitions()
    ]
    
    # Summary
    passed = sum(1 for t in tests if t.passed)
    total = len(tests)
    
    print("\n" + "="*60)
    print(f"SUMMARY: {passed}/{total} TESTS PASSED")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Framework is ready for experiments.")
    else:
        print("⚠️  Some tests failed. Check details above.")
    
    # Show results
    print("\nPassed:")
    for test in tests:
        if test.passed:
            print(f"  ✅ {test.name}")
    
    print("\nFailed:")
    for test in tests:
        if not test.passed:
            print(f"  ❌ {test.name}: {test.outcome}")
    
    # Save results
    results = {
        "summary": {"passed": passed, "total": total, "success_rate": passed/total},
        "tests": [{
            "name": t.name,
            "description": t.description,
            "expected": t.expected,
            "outcome": t.outcome,
            "passed": t.passed,
            "details": t.details
        } for t in tests]
    }
    
    Path("logs").mkdir(exist_ok=True)
    with open("logs/component_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results: logs/component_test_results.json")
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 
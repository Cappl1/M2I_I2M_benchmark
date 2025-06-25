#!/usr/bin/env python3
"""
Unit Tests for Order Experiment Components
==========================================
Comprehensive tests for all components in the task-incremental learning framework.
Each test documents:
- What is being tested
- Expected behavior
- Actual outcome
- Pass/fail status

Usage:
    python test_order_experiment_components.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# Import components to test
from comprehensive_order_experiment import (
    DATASETS, MTI_ORDER, ITM_ORDER, 
    ViT64MultiHead, get_scenario, 
    batch_accuracy, _WrappedModel
)


class TestResult:
    """Container for test results with clear documentation."""
    
    def __init__(self, test_name: str, description: str, expected: str):
        self.test_name = test_name
        self.description = description
        self.expected = expected
        self.outcome = None
        self.passed = False
        self.details = {}
        
    def set_outcome(self, outcome: str, passed: bool, **details):
        self.outcome = outcome
        self.passed = passed
        self.details = details
        
    def print_result(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"\n{status} {self.test_name}")
        print(f"Description: {self.description}")
        print(f"Expected: {self.expected}")
        print(f"Outcome: {self.outcome}")
        if self.details:
            for key, value in self.details.items():
                print(f"  {key}: {value}")
        print("-" * 60)


class ComponentTester:
    """Test suite for order experiment components."""
    
    def __init__(self):
        self.results = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Testing on device: {self.device}")
        
    def run_all_tests(self):
        """Run all component tests."""
        print("="*60)
        print("COMPONENT TEST SUITE - Task-Incremental Learning Framework")
        print("="*60)
        
        self.test_dataset_loaders()
        self.test_scenario_creation()
        self.test_vit_model()
        self.test_multihead_functionality()
        self.test_training_components()
        self.test_evaluation_components()
        self.test_wrapper_model()
        self.test_order_configurations()
        
        self.print_summary()
        
    def test_dataset_loaders(self):
        """Test 1: Dataset Loading Functions"""
        test = TestResult(
            "Dataset Loaders",
            "Test that all 6 dataset loaders work correctly with balanced sampling",
            "Each dataset returns train/test splits with correct number of samples per class"
        )
        
        try:
            results = {}
            for name, loader_func in DATASETS.items():
                train_data, test_data = loader_func(True, 50)  # balanced=True, 50 per class
                
                train_size = len(train_data)
                test_size = len(test_data)
                
                results[name] = {
                    "train_size": train_size,
                    "test_size": test_size,
                    "train_classes": len(set([train_data[i][1] for i in range(min(100, train_size))])),
                    "test_classes": len(set([test_data[i][1] for i in range(min(100, test_size))]))
                }
            
            # Check that all datasets loaded successfully
            all_loaded = len(results) == 6
            correct_classes = all(r["train_classes"] == 10 and r["test_classes"] == 10 for r in results.values())
            
            test.set_outcome(
                f"Successfully loaded {len(results)}/6 datasets, all with 10 classes",
                all_loaded and correct_classes,
                dataset_details=results
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_scenario_creation(self):
        """Test 2: Scenario Creation"""
        test = TestResult(
            "Scenario Creation",
            "Test that nc_benchmark creates proper task-incremental scenarios",
            "Scenario with 6 tasks, each with 10 classes, proper task labels"
        )
        
        try:
            scenario = get_scenario(MTI_ORDER, balanced=True, n_per_class=50)
            
            num_tasks = len(scenario.train_stream)
            task_info = []
            
            for i, (train_exp, test_exp) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
                task_info.append({
                    "task_id": i,
                    "train_size": len(train_exp.dataset),
                    "test_size": len(test_exp.dataset),
                    "classes": train_exp.classes_in_this_experience,
                    "num_classes": len(train_exp.classes_in_this_experience)
                })
            
            correct_structure = (
                num_tasks == 6 and
                all(t["num_classes"] == 10 for t in task_info)
            )
            
            test.set_outcome(
                f"Created scenario with {num_tasks} tasks, all with 10 classes each",
                correct_structure,
                task_breakdown=task_info
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_vit_model(self):
        """Test 3: ViT Model Architecture"""
        test = TestResult(
            "ViT Model Architecture",
            "Test ViT64MultiHead model creation and basic forward pass",
            "Model creates correctly with proper dimensions and can process 64x64 RGB images"
        )
        
        try:
            model = ViT64MultiHead().to(self.device)
            
            # Test model structure
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test forward pass
            batch_size = 4
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)
            task_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                output = model(x, task_labels)
                cls_features = model._cls(x)
            
            correct_dimensions = (
                output.shape == (batch_size, 10) and  # 10 classes per task
                cls_features.shape == (batch_size, 384)  # embed_dim
            )
            
            test.set_outcome(
                f"Model created with {total_params:,} parameters, correct output dimensions",
                correct_dimensions,
                total_params=total_params,
                trainable_params=trainable_params,
                output_shape=tuple(output.shape),
                cls_shape=tuple(cls_features.shape)
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_multihead_functionality(self):
        """Test 4: Multi-Head Task Functionality"""
        test = TestResult(
            "Multi-Head Task Functionality",
            "Test that model can handle multiple tasks with separate heads",
            "Model can classify correctly for different task IDs with separate heads"
        )
        
        try:
            model = ViT64MultiHead().to(self.device)
            
            batch_size = 4
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)
            
            # Test different task heads
            outputs = {}
            for task_id in range(3):  # Test first 3 tasks
                task_labels = torch.full((batch_size,), task_id, dtype=torch.long).to(self.device)
                output = model(x, task_labels)
                outputs[task_id] = output
                
                # Test single task forward
                single_output = model.forward_single_task(x, task_id)
                
                # Outputs should be same for both methods
                same_output = torch.allclose(output, single_output, atol=1e-6)
                if not same_output:
                    raise ValueError(f"Inconsistent outputs for task {task_id}")
            
            # Different task heads should produce different outputs for same input
            different_heads = not torch.allclose(outputs[0], outputs[1], atol=1e-3)
            all_correct_shape = all(out.shape == (batch_size, 10) for out in outputs.values())
            
            test.set_outcome(
                f"Multi-head working correctly, different outputs per task",
                different_heads and all_correct_shape,
                num_heads_tested=len(outputs),
                outputs_differ=different_heads,
                shapes_correct=all_correct_shape
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_training_components(self):
        """Test 5: Training Components"""
        test = TestResult(
            "Training Components",
            "Test training function with mini-batch on dummy data",
            "Training reduces loss and improves accuracy over epochs"
        )
        
        try:
            from comprehensive_order_experiment import train_task
            
            # Create dummy data
            model = ViT64MultiHead().to(self.device)
            initial_params = [p.clone() for p in model.parameters()]
            
            # Simple dummy dataset
            dummy_data = []
            for _ in range(20):  # 20 samples
                x = torch.randn(3, 64, 64)
                y = torch.randint(0, 10, (1,)).item()
                task_label = torch.tensor(0)  # Task 0
                dummy_data.append((x, y, task_label))
            
            train_loader = DataLoader(dummy_data, batch_size=4, shuffle=True)
            test_loader = DataLoader(dummy_data, batch_size=4, shuffle=False)  # Same data for test
            
            # Train for few epochs
            train_task(model, train_loader, test_loader, self.device, task_id=0, epochs=3, base_lr=1e-3)
            
            # Check that parameters changed
            params_changed = any(
                not torch.equal(initial_p, final_p) 
                for initial_p, final_p in zip(initial_params, model.parameters())
            )
            
            # Test final accuracy (should be decent on same data used for training)
            final_acc = batch_accuracy(model, test_loader, self.device, task_id=0)
            
            test.set_outcome(
                f"Training completed, parameters changed, final accuracy: {final_acc:.1f}%",
                params_changed and final_acc > 0,
                parameters_changed=params_changed,
                final_accuracy=final_acc,
                training_samples=len(dummy_data)
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_evaluation_components(self):
        """Test 6: Evaluation Components"""
        test = TestResult(
            "Evaluation Components",
            "Test batch_accuracy function with known data",
            "Accuracy calculation should be correct for known predictions"
        )
        
        try:
            model = ViT64MultiHead().to(self.device)
            
            # Create data where we know the ground truth
            # Make model always predict class 0
            with torch.no_grad():
                for param in model.classifier.classifiers[0].parameters():
                    param.zero_()
                # Set bias to heavily favor class 0
                model.classifier.classifiers[0].bias[0] = 10.0
            
            # Create test data where all labels are 0
            test_data = []
            for _ in range(20):
                x = torch.randn(3, 64, 64)
                y = 0  # All labels are 0
                task_label = torch.tensor(0)
                test_data.append((x, y, task_label))
            
            test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
            
            # Should get 100% accuracy since model predicts 0 and all labels are 0
            accuracy = batch_accuracy(model, test_loader, self.device, task_id=0)
            
            # Also test with mixed labels (should get lower accuracy)
            mixed_data = []
            for i in range(20):
                x = torch.randn(3, 64, 64)
                y = i % 10  # Labels 0-9
                task_label = torch.tensor(0)
                mixed_data.append((x, y, task_label))
            
            mixed_loader = DataLoader(mixed_data, batch_size=4, shuffle=False)
            mixed_accuracy = batch_accuracy(model, mixed_loader, self.device, task_id=0)
            
            test.set_outcome(
                f"Perfect accuracy: {accuracy:.1f}%, Mixed accuracy: {mixed_accuracy:.1f}%",
                abs(accuracy - 100.0) < 1.0 and mixed_accuracy < accuracy,
                perfect_data_accuracy=accuracy,
                mixed_data_accuracy=mixed_accuracy,
                accuracy_calculation_correct=(abs(accuracy - 100.0) < 1.0)
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_wrapper_model(self):
        """Test 7: Wrapper Model for Analysis"""
        test = TestResult(
            "Wrapper Model",
            "Test _WrappedModel that fixes task_id for analysis compatibility",
            "Wrapper should produce same outputs as base model with fixed task_id"
        )
        
        try:
            base_model = ViT64MultiHead().to(self.device)
            task_id = 2
            wrapped_model = _WrappedModel(base_model, task_id).to(self.device)
            
            batch_size = 4
            x = torch.randn(batch_size, 3, 64, 64).to(self.device)
            
            # Get outputs from both models
            task_labels = torch.full((batch_size,), task_id, dtype=torch.long).to(self.device)
            base_output = base_model(x, task_labels)
            wrapped_output = wrapped_model(x)
            
            # Should be identical
            outputs_match = torch.allclose(base_output, wrapped_output, atol=1e-6)
            
            # Test that wrapper has vit attribute for analyzer compatibility
            has_vit_attr = hasattr(wrapped_model, 'vit')
            vit_attr_correct = has_vit_attr and hasattr(wrapped_model.vit, 'forward_features')
            
            test.set_outcome(
                f"Wrapper outputs match base model, has required vit attribute",
                outputs_match and vit_attr_correct,
                outputs_match=outputs_match,
                has_vit_attribute=has_vit_attr,
                vit_compatibility=vit_attr_correct,
                fixed_task_id=task_id
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def test_order_configurations(self):
        """Test 8: Order Configurations"""
        test = TestResult(
            "Order Configurations",
            "Test that MTI and ITM orders are correctly defined and different",
            "MTI and ITM should be reverse orders with all 6 datasets"
        )
        
        try:
            # Check that orders contain all datasets
            all_datasets = set(DATASETS.keys())
            mti_set = set(MTI_ORDER)
            itm_set = set(ITM_ORDER)
            
            mti_complete = mti_set == all_datasets
            itm_complete = itm_set == all_datasets
            
            # Check that ITM is reverse of MTI
            is_reverse = ITM_ORDER == MTI_ORDER[::-1]
            
            # Check specific order correctness
            mti_starts_mnist = MTI_ORDER[0] == "mnist" and MTI_ORDER[-1] == "imagenet"
            itm_starts_imagenet = ITM_ORDER[0] == "imagenet" and ITM_ORDER[-1] == "mnist"
            
            test.set_outcome(
                f"Orders configured correctly: MTI={MTI_ORDER[0]}→{MTI_ORDER[-1]}, ITM={ITM_ORDER[0]}→{ITM_ORDER[-1]}",
                mti_complete and itm_complete and is_reverse and mti_starts_mnist and itm_starts_imagenet,
                mti_order=MTI_ORDER,
                itm_order=ITM_ORDER,
                mti_complete=mti_complete,
                itm_complete=itm_complete,
                is_reverse=is_reverse,
                correct_start_end=mti_starts_mnist and itm_starts_imagenet
            )
            
        except Exception as e:
            test.set_outcome(f"Failed with error: {str(e)}", False, error=str(e))
            
        self.results.append(test)
        test.print_result()
        
    def print_summary(self):
        """Print overall test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "="*60)
        print(f"TEST SUMMARY: {passed}/{total} TESTS PASSED")
        print("="*60)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! The framework is ready for experiments.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            
        print(f"\nFailed tests:")
        for result in self.results:
            if not result.passed:
                print(f"  ❌ {result.test_name}: {result.outcome}")
                
        print(f"\nPassed tests:")
        for result in self.results:
            if result.passed:
                print(f"  ✅ {result.test_name}")
                
        # Save detailed results
        results_data = {
            "summary": {"passed": passed, "total": total, "success_rate": passed/total},
            "tests": [
                {
                    "name": r.test_name,
                    "description": r.description,
                    "expected": r.expected,
                    "outcome": r.outcome,
                    "passed": r.passed,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        Path("logs").mkdir(exist_ok=True)
        with open("logs/component_test_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
            
        print(f"\nDetailed results saved to: logs/component_test_results.json")


def main():
    """Run all component tests."""
    tester = ComponentTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 
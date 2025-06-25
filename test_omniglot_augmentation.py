#!/usr/bin/env python3
"""
Test script to verify Omniglot replacement sampling + augmentation works.
This should demonstrate that we can get 500 samples per class from Omniglot
even though it only has ~20 original samples per class.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from collections import Counter
from scenarios.datasets.omniglot import _load_omniglot
from scenarios.utils import filter_classes
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from scenarios.utils import transform_from_gray_to_rgb

def test_omniglot_augmentation():
    print("🧪 Testing Omniglot replacement sampling + augmentation...")
    
    # Create the same transform used in the real experiments
    transform_with_resize = Compose([
        ToTensor(),
        Resize((64, 64)),
        transform_from_gray_to_rgb(),
        Normalize(mean=(0.9221,), std=(0.2681,))
    ])
    
    # Test with 500 samples per class (this should trigger replacement sampling)
    print("\n📊 Loading Omniglot with balanced=True, 500 samples per class...")
    train_dataset, test_dataset = _load_omniglot(
        transform_with_resize, 
        balanced=True, 
        number_of_samples_per_class=500
    )
    
    # Filter to 10 classes as done in experiments
    print("🔍 Filtering to first 10 classes...")
    train_filtered, test_filtered = filter_classes(train_dataset, test_dataset, list(range(10)))
    
    # Count samples per class
    print("\n📈 Analyzing train dataset...")
    train_labels = []
    for i, item in enumerate(train_filtered):
        # Handle datasets that return (x, y) or (x, y, task_id)
        if len(item) == 2:
            img, label = item
        else:
            img, label = item[0], item[1]  # Handle extra task_id or other elements
        
        train_labels.append(label)
        if i < 5:  # Show first few samples
            print(f"  Sample {i}: label={label}, image_shape={img.shape}")
    
    train_counts = Counter(train_labels)
    print(f"\n✅ Train samples per class: {dict(train_counts)}")
    print(f"📊 Total train samples: {len(train_labels)}")
    print(f"🎯 Expected: 10 classes × 500 samples = 5000 total")
    print(f"✓ Success: {len(train_labels) == 5000}")
    
    # Check that all classes have exactly 500 samples
    all_have_500 = all(count == 500 for count in train_counts.values())
    print(f"✓ All classes have exactly 500 samples: {all_have_500}")
    
    # Also check test dataset
    print("\n📈 Analyzing test dataset...")
    test_labels = []
    for item in test_filtered:
        if len(item) == 2:
            _, label = item
        else:
            _, label = item[0], item[1]
        test_labels.append(label)
    test_counts = Counter(test_labels)
    print(f"📊 Test samples per class: {dict(test_counts)}")
    print(f"📊 Total test samples: {len(test_labels)}")
    
    return all_have_500 and len(train_labels) == 5000

if __name__ == "__main__":
    try:
        success = test_omniglot_augmentation()
        if success:
            print("\n🎉 SUCCESS: Omniglot replacement sampling + augmentation is working!")
        else:
            print("\n❌ FAILED: Issues with Omniglot replacement sampling")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
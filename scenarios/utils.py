import math
from collections import defaultdict
from typing import List, Tuple, Callable, Optional

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheTensorDataset
from torch.utils.data import Subset as ClassificationSubset
from torchvision.transforms import transforms, Compose as TorchCompose, RandomRotation, RandomHorizontalFlip, ColorJitter


def _filter_classes_in_single_dataset(dataset, classes):
    # Handle different ways to access targets
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, '_dataset') and hasattr(dataset._dataset, 'targets'):
        targets = dataset._dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        targets = dataset.dataset.targets
    else:
        # Fallback: extract targets by iterating through dataset
        targets = []
        for _, target in dataset:
            targets.append(target)
    
    indices = [i for i, t in enumerate(targets) if t in classes]
    max_class = max(classes)
    class_mapping = [-1] * (max_class + 1)
    for i, c in enumerate(sorted(classes)):
        class_mapping[c] = i

    subset = ClassificationSubset(dataset, indices)
    
    # Create AvalancheDataset that preserves targets properly
    avalanche_dataset = AvalancheDataset(subset)
    
    # Make sure targets are accessible
    if not hasattr(avalanche_dataset, 'targets') or avalanche_dataset.targets is None:
        # Extract targets from the subset
        filtered_targets = [targets[i] for i in indices]
        avalanche_dataset.targets = filtered_targets
    
    return avalanche_dataset


def filter_classes(train_dataset, test_dataset, classes):
    return _filter_classes_in_single_dataset(train_dataset, classes), _filter_classes_in_single_dataset(test_dataset,
                                                                                                        classes)


def _split_classes_list(classes, no_classes_in_task) -> List:
    return [classes[i * no_classes_in_task: (i + 1) * no_classes_in_task] for i in
            range(math.floor(len(classes) / no_classes_in_task))]


def separate_into_tasks(train_dataset, test_dataset, no_classes_in_task, targets_set) -> Tuple[List, List]:
    split = _split_classes_list(targets_set, no_classes_in_task)
    tasks = [filter_classes(train_dataset, test_dataset, classes_in_task) for classes_in_task in split]
    x = [train for train, _ in tasks], [test for _, test in tasks]
    return x


def transform_from_gray_to_rgb():
    return transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)


def balance_dataset(dataset, transform, number_of_samples_per_class):
    from collections import defaultdict
    img_occurrences = defaultdict(int)
    X, y = [], []

    # Determine exactly which classes are in the incoming dataset
    all_classes = set(t for _, t in dataset)

    # First pass: collect all available samples
    class_samples = defaultdict(list)  # Store samples by class for replacement sampling
    for img, target in dataset:
        class_samples[target].append(img)
        if img_occurrences[target] < number_of_samples_per_class:
            X.append(img)
            y.append(target)
            img_occurrences[target] += 1

    # Second pass: replacement sampling + augmentation for classes that need more samples
    import random
    
    # Create augmentation transforms for replacement sampling
    augment_transforms = TorchCompose([
        RandomRotation(degrees=15),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    for target_class in all_classes:
        needed_samples = number_of_samples_per_class - img_occurrences[target_class]
        if needed_samples > 0:
            # We need more samples for this class - do replacement sampling
            class_imgs = class_samples[target_class]
            for _ in range(needed_samples):
                # Randomly sample with replacement from existing images
                sampled_img = random.choice(class_imgs)
                # Apply augmentation transforms to create variety
                if hasattr(sampled_img, 'mode'):  # PIL Image
                    augmented_img = augment_transforms(sampled_img)
                else:
                    augmented_img = sampled_img  # Fallback for non-PIL images
                X.append(augmented_img)
                y.append(target_class)
                img_occurrences[target_class] += 1
        
    import torch
    from avalanche.benchmarks.utils import make_tensor_classification_dataset, AvalancheDataset
    
    # Don't convert PIL images to tensors here - let transforms handle it
    if y:
        y = torch.tensor(y)
    
    # For PIL Images, convert to tensors to avoid hashability issues
    if X and hasattr(X[0], 'mode'):  # PIL Image check
        # Convert PIL images to tensors using the transform (which should include ToTensor)
        if transform is not None:
            # Apply transform to convert PIL to tensor
            X_tensors = []
            for img in X:
                try:
                    # Apply transform to convert PIL to tensor
                    tensor_img = transform(img)
                    X_tensors.append(tensor_img)
                except:
                    # Fallback: just convert to tensor manually
                    import torchvision.transforms as T
                    to_tensor = T.ToTensor()
                    X_tensors.append(to_tensor(img))
            X = torch.stack(X_tensors)
        else:
            # No transform provided, use basic ToTensor
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            X = torch.stack([to_tensor(img) for img in X])
        
        # Now create tensor dataset
        return make_tensor_classification_dataset(X, y, transform=None)  # Transform already applied
    else:
        # For already tensor data
        if X and not torch.is_tensor(X[0]):
            X = torch.stack([torch.tensor(x) for x in X])
        elif X:
            X = torch.stack(X)
        
        # Use Avalanche 0.4.0 compatible method - pass as separate arguments
        return make_tensor_classification_dataset(X, y, transform=transform)


def load_dataset(dataset_loader: Callable[[Optional[Callable]], AvalancheDataset], transform: Optional[Callable],
                 balanced: bool, number_of_samples_per_class=None):
    if balanced:
        return balance_dataset(dataset_loader(None), transform, number_of_samples_per_class)
    else:
        return dataset_loader(transform)


# scenarios/utils/cached_loader.py
import hashlib, inspect, os, torch
from pathlib import Path
import pickle

CACHE_DIR = Path("dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_dataset(loader_fn):
    """Decorator: cache (train_ds, test_ds) to <hash>.pt once."""
    def _wrapped(*args, **kw):
        sig = str(loader_fn.__name__) + str(args) + str(kw)
        fname = CACHE_DIR / (hashlib.md5(sig.encode()).hexdigest() + ".pt")
        if fname.exists():
            print(f"⚡ cache hit → {fname.name}")
            train_ds, test_ds = torch.load(fname)
            return train_ds, test_ds

        train_ds, test_ds = loader_fn(*args, **kw)
        
        # Try to save to cache, but handle pickling errors gracefully
        try:
            torch.save((train_ds, test_ds), fname)
            print(f"💾 cached → {fname.name}")
        except (AttributeError, TypeError, pickle.PicklingError) as e:
            # Datasets contain unpicklable objects (e.g., lambda functions in transforms)
            print(f"⚠️  skipping cache (unpicklable): {e}")
            # Remove the file if it was partially created
            if fname.exists():
                fname.unlink()
        
        return train_ds, test_ds
    return _wrapped

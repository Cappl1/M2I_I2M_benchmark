#!/usr/bin/env python3
"""
Cumulative baseline for the six‑task 64×64 vision benchmark
───────────────────────────────────────────────────────────
This script implements the *cumulative* (a.k.a. joint or upper‑bound)
strategy requested in the chat.  After every new task arrives we **add
its dataset to a growing replay pool** and train the same Vision
Transformer on the *union* of all data seen so far.

Behaviour
---------
* **Train stream**: identical order and sampling to the original
  `vit_class_incremental_full_test.py`, but instead of fine‑tuning only
  on the current experience we concatenate all past datasets with the
  current one.
* **Model**: exactly the same `ViT64SingleHead` (60‑class head).
* **Evaluation**: after each task we still test on every dataset seen so
  far, so you get a forgetting matrix comparable to the other baselines.

Usage example
-------------
```bash
python vit_cumulative_incremental.py \
       --epochs 40 --batch 64 --device cuda:0 --n_per_class 500
```

Compared with the naive baseline you should observe **much higher
accuracies** on earlier tasks (because the network keeps seeing them
during every training phase) and BWT values close to zero.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms  # noqa: F401 (needed by dataset loaders)
from avalanche.benchmarks.utils import SupportedDataset

# ───────────────────────────── datasets ────────────────────────────
from scenarios.datasets.mnist import load_mnist_with_resize
from scenarios.datasets.omniglot import _load_omniglot
from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
from scenarios.datasets.svhn import load_svhn_resized
from scenarios.datasets.cifar import load_resized_cifar10
from scenarios.datasets.load_imagenet import load_imagenet
from scenarios.utils import cache_dataset, filter_classes, transform_from_gray_to_rgb

load_mnist_with_resize         = cache_dataset(load_mnist_with_resize)
load_fashion_mnist_with_resize = cache_dataset(load_fashion_mnist_with_resize)
load_svhn_resized              = cache_dataset(load_svhn_resized)
load_resized_cifar10           = cache_dataset(load_resized_cifar10)
load_imagenet                  = cache_dataset(load_imagenet)

import torch
from torch.utils.data import Dataset

class RelabeledDataset(Dataset):
    """Wrapper to remap dataset labels to a different range"""
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if len(item) == 2:
            x, y = item
            return x, y + self.label_offset
        else:
            # Handle datasets that return (x, y, task_id)
            x, y = item[0], item[1]
            return x, y + self.label_offset

def _omniglot10(b: bool, n: int):
    from scenarios.datasets.omniglot import _load_omniglot
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    
    transform_with_resize = Compose([
        ToTensor(),
        Resize((64, 64)),
        transform_from_gray_to_rgb(),
        Normalize(mean=(0.9221,), std=(0.2681,))
    ])
    
    tr, te = _load_omniglot(transform_with_resize, balanced=b, number_of_samples_per_class=n)
    return filter_classes(tr, te, list(range(10)))

def _imagenet10(b: bool, n: int):
    tr, te = load_imagenet(balanced=b, number_of_samples_per_class=n)
    return filter_classes(tr, te, list(range(10)))

DATASETS: Dict[str, Callable[[bool, int], Tuple[SupportedDataset, SupportedDataset]]] = {
    "mnist":          lambda b, n: load_mnist_with_resize(balanced=b, number_of_samples_per_class=n),
    "omniglot":       _omniglot10,
    "fashion_mnist":  lambda b, n: load_fashion_mnist_with_resize(balanced=b, number_of_samples_per_class=n),
    "svhn":           lambda b, n: load_svhn_resized(balanced=b, number_of_samples_per_class=n),
    "cifar10":        lambda b, n: load_resized_cifar10(balanced=b, number_of_samples_per_class=n),
    "imagenet":       _imagenet10,
}

ORDER = [
    "mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"
]
TOTAL_CLASSES = len(ORDER) * 10  # 60 classes overall

# ───────────────────────────── model ───────────────────────────────
from timm.models.vision_transformer import VisionTransformer

class ViT64SingleHead(nn.Module):
    """Vision Transformer (64×64) with a single 60‑class head."""

    def __init__(
        self,
        num_classes: int = TOTAL_CLASSES,
        patch: int = 8,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=64,
            patch_size=patch,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            class_token=True,
            global_pool="token",
        )
        self.head = self.vit.head  # alias for convenience

    def forward(self, x):
        return self.vit(x)

# ──────────────────────────── utilities ────────────────────────────

def batch_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            if len(batch) >= 2:
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x, y = batch  # type: ignore[misc]
                x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


# ───────── wrapper to slice logits for analyzer (10 per task) ──
class TaskSliceWrapper(nn.Module):
    def __init__(self, base: ViT64SingleHead, tid: int):
        super().__init__(); self.base, self.tid = base, tid
    def forward(self, x):
        out = self.base(x)
        start = self.tid * 10; end = start + 10
        return out[:, start:end]


def train_epochs(model: nn.Module, loader: DataLoader, device: str, *,
                 epochs: int = 40, lr: float = 3e-4) -> None:
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce    = nn.CrossEntropyLoss()

    for ep in range(epochs):
        for batch in loader:
            # supports datasets that may return (x, y) or (x, y, task_id)
            x, y = batch[0].to(device), batch[1].to(device)
            loss = ce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

# ───────────────────────────── analysis import ────────────────────────────
from analysis.vit_class_projection import ViTClassProjectionAnalyzer

# ───────────────────────────── main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="epochs per incremental step")
    parser.add_argument("--batch",  type=int, default=64, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_per_class", type=int, default=500, help="limit per class (use -1 for full)")
    parser.add_argument("--logdir", type=str, default="logs/task_incremental_cumlative_baseline")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device: str = args.device

    # Pre‑load all datasets once so we can reuse them
    train_ds: List[SupportedDataset] = []
    test_ds:  List[SupportedDataset] = []
    for key in ORDER:
        tr, te = DATASETS[key](True, args.n_per_class)
        train_ds.append(tr); test_ds.append(te)

    model = ViT64SingleHead().to(device)
    acc_matrix: List[List[float]] = []
    train_acc_matrix: List[List[float]] = []
    proj_scores = defaultdict(dict)

    cumulative_train_pool = []

    for tid, (current_tr, current_te) in enumerate(zip(train_ds, test_ds)):
        print(f"\n=== Task {tid}: {ORDER[tid]} ===")

        # Add relabeled dataset to cumulative pool
        label_offset = tid * 10  # 0, 10, 20, 30, 40, 50
        relabeled_train = RelabeledDataset(current_tr, label_offset)
        cumulative_train_pool.append(relabeled_train)
        
        cum_dataset = ConcatDataset(cumulative_train_pool)
        cum_loader = DataLoader(cum_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        # Train
        train_epochs(model, cum_loader, device, epochs=args.epochs)

        # Evaluate on all tasks seen so far (test) - also need to relabel test sets
        test_row = []
        for k in range(tid + 1):
            test_offset = k * 10
            relabeled_test = RelabeledDataset(test_ds[k], test_offset)
            loader = DataLoader(relabeled_test, batch_size=args.batch, num_workers=2)
            acc = batch_accuracy(model, loader, device)
            test_row.append(acc)
        acc_matrix.append(test_row)
        
        # Evaluate on all tasks seen so far (train)
        train_row = []
        for k in range(tid + 1):
            train_offset = k * 10
            relabeled_train_eval = RelabeledDataset(train_ds[k], train_offset)
            loader = DataLoader(relabeled_train_eval, batch_size=args.batch, num_workers=2)
            acc = batch_accuracy(model, loader, device)
            train_row.append(acc)
        train_acc_matrix.append(train_row)
        
        print("Test  accuracies on tasks 0‑", tid, ":", [f"{a:.2f}" for a in test_row])
        print("Train accuracies on tasks 0‑", tid, ":", [f"{a:.2f}" for a in train_row])

        # Projection analysis for all tasks seen so far
        analyzer = ViTClassProjectionAnalyzer(model, device)
        
        for k in range(tid + 1):
            # Create properly relabeled datasets for analysis
            train_offset = k * 10
            test_offset = k * 10
            relabeled_train_analysis = RelabeledDataset(train_ds[k], train_offset)
            relabeled_test_analysis = RelabeledDataset(test_ds[k], test_offset)
            
            train_loader = DataLoader(relabeled_train_analysis, batch_size=args.batch, num_workers=2)
            test_loader = DataLoader(relabeled_test_analysis, batch_size=args.batch, num_workers=2)
            
            for split, loader in [("train", train_loader), ("test", test_loader)]:
                scores = analyzer.analyze_task_representations(
                    loader, 
                    task_id=k, 
                    num_classes_per_task=10
                )
                proj_scores[f"after_task{tid}_analyzing_task{k}_{split}"] = scores


    # ───────── save results ─────────
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.logdir) / "results.json"
    json.dump({
        "order": ORDER, 
        "test_accuracies": acc_matrix, 
        "train_accuracies": train_acc_matrix,
        "projection_scores": proj_scores
    }, out_file.open("w"), indent=2)
    print("\nSaved cumulative results →", out_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Class‑incremental baseline + representation analysis on six 64×64 datasets
---------------------------------------------------------------------------
Datasets/tasks (10 classes each, in this order)
  0. MNIST          1. Omniglot‑10  2. Fashion‑MNIST
  3. SVHN           4. CIFAR‑10     5. ImageNet‑10

This script trains **one ViT‑64 with a single 60‑class head** that expands as
new tasks arrive (class‑incremental learning). After each task it logs:
  • test accuracy on *all* tasks seen so far (forgetting matrix)
  • CLS‑token projection quality via `ViTClassProjectionAnalyzer`.

Results are saved to `logs/vit_class_incremental/results.json`.

Run example
```
python vit_class_incremental_full_test.py \
       --epochs 40 --batch 64 --device auto --n_per_class 500
```
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import SupportedDataset

# ─────────────────────────── dataset loaders ──
from scenarios.datasets.mnist import load_mnist_with_resize
from scenarios.datasets.omniglot import load_resized_omniglot
from scenarios.datasets.fashion_mnist import load_fashion_mnist_with_resize
from scenarios.datasets.svhn import load_svhn_resized
from scenarios.datasets.cifar import load_resized_cifar10
from scenarios.datasets.load_imagenet import load_imagenet
# ── caching wrapper ───────────────────────────────────────────
from scenarios.utils import cache_dataset

# apply decorator once so every later call is cached to disk
load_mnist_with_resize           = cache_dataset(load_mnist_with_resize)
load_resized_omniglot            = cache_dataset(load_resized_omniglot)
load_fashion_mnist_with_resize   = cache_dataset(load_fashion_mnist_with_resize)
load_svhn_resized                = cache_dataset(load_svhn_resized)
load_resized_cifar10             = cache_dataset(load_resized_cifar10)
load_imagenet                    = cache_dataset(load_imagenet)
from scenarios.utils import filter_classes


def _omniglot10(b: bool, n: int):
    from scenarios.datasets.omniglot import _load_omniglot
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    from scenarios.utils import transform_from_gray_to_rgb
    
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
    "mnist": lambda b, n: load_mnist_with_resize(b, n),
    "omniglot": _omniglot10,
    "fashion_mnist": lambda b, n: load_fashion_mnist_with_resize(b, n),
    "svhn": lambda b, n: load_svhn_resized(b, n),
    "cifar10": lambda b, n: load_resized_cifar10(b, n),
    "imagenet": _imagenet10,
}

ORDER = ["mnist", "omniglot", "fashion_mnist", "svhn", "cifar10", "imagenet"]
TOTAL_CLASSES = len(ORDER) * 10  # 60

# ───────────────────── scenario (class‑incremental) ──

def get_scenario(n_per_class: int):
    train, test = [], []
    for key in ORDER:
        tr, te = DATASETS[key](True, n_per_class)
        train.append(tr)
        test.append(te)
    return nc_benchmark(
        train_dataset=train,
        test_dataset=test,
        task_labels=False,                  # single head → no task labels
        one_dataset_per_exp=True,
        class_ids_from_zero_from_first_exp=True,
        shuffle=False,
        n_experiences=0,
    )

# ───────────────────────────── model ──
from timm.models.vision_transformer import VisionTransformer


class ViT64SingleHead(nn.Module):
    """ViT‑64 with one 60‑class head for class‑incremental learning."""

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
            num_classes=num_classes,  # built‑in classifier head
            embed_dim=dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            class_token=True,
            global_pool="token",
        )
        # alias for analyzer convenience
        self.head = self.vit.head

    def forward(self, x):
        return self.vit(x)

# ─────────────────────── trainer & utils ──

def batch_accuracy(model, loader, device):
    model.eval(); tot = corr = 0
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]  # Handle extra task_id or other elements
            x, y = x.to(device), y.to(device)
            corr += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
    return 100 * corr / tot if tot else 0.


def train_task(model, tr_loader, te_loader, device, epochs=40, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        model.train(); corr = tot = 0
        for batch in tr_loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]  # Handle extra task_id or other elements
            x, y = x.to(device), y.to(device)
            loss = ce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            corr += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
        sched.step()
        
        # Evaluation phase
        if ep % 10 == 9:  # Every 10 epochs
            train_acc = batch_accuracy(model, tr_loader, device)
            test_acc = batch_accuracy(model, te_loader, device)
            print(f"    epoch {ep+1:2d}: train {train_acc:5.1f}%  test {test_acc:5.1f}%")


# ───────── wrapper to slice logits for analyzer (10 per task) ──
class TaskSliceWrapper(nn.Module):
    def __init__(self, base: ViT64SingleHead, tid: int):
        super().__init__(); self.base, self.tid = base, tid
    def forward(self, x):
        out = self.base(x)
        start = self.tid * 10; end = start + 10
        return out[:, start:end]


from analysis.vit_class_projection import ViTClassProjectionAnalyzer

# ───────────────────────────── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--logdir", type=str, default="logs/vit_class_incremental")
    args = ap.parse_args()

    device = "cuda:3"

    scenario = get_scenario(args.n_per_class)
    print("Tasks:", ORDER)

    model = ViT64SingleHead().to(device)
    acc_mat: List[List[float]] = []
    proj_scores = defaultdict(dict)

    for tid, (tr_exp, te_exp) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
        print(f"\n=== Task {tid}: {tr_exp.classes_in_this_experience} ===")
        
        tr_loader = DataLoader(tr_exp.dataset, batch_size=args.batch, shuffle=True)
        te_loader = DataLoader(te_exp.dataset, batch_size=args.batch)
        
        # Train with proper evaluation
        train_task(model, tr_loader, te_loader, device, epochs=args.epochs)
        
        # Evaluate on all seen tasks
        row = []
        for k in range(tid + 1):
            loader = DataLoader(scenario.test_stream[k].dataset, batch_size=args.batch)
            acc = batch_accuracy(model, loader, device)
            row.append(acc)
        acc_mat.append(row)
        print("Accuracies on tasks 0 to", tid, ":", row)
        
        # For projection analysis, use the unwrapped model
        analyzer = ViTClassProjectionAnalyzer(model, device)
        
        # Analyze with proper task context
        for split, loader in [("train", tr_loader), ("test", te_loader)]:
            scores = analyzer.analyze_task_representations(
                loader, 
                task_id=tid, 
                num_classes_per_task=10
            )
            proj_scores[f"task{tid}_{split}"] = scores

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    out = Path(args.logdir) / "results.json"
    json.dump({"order": ORDER, "accuracies": acc_mat, "projection_scores": proj_scores}, out.open("w"), indent=2)
    print("\nSaved results →", out)


if __name__ == "__main__":
    main()

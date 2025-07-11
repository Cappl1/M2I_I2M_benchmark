"""Base strategy framework for continual learning experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import random
import copy
from collections import deque


class BaseStrategy(ABC):
    """Base class for continual learning strategies."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.current_task = 0
        
    @abstractmethod
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train on a single task and return metrics."""
        pass
    
    def _base_train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                         criterion: nn.Module) -> Tuple[float, float]:
        """Base training loop for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        
        for batch in train_loader:
            # Safe batch unpacking
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]  # Take first two elements
            else:
                x, y = batch  # Direct unpacking
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        accuracy = 100.0 * correct / total
        return total_loss / len(train_loader), accuracy
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer based on config."""
        lr = self.config.get('lr', 0.001)
        optimizer_name = self.config.get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=self.config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _analyze_all_tasks(self, analyzer: Any, all_datasets: Optional[Dict[str, Any]], 
                          order: Optional[List[str]], current_task: int, epoch: int) -> Dict[str, Any]:
        """Analyze representations for all seen tasks."""
        if all_datasets is None or order is None:
            return {}
            
        epoch_analysis = {}
        
        for j in range(current_task + 1):
            dataset_name = order[j]
            _, test_ds = all_datasets[dataset_name]
            
            # Get accuracy
            acc = self._evaluate_dataset(test_ds)
            
            # Analyze representations
            loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            repr_results = analyzer.analyze_task_representations(
                loader, j, 
                num_classes_per_task=self.config.get('num_classes', 10),
                max_batches=50,
                sample_tokens=True
            )
            
            epoch_analysis[f'task_{j}_{dataset_name}'] = {
                'accuracy': acc,
                'representations': repr_results
            }
            
            status = "CURRENT" if j == current_task else "PREVIOUS"
            print(f"      → {dataset_name} ({status}): {acc:.1f}%")
        
        return epoch_analysis
    
    def _evaluate_dataset(self, test_ds: Dataset) -> float:
        """Evaluate accuracy on a dataset."""
        self.model.eval()
        correct = total = 0
        
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        with torch.inference_mode():
            for batch in loader:
                # Safe batch unpacking
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]  # Take first two elements
                else:
                    x, y = batch  # Direct unpacking
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total


class NaiveStrategy(BaseStrategy):
    """Naive strategy: just train on current task."""
    
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train on current task only."""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('minibatch_size', 128), 
            shuffle=True
        )
        
        optimizer = self._build_optimizer()
        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get('epochs', 50)
        analysis_freq = self.config.get('analysis_freq', 10)
        
        trajectory_data = {}
        
        for epoch in range(epochs):
            loss, acc = self._base_train_epoch(train_loader, optimizer, criterion)
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}, Acc={acc:.1f}%")
        
        self.current_task = task_id + 1
        return {'trajectory': trajectory_data}
    



class ReplayStrategy(BaseStrategy):
    """Experience replay strategy with reservoir sampling."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(model, config, device)
        self.memory_size = config.get('memory_size', 500)  # samples per task
        self.replay_batch_ratio = config.get('replay_batch_ratio', 0.5)  # ratio of replay samples in batch
        self.memory_per_task = {}  # {task_id: (data, labels)}
        
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train with experience replay."""
        
        # Store samples from current task
        self._update_memory(train_dataset, task_id)
        
        # Create mixed dataloader if we have previous tasks
        if task_id > 0:
            train_loader = self._create_replay_dataloader(train_dataset, task_id)
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.get('minibatch_size', 128), 
                shuffle=True
            )
        
        optimizer = self._build_optimizer()
        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get('epochs', 50)
        analysis_freq = self.config.get('analysis_freq', 10)
        
        trajectory_data = {}
        
        for epoch in range(epochs):
            loss, acc = self._train_epoch_with_replay(train_loader, optimizer, criterion, task_id)
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}, Acc={acc:.1f}%")
        
        self.current_task = task_id + 1
        return {'trajectory': trajectory_data, 'memory_stats': self._get_memory_stats()}
    
    def _update_memory(self, dataset: Dataset, task_id: int):
        """Update memory buffer with reservoir sampling."""
        # Get all data from current task
        data_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
        all_data = []
        all_labels = []
        
        for batch in data_loader:
            # Safe batch unpacking
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            all_data.append(x)
            all_labels.append(y)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Reservoir sampling
        n_samples = min(self.memory_size, len(all_data))
        indices = torch.randperm(len(all_data))[:n_samples]
        
        self.memory_per_task[task_id] = (
            all_data[indices].cpu(),
            all_labels[indices].cpu()
        )
        
        print(f"    → Stored {n_samples} samples from task {task_id} in memory")
    
    def _create_replay_dataloader(self, current_dataset: Dataset, current_task: int) -> DataLoader:
        """Create dataloader mixing current task with replay samples."""
        batch_size = self.config.get('minibatch_size', 128)
        replay_batch_size = int(batch_size * self.replay_batch_ratio)
        current_batch_size = batch_size - replay_batch_size
        
        # Simple approach: create separate datasets and combine in training
        class SimpleReplayDataset(Dataset):
            def __init__(self, current_data, memory_dict):
                self.current_data = current_data
                self.memory_dict = memory_dict
                
                # Concatenate all memory data
                if memory_dict:
                    all_mem_x = []
                    all_mem_y = []
                    for tid, (mem_x, mem_y) in memory_dict.items():
                        all_mem_x.append(mem_x)
                        all_mem_y.append(mem_y)
                    self.memory_x = torch.cat(all_mem_x, dim=0)
                    self.memory_y = torch.cat(all_mem_y, dim=0)
                    self.memory_len = len(self.memory_x)
                else:
                    self.memory_x = None
                    self.memory_y = None
                    self.memory_len = 0
            
            def __len__(self):
                return len(self.current_data)
            
            def get_memory_batch(self, batch_size):
                """Get a random batch from memory."""
                if self.memory_len == 0 or self.memory_x is None or self.memory_y is None:
                    return None, None
                indices = torch.randint(0, self.memory_len, (batch_size,))
                mem_x = self.memory_x[indices]
                mem_y = self.memory_y[indices]
                # Ensure labels are tensors
                if not isinstance(mem_y, torch.Tensor):
                    mem_y = torch.tensor(mem_y)
                return mem_x, mem_y
            
            def __getitem__(self, idx):
                # Just return current data item
                item = self.current_data[idx]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    x, y = item[0], item[1]
                    # Ensure label is tensor
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y)
                    return x, y
                else:
                    raise ValueError(f"Unexpected item format in dataset: {type(item)}")
        
        replay_dataset = SimpleReplayDataset(current_dataset, self.memory_per_task)
        
        # Store dataset reference for use in training
        self._current_replay_dataset = replay_dataset
        
        # Return regular dataloader for current data
        return DataLoader(
            current_dataset,
            batch_size=current_batch_size,
            shuffle=True
        )
    
    def _train_epoch_with_replay(self, train_loader: DataLoader, 
                                optimizer: torch.optim.Optimizer,
                                criterion: nn.Module,
                                task_id: int) -> Tuple[float, float]:
        """Training loop that handles mixed batches."""
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        
        # Check if we have replay data
        has_replay = task_id > 0 and hasattr(self, '_current_replay_dataset')
        
        for batch_idx, batch in enumerate(train_loader):
            # Get current task data
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                curr_x, curr_y = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            curr_x = curr_x.to(self.device)
            curr_y = curr_y.to(self.device)
            
            # Get replay data if available
            if has_replay:
                batch_size = curr_x.size(0)
                # Calculate replay batch size more safely
                replay_batch_size = max(1, int(batch_size * self.replay_batch_ratio))
                
                mem_x, mem_y = self._current_replay_dataset.get_memory_batch(replay_batch_size)
                
                if mem_x is not None and mem_y is not None:
                    mem_x = mem_x.to(self.device)
                    mem_y = mem_y.to(self.device)
                    
                    # Combine current and memory samples
                    x = torch.cat([curr_x, mem_x], dim=0)
                    y = torch.cat([curr_y, mem_y], dim=0)
                else:
                    x, y = curr_x, curr_y
            else:
                x, y = curr_x, curr_y
            
            # Standard training step
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        accuracy = 100.0 * correct / total
        return total_loss / len(train_loader), accuracy
    
    def _get_memory_stats(self) -> Dict:
        """Get statistics about memory buffer."""
        stats = {}
        total_samples = 0
        
        for task_id, (data, labels) in self.memory_per_task.items():
            stats[f'task_{task_id}_samples'] = len(data)
            total_samples += len(data)
        
        stats['total_samples'] = total_samples
        return stats

class CumulativeStrategy(BaseStrategy):
    """Cumulative/Joint training strategy - upper bound baseline."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(model, config, device)
        self.all_train_data = []  # List of all training datasets
        
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train on all tasks seen so far (cumulative)."""
        
        # Add current task to cumulative dataset
        self.all_train_data.append(train_dataset)
        
        # Create combined dataset
        combined_dataset = ConcatDataset(self.all_train_data)
        
        print(f"    → Training on cumulative dataset ({len(combined_dataset)} samples from {len(self.all_train_data)} tasks)")
        
        train_loader = DataLoader(
            combined_dataset, 
            batch_size=self.config.get('minibatch_size', 128), 
            shuffle=True
        )
        
        optimizer = self._build_optimizer()
        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get('epochs', 50)
        analysis_freq = self.config.get('analysis_freq', 10)
        
        trajectory_data = {}
        
        for epoch in range(epochs):
            loss, acc = self._base_train_epoch(train_loader, optimizer, criterion)
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}, Acc={acc:.1f}%")
        
        self.current_task = task_id + 1
        return {'trajectory': trajectory_data, 'cumulative_size': len(combined_dataset)}


class KaizenStrategy(BaseStrategy):
    """Kaizen strategy with self-supervised learning and knowledge distillation."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(model, config, device)
        self.previous_model = None
        self.memory_size = config.get('memory_size', 500)
        self.memory_per_task = {}
        self.ssl_method = config.get('ssl_method', 'simclr')  # or 'byol', 'simsiam'
        self.kd_weight = config.get('kd_weight', 1.0)
        self.ssl_weight = config.get('ssl_weight', 1.0)
        
        # Build SSL projector/predictor heads
        self._build_ssl_heads()
        
    def _build_ssl_heads(self):
        """Build SSL projection and prediction heads."""
        # Get feature dimension from model
        if hasattr(self.model, 'base') and hasattr(self.model.base, 'vit'):
            vit = self.model.base.vit
            feat_dim = vit.embed_dims[-1] if hasattr(vit, 'embed_dims') else 768
        else:
            feat_dim = 768  # Default ViT dimension
        
        # Projector for SSL (current task)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)
        
        # Predictor for SSL  
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)
        
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train with Kaizen strategy."""
        
        # Store previous model for distillation
        if task_id > 0:
            self.previous_model = copy.deepcopy(self.model)
            self.previous_model.eval()
            for param in self.previous_model.parameters():
                param.requires_grad = False
        
        # Update memory
        if self.memory_size > 0:
            self._update_memory(train_dataset, task_id)
        
        # Create dataloader with augmentations
        train_loader = self._create_ssl_dataloader(train_dataset, task_id)
        
        # Setup optimizers
        feature_params = []
        classifier_params = []
        
        # Separate feature extractor and classifier parameters
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'head' in name:
                classifier_params.append(param)
            else:
                feature_params.append(param)
        
        # Add SSL heads to feature optimizer
        ssl_params = list(self.projector.parameters()) + list(self.predictor.parameters())
        
        feature_optimizer = torch.optim.Adam(
            feature_params + ssl_params, 
            lr=self.config.get('lr', 0.001)
        )
        classifier_optimizer = torch.optim.Adam(
            classifier_params, 
            lr=self.config.get('lr', 0.001)
        )
        
        epochs = self.config.get('epochs', 50)
        analysis_freq = self.config.get('analysis_freq', 10)
        trajectory_data = {}
        
        for epoch in range(epochs):
            loss_dict = self._train_epoch_kaizen(
                train_loader, feature_optimizer, classifier_optimizer, task_id
            )
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: "
                      f"SSL={loss_dict['ssl']:.3f}, "
                      f"CE={loss_dict['ce']:.3f}, "
                      f"KD={loss_dict['kd']:.3f}, "
                      f"Acc={loss_dict['acc']:.1f}%")
        
        self.current_task = task_id + 1
        return {'trajectory': trajectory_data, 'loss_components': loss_dict}
    
    def _create_ssl_dataloader(self, dataset: Dataset, task_id: int) -> DataLoader:
        """Create dataloader with SSL augmentations."""
        from torchvision import transforms
        
        # Define SSL augmentations
        ssl_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        class SSLDataset(Dataset):
            def __init__(self, base_dataset, transform, memory_data=None):
                self.base_dataset = base_dataset
                self.transform = transform
                self.memory_data = memory_data
                
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                # Safe batch unpacking - same pattern as used elsewhere in strategies.py
                item = self.base_dataset[idx]
                if isinstance(item, (list, tuple)):
                    img, label = item[0], item[1]  # Take first two elements
                else:
                    img, label = item  # Direct unpacking
                
                # Create two augmented views
                view1 = self.transform(img)
                view2 = self.transform(img)
                
                # Always return consistent format - no None values to avoid collate issues
                # If we have memory data, occasionally include memory sample
                has_memory = False
                mem_view1 = view1  # Default to current views
                mem_view2 = view2
                mem_label = label
                
                if self.memory_data and len(self.memory_data) > 0:
                    if random.random() < 0.5:  # 50% chance to include memory
                        # Random memory sample
                        mem_idx = random.randint(0, len(self.memory_data) - 1)
                        mem_img, mem_label = self.memory_data[mem_idx]
                        mem_view1 = self.transform(mem_img)
                        mem_view2 = self.transform(mem_img)
                        has_memory = True
                
                # Return consistent format with memory flag
                return {
                    'view1': view1,
                    'view2': view2, 
                    'label': label,
                    'mem_view1': mem_view1,
                    'mem_view2': mem_view2,
                    'mem_label': mem_label,
                    'has_memory': has_memory
                }
        
        # Prepare memory data if available
        memory_data = []
        if task_id > 0 and self.memory_per_task:
            for tid, (data, labels) in self.memory_per_task.items():
                for i in range(len(data)):
                    memory_data.append((data[i], labels[i]))
        
        ssl_dataset = SSLDataset(dataset, ssl_transform, memory_data)
        return DataLoader(
            ssl_dataset, 
            batch_size=self.config.get('minibatch_size', 128), 
            shuffle=True
        )
    
    def _train_epoch_kaizen(self, train_loader: DataLoader, 
                           feature_optimizer: torch.optim.Optimizer,
                           classifier_optimizer: torch.optim.Optimizer,
                           task_id: int) -> Dict[str, float]:
        """Kaizen training loop."""
        self.model.train()
        
        total_ssl_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        correct = total = 0
        
        for batch in train_loader:
            # Unpack dictionary batch format
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Memory samples (always present, but may be duplicates of current views)
            mem_view1 = batch['mem_view1'].to(self.device)
            mem_view2 = batch['mem_view2'].to(self.device)
            mem_labels = batch['mem_label'].to(self.device)
            has_memory = batch['has_memory']  # Boolean tensor indicating which samples have real memory
            
            # === Feature Extractor Training (SSL + KD) ===
            feature_optimizer.zero_grad()
            
            # Get features from both views
            features1 = self._get_features(self.model, view1)
            features2 = self._get_features(self.model, view2)
            
            # SSL loss (current task learning)
            z1 = self.projector(features1)
            z2 = self.projector(features2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            
            ssl_loss = self._compute_ssl_loss(p1, z2) + self._compute_ssl_loss(p2, z1)
            
            # Knowledge distillation for features (if not first task)
            kd_feat_loss = 0.0
            if self.previous_model is not None:
                with torch.no_grad():
                    prev_features1 = self._get_features(self.previous_model, view1)
                
                # Match current features to previous features
                kd_feat_loss = F.mse_loss(features1, prev_features1)
            
            # Total feature loss
            feat_loss = self.ssl_weight * ssl_loss + self.kd_weight * kd_feat_loss
            feat_loss.backward()
            feature_optimizer.step()
            
            # === Classifier Training (CE + KD) ===
            classifier_optimizer.zero_grad()
            
            # Forward pass for classification
            with torch.no_grad():
                features = self._get_features(self.model, view1)
            
            logits = self._get_classifier_output(self.model, features)
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, labels)
            
            # Knowledge distillation for classifier
            kd_class_loss = 0.0
            if self.previous_model is not None:
                with torch.no_grad():
                    prev_logits = self._get_classifier_output(self.previous_model, features)
                
                # Distillation loss
                kd_class_loss = self._distillation_loss(logits, prev_logits)
            
            # Total classifier loss
            class_loss = ce_loss + self.kd_weight * kd_class_loss
            class_loss.backward()
            classifier_optimizer.step()
            
            # Track metrics
            total_ssl_loss += float(ssl_loss)
            total_ce_loss += float(ce_loss)
            total_kd_loss += float(kd_feat_loss + kd_class_loss) if self.previous_model else 0
            
            pred = logits.argmax(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        
        return {
            'ssl': total_ssl_loss / len(train_loader),
            'ce': total_ce_loss / len(train_loader),
            'kd': total_kd_loss / len(train_loader),
            'acc': 100.0 * correct / total
        }
    
    def _get_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from model."""
        # Handle different model architectures
        if hasattr(model, 'base'):
            features = model.base(x)
        elif hasattr(model, 'features'):
            features = model.features(x)
        else:
            # Forward through all layers except classifier
            for name, module in model.named_children():
                if 'classifier' not in name and 'head' not in name:
                    x = module(x)
            features = x
        
        return features
    
    def _get_classifier_output(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """Get classifier output from features."""
        if hasattr(model, 'classifier'):
            return model.classifier(features)
        elif hasattr(model, 'head'):
            return model.head(features)
        else:
            # Find classifier module
            for name, module in model.named_children():
                if 'classifier' in name or 'head' in name:
                    return module(features)
        return features
    
    def _compute_ssl_loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute SSL loss (SimSiam style)."""
        # Negative cosine similarity
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    
    def _distillation_loss(self, student_logits: torch.Tensor, 
                          teacher_logits: torch.Tensor,
                          temperature: float = 3.0) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        student_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    def _update_memory(self, dataset: Dataset, task_id: int):
        """Update memory buffer with reservoir sampling."""
        # Get all data from current task
        data_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
        all_data = []
        all_labels = []
        
        for batch in data_loader:
            # Safe batch unpacking
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]  # Take first two elements
            else:
                x, y = batch  # Direct unpacking
            all_data.append(x)
            all_labels.append(y)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Reservoir sampling
        n_samples = min(self.memory_size, len(all_data))
        indices = torch.randperm(len(all_data))[:n_samples]
        
        self.memory_per_task[task_id] = (
            all_data[indices].cpu(),
            all_labels[indices].cpu()
        )
        
        print(f"    → Stored {n_samples} samples from task {task_id} in memory")
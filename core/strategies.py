"""Base strategy framework for continual learning experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from collections import deque
import random
import copy
import numpy as np
from torchvision import transforms


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
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            # Handle multi-head models with our clean implementation
            scenario_type = self.config.get('scenario_type', 'class_incremental')
            if scenario_type == 'task_incremental' and hasattr(self, 'current_task_id'):
                # Use our clean forward_single_task method
                if hasattr(self.model, 'forward_single_task'):
                    outputs = self.model.forward_single_task(x, self.current_task_id)
                else:
                    outputs = self.model(x)
            else:
                # Class incremental learning
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
            acc = self._evaluate_dataset(test_ds, task_id=j)
            
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
    
    def _evaluate_dataset(self, test_ds: Dataset, task_id: Optional[int] = None) -> float:
        """Evaluate accuracy on a dataset."""
        import numpy as np
        from collections import Counter
        self.model.eval()
        correct = total = 0
        all_preds = []
        all_labels = []
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        # Check if this is a multi-head model
        scenario_type = self.config.get('scenario_type', 'class_incremental')
        is_multi_head = scenario_type == 'task_incremental'
        
        with torch.inference_mode():
            for batch in loader:
                # Safe batch unpacking
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]  # Take first two elements
                else:
                    x, y = batch  # Direct unpacking
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass with our clean implementation
                if is_multi_head and task_id is not None:
                    # Multi-head model: use specific task head
                    if hasattr(self.model, 'forward_single_task'):
                        outputs = self.model.forward_single_task(x, task_id)
                    else:
                        outputs = self.model(x)
                else:
                    # Single head model
                    outputs = self.model(x)
                
                pred = outputs.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        acc = 100.0 * correct / total
        # Debug prints (keep existing debug code)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        print(f"[DEBUG] Evaluation: Accuracy={acc:.2f}%")
        print(f"[DEBUG] Predicted class histogram: {Counter(all_preds)}")
        print(f"[DEBUG] True label histogram: {Counter(all_labels)}")
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_labels, all_preds)
            print(f"[DEBUG] Confusion matrix:\n{cm}")
        except ImportError:
            print("[DEBUG] sklearn not installed, skipping confusion matrix.")
        return acc


class NaiveStrategy(BaseStrategy):
    """Naive strategy: just train on current task."""
    
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train on current task only."""
        self.current_task_id = task_id
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
        self.current_task_id = task_id
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
        self.current_task_id = task_id
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



class LARS(torch.optim.Optimizer):
    """LARS optimizer for large batch training."""
    
    def __init__(self, params, lr=1.0, momentum=0.9, dampening=0, weight_decay=1e-6, 
                 nesterov=False, trust_coefficient=0.001, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust coefficient: {trust_coefficient}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov, 
                       trust_coefficient=trust_coefficient, eps=eps)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coefficient = group['trust_coefficient']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)

                # Compute adaptive learning rate
                if param_norm != 0 and grad_norm != 0:
                    adaptive_lr = trust_coefficient * param_norm / (grad_norm + eps)
                    adaptive_lr = min(adaptive_lr, 1.0)  # Cap at 1.0
                else:
                    adaptive_lr = 1.0

                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=adaptive_lr)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                p.data.add_(grad, alpha=-group['lr'])

        return loss
    
class KaizenStrategy(BaseStrategy):
    """Kaizen strategy implementing the exact approach from the paper."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(model, config, device)
        self.previous_model = None
        
        # Memory configuration - exactly as in paper
        self.memory_size_percent = config.get('memory_size_percent', 1)  # 1% as in paper
        self.memory_buffer = {}  # {task_id: [(x, y), ...]}
        
        # SSL method and loss weights - as specified in paper
        self.ssl_method = config.get('ssl_method', 'simclr')
        self.kd_classifier_weight = config.get('kd_classifier_weight', 2.0)  # Paper uses 2.0
        
        # Detect scenario type
        self.scenario_type = config.get('scenario_type', 'class_incremental')
        self.is_multi_head = self.scenario_type == 'task_incremental'
        
        # Build SSL heads
        self._build_ssl_heads()
        
    def _build_ssl_heads(self):
        """Build SSL projection heads exactly as in paper."""
        feat_dim = self._detect_feature_dim()
        print(f"[DEBUG] Detected feature dimension: {feat_dim}")
        
        # Projector for current task SSL (h^T in paper)
        self.projector_ssl = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)
        
        if self.ssl_method in ['byol']:
            self.predictor_ssl = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            ).to(self.device)
            print(f"[DEBUG] Built BYOL predictor: 256 -> 128 -> 256")
        else:
            self.predictor_ssl = None
        
        print(f"[DEBUG] Built SSL projector: {feat_dim} -> 512 -> 256")
        
        # Momentum feature extractor for SSL methods that need it
        if self.ssl_method in ['byol', 'moco', 'mocov2+']:
            self.momentum_feature_extractor = copy.deepcopy(self._get_feature_extractor())
            self.momentum_feature_extractor.eval()
            for param in self.momentum_feature_extractor.parameters():
                param.requires_grad = False
                
            # ADD: Momentum projector for methods that need it
            self.momentum_projector = copy.deepcopy(self.projector_ssl)
            self.momentum_projector.eval()
            for param in self.momentum_projector.parameters():
                param.requires_grad = False
        else:
            self.momentum_feature_extractor = None
            self.momentum_projector = None
    
    def _detect_feature_dim(self):
        """Detect feature dimension by forward pass."""
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
            features = self._extract_features(dummy_input)
            feat_dim = features.shape[-1]
        self.model.train()
        return feat_dim

    def _get_feature_extractor(self):
        """Get the feature extractor module."""
        # Handle ViT64SingleHead and ViT64MultiHead models
        if hasattr(self.model, 'vit') and (hasattr(self.model, 'heads') or hasattr(self.model, 'head')):
            return self.model.vit
        elif hasattr(self.model, 'base'):
            return self.model.base
        elif hasattr(self.model, 'features'):
            return self.model.features
        else:
            # Create a feature extractor by excluding classifier
            modules = []
            for name, module in self.model.named_children():
                if not any(keyword in name.lower() for keyword in ['classifier', 'head', 'fc']):
                    modules.append(module)
            return nn.Sequential(*modules)
    
    def _get_classifier_output(self, features, task_id=None):
        """Get classifier output respecting multi-head configuration."""
        if self.is_multi_head and task_id is not None and hasattr(self.model, 'forward_single_task'):
            # For multi-head, we need to reconstruct the input
            # This is a limitation of the interface, so we'll use direct head access
            if hasattr(self.model, 'heads') and hasattr(self.model.heads, str(task_id)):
                head = getattr(self.model.heads, str(task_id))
                return head(features)
        
        # Single head or fallback
        if hasattr(self.model, 'head'):
            return self.model.head(features)
        elif hasattr(self.model, 'classifier'):
            return self.model.classifier(features)
        else:
            for name, module in self.model.named_modules():
                if any(keyword in name.lower() for keyword in ['classifier', 'head']) and isinstance(module, nn.Linear):
                    return module(features)
            raise AttributeError("No classifier found in model")
    
    def train_task(self, train_dataset: Dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train with Kaizen strategy exactly as described in paper."""
        self.current_task_id = task_id
        
        # Store previous model for knowledge distillation (step 1 in paper)
        if task_id > 0 and self.model is not None:
            self.previous_model = copy.deepcopy(self.model)
            self.previous_model.eval()
            for param in self.previous_model.parameters():
                param.requires_grad = False

        # Create dataloader with replay integration (step 2) 
        train_loader = self._create_replay_dataloader(train_dataset, task_id)
        
        # Setup optimizers as in paper
        optimizer = self._build_optimizer()
        
        # Training loop
        epochs = self.config.get('epochs', 50)
        analysis_freq = self.config.get('analysis_freq', 10)
        trajectory_data = {}
        
        for epoch in range(epochs):
            loss_dict = self._train_epoch_kaizen(train_loader, optimizer, task_id)
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: "
                      f"Total={loss_dict['total']:.3f}, "
                      f"SSL_KD={loss_dict['ssl_kd']:.3f}, "
                      f"CE_KD={loss_dict['ce_kd']:.3f}, "
                      f"CE_CT={loss_dict['ce_ct']:.3f}, "
                      f"SSL_CT={loss_dict['ssl_ct']:.3f}")
        
        # Update memory buffer AFTER training is complete (step 3)
        if self.memory_size_percent > 0:
            self._update_memory_buffer(train_dataset, task_id)
        
        self.current_task = task_id + 1
        return {'trajectory': trajectory_data, 'loss_components': loss_dict}
    
    def _update_memory_buffer(self, dataset: Dataset, task_id: int):
        """Update memory buffer using reservoir sampling as in paper."""
        # Calculate memory size as percentage of dataset
        total_samples = len(dataset)
        memory_size = max(1, int(total_samples * self.memory_size_percent / 100))
        
        # Get all samples from current task
        all_samples = []
        data_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            
            # Store individual samples
            for i in range(x.size(0)):
                all_samples.append((x[i].cpu(), y[i].cpu()))
        
        # Reservoir sampling
        if len(all_samples) <= memory_size:
            selected_samples = all_samples
        else:
            selected_samples = random.sample(all_samples, memory_size)
        
        self.memory_buffer[task_id] = selected_samples
        
        print(f"    → Stored {len(selected_samples)} samples from task {task_id} in memory "
              f"({self.memory_size_percent}% of {total_samples})")
    
    def _create_replay_dataloader(self, current_dataset: Dataset, task_id: int) -> DataLoader:
        """Create dataloader that mixes current task with replay samples as in paper."""
        
        class ReplayMixedDataset(Dataset):
            def __init__(self, current_data, memory_buffer, task_id):
                self.current_data = current_data
                self.memory_buffer = memory_buffer
                self.task_id = task_id
                
                # Flatten all memory samples (only for previous tasks)
                self.memory_samples = []
                for tid, samples in memory_buffer.items():
                    if tid < self.task_id:  # Only include previous tasks
                        self.memory_samples.extend(samples)
                
                if self.memory_samples:
                    print(f"    → Created mixed dataset: {len(current_data)} current + "
                          f"{len(self.memory_samples)} replay samples")
                else:
                    print(f"    → Created dataset: {len(current_data)} current samples (no replay)")
            
            def __len__(self):
                return len(self.current_data)
            
            def __getitem__(self, idx):
                # Get current task sample
                item = self.current_data[idx]
                if isinstance(item, (tuple, list)):
                    curr_x, curr_y = item[0], item[1]
                else:
                    curr_x, curr_y = item, 0
                
                # Ensure curr_y is a tensor
                if not isinstance(curr_y, torch.Tensor):
                    curr_y = torch.tensor(curr_y, dtype=torch.long)
                
                # Decide whether to include replay sample (paper doesn't specify exact ratio)
                # We'll include replay with 50% probability when available
                has_replay = len(self.memory_samples) > 0 and random.random() < 0.5
                
                if has_replay:
                    # Get random replay sample
                    replay_idx = random.randint(0, len(self.memory_samples) - 1)
                    replay_x, replay_y = self.memory_samples[replay_idx]
                    
                    # Convert to proper tensor format
                    if isinstance(replay_x, torch.Tensor):
                        if replay_x.dim() == 3:  # Add batch dimension if missing
                            replay_x = replay_x.unsqueeze(0)
                    replay_x = replay_x.squeeze(0) if replay_x.dim() == 4 else replay_x
                    
                    # Ensure replay_y is a tensor
                    if not isinstance(replay_y, torch.Tensor):
                        replay_y = torch.tensor(replay_y, dtype=torch.long)
                    
                    # Ensure 3 channels
                    if replay_x.shape[0] == 1:
                        replay_x = replay_x.repeat(3, 1, 1)
                    
                    return {
                        'current_x': curr_x,
                        'current_y': curr_y,
                        'replay_x': replay_x,
                        'replay_y': replay_y,
                        'has_replay': torch.tensor(1, dtype=torch.long)
                    }
                else:
                    return {
                        'current_x': curr_x,
                        'current_y': curr_y,
                        'replay_x': curr_x,  # Dummy
                        'replay_y': curr_y,  # Dummy
                        'has_replay': torch.tensor(0, dtype=torch.long)
                    }
        
        mixed_dataset = ReplayMixedDataset(current_dataset, self.memory_buffer, task_id)
        
        return DataLoader(
            mixed_dataset,
            batch_size=self.config.get('minibatch_size', 128),
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues with mixed data
            pin_memory=True
        )
    
    def _train_epoch_kaizen(self, train_loader: DataLoader, 
                           optimizer: torch.optim.Optimizer,
                           task_id: int) -> Dict[str, float]:
        """Training epoch implementing exact Kaizen algorithm from paper."""
        self.model.train()
        
        # Track individual loss components as in paper
        total_losses = {
            'ssl_kd': 0.0,    # L^KD_FE - Feature extractor knowledge distillation
            'ce_kd': 0.0,     # L^KD_C - Classifier knowledge distillation  
            'ce_ct': 0.0,     # L^CT_C - Current task classification
            'ssl_ct': 0.0,    # L^CT_FE - Current task SSL
            'total': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack mixed batch
            current_x = batch_data['current_x'].to(self.device)
            current_y = batch_data['current_y'].to(self.device)
            replay_x = batch_data['replay_x'].to(self.device)
            replay_y = batch_data['replay_y'].to(self.device)
            has_replay = batch_data['has_replay'].to(self.device)
            
            # Combine current and replay samples
            batch_size = current_x.size(0)
            replay_mask = has_replay.bool()
            
            # NICHT concatenate wenn kein replay
            if has_replay.any():
                all_x = torch.cat([current_x, replay_x], dim=0)  
                all_y = torch.cat([current_y, replay_y], dim=0)
            else:
                all_x = current_x  # NUR current samples!
                all_y = current_y
                        
            # Create augmented views for SSL (as in Algorithm 1)
            x1_aug = self._apply_ssl_augmentation(all_x)
            x2_aug = self._apply_ssl_augmentation(all_x)
            
            optimizer.zero_grad()
            
            # Forward pass through feature extractors
            z_o = self._extract_features(x1_aug)  # Current feature extractor (f_o)
            
            # Momentum/Target feature extractor (f_t) 
            if self.momentum_feature_extractor is not None:
                with torch.no_grad():
                    # Direct call - we know momentum_feature_extractor is a ViT model
                    z_t_features = self.momentum_feature_extractor.forward_features(x2_aug)
                    z_t = z_t_features[:, 0]  # Extract CLS token
            else:
                z_t = self._extract_features(x2_aug)
            
            # === Loss Component 1: SSL Current Task (L^CT_FE) ===
            p_t = self.projector_ssl(z_o)  # h^T(f_o(x1))
            
            # Project target features too for SSL comparison
            if self.momentum_feature_extractor is not None and self.momentum_projector is not None:
                # For BYOL/MoCo style - use momentum projector
                with torch.no_grad():
                    proj_target = self.momentum_projector(z_t)
                
                # For BYOL: apply predictor to online projection
                if self.ssl_method == 'byol' and self.predictor_ssl is not None:
                    pred_online = self.predictor_ssl(p_t)
                    ssl_ct_loss = self._compute_ssl_loss(pred_online, proj_target, self.ssl_method)
                else:
                    ssl_ct_loss = self._compute_ssl_loss(p_t, proj_target, self.ssl_method)
            else:
                # For SimCLR style - project both sides with same projector
                proj_target = self.projector_ssl(z_t)
                ssl_ct_loss = self._compute_ssl_loss(p_t, proj_target, self.ssl_method)
            
            # === Loss Component 2: Feature Extractor Knowledge Distillation (L^KD_FE) ===
            ssl_kd_loss = 0.0
            if self.previous_model is not None:
                with torch.no_grad():
                    z_p = self._extract_features_from_model(self.previous_model, x1_aug)  # f_{t-1}(x1)
                
                # Feature-level distillation (MSE loss in feature space)
                ssl_kd_loss = F.mse_loss(z_o, z_p)
            
            # === Loss Component 3: Current Task Classification (L^CT_C) ===
            # Important: Stop gradients to feature extractor as in paper
            logits_current = self._get_classifier_output(z_o.detach(), task_id)
            
            # Only compute CE loss for labeled samples (paper mentions label availability)
            ce_ct_loss = F.cross_entropy(logits_current, all_y)
            
            # === Loss Component 4: Classifier Knowledge Distillation (L^KD_C) ===
            ce_kd_loss = 0.0
            if self.previous_model is not None:
                with torch.no_grad():
                    prev_features = self._extract_features_from_model(self.previous_model, x1_aug)
                    prev_logits = self._get_classifier_output_from_model(self.previous_model, prev_features)
                
                # Knowledge distillation loss
                ce_kd_loss = F.cross_entropy(logits_current, F.softmax(prev_logits, dim=1))
            
            # === Total Loss (Equation 1 from paper) ===
            total_loss = (ssl_kd_loss + 
                         self.kd_classifier_weight * ce_kd_loss + 
                         ce_ct_loss + 
                         ssl_ct_loss)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update momentum feature extractor if using BYOL/MoCo
            if self.momentum_feature_extractor is not None:
                self._update_momentum_feature_extractor()
            
            # Track losses
            total_losses['ssl_kd'] += ssl_kd_loss.item() if isinstance(ssl_kd_loss, torch.Tensor) else 0
            total_losses['ce_kd'] += ce_kd_loss.item() if isinstance(ce_kd_loss, torch.Tensor) else 0
            total_losses['ce_ct'] += ce_ct_loss.item()
            total_losses['ssl_ct'] += ssl_ct_loss.item()
            total_losses['total'] += total_loss.item()
            
            num_batches += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def _apply_ssl_augmentation(self, x):
        """SSL augmentations for normalized data."""
        batch_size = x.size(0)
        device = x.device
        x_aug = x.clone()
        
        # Flip (works in any range)
        flip_mask = torch.rand(batch_size, device=device) > 0.5
        x_aug[flip_mask] = torch.flip(x_aug[flip_mask], dims=[3])
        
        # Brightness for normalized data (NO clamping to [0,1]!)
        bright_mask = torch.rand(batch_size, device=device) < 0.8
        if bright_mask.any():
            brightness = 1.0 + 0.3 * (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5)
            bright_expand = bright_mask.view(-1, 1, 1, 1).expand_as(x_aug)
            x_aug = torch.where(bright_expand, x_aug * brightness, x_aug)  # NO clamp!
        
        # Contrast (works with mean centering)  
        contrast_mask = torch.rand(batch_size, device=device) < 0.8
        if contrast_mask.any():
            contrast = 1.0 + 0.3 * (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5)
            mean_vals = x_aug.mean(dim=(2, 3), keepdim=True)
            contrasted = (x_aug - mean_vals) * contrast + mean_vals  # NO clamp!
            contrast_expand = contrast_mask.view(-1, 1, 1, 1).expand_as(x_aug)
            x_aug = torch.where(contrast_expand, contrasted, x_aug)
        
        return x_aug
    
    def _compute_ssl_loss(self, pred, target, method='simclr'):
        """Compute SSL loss based on method."""
        temperature = self.config.get('ssl_temperature', 0.1)
        if method == 'simclr':
            return self._simclr_loss(pred, target, temperature)
        elif method in ['byol', 'simsiam']:
            return self._byol_loss(pred, target)
        elif method in ['moco', 'mocov2+']:
            return self._moco_loss(pred, target, temperature)
        elif method == 'vicreg':
            return self._vicreg_loss(pred, target)
        else:
            return self._simclr_loss(pred, target, temperature)  # Default
    
    def _simclr_loss(self, z1, z2, temperature=0.1):
        """SimCLR NT-Xent loss with numerical stability fixes."""
        batch_size = z1.shape[0]
        
        # Check for degenerate cases
        if torch.allclose(z1, z2, atol=1e-6):
            # Add small noise to break symmetry
            z2 = z2 + torch.randn_like(z2) * 1e-4
        
        # Normalize with numerical stability
        z1 = F.normalize(z1, dim=1, eps=1e-8)
        z2 = F.normalize(z2, dim=1, eps=1e-8)
        
        # Concatenate
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create masks
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        # Positive pairs mask
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=self.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        
        # Compute loss
        pos_sim = similarity_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = similarity_matrix[mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    # Add this debug method to your KaizenStrategy class
    def debug_simclr_loss(self, z1, z2, temperature=0.1):
        """Debug SimCLR loss to find NaN source."""
        print(f"\n=== SimCLR Debug ===")
        print(f"Input z1 shape: {z1.shape}, range: [{z1.min():.3f}, {z1.max():.3f}]")
        print(f"Input z2 shape: {z2.shape}, range: [{z2.min():.3f}, {z2.max():.3f}]")
        
        # Check for NaN/Inf in inputs
        if torch.isnan(z1).any():
            print("❌ NaN detected in z1!")
            return torch.tensor(float('nan'))
        if torch.isnan(z2).any():
            print("❌ NaN detected in z2!")
            return torch.tensor(float('nan'))
        
        batch_size = z1.shape[0]
        
        # Normalize with debug
        z1_norm = F.normalize(z1, dim=1, eps=1e-8)
        z2_norm = F.normalize(z2, dim=1, eps=1e-8)
        
        print(f"After normalize z1: [{z1_norm.min():.3f}, {z1_norm.max():.3f}]")
        print(f"After normalize z2: [{z2_norm.min():.3f}, {z2_norm.max():.3f}]")
        
        # Check norms
        z1_norms = z1.norm(dim=1)
        z2_norms = z2.norm(dim=1)
        print(f"Original norms z1: [{z1_norms.min():.3f}, {z1_norms.max():.3f}]")
        print(f"Original norms z2: [{z2_norms.min():.3f}, {z2_norms.max():.3f}]")
        
        if z1_norms.min() < 1e-6 or z2_norms.min() < 1e-6:
            print("❌ Very small norms detected - will cause NaN in normalization!")
        
        # Concatenate
        representations = torch.cat([z1_norm, z2_norm], dim=0)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        print(f"Similarity matrix range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # Apply temperature
        logits_raw = similarity_matrix / temperature
        print(f"After temperature ({temperature}): [{logits_raw.min():.3f}, {logits_raw.max():.3f}]")
        
        if logits_raw.max() > 20:
            print("❌ Logits too large - will cause overflow in softmax!")
        
        # Create masks
        mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z1.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        
        # Get positive and negative similarities
        pos_sim = logits_raw[pos_mask].view(2 * batch_size, 1)
        neg_sim = logits_raw[mask].view(2 * batch_size, -1)
        
        print(f"Positive sims: [{pos_sim.min():.3f}, {pos_sim.max():.3f}]")
        print(f"Negative sims: [{neg_sim.min():.3f}, {neg_sim.max():.3f}]")
        
        # Final logits
        final_logits = torch.cat([pos_sim, neg_sim], dim=1)
        print(f"Final logits: [{final_logits.min():.3f}, {final_logits.max():.3f}]")
        
        # Labels
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
        
        # Compute loss
        loss = F.cross_entropy(final_logits, labels)
        print(f"Final loss: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("❌ NaN loss detected!")
        
        return loss
    
    def _byol_loss(self, p, z):
        """BYOL loss (negative cosine similarity)."""
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()
    
    def _moco_loss(self, q, k, temperature=0.07):
        """MoCo loss."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # For simplicity, use current batch as negatives
        l_neg = torch.einsum('nc,ck->nk', [q, k.T])
        
        logits = torch.cat([l_pos, l_neg], dim=1) / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        
        return F.cross_entropy(logits, labels)
    
    def _vicreg_loss(self, x, y):
        """VICReg loss (simplified)."""
        # Variance
        std_x = torch.sqrt(x.var(dim=0) + 1e-04)
        std_y = torch.sqrt(y.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
        
        # Invariance  
        inv_loss = F.mse_loss(x, y)
        
        # Covariance
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        
        cov_loss = (cov_x.fill_diagonal_(0).pow(2).sum() + 
                   cov_y.fill_diagonal_(0).pow(2).sum()) / x.shape[1]
        
        return 25 * inv_loss + 25 * std_loss + cov_loss
    
    def _update_momentum_feature_extractor(self, momentum=0.999):
        """Update momentum feature extractor and projector for BYOL/MoCo."""
        if self.momentum_feature_extractor is None:
            return
            
        # Update momentum feature extractor
        current_extractor = self._get_feature_extractor()
        for param_q, param_k in zip(current_extractor.parameters(), 
                                self.momentum_feature_extractor.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
        # Update momentum projector 
        if self.momentum_projector is not None:
            for param_q, param_k in zip(self.projector_ssl.parameters(), 
                                    self.momentum_projector.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
    def _extract_features(self, x):
        """Extract features from current model."""
        # Handle ViT64MultiHead model specifically  
        if hasattr(self.model, 'vit') and hasattr(self.model, 'heads'):
            # This is our ViT64MultiHead model
            features = self.model.vit.forward_features(x)  # (batch, num_tokens, embed_dim)
            cls_token = features[:, 0]  # Extract CLS token (batch, embed_dim)
            return cls_token
        elif hasattr(self.model, 'vit'):
            # Handle other ViT architectures
            features = self.model.vit.forward_features(x)
            if len(features.shape) == 3:  # (batch, num_tokens, embed_dim)
                return features[:, 0]  # Extract CLS token
            return features
        elif hasattr(self.model, 'base'):
            return self.model.base(x)
        elif hasattr(self.model, 'features'):
            return self.model.features(x)
        else:
            features = x
            for name, module in self.model.named_children():
                if not any(keyword in name.lower() for keyword in ['classifier', 'head', 'fc']):
                    features = module(features)
            return features

    def _extract_features_from_model(self, model, x):
        """Extract features from a specific model."""
        # Handle ViT64MultiHead model specifically
        if hasattr(model, 'vit') and hasattr(model, 'heads'):
            # This is our ViT64MultiHead model
            features = model.vit.forward_features(x)  # (batch, num_tokens, embed_dim)
            cls_token = features[:, 0]  # Extract CLS token (batch, embed_dim)
            return cls_token
        elif hasattr(model, 'forward_features'):  # THIS IS THE FIX!
            # This is a pure ViT model (like momentum_feature_extractor)
            features = model.forward_features(x)
            cls_token = features[:, 0]  # Extract CLS token
            return cls_token
        elif hasattr(model, 'vit'):
            # Handle other ViT architectures
            features = model.vit.forward_features(x)
            if len(features.shape) == 3:  # (batch, num_tokens, embed_dim)
                return features[:, 0]  # Extract CLS token
            return features
        elif hasattr(model, 'base'):
            return model.base(x)
        elif hasattr(model, 'features'):
            return model.features(x)
        else:
            features = x
            for name, module in model.named_children():
                if not any(keyword in name.lower() for keyword in ['classifier', 'head', 'fc']):
                    features = module(features)
            return features
    
    def _get_classifier_output_from_model(self, model, features):
        """Get classifier output from specific model."""
        if hasattr(model, 'head'):
            return model.head(features)
        elif hasattr(model, 'classifier'):
            return model.classifier(features)
        else:
            for name, module in model.named_modules():
                if any(keyword in name.lower() for keyword in ['classifier', 'head']) and isinstance(module, nn.Linear):
                    return module(features)
            raise AttributeError("No classifier found")
    
    def _build_optimizer(self):
        """Build optimizer for all trainable parameters."""
        # Collect all trainable parameters
        params = list(self.model.parameters()) + list(self.projector_ssl.parameters())
        
        lr = self.config.get('lr', 0.00001)
        optimizer_name = self.config.get('optimizer', 'lars')
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == 'lars':
            # LARS optimizer implementation for large batch SSL training
            return LARS(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
        else:
            return torch.optim.Adam(params, lr=lr)


# Configuration helper function
def create_kaizen_config(base_config):
    """Create Kaizen configuration matching paper settings."""
    kaizen_config = {
        **base_config,
        'strategy_name': 'Kaizen',
        'ssl_method': 'byol',  # Can be 'simclr', 'byol', 'moco', 'mocov2+', 'vicreg'
        'memory_size_percent': 1,  # 1% as in paper
        'kd_classifier_weight': 2.0,  # Paper uses 2.0 for L^KD_C
        'epochs': 500,  # Paper uses 500 for CIFAR-100 5 tasks
        'minibatch_size': 256,  # Paper uses 256 for CIFAR-100
        'lr': 0.001,
        'optimizer': 'lars',
    }
    return kaizen_config



class ReLUDown(nn.Module):
    """ReLUDown activation function from RDBP paper.
    
    Formula: f(x) = max(0, x) - max(0, -x + d), where d < 0
    This preserves plasticity by preventing neuron dormancy.
    """
    
    def __init__(self, hinge_point: float = -3.0):
        super().__init__()
        self.hinge_point = hinge_point
        assert hinge_point < 0, "Hinge point must be negative"
    
    def forward(self, x):
        return F.relu(x) - F.relu(-x + self.hinge_point)


class RDBPStrategy(BaseStrategy):
    """RDBP Strategy implementing ReLUDown + Decreasing Backpropagation.
    
    This strategy balances plasticity and stability by:
    1. Using ReLUDown activation to maintain plasticity
    2. Applying decreasing backpropagation to protect early layers
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        super().__init__(model, config, device)
        
        # RDBP hyperparameters from paper
        self.hinge_point = config.get('hinge_point', -3.0)
        self.decrease_factor = config.get('decrease_factor', 0.15)
        self.speed_factor = config.get('speed_factor', 1.005)
        
        # Replace ReLU activations with ReLUDown
        self._replace_activations()
        
        # Store layer information for gradient scaling
        self._setup_layer_info()
        
        # Track current task for gradient scaling
        self.current_task_num = 0
        
    def _replace_activations(self):
        """Replace ReLU activations in the model with ReLUDown."""
        def replace_relu_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, ReLUDown(self.hinge_point))
                else:
                    replace_relu_recursive(child)
        
        replace_relu_recursive(self.model)
        print(f"    → Replaced ReLU activations with ReLUDown (hinge_point={self.hinge_point})")
    
    def _setup_layer_info(self):
        """Setup layer information for gradient scaling."""
        self.layer_modules = []
        self.layer_names = []
        
        # Collect all trainable layers (excluding activations, pooling, etc.)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layer_modules.append(module)
                self.layer_names.append(name)
        
        print(f"    → Found {len(self.layer_modules)} trainable layers for DBP")
        for i, name in enumerate(self.layer_names):
            print(f"      Layer {i}: {name}")
    
    def _compute_gradient_factor(self, layer_idx: int, task_num: int) -> float:
        """Compute gradient scaling factor for Decreasing Backpropagation.
        
        Formula from paper:
        bpdecrease(n, l, f, a) = bpstandard * (1 - (l * f) + (l * f) * a^(-n))
        
        Note: We invert layer_idx so that earlier layers (closer to input) get 
        higher l values and thus more protection, while later layers (closer to output)
        get lower l values and thus less protection.
        
        Args:
            layer_idx: Layer index from _setup_layer_info (0 = first encountered layer)
            task_num: Current task number (0-indexed)
            
        Returns:
            Gradient scaling factor
        """
        # Invert layer numbering: earlier layers get higher l values
        l = len(self.layer_modules) - 1 - layer_idx
        n = task_num
        f = self.decrease_factor
        a = self.speed_factor
        
        # Apply formula for all tasks - it naturally handles task 0 correctly
        factor = 1 - (l * f) + (l * f) * (a ** (-n))
        return max(factor, 0.0)  # Ensure non-negative
    
    def _apply_decreasing_backprop(self, task_num: int):
        """Apply gradient scaling to implement Decreasing Backpropagation."""
        for layer_idx, module in enumerate(self.layer_modules):
            gradient_factor = self._compute_gradient_factor(layer_idx, task_num)
            
            # Scale gradients for all parameters in this layer
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.data *= gradient_factor
    
    def train_task(self, train_dataset, task_id: int, 
                   analyzer: Optional[Any] = None,
                   all_datasets: Optional[Dict[str, Any]] = None, 
                   order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train on a single task using RDBP strategy."""
        self.current_task_id = task_id
        self.current_task_num = task_id
        
        print(f"    → Training task {task_id} with RDBP strategy")
        print(f"      Hinge point: {self.hinge_point}")
        print(f"      Decrease factor: {self.decrease_factor}")
        print(f"      Speed factor: {self.speed_factor}")
        
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
        gradient_factors = self._get_current_gradient_factors()
        
        print(f"      Current gradient factors by layer:")
        for i, (name, factor) in enumerate(zip(self.layer_names, gradient_factors)):
            # Show both the storage index and the effective l value used in formula
            l_value = len(self.layer_modules) - 1 - i
            position = "input" if i == 0 else "output" if i == len(self.layer_modules)-1 else "middle"
            print(f"        Layer {i} ({name}, l={l_value}, {position}): {factor:.3f}")
        
        for epoch in range(epochs):
            loss, acc = self._train_epoch_rdbp(train_loader, optimizer, criterion, task_id)
            
            # Periodic analysis
            if analyzer and (epoch + 1) % analysis_freq == 0:
                epoch_analysis = self._analyze_all_tasks(
                    analyzer, all_datasets, order, task_id, epoch + 1
                )
                trajectory_data[f'epoch_{epoch + 1}'] = epoch_analysis
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}, Acc={acc:.1f}%")
        
        self.current_task = task_id + 1
        return {
            'trajectory': trajectory_data,
            'gradient_factors': gradient_factors,
            'rdbp_config': {
                'hinge_point': self.hinge_point,
                'decrease_factor': self.decrease_factor,
                'speed_factor': self.speed_factor
            }
        }
    
    def _train_epoch_rdbp(self, train_loader: DataLoader, 
                         optimizer: torch.optim.Optimizer,
                         criterion: nn.Module,
                         task_id: int) -> Tuple[float, float]:
        """Training epoch with RDBP modifications."""
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        
        for batch in train_loader:
            # Safe batch unpacking
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass (using ReLUDown activations)
            scenario_type = self.config.get('scenario_type', 'class_incremental')
            if scenario_type == 'task_incremental' and hasattr(self, 'current_task_id'):
                if hasattr(self.model, 'forward_single_task'):
                    outputs = self.model.forward_single_task(x, self.current_task_id)
                else:
                    outputs = self.model(x)
            else:
                outputs = self.model(x)
            
            loss = criterion(outputs, y)
            loss.backward()
            
            # Apply Decreasing Backpropagation
            self._apply_decreasing_backprop(task_id)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        accuracy = 100.0 * correct / total
        return total_loss / len(train_loader), accuracy
    
    def _get_current_gradient_factors(self) -> List[float]:
        """Get current gradient factors for all layers."""
        factors = []
        for layer_idx in range(len(self.layer_modules)):
            factor = self._compute_gradient_factor(layer_idx, self.current_task_num)
            factors.append(factor)
        return factors


def create_rdbp_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create RDBP configuration with paper defaults."""
    rdbp_config = {
        **base_config,
        'strategy_name': 'RDBP',
        
        # ReLUDown parameters
        'hinge_point': -3.0,  # d parameter from paper
        
        # Decreasing Backpropagation parameters  
        'decrease_factor': 0.15,  # f parameter from paper
        'speed_factor': 1.005,    # a parameter from paper
        
        # Training parameters from paper
        'epochs': 50,  # Paper uses 250 epochs for Continual ImageNet
        'minibatch_size': 128,  # Paper uses 100 for Continual ImageNet
        'lr': 0.0003,
        'optimizer': 'adam',
    }
    return rdbp_config

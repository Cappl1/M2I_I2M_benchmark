from avalanche.benchmarks.datasets import Omniglot
from avalanche.benchmarks.utils import AvalancheTensorDataset, make_tensor_classification_dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, Compose, ToTensor, RandomCrop, Normalize
import torch

from paths import DATA_PATH
from scenarios.utils import transform_from_gray_to_rgb, load_dataset

transform_not_resize = Compose([
    RandomCrop(105, padding=4),
    ToTensor(),
    transform_from_gray_to_rgb(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])

transform_with_resize = Compose([
    ToTensor(),
    Resize((64, 64)),
    transform_from_gray_to_rgb(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])


def _load_omniglot(transform_func, balanced: bool = False, number_of_samples_per_class=None):
    def create_avalanche_dataset(X, y, transform=None):
        from avalanche.benchmarks.utils import make_tensor_classification_dataset
        
        if isinstance(y, list):
            y = torch.tensor(y)
            
        # For PIL Images, pass them directly and let transforms handle conversion
        if isinstance(X, list):
            # Create dataset with raw PIL images, transforms will handle tensor conversion
            from avalanche.benchmarks.utils import AvalancheDataset
            from torch.utils.data import Dataset
            
            # Create a simple dataset wrapper that keeps PIL images
            class PILDataset(Dataset):
                def __init__(self, images, labels):
                    self.images = images
                    self.labels = labels
                    self.targets = labels  # Add targets attribute for compatibility
                
                def __len__(self):
                    return len(self.images)
                
                def __getitem__(self, idx):
                    img = self.images[idx]
                    label = self.labels[idx] if isinstance(self.labels, list) else self.labels[idx].item()
                    return img, label
            
            pil_dataset = PILDataset(X, y)
            return AvalancheDataset(pil_dataset, transform_groups={'train': transform, 'eval': transform})
        else:
            return make_tensor_classification_dataset(X, y, transform=transform)
    
    dataset = Omniglot(root=f'{DATA_PATH}/data', download=True, train=True)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.targets, train_size=0.8, test_size=0.2)
    train = load_dataset(lambda transform: create_avalanche_dataset(X_train, y_train, transform),
                         transform=transform_func, balanced=balanced,
                         number_of_samples_per_class=number_of_samples_per_class)
    test = load_dataset(lambda transform: create_avalanche_dataset(X_test, y_test, transform),
                        transform=transform_func, balanced=balanced,
                        number_of_samples_per_class=number_of_samples_per_class)
    return train, test


def load_omniglot():
    return _load_omniglot(transform_not_resize)


def load_resized_omniglot():
    return _load_omniglot(transform_with_resize)


if __name__ == '__main__':
    _load_omniglot(transform_not_resize) 